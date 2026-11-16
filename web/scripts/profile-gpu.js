// Injected into the inference worker through CDP; never shipped with the application.
(() => {
  const encoders = new WeakMap();
  const buffers = new WeakMap();
  const pipelines = new WeakMap();
  const pending = new Set();
  const totals = new Map();
  const createPipeline = GPUDevice.prototype.createComputePipeline;
  GPUDevice.prototype.createComputePipeline = function (descriptor) {
    const pipeline = createPipeline.call(this, descriptor);
    pipelines.set(
      pipeline,
      descriptor.label || descriptor.compute.entryPoint || "compute",
    );
    return pipeline;
  };
  const createPipelineAsync = GPUDevice.prototype.createComputePipelineAsync;
  GPUDevice.prototype.createComputePipelineAsync = async function (descriptor) {
    const pipeline = await createPipelineAsync.call(this, descriptor);
    pipelines.set(
      pipeline,
      descriptor.label || descriptor.compute.entryPoint || "compute",
    );
    return pipeline;
  };
  const createEncoder = GPUDevice.prototype.createCommandEncoder;
  GPUDevice.prototype.createCommandEncoder = function (descriptor) {
    if (!this.features.has("timestamp-query"))
      throw new Error(
        "GPU profiling requires timestamp-query on the inference device",
      );
    const encoder = createEncoder.call(this, descriptor);
    encoders.set(encoder, { device: this, passes: [] });
    return encoder;
  };
  const begin = GPUCommandEncoder.prototype.beginComputePass;
  GPUCommandEncoder.prototype.beginComputePass = function (descriptor = {}) {
    const state = encoders.get(this);
    if (descriptor.timestampWrites)
      throw new Error("GPU profiling is already active");
    // WebGPU caps one query set at 8192 queries; every measured pass uses two.
    if (state.passes.length >= 4096)
      throw new Error("Too many compute passes in one encoder");
    state.queries ??= state.device.createQuerySet({
      type: "timestamp",
      count: 8192,
    });
    const index = state.passes.length * 2;
    const pass = begin.call(this, {
      ...descriptor,
      timestampWrites: {
        querySet: state.queries,
        beginningOfPassWriteIndex: index,
        endOfPassWriteIndex: index + 1,
      },
    });
    const names = [];
    let kernel;
    const setPipeline = pass.setPipeline.bind(pass);
    pass.setPipeline = (pipeline) => {
      kernel = pipelines.get(pipeline) || pipeline.label;
      setPipeline(pipeline);
    };
    const dispatch = pass.dispatchWorkgroups.bind(pass);
    pass.dispatchWorkgroups = (...dimensions) => {
      names.push({ kernel, dimensions });
      dispatch(...dimensions);
    };
    state.passes.push(names);
    return pass;
  };
  const finish = GPUCommandEncoder.prototype.finish;
  GPUCommandEncoder.prototype.finish = function (descriptor) {
    const state = encoders.get(this);
    if (state.passes.length) {
      const bytes = state.passes.length * 2 * BigUint64Array.BYTES_PER_ELEMENT;
      state.resolved = state.device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      state.readback = state.device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      this.resolveQuerySet(
        state.queries,
        0,
        state.passes.length * 2,
        state.resolved,
        0,
      );
      this.copyBufferToBuffer(state.resolved, 0, state.readback, 0, bytes);
    }
    const buffer = finish.call(this, descriptor);
    if (state.passes.length) buffers.set(buffer, state);
    return buffer;
  };
  const submit = GPUQueue.prototype.submit;
  GPUQueue.prototype.submit = function (commands) {
    const list = Array.from(commands);
    submit.call(this, list);
    for (const command of list) {
      const state = buffers.get(command);
      if (!state) continue;
      buffers.delete(command);
      const task = (async () => {
        try {
          await state.readback.mapAsync(GPUMapMode.READ);
          const times = new BigUint64Array(state.readback.getMappedRange());
          state.passes.forEach((dispatches, i) => {
            const key = JSON.stringify(dispatches);
            const total = totals.get(key) || {
              dispatches,
              passes: 0,
              gpuMs: 0,
            };
            total.passes++;
            total.gpuMs += Number(times[i * 2 + 1] - times[i * 2]) / 1e6;
            totals.set(key, total);
          });
        } finally {
          state.readback.destroy();
          state.resolved.destroy();
          state.queries.destroy();
        }
      })();
      pending.add(task);
      task.finally(() => pending.delete(task));
    }
  };
  self.speechGpuProfileTake = async () => {
    await Promise.all(Array.from(pending));
    const result = Array.from(totals.values()).sort(
      (a, b) => b.gpuMs - a.gpuMs,
    );
    totals.clear();
    return result;
  };
})();
