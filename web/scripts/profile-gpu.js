// Injected into the inference worker through CDP; never shipped with the application.
(() => {
  const encoders = new WeakMap();
  const buffers = new WeakMap();
  const pipelines = new WeakMap();
  const pending = new Set();
  const failures = [];
  const totals = new Map();
  let testDevice;
  let phase = "preparation";
  const postMessage = self.postMessage;
  self.postMessage = function (event, ...rest) {
    if (["ready", "conditioned", "done", "error"].includes(event.kind))
      console.info("Speech diagnostic phase", performance.now(), event.kind);
    if (event.kind === "ready") phase = "conditioning";
    if (event.kind === "conditioned") phase = "inference";
    return postMessage.call(this, event, ...rest);
  };
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
    const encoder = createEncoder.call(this, descriptor);
    // Capture steady-state generation, excluding model upload and conditioning.
    if (phase !== "inference" && this !== testDevice) return encoder;
    if (!this.features.has("timestamp-query"))
      throw new Error(
        "GPU profiling requires timestamp-query on the inference device",
      );
    encoders.set(encoder, { device: this, passes: [] });
    return encoder;
  };
  const begin = GPUCommandEncoder.prototype.beginComputePass;
  GPUCommandEncoder.prototype.beginComputePass = function (descriptor = {}) {
    const state = encoders.get(this);
    // Encoders created before CDP injection belong to startup outside this capture.
    if (!state) return begin.call(this, descriptor);
    if (descriptor.timestampWrites)
      throw new Error("GPU profiling is already active");
    const queries = state.device.createQuerySet({
      type: "timestamp",
      count: 2,
    });
    const pass = begin.call(this, {
      ...descriptor,
      timestampWrites: {
        querySet: queries,
        beginningOfPassWriteIndex: 0,
        endOfPassWriteIndex: 1,
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
    state.passes.push({ dispatches: names, queries, phase });
    return pass;
  };
  const finish = GPUCommandEncoder.prototype.finish;
  GPUCommandEncoder.prototype.finish = function (descriptor) {
    const state = encoders.get(this);
    if (!state) return finish.call(this, descriptor);
    if (state.passes.length) {
      // WebGPU requires query resolve destination offsets aligned to 256 bytes.
      const bytes = state.passes.length * 256;
      state.resolved = state.device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      state.readback = state.device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      state.passes.forEach(({ queries }, i) =>
        this.resolveQuerySet(queries, 0, 2, state.resolved, i * 256),
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
          state.passes.forEach(({ dispatches, phase }, i) => {
            const key = JSON.stringify({ dispatches, phase });
            const total = totals.get(key) || {
              dispatches,
              phase,
              passes: 0,
              gpuMs: 0,
            };
            total.passes++;
            total.gpuMs += Number(times[i * 32 + 1] - times[i * 32]) / 1e6;
            totals.set(key, total);
          });
        } finally {
          state.readback.destroy();
          state.resolved.destroy();
          state.passes.forEach(({ queries }) => queries.destroy());
        }
      })();
      pending.add(task);
      task.then(
        () => pending.delete(task),
        (reason) => {
          pending.delete(task);
          failures.push(reason);
        },
      );
    }
  };
  self.speechGpuProfileTake = async () => {
    await Promise.allSettled(Array.from(pending));
    if (failures.length)
      throw new AggregateError(failures.splice(0), "GPU profiling failed");
    const result = Array.from(totals.values()).sort(
      (a, b) => b.gpuMs - a.gpuMs,
    );
    totals.clear();
    return result;
  };
  self.speechGpuProfileSelfTest = async () => {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice({
      requiredFeatures: ["timestamp-query"],
    });
    testDevice = device;
    device.pushErrorScope("validation");
    const storage = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readback = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const module = device.createShaderModule({
        code: "@group(0) @binding(0) var<storage, read_write> value: array<u32>; @compute @workgroup_size(1) fn main() { value[0] = 7u; }",
      });
      const pipeline = device.createComputePipeline({
        label: "profiler-self-test",
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
      const group = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: storage } }],
      });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(1);
      pass.end();
      encoder.copyBufferToBuffer(storage, 0, readback, 0, 4);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const value = new Uint32Array(readback.getMappedRange())[0];
      const result = await self.speechGpuProfileTake();
      const error = await device.popErrorScope();
      if (error) throw new Error(error.message);
      if (value !== 7 || result.length !== 1 || result[0].passes !== 1)
        throw new Error("GPU profiler self-test failed");
      return result;
    } finally {
      readback.destroy();
      storage.destroy();
      testDevice = undefined;
      device.destroy();
    }
  };
})();
