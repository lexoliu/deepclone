(() => {
  const create = GPUDevice.prototype.createShaderModule;
  GPUDevice.prototype.createShaderModule = function (descriptor) {
    const module = create.call(this, descriptor);
    const features = Array.from(this.features);
    void module.getCompilationInfo().then(({ messages }) => {
      if (messages.some((message) => message.type === "error")) {
        console.error(
          "Speech shader failure",
          JSON.stringify({
            code: descriptor.code,
            label: descriptor.label,
            features,
            messages: messages.map(
              ({ message, type, lineNum, linePos, offset, length }) => ({
                message,
                type,
                lineNum,
                linePos,
                offset,
                length,
              }),
            ),
          }),
        );
      }
    });
    return module;
  };
})();
