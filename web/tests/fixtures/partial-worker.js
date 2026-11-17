// Test-only PCM makes stop-path sample retention directly observable.
self.onmessage = ({ data: command }) => {
  const event =
    command.kind === "load"
      ? { kind: "ready" }
      : command.kind === "reference"
        ? { kind: "conditioned" }
        : command.kind === "start"
          ? { kind: "started" }
          : command.kind === "text"
            ? { kind: "accepted", id: command.id }
            : null;
  if (event) self.postMessage(event);
  if (command.kind === "text") {
    self.postMessage({
      kind: "audio",
      id: 0,
      samples: new Float32Array([0.25, -0.5, 0.75, -1]),
      sampleRate: 24000,
    });
    if (command.text.includes("fail"))
      self.postMessage({
        kind: "error",
        message: "Test frame budget exhausted",
      });
  }
};
