self.onmessage = (message) => {
  switch (message.data.kind) {
    case "load":
      self.postMessage({ kind: "ready" });
      break;
    case "reference":
      self.postMessage({ kind: "conditioned" });
      break;
    case "start":
      self.postMessage({ kind: "started" });
      // A blocked inference thread cannot acknowledge cancellation messages.
      while (true) {}
  }
};
