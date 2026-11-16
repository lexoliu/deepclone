self.onmessage = ({ data }) => {
  if (data.kind === "load") self.postMessage({ kind: "ready" });
  if (data.kind === "reference") throw new Error("Reference execution crashed");
};
