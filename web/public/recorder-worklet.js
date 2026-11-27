class PcmRecorderProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channels = [];
    const transfer = [];
    for (let c = 0; c < input.length; c++) {
      const copy = new Float32Array(input[c]);
      channels.push(copy);
      transfer.push(copy.buffer);
    }
    this.port.postMessage(channels, transfer);
    return true;
  }
}

registerProcessor("pcm-recorder", PcmRecorderProcessor);
