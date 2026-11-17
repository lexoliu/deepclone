# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.0](https://github.com/lexoliu/deepclone/releases/tag/v0.0.0) - 2026-11-17

### Added

- guide multilingual reference voice recording
- share cloning sessions with a browser WebGPU workspace
- add Burn Qwen transformer and audio components
- [**breaking**] stream text and PCM through continuous cloning sessions
- download verified Core ML voice cloning bundles
- separate acoustic generation from waveform decoding
- integrate Rust cloning pipeline and audio model exports
- add stateful Core ML decoder and quantized export foundation

### Fixed

- pin validated incremental Core ML model bundles
- compare dual-mono native waveform outputs
- keep incremental waveform projections on ANE
- preserve acoustic continuity across generation contexts
- retain structured streaming performance measurements
- expose detailed streaming diagnostics in the CLI
- manage bounded utterances for long text streams
- retain generated audio after interruption
- identify the executing backend in audio metadata
- preserve softmax reduction while specializing attention products
- validate browser GPU profiling and isolate cached artifacts
- use corrected WebGPU capability and timing revisions
- discard failed browser workers and report Wasm panic locations
- initialize Wasm constructors once before browser inference
- persist browser voices and terminate cancelled inference workers
- prepare browser voices in the background without technical settings
- reuse the browser GPU runtime across model loads
- load byte-addressable large tensors on wasm32
- cancel idle text producers when inference stops
- preserve wide gated activations in WebGPU decoding
- await browser inference cancellation before reuse
- declare stream timer race feature explicitly
- enable browser entropy and scope native imports
- distinguish input EOF from producer failure
- preserve acoustic codes and sampling state at frame limits
- generate the initial release changelog
- isolate release staging from compiler caches
- attach the tested release PR base commit
- preserve binary checksums in release installers
- provision the workflow linter toolchain explicitly
- install workflow linter from its Go module
- bind resumable exports to verified inputs
- retain full precision for reference conditioning
- preserve dynamic audio masks across compute precisions
- release inference temporaries and record comparison sampling
- preserve Core ML cache layout and waveform fidelity

### Other

- Merge branch 'fix/coreml-ane-abort' into fix/long-text
- integrate incremental waveform state with continuous text
- cover single-frame codebooks at model dimensions
- retain incremental Core ML waveform state
- reproduce interleaved Core ML decoder aborts
- isolate Core ML decoder replay with voice conditioning
- record native long-stream validation stages
- execute pointwise codec convolutions as tiled projections
- retain acoustic codes when native long-form decoding fails
- make context scheduling transitions explicit
- Merge main into perf/webgpu-tiled-projection
- *(burn)* tile batched linear projections in shared memory
- *(burn)* retain model storage for gated projection weights
- read projection weights without temporary dtype conversions
- cover attention cache growth and reset across submissions
- fuse single-query attention over live GPU caches
- specialize single-token projections for GPU vector reductions
- sort sampling vocabularies directly on the GPU
- keep autoregressive token selection on the execution device
- combine shared-input transformer projections
- retain prepared voice codec checkpoints across utterances
- reuse causal codec state across streaming audio prefixes
- validate browser kernel capabilities and batch autotune measurements
- preserve shader diagnostics across worker termination
- capture invalid browser compute shaders for isolated validation
- keep browser kernel autotuning disabled until validated
- evaluate hardware-selected Burn compute kernels
- retain the faster transposed convolution
- lower causal transposed convolutions into polyphase kernels
- measure codec reconstruction and WebGPU round trips
- reuse unchanged browser voice conditioning
- broadcast grouped attention without repeating KV heads
- batch causal conditioning prefixes and remove index uploads
- use valid PCM input for blocked worker cancellation
- keep pending future imports scoped to tests
- prepare wide projection weights once when loading
- share preview and release packaging
- gate binary publishing on verified release PR merges
- initialize dual-licensed project
