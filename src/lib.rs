//! Native voice cloning components for Apple hardware.
#[cfg(feature = "download")]
pub mod models;

#[cfg(feature = "sampling")]
pub mod sampling;

#[cfg(all(feature = "coreml", target_vendor = "apple"))]
pub mod decoder;

#[cfg(feature = "inference")]
pub mod generation;

#[cfg(feature = "inference")]
pub mod mel;

#[cfg(feature = "inference")]
pub mod inference;

#[cfg(feature = "inference")]
mod text;
#[cfg(feature = "inference")]
mod utterances;

#[cfg(feature = "streaming")]
pub mod streaming;

#[cfg(feature = "burn")]
pub mod burn;

#[cfg(all(feature = "coreml", feature = "inference", target_vendor = "apple"))]
pub mod coreml;

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
pub mod browser;
