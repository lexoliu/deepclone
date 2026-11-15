//! Native voice cloning components for Apple hardware.
#[cfg(feature = "sampling")]
pub mod sampling;

#[cfg(all(feature = "coreml", target_vendor = "apple"))]
pub mod decoder;

#[cfg(all(feature = "generation", target_vendor = "apple"))]
pub mod generation;

#[cfg(feature = "inference")]
pub mod mel;

#[cfg(all(feature = "inference", target_vendor = "apple"))]
pub mod inference;
