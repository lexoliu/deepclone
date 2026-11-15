//! Native voice cloning components for Apple hardware.
#[cfg(feature = "sampling")]
pub mod sampling;

#[cfg(all(feature = "coreml", target_vendor = "apple"))]
pub mod decoder;
