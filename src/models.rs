//! Verified model delivery, independent of the inference runtime.
//!
//! Reuse an application's [`Store`] and [`Transport`], or enable `http` to
//! construct a store with `Zenwave`. Downloading does not load Core ML models.

#[cfg(feature = "http")]
pub use ondevice::Zenwave;
pub use ondevice::{Archive, Error, Pin, Progress, Sha256, Store, Transport};
use std::path::PathBuf;

/// Batched 0.6B FP16 model for ANE-backed streaming, with FP32 conditioning.
///
/// ```no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// # #[cfg(feature = "http")] {
/// use deepclone::models::{qwen3_tts_0_6b, Store, Zenwave};
/// let store = Store::new("models/cache", Zenwave);
/// let installed = qwen3_tts_0_6b().download(&store, |_| {}).await?;
/// // Pass installed.transformers() and installed.audio() to Engine::open.
/// # }
/// # Ok(()) }
/// ```
pub const fn qwen3_tts_0_6b() -> Bundle<'static> {
    Bundle::new(Pin {
        url: "https://huggingface.co/lexoliu/deepclone-qwen3-tts-coreml/resolve/b92456a0bb21dc18cad07a6828ff391deb38ae3f/qwen3-tts-0.6b-coreml-realtime.tar.zst",
        sha256: Sha256::from_hex(
            "ed7ee326b29aa966ce61562776cbe8db3cc16f0dd73fb80346f49b5e5e0c1e7b",
        ),
        size: 1_882_140_090,
        kind: Archive,
    })
}

/// Fixed-shape 0.6B FP16 model for CPU execution, with FP32 conditioning.
pub const fn qwen3_tts_0_6b_fixed_shape() -> Bundle<'static> {
    Bundle::new(Pin {
        url: "https://huggingface.co/lexoliu/deepclone-qwen3-tts-coreml/resolve/d132983a8c77090eb29362603e68fdc50a789499/qwen3-tts-0.6b-coreml.tar.zst",
        sha256: Sha256::from_hex(
            "2d9be296a0c7c2159ef2029be1d5945b7069c976d6edc219483807667396a4c9",
        ),
        size: 1_882_818_272,
        kind: Archive,
    })
}

/// Published 1.7B FP16 model with FP32 reference conditioning.
pub const fn qwen3_tts_1_7b() -> Bundle<'static> {
    Bundle::new(Pin {
        url: "https://huggingface.co/lexoliu/deepclone-qwen3-tts-coreml/resolve/d132983a8c77090eb29362603e68fdc50a789499/qwen3-tts-1.7b-coreml.tar.zst",
        sha256: Sha256::from_hex(
            "eba1b707dbf4c1fb781eb8adecb6ab950aeacef60e2ee54e4322cb884ae7130f",
        ),
        size: 3_497_929_525,
        kind: Archive,
    })
}

/// An archive containing deepclone's `transformers/` and `audio/` directories.
#[derive(Debug, Clone, Copy)]
pub struct Bundle<'a> {
    pin: Pin<'a, Archive>,
}

/// Installed paths accepted by `inference::Engine::open`.
#[derive(Debug, Clone)]
pub struct Installed {
    root: PathBuf,
}

impl Installed {
    /// The complete, verified archive installation.
    pub fn root(&self) -> &std::path::Path {
        &self.root
    }

    /// Autoregressive model graphs, tokenizer and configuration.
    pub fn transformers(&self) -> PathBuf {
        self.root.join("transformers")
    }

    /// Reference conditioning and waveform decoding models.
    pub fn audio(&self) -> PathBuf {
        self.root.join("audio")
    }
}

impl<'a> Bundle<'a> {
    /// Describe a custom export packaged with deepclone's model layout.
    pub const fn new(pin: Pin<'a, Archive>) -> Self {
        Self { pin }
    }

    /// Immutable location, compressed length and checksum of the archive.
    pub const fn pin(&self) -> &Pin<'a, Archive> {
        &self.pin
    }

    /// Download, verify and install, or reuse an existing cache entry.
    ///
    /// The store owns installation locking and cancellation cleanup. The
    /// caller owns the cache location, transport and progress reporting.
    pub async fn download<T: Transport>(
        &self,
        store: &Store<T>,
        progress: impl FnMut(Progress) + Send,
    ) -> Result<Installed, Error<T::Error>> {
        Ok(Installed {
            root: store.ensure(&self.pin, progress).await?,
        })
    }
}
