//! Verified model delivery, independent of the inference runtime.
//!
//! Reuse an application's [`Store`] and [`Transport`], or enable `http` to
//! construct a store with `Zenwave`. Downloading does not load Core ML models.

#[cfg(feature = "http")]
pub use ondevice::Zenwave;
pub use ondevice::{Archive, Error, Pin, Progress, Sha256, Store, Transport};
use std::path::PathBuf;

/// Published 0.6B FP16 model with FP32 reference conditioning.
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
        url: "https://huggingface.co/lexoliu/deepclone-qwen3-tts-coreml/resolve/c22b483de06b287069a3f780a535da11ee032b8c/qwen3-tts-0.6b-coreml.tar.zst",
        sha256: Sha256::from_hex(
            "48205c5b98e8b632ca6a22040c8a41a1f803f2d885c6b04d548f7b22e7822e63",
        ),
        size: 1_882_797_495,
        kind: Archive,
    })
}

/// Published 1.7B FP16 model with FP32 reference conditioning.
pub const fn qwen3_tts_1_7b() -> Bundle<'static> {
    Bundle::new(Pin {
        url: "https://huggingface.co/lexoliu/deepclone-qwen3-tts-coreml/resolve/c22b483de06b287069a3f780a535da11ee032b8c/qwen3-tts-1.7b-coreml.tar.zst",
        sha256: Sha256::from_hex(
            "3cf2e907ecf03ca6bf7f35c785f315f79c5ac06da7309a3db2e8aec50c31073a",
        ),
        size: 3_497_919_717,
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
