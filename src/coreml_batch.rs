use std::num::NonZeroUsize;

/// Ordered model input lengths. One is always supported, so partitioning
/// never needs to insert padding into causal state.
#[derive(Clone)]
#[cfg_attr(feature = "inference", derive(serde::Deserialize))]
#[cfg_attr(feature = "inference", serde(try_from = "Vec<NonZeroUsize>"))]
pub(crate) struct BatchSizes(Vec<NonZeroUsize>);

impl Default for BatchSizes {
    fn default() -> Self {
        Self(vec![NonZeroUsize::MIN])
    }
}

impl TryFrom<Vec<NonZeroUsize>> for BatchSizes {
    type Error = &'static str;

    fn try_from(sizes: Vec<NonZeroUsize>) -> Result<Self, Self::Error> {
        if sizes.first() != Some(&NonZeroUsize::MIN)
            || sizes.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err("batch sizes must increase from one");
        }
        Ok(Self(sizes))
    }
}

impl BatchSizes {
    pub fn fits(&self, capacity: usize) -> bool {
        self.0.iter().all(|size| size.get() <= capacity)
    }

    pub fn flexible(&self) -> bool {
        self.0.len() > 1
    }

    #[cfg(feature = "inference")]
    pub fn batch(&self, remaining: NonZeroUsize) -> usize {
        self.0
            .iter()
            .rev()
            .find(|&&size| size <= remaining)
            .map_or(1, |size| size.get())
    }
}
