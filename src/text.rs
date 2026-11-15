//! Incremental Qwen tokenization. Transport boundaries never become token boundaries.
use crate::inference::Error;
use tokenizers::{
    Normalizer, OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, Tokenizer,
};

pub(crate) struct TextBuffer<'a> {
    tokenizer: &'a Tokenizer,
    tail: String,
    limit: usize,
}

impl<'a> TextBuffer<'a> {
    pub(crate) fn new(tokenizer: &'a Tokenizer, limit: usize) -> Self {
        Self {
            tokenizer,
            tail: String::new(),
            limit,
        }
    }

    pub(crate) fn push(&mut self, text: &str) -> Result<Vec<u32>, Error> {
        self.tail.push_str(text);
        let mut split = PreTokenizedString::from(self.tail.as_str());
        if let Some(normalizer) = self.tokenizer.get_normalizer() {
            split
                .normalize(|s| normalizer.normalize(s))
                .map_err(|e| Error::Tokenizer(e.to_string()))?;
        }
        if let Some(pre) = self.tokenizer.get_pre_tokenizer() {
            pre.pre_tokenize(&mut split)
                .map_err(|e| Error::Tokenizer(e.to_string()))?;
        }
        let splits = split.get_splits(OffsetReferential::Original, OffsetType::Byte);
        // Qwen's regex can regroup the last whitespace/letter run, and NFC can
        // change its last scalar. Keep both trailing pre-tokenizer runs intact.
        let mut boundary = splits.len().checked_sub(2).map_or(0, |i| splits[i].1.0);
        // Added tokens are recognized before pre-tokenization, including tokens
        // split across network messages. Do not commit any prefix of one.
        for token in self.tokenizer.get_added_tokens_decoder().values() {
            for (start, _) in self.tail.char_indices() {
                if start >= boundary {
                    break;
                }
                let remaining = &self.tail[start..];
                if token.content.starts_with(remaining)
                    || (remaining.starts_with(&token.content)
                        && start + token.content.len() > boundary)
                {
                    boundary = start;
                    if token.lstrip {
                        boundary = self.tail[..boundary].trim_end().len();
                    }
                    break;
                }
            }
        }
        let ids = self
            .tokenizer
            .encode(&self.tail[..boundary], false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?
            .get_ids()
            .to_vec();
        self.tail.drain(..boundary);
        if self.tail.len() > self.limit {
            return Err(Error::TextCapacity(self.limit));
        }
        Ok(ids)
    }

    pub(crate) fn finish(&mut self) -> Result<Vec<u32>, Error> {
        let ids = self
            .tokenizer
            .encode(self.tail.as_str(), false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?
            .get_ids()
            .to_vec();
        self.tail.clear();
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::{
        AddedToken,
        models::bpe::{BPE, Vocab},
        normalizers::unicode::NFC,
        pre_tokenizers::{
            byte_level::ByteLevel,
            sequence::Sequence,
            split::{Split, SplitPattern},
        },
    };

    fn tokenizer() -> Tokenizer {
        let mut alphabet: Vec<_> = ByteLevel::alphabet().into_iter().collect();
        alphabet.sort();
        let vocab: Vocab = alphabet
            .into_iter()
            .enumerate()
            .map(|(i, c)| (c.to_string(), i as u32))
            .collect();
        let mut tokenizer = Tokenizer::new(
            BPE::builder()
                .vocab_and_merges(vocab, vec![])
                .build()
                .unwrap(),
        );
        tokenizer.with_normalizer(Some(NFC)).unwrap();
        // Qwen's published regex; the test vocabulary is deliberately tiny.
        let split = Split::new(SplitPattern::Regex(r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+".into()), tokenizers::SplitDelimiterBehavior::Isolated, false).unwrap();
        tokenizer.with_pre_tokenizer(Some(Sequence::new(vec![
            split.into(),
            ByteLevel::new(false, true, false).into(),
        ])));
        tokenizer
            .add_special_tokens([AddedToken::from("<|im_end|>", true)])
            .unwrap();
        tokenizer
    }

    fn check(tokenizer: Tokenizer) {
        for text in [
            "今天的阳光很好，我们沿着河边慢慢散步，聊着接下来的旅行计划。",
            "Hello, the next step isn't a reset. CAFÉ Cafe\u{301} 12345!\n",
            "こんにちは。規則への信頼が安心につながります。",
            "안녕하세요. 규칙에 대한 신뢰가 중요합니다. 한글",
            "Before <|im_end|> after\r\n   next / punctuation!!!\n",
        ] {
            let expected = tokenizer.encode(text, false).unwrap().get_ids().to_vec();
            let boundaries: Vec<_> = text
                .char_indices()
                .map(|(i, _)| i)
                .chain([text.len()])
                .collect();
            for &split in &boundaries {
                let mut buffer = TextBuffer::new(&tokenizer, 65536);
                let mut actual = buffer.push(&text[..split]).unwrap();
                actual.extend(buffer.push(&text[split..]).unwrap());
                actual.extend(buffer.finish().unwrap());
                assert_eq!(actual, expected, "split at {split}: {text}");
            }
            let mut buffer = TextBuffer::new(&tokenizer, 65536);
            let mut actual = Vec::new();
            for ch in text.chars() {
                actual.extend(buffer.push(&ch.to_string()).unwrap());
            }
            actual.extend(buffer.finish().unwrap());
            assert_eq!(actual, expected, "one scalar per chunk: {text}");
        }
    }

    #[test]
    fn transport_boundaries_preserve_normalization_and_added_tokens() {
        check(tokenizer());
    }

    #[test]
    fn unbounded_unstable_text_fails_explicitly() {
        let tokenizer = tokenizer();
        let mut buffer = TextBuffer::new(&tokenizer, 4);
        assert!(matches!(
            buffer.push("longword"),
            Err(Error::TextCapacity(4))
        ));
    }

    #[test]
    #[ignore = "requires the published Qwen tokenizer"]
    fn published_multilingual_tokenizer_is_chunk_invariant() {
        let path = std::env::var_os("DEEPCLONE_TOKENIZER").expect("set DEEPCLONE_TOKENIZER");
        check(Tokenizer::from_file(path).unwrap());
    }
}
