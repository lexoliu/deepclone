//! Transport-independent Unicode sentence and tokenizer capacity boundaries.
use crate::inference::Error;
use tokenizers::Tokenizer;
use unicode_segmentation::UnicodeSegmentation;

pub(crate) struct Splitter {
    pending: String,
    checked: usize,
    limit: usize,
    bytes: usize,
}
impl Splitter {
    pub(crate) fn new(limit: usize, bytes: usize) -> Self {
        Self {
            pending: String::new(),
            checked: 0,
            limit,
            bytes,
        }
    }
    pub(crate) fn push(&mut self, text: &str, tokenizer: &Tokenizer) -> Result<Vec<String>, Error> {
        self.pending.push_str(text);
        let result = self.extract(tokenizer, false)?;
        if self.pending.len() > self.bytes {
            return Err(Error::TextCapacity(self.bytes));
        }
        Ok(result)
    }
    pub(crate) fn finish(&mut self, tokenizer: &Tokenizer) -> Result<Vec<String>, Error> {
        self.extract(tokenizer, true)
    }
    fn extract(&mut self, tokenizer: &Tokenizer, finished: bool) -> Result<Vec<String>, Error> {
        let mut result = Vec::new();
        loop {
            // Keep the final sentence and grapheme open until lookahead or EOF.
            let added = crate::text::AddedBoundaries::new(tokenizer, &self.pending, !finished);
            let sentence = self
                .pending
                .split_sentence_bound_indices()
                .nth(1)
                .map(|(i, _)| i)
                .filter(|&end| added.before(end) == end)
                .or_else(|| finished.then_some(self.pending.len()));
            let mut capacity = None;
            let boundaries = self
                .pending
                .grapheme_indices(true)
                .map(|(i, _)| i)
                .chain(finished.then_some(self.pending.len()));
            for end in boundaries {
                if end <= self.checked {
                    continue;
                }
                if sentence.is_some_and(|sentence| end > sentence) {
                    break;
                }
                if added.before(end) < end {
                    continue;
                }
                let count = tokenizer
                    .encode(&self.pending[..end], false)
                    .map_err(|e| Error::Tokenizer(e.to_string()))?
                    .len();
                // Evaluate prefixes in order: BPE counts need not be monotonic.
                // This makes a capacity boundary independent of transport chunks.
                if count > self.limit {
                    if self.checked == 0 {
                        return Err(Error::SegmentCapacity(self.limit));
                    }
                    capacity = Some(self.checked);
                    break;
                }
                self.checked = end;
            }
            let end = match capacity.or(sentence) {
                Some(end) if end > 0 => end,
                _ => break,
            };
            let remaining = self.pending.split_off(end);
            let text = std::mem::replace(&mut self.pending, remaining);
            self.checked = 0;
            if !text.trim().is_empty() {
                result.push(text);
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::tests::tokenizer;

    fn split(
        tokenizer: &Tokenizer,
        parts: impl IntoIterator<Item = String>,
        limit: usize,
    ) -> Vec<String> {
        let mut split = Splitter::new(limit, usize::MAX);
        let mut result = Vec::new();
        for part in parts {
            result.extend(split.push(&part, tokenizer).unwrap());
        }
        result.extend(split.finish(tokenizer).unwrap());
        result
    }
    #[test]
    fn unicode_sentences_and_capacity_are_transport_independent() {
        let tokenizer = tokenizer();
        for text in [
            "法律让人安心。我们相信规则。明天一起出发！",
            "Dr. Smith walks home. CAFÉ Cafe\u{301} is next! Why wait?\nNew paragraph.",
            "今日は静かな道を歩きます。明日も一緒に出かけましょう。",
            "오늘은 조용한 길을 걸어요. 내일도 함께 걸어요!",
            "法律への信頼규칙에대한신뢰longwordwithoutanypunctuationCafe\u{301}",
            "Before <|im_end|> after. The next sentence.",
        ] {
            for limit in [16, 64, 128] {
                let expected = split(&tokenizer, [text.to_owned()], limit);
                assert_eq!(expected.concat(), text);
                for part in &expected {
                    assert!(tokenizer.encode(part.as_str(), false).unwrap().len() <= limit);
                }
                for (at, _) in text.char_indices() {
                    assert_eq!(
                        split(
                            &tokenizer,
                            [text[..at].to_owned(), text[at..].to_owned()],
                            limit
                        ),
                        expected,
                        "split at {at}: {text}"
                    );
                }
                assert_eq!(
                    split(&tokenizer, text.chars().map(|ch| ch.to_string()), limit),
                    expected
                );
            }
        }
    }
    #[test]
    fn complete_sentences_are_available_before_eof() {
        let tokenizer = tokenizer();
        let mut split = Splitter::new(128, usize::MAX);
        assert!(split.push("Hello", &tokenizer).unwrap().is_empty());
        let completed = split.push(" world. Next", &tokenizer).unwrap();
        assert_eq!(completed, ["Hello world. "]);
        assert_eq!(split.finish(&tokenizer).unwrap(), ["Next"]);
    }
    #[test]
    fn oversized_grapheme_fails_without_splitting_it() {
        let tokenizer = tokenizer();
        let mut split = Splitter::new(1, usize::MAX);
        assert!(matches!(
            split.push("你a", &tokenizer),
            Err(Error::SegmentCapacity(1))
        ));
    }

    #[test]
    fn unfinished_grapheme_obeys_the_byte_budget() {
        let tokenizer = tokenizer();
        let mut split = Splitter::new(128, 4);
        assert!(matches!(
            split.push("a\u{301}\u{301}", &tokenizer),
            Err(Error::TextCapacity(4))
        ));
    }

    #[test]
    #[ignore = "requires the published Qwen tokenizer"]
    fn published_tokenizer_preserves_multilingual_capacity_boundaries() {
        let path = std::env::var_os("DEEPCLONE_TOKENIZER").unwrap();
        let tokenizer = Tokenizer::from_file(path).unwrap();
        for text in [
            "法律让人安心。带着信任走向明天！",
            "Let us take a quiet walk and decide what comes next. Tomorrow we will meet again.",
            "今日は静かな道を歩きながら、明日のことを考えましょう。明日も一緒に歩きます。",
            "오늘은 조용한 길을 걸으며 내일을 생각해 봅시다. 내일도 함께 걸어요.",
        ] {
            let expected = split(&tokenizer, [text.to_owned()], 16);
            assert_eq!(expected.concat(), text);
            assert_eq!(
                split(&tokenizer, text.chars().map(|ch| ch.to_string()), 16),
                expected
            );
            for part in expected {
                assert!(tokenizer.encode(part, false).unwrap().len() <= 16);
            }
        }
    }
}
