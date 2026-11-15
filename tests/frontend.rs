#![cfg(feature = "inference")]
use deepclone::mel::{Filters, Spectrogram};
use serde::Deserialize;
use std::{fs::File, io::BufReader, path::PathBuf};

#[derive(Deserialize)]
struct Fixture {
    audio: Vec<f32>,
    mel: Vec<f32>,
    texts: Vec<Text>,
}
#[derive(Deserialize)]
struct Text {
    text: String,
    ids: Vec<u32>,
}

#[test]
#[ignore = "requires locally generated real-audio frontend fixtures"]
fn real_audio_and_multilingual_tokens_match_official_frontend() {
    let root = PathBuf::from(
        std::env::var_os("DEEPCLONE_FRONTEND_FIXTURE").expect("set DEEPCLONE_FRONTEND_FIXTURE"),
    );
    let fixture: Fixture = serde_json::from_reader(BufReader::new(
        File::open(root.join("frontend.json")).unwrap(),
    ))
    .unwrap();
    let filters: Filters =
        serde_json::from_reader(BufReader::new(File::open(root.join("mel.json")).unwrap()))
            .unwrap();
    let mel = Spectrogram::new(filters)
        .unwrap()
        .compute(&fixture.audio)
        .unwrap();
    assert_eq!(mel.len(), fixture.mel.len());
    let maximum = mel
        .iter()
        .zip(&fixture.mel)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    // Equivalent float32 FFT implementations can differ in rounding. A 1e-3
    // log-magnitude error is far below the neural graph's float16 resolution.
    assert!(maximum < 1e-3, "maximum log-mel error: {maximum}");
    let tokenizer = tokenizers::Tokenizer::from_file(root.join("tokenizer.json")).unwrap();
    for item in fixture.texts {
        assert_eq!(
            tokenizer.encode(item.text, false).unwrap().get_ids(),
            item.ids
        );
    }
}
