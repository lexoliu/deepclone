#![cfg(all(feature = "coreml", target_vendor = "apple"))]
use deepclone::decoder::Decoder;
use ondevice_coreml::{ComputeUnits, Tensor, f16};

#[test]
#[ignore = "requires a locally exported Qwen decoder chunk"]
fn native_state_accepts_float16_inputs_and_resets() {
    let artifact =
        std::env::var_os("DEEPCLONE_DECODER_ARTIFACT").expect("set DEEPCLONE_DECODER_ARTIFACT");
    let mut decoder =
        Decoder::open([artifact], 1_000_000.0, ComputeUnits::CpuAndNeuralEngine).unwrap();
    let input = Tensor::from_slice(
        &[1, decoder.hidden_size(), 1, 1],
        &vec![f16::ZERO; decoder.hidden_size()],
    )
    .unwrap();
    let first = decoder.step(input.clone()).unwrap().read();
    decoder.reset();
    let second = decoder.step(input).unwrap().read();
    assert_eq!(first.data(), second.data());
    assert!(first.data().iter().all(|x| x.is_finite()));
}

#[test]
#[ignore = "requires locally exported text embedding and decoder models"]
fn native_embedding_output_feeds_stateful_decoder() {
    use ondevice_coreml::{Inputs, Model};
    let artifact = std::path::PathBuf::from(
        std::env::var_os("DEEPCLONE_DECODER_ARTIFACT").expect("set DEEPCLONE_DECODER_ARTIFACT"),
    );
    let mut embedding = Model::open(artifact.parent().unwrap().join("text-embedding.mlmodelc"))
        .unwrap()
        .function(None, ComputeUnits::CpuAndNeuralEngine)
        .unwrap();
    let token = Tensor::from_slice(&[1, 1], &[151644i32]).unwrap();
    let mut inputs = Inputs::new();
    inputs.insert("token", &token).unwrap();
    let hidden = embedding
        .predict(&inputs)
        .unwrap()
        .tensor::<f16>("hidden")
        .unwrap();
    let original = hidden.read();
    let next_token = Tensor::from_slice(&[1, 1], &[77091i32]).unwrap();
    let mut next_inputs = Inputs::new();
    next_inputs.insert("token", &next_token).unwrap();
    let _next_output = embedding.predict(&next_inputs).unwrap();
    assert_eq!(hidden.read().data(), original.data());
    let mut decoder =
        Decoder::open([artifact], 1_000_000.0, ComputeUnits::CpuAndNeuralEngine).unwrap();
    let native = decoder.step(hidden.clone());
    decoder.reset();
    let copied = Tensor::from_slice(hidden.shape(), hidden.read().data()).unwrap();
    decoder.step(copied).expect("copied embedding must predict");
    native.expect("native embedding must predict");
}
