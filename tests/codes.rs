#![cfg(all(feature = "inference", target_vendor = "apple"))]

use deepclone::inference::AcousticCodes;
use std::num::NonZeroUsize;

#[test]
fn code_matrix_preserves_frame_order_and_reference_boundary() {
    let values = vec![10, 11, 20, 21, 30, 31];
    let codes = AcousticCodes::new(NonZeroUsize::new(2).unwrap(), values.clone(), 1).unwrap();
    assert_eq!(codes.values(), values);
    assert_eq!(codes.frames(), 3);
    assert_eq!(codes.reference_frames(), 1);
    assert_eq!(codes.generated_frames(), 2);
    let exported = serde_json::to_value(&codes).unwrap();
    assert_eq!(exported["groups"], 2);
    assert_eq!(exported["values"], serde_json::json!(values));
}

#[test]
fn malformed_matrix_and_out_of_bounds_reference_are_rejected() {
    let groups = NonZeroUsize::new(2).unwrap();
    assert!(AcousticCodes::new(groups, vec![1, 2, 3], 0).is_err());
    assert!(AcousticCodes::new(groups, vec![1, 2], 2).is_err());
}
