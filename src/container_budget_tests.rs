use super::container_budget::{AllocationClass, AllocationToken, ParseContext};
use crate::limits::NativeDecodeLimits;

fn limits(metadata: usize, payload: usize) -> NativeDecodeLimits {
    NativeDecodeLimits::new(
        payload,
        64,
        64,
        4096,
        usize::MAX,
        metadata,
        usize::MAX,
        32,
        32,
        32,
        8,
        8,
    )
}

#[test]
fn reserve_within_existing_capacity_does_not_charge_capacity() {
    let bounds = limits(8, 4096);
    let mut context = ParseContext::native_still(&bounds);
    let mut values: Vec<u8> = Vec::with_capacity(8);
    let mut token = AllocationToken::new(AllocationClass::Metadata);
    context
        .try_reserve_class_with_token(
            &mut values,
            &mut token,
            1,
            AllocationClass::Metadata,
            "existing capacity",
        )
        .unwrap();
    assert_eq!(values.capacity(), 8);
    assert_eq!(context.accounting().metadata_live, 8);
    assert_eq!(context.accounting().metadata_peak, 8);
}

#[test]
fn replacement_peak_includes_old_and_new_capacity() {
    let bounds = limits(16, 4096);
    let mut context = ParseContext::native_still(&bounds);
    let mut values = Vec::with_capacity(8);
    values.extend(0_u8..8);
    let mut token = AllocationToken::new(AllocationClass::Metadata);
    let error = context
        .try_reserve_with_token(&mut values, &mut token, 8, "replacement")
        .unwrap_err();
    assert!(error.to_string().contains("replacement allocation"));
    assert_eq!(values.len(), 8);
    assert_eq!(values.capacity(), 8);
    assert_eq!(context.accounting().metadata_live, 0);
}

#[test]
fn successful_replacement_releases_old_owner_after_peak() {
    let bounds = limits(24, 4096);
    let mut context = ParseContext::native_still(&bounds);
    let mut values = Vec::with_capacity(8);
    values.extend(0_u8..8);
    let mut token = AllocationToken::new(AllocationClass::Metadata);
    context
        .try_reserve_class_with_token(
            &mut values,
            &mut token,
            8,
            AllocationClass::Metadata,
            "replacement",
        )
        .unwrap();
    assert_eq!(values.len(), 8);
    assert!(values.capacity() >= 16);
    assert_eq!(context.accounting().metadata_live, values.capacity());
    assert!(context.accounting().metadata_peak >= 24);
}

#[test]
fn payload_append_within_capacity_has_no_new_charge() {
    let bounds = limits(1, 4096);
    let mut context = ParseContext::native_still(&bounds);
    let mut payload = Vec::with_capacity(4096);
    payload.extend_from_slice(&[0_u8; 16]);
    let mut token = AllocationToken::new(AllocationClass::Payload);
    context
        .try_reserve_class_with_token(
            &mut payload,
            &mut token,
            4080,
            AllocationClass::Payload,
            "payload",
        )
        .unwrap();
    assert_eq!(payload.capacity(), 4096);
    assert_eq!(context.accounting().payload_live, 4096);
}

#[test]
fn tokenless_existing_storage_is_rejected_without_adoption() {
    let bounds = limits(24, 4096);
    let mut context = ParseContext::native_still(&bounds);
    let mut values = vec![7_u8; 8];
    let before = context.accounting();
    let pointer = values.as_ptr();
    let error = context
        .try_reserve(&mut values, 8, "untracked existing storage")
        .unwrap_err();
    assert!(error.to_string().contains("requires an owner token"));
    assert_eq!(values.as_ptr(), pointer);
    assert_eq!(values, vec![7_u8; 8]);
    assert_eq!(context.accounting(), before);
}

#[test]
fn every_tokenless_classified_wrapper_rejects_existing_storage() {
    for class in [
        AllocationClass::Metadata,
        AllocationClass::Icc,
        AllocationClass::Payload,
    ] {
        let bounds = limits(64, 64);
        let mut context = ParseContext::native_still(&bounds);
        let mut values = vec![7_u8; 8];
        let before = context.accounting();
        let error = context
            .try_reserve_class(&mut values, 8, class, "classified existing storage")
            .unwrap_err();
        assert!(error.to_string().contains("requires an owner token"));
        assert_eq!(context.accounting(), before);
    }
}

#[test]
fn explicit_adoption_checks_budget_before_spare_capacity() {
    for class in [
        AllocationClass::Metadata,
        AllocationClass::Icc,
        AllocationClass::Payload,
    ] {
        let bounds = limits(7, 7);
        let mut context = ParseContext::native_still(&bounds);
        let mut values = vec![7_u8; 8];
        let mut token = AllocationToken::new(class);
        let before = context.accounting();
        let error = context
            .try_reserve_class_with_token(&mut values, &mut token, 0, class, "adoption")
            .unwrap_err();
        assert!(error.to_string().contains("exceeds limit"));
        assert_eq!(context.accounting(), before);
        assert_eq!(token, AllocationToken::new(class));
    }
}

#[test]
fn stale_capacity_token_is_rejected_before_spare_capacity() {
    let bounds = limits(64, 64);
    let mut context = ParseContext::native_still(&bounds);
    let mut values = Vec::<u8>::new();
    let mut token = AllocationToken::new(AllocationClass::Metadata);
    context
        .try_reserve_with_token(&mut values, &mut token, 8, "tracked owner")
        .unwrap();
    values.extend_from_slice(&[7; 8]);
    values.reserve_exact(8);
    assert!(values.capacity() >= 16);
    let before = context.accounting();
    let token_before = format!("{token:?}");
    let error = context
        .try_reserve_with_token(&mut values, &mut token, 0, "stale owner")
        .unwrap_err();
    assert!(error.to_string().contains("token capacity is stale"));
    assert_eq!(context.accounting(), before);
    assert_eq!(format!("{token:?}"), token_before);
}
