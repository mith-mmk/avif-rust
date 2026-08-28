use super::container_budget::ParseContext;
use super::*;

fn limits(payload: usize) -> NativeDecodeLimits {
    NativeDecodeLimits::new(
        payload,
        64,
        64,
        4096,
        8192,
        1 << 20,
        1 << 20,
        64,
        64,
        32,
        8,
        1,
    )
}

fn direct_state(method: u16, extents: Vec<ItemExtent>) -> MetaState {
    MetaState {
        item_locations: vec![ItemLocation {
            item_id: 1,
            base_offset: 0,
            extents,
        }],
        item_construction_methods: vec![(1, method)],
        ..MetaState::default()
    }
}

#[test]
fn native_direct_payload_copies_all_method_zero_extents_without_stack() {
    let source = b"0123456789";
    let state = direct_state(
        0,
        vec![
            ItemExtent {
                offset: 1,
                length: 3,
            },
            ItemExtent {
                offset: 7,
                length: 2,
            },
        ],
    );
    let bounds = limits(8);
    let mut context = ParseContext::native_still(&bounds);

    let payload = super::item_payload_with_context(source, &state, 1, &mut context).unwrap();

    assert_eq!(payload, b"12378");
    assert_eq!(context.accounting().payload_live, payload.capacity());
}

#[test]
fn native_direct_payload_copies_method_one_idat_extents() {
    let state = MetaState {
        idat_payload: Some(b"abcdefgh".to_vec()),
        item_locations: vec![ItemLocation {
            item_id: 1,
            base_offset: 0,
            extents: vec![ItemExtent {
                offset: 2,
                length: 4,
            }],
        }],
        item_construction_methods: vec![(1, 1)],
        ..MetaState::default()
    };
    let bounds = limits(4);
    let mut context = ParseContext::native_still(&bounds);

    let payload = super::native_direct_item_payload(b"unused", &state, 1, &mut context).unwrap();

    assert_eq!(payload, b"cdef");
    assert_eq!(context.accounting().payload_live, payload.capacity());
}

#[test]
fn native_direct_payload_rejects_invalid_extent_before_payload_reserve() {
    let state = direct_state(
        0,
        vec![ItemExtent {
            offset: 8,
            length: 3,
        }],
    );
    let bounds = limits(8);
    let mut context = ParseContext::native_still(&bounds);

    let error =
        super::native_direct_item_payload(b"0123456789", &state, 1, &mut context).unwrap_err();

    assert!(matches!(error, DecoderError::NotEnoughData(_)));
    assert_eq!(context.accounting().payload_live, 0);
}

#[test]
fn native_item_offset_is_unsupported_before_payload_allocation() {
    let state = direct_state(
        2,
        vec![ItemExtent {
            offset: 0,
            length: 1,
        }],
    );
    let bounds = limits(1);
    let mut context = ParseContext::native_still(&bounds);

    let error = super::item_payload_with_context(b"x", &state, 1, &mut context).unwrap_err();

    assert!(matches!(error, DecoderError::Unsupported(_)));
    assert_eq!(context.accounting().payload_live, 0);
}
