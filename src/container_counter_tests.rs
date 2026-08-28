use super::container_budget::ParseContext;
use super::*;

fn limits(items: usize, properties: usize) -> NativeDecodeLimits {
    NativeDecodeLimits::new(
        1 << 20,
        64,
        64,
        4096,
        8192,
        1 << 20,
        1 << 20,
        items,
        properties,
        32,
        8,
        1,
    )
}

fn boxed(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut result = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
    result.extend_from_slice(kind);
    result.extend_from_slice(payload);
    result
}

fn one_iinf() -> Vec<u8> {
    let mut payload = vec![0, 0, 0, 0, 0, 1];
    payload.extend(boxed(
        b"infe",
        &[2, 0, 0, 0, 0, 1, 0, 0, b'a', b'v', b'0', b'1', 0],
    ));
    boxed(b"iinf", &payload)
}

fn one_iloc() -> Vec<u8> {
    boxed(b"iloc", &[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0])
}

fn one_ipma(item_id: u16, association: bool) -> Vec<u8> {
    let mut payload = vec![0, 0, 0, 0, 0, 0, 0, 1, 0, 1, association as u8];
    payload[8..10].copy_from_slice(&item_id.to_be_bytes());
    if association {
        payload.push(1);
    }
    boxed(b"iprp", &boxed(b"ipma", &payload))
}

fn one_property() -> Vec<u8> {
    boxed(b"iprp", &boxed(b"ipco", &boxed(b"av1C", &[0x71; 513])))
}

fn parse_child(
    bytes: &[u8],
    state: &mut MetaState,
    context: &mut ParseContext<'_>,
) -> Result<(), DecoderError> {
    parse_meta_children_with_context(bytes, bytes, 0, state, context, 0)
}

#[test]
fn cumulative_container_counts_reject_each_kind_before_reserve() {
    let cases = [
        (one_iinf(), "iinf"),
        (one_iloc(), "iloc"),
        (one_property(), "property"),
        (one_ipma(1, false), "ipma entry"),
    ];
    for (bytes, label) in cases {
        let bounds = limits(1, 1);
        let mut context = ParseContext::native_still(&bounds);
        let mut state = MetaState::default();
        parse_child(&bytes, &mut state, &mut context).unwrap();
        let before = context.accounting();
        let result = parse_child(&bytes, &mut state, &mut context);
        assert!(
            matches!(result, Err(DecoderError::InvalidParam(_))),
            "{label}: {result:?}"
        );
        assert_eq!(
            context.accounting(),
            before,
            "{label}: rejected count changed ownership"
        );
    }
}

#[test]
fn cumulative_ipma_association_count_is_checked_atomically() {
    let bounds = limits(8, 1);
    let mut context = ParseContext::native_still(&bounds);
    let mut state = MetaState::default();
    parse_child(&one_ipma(1, true), &mut state, &mut context).unwrap();
    let before = context.accounting();
    let result = parse_child(&one_ipma(2, true), &mut state, &mut context);
    assert!(matches!(result, Err(DecoderError::InvalidParam(_))));
    assert_eq!(context.accounting(), before);
}

#[test]
fn different_container_count_kinds_use_independent_limits() {
    let bounds = limits(1, 1);
    let mut context = ParseContext::native_still(&bounds);
    let mut state = MetaState::default();
    for bytes in [one_iinf(), one_iloc(), one_property(), one_ipma(1, true)] {
        parse_child(&bytes, &mut state, &mut context).unwrap();
    }
    assert_eq!(state.item_infos.len(), 1);
    assert_eq!(state.item_locations.len(), 1);
    assert_eq!(state.item_properties.len(), 1);
    assert_eq!(state.item_property_associations.len(), 1);
}

#[test]
fn ipma_count_admission_rejects_atomically_and_allows_retry() {
    let bounds = limits(2, 1);
    let mut context = ParseContext::native_still(&bounds);
    context.admit_ipma(1, 1).unwrap();
    let before = format!("{context:?}");
    assert!(context.admit_ipma(1, 1).is_err());
    assert_eq!(format!("{context:?}"), before);
    context.admit_ipma(1, 0).unwrap();
    assert!(context.admit_ipma(1, 0).is_err());
}

#[test]
fn count_arithmetic_overflow_does_not_change_owners_or_work() {
    let bounds = limits(usize::MAX, usize::MAX);
    let mut context = ParseContext::native_still(&bounds);
    context.admit_iinf_entries(usize::MAX).unwrap();
    context.admit_iloc_entries(usize::MAX).unwrap();
    context.admit_ipma(1, usize::MAX).unwrap();
    for kind in 0..4 {
        let before = format!("{context:?}");
        let result = match kind {
            0 => context.admit_iinf_entries(1),
            1 => context.admit_iloc_entries(1),
            2 => context.admit_ipma(1, 1),
            _ => context.admit_ipma(usize::MAX, 0),
        };
        assert!(matches!(result, Err(DecoderError::InvalidParam(_))));
        assert_eq!(format!("{context:?}"), before);
        assert_eq!(
            context.accounting(),
            super::container_budget::ParseAccounting::default()
        );
    }
}

#[test]
fn count_work_survives_accounting_rollback_and_metadata_replacement() {
    let bounds = limits(1, 1);
    let mut context = ParseContext::native_still(&bounds);
    let checkpoint = context.checkpoint();
    context.admit_iinf_entries(1).unwrap();
    context.rollback(checkpoint);
    assert!(context.admit_iinf_entries(1).is_err());

    let bytes = one_iinf();
    let mut replacement_context = ParseContext::native_still(&bounds);
    let mut old_state = MetaState::default();
    let mut replacement_state = MetaState::default();
    parse_child(&bytes, &mut old_state, &mut replacement_context).unwrap();
    assert!(parse_child(&bytes, &mut replacement_state, &mut replacement_context).is_err());
    assert_eq!(old_state.item_infos.len(), 1);
    assert!(replacement_state.item_infos.is_empty());
}

#[test]
fn legacy_count_admission_is_a_complete_noop() {
    let mut context = ParseContext::legacy();
    let before = format!("{context:?}");
    for _ in 0..2 {
        context.admit_iinf_entries(usize::MAX).unwrap();
        context.admit_iloc_entries(usize::MAX).unwrap();
        context.admit_ipma(usize::MAX, usize::MAX).unwrap();
        context.admit_property().unwrap();
    }
    assert_eq!(format!("{context:?}"), before);
}

#[test]
fn legacy_ipma_keeps_the_existing_truncation_error() {
    let bytes = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1];
    assert_eq!(
        parse_ipma(&bytes).unwrap_err(),
        DecoderError::NotEnoughData("ipma association count exceeds the remaining payload".into())
    );
}
