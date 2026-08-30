use super::container_budget::ParseContext;
use super::{MetaState, OwnedItemLocations};
use crate::limits::NativeDecodeLimits;

fn limits(max_live: Option<usize>) -> NativeDecodeLimits {
    let limits = NativeDecodeLimits::new(4096, 64, 64, 4096, 4096, 4096, 4096, 32, 32, 32, 8, 1);
    match max_live {
        Some(bytes) => limits
            .with_max_live_allocation_bytes(bytes)
            .expect("positive live limit"),
        None => limits,
    }
}

fn boxed(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let size = u32::try_from(payload.len() + 8).expect("test box fits");
    let mut output = Vec::with_capacity(payload.len() + 8);
    output.extend_from_slice(&size.to_be_bytes());
    output.extend_from_slice(kind);
    output.extend_from_slice(payload);
    output
}

fn iloc_fixture(version: u8, items: &[&[(u32, u32)]]) -> Vec<u8> {
    let mut payload = vec![version, 0, 0, 0, 0x44, 0];
    if version < 2 {
        payload.extend_from_slice(
            &u16::try_from(items.len())
                .expect("test item count fits")
                .to_be_bytes(),
        );
    } else {
        payload.extend_from_slice(
            &u32::try_from(items.len())
                .expect("test item count fits")
                .to_be_bytes(),
        );
    }
    for (ordinal, extents) in items.iter().enumerate() {
        if version < 2 {
            payload.extend_from_slice(
                &u16::try_from(ordinal + 1)
                    .expect("test item id fits")
                    .to_be_bytes(),
            );
        } else {
            payload.extend_from_slice(
                &u32::try_from(ordinal + 1)
                    .expect("test item id fits")
                    .to_be_bytes(),
            );
        }
        if version == 1 || version == 2 {
            payload.extend_from_slice(&[0, 0]);
        }
        payload.extend_from_slice(&[0, 0]);
        payload.extend_from_slice(
            &u16::try_from(extents.len())
                .expect("test extent count fits")
                .to_be_bytes(),
        );
        for &(offset, length) in *extents {
            payload.extend_from_slice(&offset.to_be_bytes());
            payload.extend_from_slice(&length.to_be_bytes());
        }
    }
    boxed(b"iloc", &payload)
}

fn parse_iloc(
    input: &[u8],
    state: &mut MetaState,
    context: &mut ParseContext<'_>,
) -> Result<(), crate::DecoderError> {
    super::parse_meta_children_with_context(input, input, 0, state, context, 0)
}

fn take_owned(state: &mut MetaState) -> OwnedItemLocations {
    OwnedItemLocations {
        locations: std::mem::take(&mut state.item_locations),
        construction_methods: std::mem::take(&mut state.item_construction_methods),
        extent_indexes: std::mem::take(&mut state.item_extent_indexes),
        locations_token: std::mem::take(&mut state.item_locations_token),
        construction_methods_token: std::mem::take(&mut state.item_construction_methods_token),
        extent_indexes_token: std::mem::take(&mut state.item_extent_indexes_token),
        native_owners: state.item_location_owners.take(),
    }
}

#[test]
fn native_iloc_moves_nested_extent_tokens_and_retires_all_values() {
    let input = iloc_fixture(2, &[&[], &[(3, 2), (8, 0)]]);
    let bounds = limits(None);
    let mut context = ParseContext::native_still(&bounds);
    let mut state = MetaState::default();
    parse_iloc(&input, &mut state, &mut context).unwrap();

    let owners = state.item_location_owners.as_ref().unwrap();
    assert_eq!(owners.entries.len(), state.item_locations.len());
    assert_eq!(owners.entries.len(), state.item_extent_indexes.len());
    for ((location, (_, indexes)), ticket) in state
        .item_locations
        .iter()
        .zip(&state.item_extent_indexes)
        .zip(&owners.entries)
    {
        assert_eq!(
            ticket.extents.charged_capacity_bytes,
            location.extents.capacity() * std::mem::size_of::<super::ItemExtent>()
        );
        assert_eq!(
            ticket.indexes.charged_capacity_bytes,
            indexes.capacity() * std::mem::size_of::<u64>()
        );
    }

    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    observation.track_raw_slot(
        0,
        state.item_locations.as_ptr().cast(),
        state.item_locations.capacity() * std::mem::size_of::<super::ItemLocation>(),
    );
    observation.track_raw_slot(
        1,
        state.item_construction_methods.as_ptr().cast(),
        state.item_construction_methods.capacity() * std::mem::size_of::<(u32, u16)>(),
    );
    observation.track_raw_slot(
        2,
        state.item_extent_indexes.as_ptr().cast(),
        state.item_extent_indexes.capacity() * std::mem::size_of::<(u32, Vec<u64>)>(),
    );
    observation.track_raw_slot(
        3,
        owners.entries.as_ptr().cast(),
        owners.entries.capacity() * std::mem::size_of::<super::iloc_owners::ExtentOwnerTickets>(),
    );
    observation.track_raw_slot(
        4,
        state.item_locations[1].extents.as_ptr().cast(),
        state.item_locations[1].extents.capacity() * std::mem::size_of::<super::ItemExtent>(),
    );
    observation.track_raw_slot(
        5,
        state.item_extent_indexes[1].1.as_ptr().cast(),
        state.item_extent_indexes[1].1.capacity() * std::mem::size_of::<u64>(),
    );

    take_owned(&mut state).retire(&mut context).unwrap();
    let masks = observation.release_drop_masks();
    assert!(masks[0][0] && masks[0][1] && masks[0][2] && masks[0][4] && masks[0][5]);
    assert!(!masks[0][3], "sidecar remains live until backing release");
    assert!(masks[4][3], "sidecar drops before its backing debit");
    drop(observation);
    assert_eq!(context.accounting().metadata_live, 0);
    assert_eq!(context.accounting().aggregate_live, 0);
}

#[test]
fn native_iloc_exact_limit_succeeds_and_one_under_rejects_before_copy() {
    let input = iloc_fixture(0, &[&[(0, 4), (8, 0)]]);
    let probe_bounds = limits(None);
    let mut probe_context = ParseContext::native_still(&probe_bounds);
    let mut probe_state = MetaState::default();
    parse_iloc(&input, &mut probe_state, &mut probe_context).unwrap();
    let exact = probe_context.accounting().metadata_live;
    take_owned(&mut probe_state)
        .retire(&mut probe_context)
        .unwrap();

    let exact_bounds = limits(Some(exact));
    let mut exact_context = ParseContext::native_still(&exact_bounds);
    let mut exact_state = MetaState::default();
    parse_iloc(&input, &mut exact_state, &mut exact_context).unwrap();
    take_owned(&mut exact_state)
        .retire(&mut exact_context)
        .unwrap();

    let under_bounds = limits(Some(exact - 1));
    let mut under_context = ParseContext::native_still(&under_bounds);
    let mut under_state = MetaState::default();
    let error = parse_iloc(&input, &mut under_state, &mut under_context).unwrap_err();
    assert!(matches!(error, crate::DecoderError::InvalidParam(_)));
    assert_eq!(under_context.accounting().metadata_live, 0);
    assert_eq!(under_context.accounting().aggregate_live, 0);
}

#[test]
fn native_iloc_replacement_drops_inner_and_outer_candidates_before_restore() {
    let old_input = iloc_fixture(1, &[&[(0, 4)]]);
    let incoming = iloc_fixture(1, &[&[(4, 4), (12, 0)]]);
    let probe_bounds = limits(None);
    let mut old_probe_context = ParseContext::native_still(&probe_bounds);
    let mut old_probe_state = MetaState::default();
    parse_iloc(&old_input, &mut old_probe_state, &mut old_probe_context).unwrap();
    let old_bytes = old_probe_context.accounting().metadata_live;
    take_owned(&mut old_probe_state)
        .retire(&mut old_probe_context)
        .unwrap();
    let mut incoming_probe_context = ParseContext::native_still(&probe_bounds);
    let mut incoming_probe_state = MetaState::default();
    parse_iloc(
        &incoming,
        &mut incoming_probe_state,
        &mut incoming_probe_context,
    )
    .unwrap();
    let incoming_bytes = incoming_probe_context.accounting().metadata_live;
    take_owned(&mut incoming_probe_state)
        .retire(&mut incoming_probe_context)
        .unwrap();

    let bounds = limits(Some(old_bytes + incoming_bytes));
    let mut context = ParseContext::native_still(&bounds);
    let mut state = MetaState::default();
    parse_iloc(&old_input, &mut state, &mut context).unwrap();
    let checkpoint = context.checkpoint();
    let old_pointer = state.item_locations.as_ptr();
    let old_token = state.item_locations_token.charged_capacity_bytes;
    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let _all_candidates = crate::test_allocation_observer::track_all_candidates();
    let _excess =
        crate::test_allocation_observer::force_fresh_capacity_extra("iloc extent owner tickets", 8);

    let error = parse_iloc(&incoming, &mut state, &mut context).unwrap_err();
    assert!(matches!(error, crate::DecoderError::InvalidParam(_)));
    assert_eq!(observation.restore_count(), 2);
    let masks = observation.restore_drop_masks();
    assert!(
        !masks[0][0] && masks[0][3],
        "inner restore must retain outer candidates while dropping sidecar"
    );
    assert!(
        masks[1][0] && masks[1][1] && masks[1][2] && masks[1][3],
        "outer restore must observe every incoming candidate dropped"
    );
    assert_eq!(context.checkpoint(), checkpoint);
    assert_eq!(state.item_locations.as_ptr(), old_pointer);
    assert_eq!(state.item_locations_token.charged_capacity_bytes, old_token);
    drop(observation);
    drop(_all_candidates);
    drop(_excess);

    parse_iloc(&incoming, &mut state, &mut context).unwrap();
    take_owned(&mut state).retire(&mut context).unwrap();
    assert_eq!(context.accounting().metadata_live, 0);
}

#[test]
fn iloc_legacy_method_two_and_native_rejection_remain_unchanged() {
    let payload = vec![
        1, 0, 0, 0, 0x44, 0, 0, 1, 0, 1, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
    ];
    let legacy = super::parse_iloc_with_methods(&payload).unwrap();
    assert_eq!(legacy.1, vec![(1, 2)]);

    let input = boxed(b"iloc", &payload);
    let bounds = limits(None);
    let mut context = ParseContext::native_still(&bounds);
    let mut state = MetaState::default();
    let error = parse_iloc(&input, &mut state, &mut context).unwrap_err();
    assert!(matches!(error, crate::DecoderError::Unsupported(_)));
    assert_eq!(context.accounting().metadata_live, 0);
}

/// The fixed iloc owner matrix deliberately derives its budget from Rust's
/// layout rather than learning it from the parser's ledger.  This keeps the
/// regression useful when an owner is accidentally omitted from accounting.
mod fixed_iloc_boundary {
    use super::{MetaState, boxed, parse_iloc};
    use crate::limits::NativeDecodeLimits;
    use crate::test_allocation_observer::{self as observe, Observation};
    use std::mem::size_of;

    fn bounds(maximum: usize) -> NativeDecodeLimits {
        NativeDecodeLimits::new(
            1 << 20,
            64,
            64,
            4096,
            8192,
            1 << 20,
            1 << 20,
            128,
            32,
            32,
            8,
            1,
        )
        .with_max_live_allocation_bytes(maximum)
        .expect("test live limit")
    }

    fn put(value: &mut Vec<u8>, width: u8, number: u64) {
        if width != 0 {
            value.extend_from_slice(&number.to_be_bytes()[8 - usize::from(width)..]);
        }
    }

    fn fixture(version: u8, width: u8, method: u16, counts: &[usize]) -> Vec<u8> {
        let mut payload = vec![
            version,
            0,
            0,
            0,
            (width << 4) | width,
            (width << 4) | if version == 0 { 0 } else { width },
        ];
        put(
            &mut payload,
            if version < 2 { 2 } else { 4 },
            counts.len() as u64,
        );
        for (ordinal, &count) in counts.iter().enumerate() {
            put(
                &mut payload,
                if version < 2 { 2 } else { 4 },
                (ordinal + 1) as u64,
            );
            if version != 0 {
                put(&mut payload, 2, method as u64);
            }
            put(&mut payload, 2, 0);
            put(&mut payload, width, 9 + ordinal as u64);
            put(&mut payload, 2, count as u64);
            for extent in 0..count {
                if version != 0 {
                    put(&mut payload, width, (extent + 1) as u64);
                }
                put(&mut payload, width, (extent * 3) as u64);
                put(&mut payload, width, if extent == 1 { 0 } else { 2 });
            }
        }
        boxed(b"iloc", &payload)
    }

    fn exact(counts: &[usize]) -> usize {
        counts.len()
            * (size_of::<super::super::ItemLocation>()
                + size_of::<(u32, u16)>()
                + size_of::<(u32, Vec<u64>)>()
                + size_of::<super::super::iloc_owners::ExtentOwnerTickets>())
            + counts.iter().sum::<usize>()
                * (size_of::<super::super::ItemExtent>() + size_of::<u64>())
    }

    fn shape(
        state: &MetaState,
        version: u8,
        width: u8,
        method: u16,
        counts: &[usize],
        native: bool,
    ) {
        assert_eq!(state.item_locations.len(), counts.len());
        assert_eq!(state.item_construction_methods.len(), counts.len());
        assert_eq!(state.item_extent_indexes.len(), counts.len());
        for (ordinal, &count) in counts.iter().enumerate() {
            let location = &state.item_locations[ordinal];
            let (id, indexes) = &state.item_extent_indexes[ordinal];
            assert_eq!(location.item_id, (ordinal + 1) as u32);
            assert_eq!(*id, location.item_id);
            assert_eq!(
                state.item_construction_methods[ordinal],
                (location.item_id, if version == 0 { 0 } else { method })
            );
            assert_eq!(
                location.base_offset,
                if width == 0 { 0 } else { 9 + ordinal as u64 }
            );
            assert_eq!(location.extents.len(), count);
            assert_eq!(indexes.len(), count);
            for (extent, (item_extent, &index)) in
                location.extents.iter().zip(indexes).enumerate().take(count)
            {
                assert_eq!(
                    item_extent.offset,
                    if width == 0 { 0 } else { (extent * 3) as u64 }
                );
                assert_eq!(
                    item_extent.length,
                    if width == 0 || extent == 1 { 0 } else { 2 }
                );
                assert_eq!(
                    index,
                    if width == 0 || version == 0 {
                        1
                    } else {
                        (extent + 1) as u64
                    }
                );
            }
        }
        if native {
            let owners = state.item_location_owners.as_ref().unwrap();
            assert_eq!(owners.entries.len(), counts.len());
            assert_eq!(
                owners.backing_token.charged_capacity_bytes,
                owners.entries.capacity()
                    * size_of::<super::super::iloc_owners::ExtentOwnerTickets>()
            );
            for (ordinal, owner) in owners.entries.iter().enumerate() {
                assert_eq!(owner.extents.class, super::super::AllocationClass::Metadata);
                assert_eq!(owner.indexes.class, super::super::AllocationClass::Metadata);
                assert_eq!(
                    owner.extents.charged_capacity_bytes,
                    state.item_locations[ordinal].extents.capacity()
                        * size_of::<super::super::ItemExtent>()
                );
                assert_eq!(
                    owner.indexes.charged_capacity_bytes,
                    state.item_extent_indexes[ordinal].1.capacity() * size_of::<u64>()
                );
            }
        } else {
            assert!(state.item_location_owners.is_none());
            assert_eq!(state.item_locations_token.charged_capacity_bytes, 0);
        }
    }

    fn pointers(state: &MetaState) -> [usize; 6] {
        [
            state.item_locations.as_ptr() as usize,
            state.item_construction_methods.as_ptr() as usize,
            state.item_extent_indexes.as_ptr() as usize,
            state.item_locations[0].extents.as_ptr() as usize,
            state.item_extent_indexes[0].1.as_ptr() as usize,
            state
                .item_location_owners
                .as_ref()
                .map_or(0, |owners| owners.entries.as_ptr() as usize),
        ]
    }

    fn register(observation: &Observation, state: &MetaState) {
        observation.track_raw_slot(
            0,
            state.item_locations.as_ptr().cast(),
            state.item_locations.capacity() * size_of::<super::super::ItemLocation>(),
        );
        observation.track_raw_slot(
            1,
            state.item_construction_methods.as_ptr().cast(),
            state.item_construction_methods.capacity() * size_of::<(u32, u16)>(),
        );
        observation.track_raw_slot(
            2,
            state.item_extent_indexes.as_ptr().cast(),
            state.item_extent_indexes.capacity() * size_of::<(u32, Vec<u64>)>(),
        );
        observation.track_raw_slot(
            3,
            state.item_locations[0].extents.as_ptr().cast(),
            state.item_locations[0].extents.capacity() * size_of::<super::super::ItemExtent>(),
        );
        observation.track_raw_slot(
            4,
            state.item_extent_indexes[0].1.as_ptr().cast(),
            state.item_extent_indexes[0].1.capacity() * size_of::<u64>(),
        );
        let owners = state.item_location_owners.as_ref().unwrap();
        observation.track_raw_slot(
            5,
            owners.entries.as_ptr().cast(),
            owners.entries.capacity() * size_of::<super::super::iloc_owners::ExtentOwnerTickets>(),
        );
    }

    fn release(state: MetaState, context: &mut super::super::container_budget::ParseContext<'_>) {
        super::super::release_native_parser_metadata(context, state).unwrap();
    }

    #[test]
    fn c2_iloc_1_full_shapes_original_tickets_fixed_exact_and_under() {
        for version in [0, 1, 2] {
            for width in [0, 4, 8] {
                for counts in [&[][..], &[0][..], &[1, 3][..]] {
                    let input = fixture(version, width, if version == 0 { 0 } else { 1 }, counts);
                    let before = input.clone();
                    let amount = exact(counts);
                    let exact_bounds = bounds(amount.max(1));
                    let mut context =
                        super::super::container_budget::ParseContext::native_still(&exact_bounds);
                    let mut state = MetaState::default();
                    parse_iloc(&input, &mut state, &mut context).unwrap();
                    shape(
                        &state,
                        version,
                        width,
                        if version == 0 { 0 } else { 1 },
                        counts,
                        true,
                    );
                    assert_eq!(context.accounting().metadata_live, amount);
                    assert_eq!(context.accounting().aggregate_peak, amount);
                    release(state, &mut context);
                    assert_eq!(context.accounting().aggregate_live, 0);
                    assert!(context.take_native_owners().is_none());
                    assert_eq!(input, before);

                    if amount > 1 {
                        let under_bounds = bounds(amount - 1);
                        let mut under_context =
                            super::super::container_budget::ParseContext::native_still(
                                &under_bounds,
                            );
                        let mut under_state = MetaState::default();
                        assert!(parse_iloc(&input, &mut under_state, &mut under_context).is_err());
                        assert_eq!(under_context.accounting().aggregate_live, 0);
                        assert!(under_state.item_locations.is_empty());
                    }
                }
            }
        }
    }

    #[test]
    fn c2_iloc_2_real_owner_denials_preserve_all_old_state_and_retry() {
        let old = fixture(0, 4, 0, &[1]);
        let incoming = fixture(2, 8, 1, &[3]);
        let before = incoming.clone();
        let sizes = [
            size_of::<super::super::ItemLocation>(),
            size_of::<(u32, u16)>(),
            size_of::<(u32, Vec<u64>)>(),
            7 * size_of::<super::super::iloc_owners::ExtentOwnerTickets>(),
            3 * size_of::<super::super::ItemExtent>(),
            3 * size_of::<u64>(),
        ];
        for (owner, size) in sizes.into_iter().enumerate() {
            let decode_bounds = bounds(1 << 20);
            let mut context =
                super::super::container_budget::ParseContext::native_still(&decode_bounds);
            let mut state = MetaState::default();
            parse_iloc(&old, &mut state, &mut context).unwrap();
            let old_pointers = pointers(&state);
            let checkpoint = context.checkpoint();
            let old_shape = format!("{state:?}");
            let observation = Observation::begin(size, true);
            let extra = if owner == 3 {
                Some(observe::force_fresh_capacity_extra(
                    "iloc extent owner tickets",
                    7,
                ))
            } else {
                None
            };
            let result = parse_iloc(&incoming, &mut state, &mut context);
            assert!(
                matches!(result, Err(crate::DecoderError::InvalidParam(_))),
                "owner {owner}: {result:?}"
            );
            assert_eq!(observation.requests(), 1, "owner {owner}");
            assert!(observation.restore_count() >= 1);
            drop(result);
            drop(observation);
            drop(extra);
            assert_eq!(context.checkpoint(), checkpoint);
            assert_eq!(pointers(&state), old_pointers);
            assert_eq!(format!("{state:?}"), old_shape);

            parse_iloc(&incoming, &mut state, &mut context).unwrap();
            shape(&state, 2, 8, 1, &[3], true);
            assert_eq!(context.accounting().aggregate_live, exact(&[3]));
            assert!(context.accounting().aggregate_peak >= exact(&[1]) + exact(&[3]));
            release(state, &mut context);
            assert_eq!(context.accounting().aggregate_live, 0);
            assert_eq!(incoming, before);
        }
    }

    #[test]
    fn c2_iloc_3_real_replacement_and_final_debits_follow_actual_owner_drop() {
        for final_drop in [false, true] {
            let old = fixture(1, 4, 1, &[3]);
            let incoming = fixture(2, 8, 1, &[2]);
            let decode_bounds = bounds(exact(&[3]) + exact(&[2]));
            let mut context =
                super::super::container_budget::ParseContext::native_still(&decode_bounds);
            let mut state = MetaState::default();
            parse_iloc(&old, &mut state, &mut context).unwrap();
            let observation = Observation::begin(1, false);
            register(&observation, &state);
            if final_drop {
                release(state, &mut context);
                state = MetaState::default();
            } else {
                parse_iloc(&incoming, &mut state, &mut context).unwrap();
            }
            assert_eq!(observation.release_count(), if final_drop { 11 } else { 6 });
            let masks = observation.release_drop_masks();
            let drops = observation.registered_drops();
            assert_eq!(&drops[..6], &[1; 6]);
            for mask in &masks[..6] {
                assert!(mask[..5].iter().all(|dropped| *dropped));
            }
            assert!(!masks[0][5] && !masks[1][5]);
            assert!(masks[2][5]);
            drop(observation);
            if !final_drop {
                shape(&state, 2, 8, 1, &[2], true);
                release(state, &mut context);
            }
            assert_eq!(context.accounting().aggregate_live, 0);
        }
    }

    #[test]
    fn c2_iloc_4_excess_sidecar_at_real_inner_outer_restores_same_state_retry() {
        let old = fixture(0, 4, 0, &[1]);
        let incoming = fixture(2, 8, 1, &[3]);
        let before = incoming.clone();
        let decode_bounds = bounds(exact(&[1]) + exact(&[3]));
        let mut context =
            super::super::container_budget::ParseContext::native_still(&decode_bounds);
        let mut state = MetaState::default();
        parse_iloc(&old, &mut state, &mut context).unwrap();
        let old_pointers = pointers(&state);
        let checkpoint = context.checkpoint();
        let old_shape = format!("{state:?}");
        let observation = Observation::begin(1, false);
        let candidates = observe::track_all_candidates();
        let extra = observe::force_fresh_capacity_extra("iloc extent owner tickets", 8);
        assert!(parse_iloc(&incoming, &mut state, &mut context).is_err());
        assert_eq!(observation.restore_count(), 2);
        assert_eq!(observation.drops(), 1);
        let masks = observation.restore_drop_masks();
        assert_eq!(&masks[0][..4], &[false, false, false, true]);
        assert_eq!(&masks[1][..4], &[true; 4]);
        assert_eq!(&observation.registered_drops()[..4], &[1; 4]);
        drop(observation);
        drop(candidates);
        drop(extra);
        assert_eq!(pointers(&state), old_pointers);
        assert_eq!(format!("{state:?}"), old_shape);
        assert_eq!(context.checkpoint(), checkpoint);
        parse_iloc(&incoming, &mut state, &mut context).unwrap();
        shape(&state, 2, 8, 1, &[3], true);
        release(state, &mut context);
        assert_eq!(context.accounting().aggregate_live, 0);
        assert_eq!(incoming, before);
    }

    #[test]
    fn c2_iloc_5_legacy_repeated_shape_and_existing_errors_unchanged() {
        let input = fixture(0, 4, 0, &[1]);
        let incoming = fixture(2, 8, 2, &[3]);
        let mut context = super::super::container_budget::ParseContext::legacy();
        let mut state = MetaState::default();
        parse_iloc(&input, &mut state, &mut context).unwrap();
        let observation = Observation::begin(
            7 * size_of::<super::super::iloc_owners::ExtentOwnerTickets>(),
            true,
        );
        let extra = observe::force_fresh_capacity_extra("iloc extent owner tickets", 7);
        parse_iloc(&incoming, &mut state, &mut context).unwrap();
        assert_eq!(observation.requests(), 0);
        drop(observation);
        drop(extra);
        shape(&state, 2, 8, 2, &[3], false);

        let before = format!("{state:?}");
        let mut malformed = fixture(2, 8, 1, &[2]);
        malformed.pop();
        let length = malformed.len() as u32;
        malformed[..4].copy_from_slice(&length.to_be_bytes());
        assert!(parse_iloc(&malformed, &mut state, &mut context).is_err());
        assert_eq!(format!("{state:?}"), before);

        let native_bounds = bounds(1 << 20);
        let mut native = super::super::container_budget::ParseContext::native_still(&native_bounds);
        let mut native_state = MetaState::default();
        assert!(matches!(
            parse_iloc(&incoming, &mut native_state, &mut native),
            Err(crate::DecoderError::Unsupported(_))
        ));
        assert_eq!(native.accounting().aggregate_live, 0);
        for (offset, value, message) in [
            (8usize, 3u8, "iloc version 3 is not supported"),
            (12, 0x14, "iloc offset_size=1"),
            (
                19,
                1,
                "external AVIF item data references are not supported",
            ),
        ] {
            let mut bad = input.clone();
            bad[offset] = value;
            let legacy = super::super::parse_iloc_with_indexes(&bad[8..]).unwrap_err();
            assert!(matches!(
                legacy,
                crate::DecoderError::Unsupported(ref message_value) if message_value == message
            ));
            let error = parse_iloc(&bad, &mut native_state, &mut native).unwrap_err();
            assert_eq!(format!("{error:?}"), format!("{legacy:?}"));
        }
    }
}
