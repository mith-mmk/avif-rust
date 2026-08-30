use crate::DecoderError;
use crate::av1::decode::FrameDecodePlan;
use crate::av1::frame::FrameHeader;
use crate::av1::sequence::SequenceHeader;
use crate::av1::tile_decode::public_api::decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options_with_references_and_cdf_and_motion_and_budget;
use crate::av1::tile_group::TileGroup;
use crate::av1::{
    alloc_frame_buffers, build_still_decode_plan, parse_frame_header, parse_sequence_header,
    parse_tile_group,
};
use crate::container::{DecodeBudget, parse_avif};
use crate::obu::{ObuType, find_obu_payload};

fn sample_decode_inputs() -> Option<(
    Vec<u8>,
    SequenceHeader,
    FrameHeader,
    TileGroup,
    FrameDecodePlan,
)> {
    let data = crate::test_support::wml2viewer_avif()?;
    let info = parse_avif(&data).ok()?;
    let sequence_payload =
        find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader).ok()??;
    let sequence = parse_sequence_header(sequence_payload).ok()?;
    let frame_payload = find_obu_payload(&info.primary_item_payload, ObuType::Frame)
        .ok()??
        .to_vec();
    let frame = parse_frame_header(&frame_payload, &sequence).ok()?;
    let tile_group = parse_tile_group(
        &frame_payload,
        frame.uncompressed_header_bits,
        &frame.tile_info,
    )
    .ok()?;
    let plan = build_still_decode_plan(&sequence, &frame, &tile_group).ok()?;
    Some((frame_payload, sequence, frame, tile_group, plan))
}

fn decode_sample(budget: &mut DecodeBudget) -> Result<usize, DecoderError> {
    let Some((frame_payload, sequence, frame, tile_group, plan)) = sample_decode_inputs() else {
        return Err(DecoderError::InvalidParam(
            "palette sample fixture is unavailable".to_string(),
        ));
    };
    let mut buffers = alloc_frame_buffers(&plan)?;
    let (_prefix, state, cdfs, motion) =
        decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options_with_references_and_cdf_and_motion_and_budget(
            &frame_payload,
            &tile_group,
            &sequence,
            &frame,
            &plan,
            &mut buffers,
            4096,
            true,
            false,
            std::array::from_fn(|_| None),
            None,
            true,
            false,
            None,
            budget,
        )?;
    state.release(budget)?;
    drop(motion);
    Ok(cdfs.len())
}

#[test]
fn strict_palette_path_accounts_cumulative_owners_and_retries_after_failure() {
    let mut reference_budget = DecodeBudget::new(None);
    crate::av1::tile_decode::reset_observed_palette_allocation_labels();
    let cdf_count =
        decode_sample(&mut reference_budget).expect("strict palette fixture must decode");
    let exact_bytes = reference_budget.accounting().aggregate_peak;
    let reference_labels = crate::av1::tile_decode::observed_palette_allocation_labels();
    assert!(reference_labels.iter().all(|count| *count > 0));
    assert_eq!(reference_budget.accounting().aggregate_live, 0);
    assert!(cdf_count > 0, "state-collecting strict route must run");
    assert!(
        exact_bytes > 1,
        "strict palette route must have a nonzero peak"
    );

    // M-1 is rejected by the complete strict plan before any palette/cache
    // candidate is created.  The forced label makes an accidental palette
    // allocation observable without counting ordinary diagnostic buffers.
    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let _all_candidates = crate::test_allocation_observer::track_all_candidates();
    crate::av1::tile_decode::reset_observed_palette_allocation_labels();
    let under_extra = crate::test_allocation_observer::force_fresh_capacity_extra(
        "native AV1 palette color map",
        1 << 20,
    );
    let mut under_budget = DecodeBudget::new(Some(exact_bytes - 1));
    let error = decode_sample(&mut under_budget).expect_err(
        "one byte below the measured cumulative palette/cache peak must fail before allocation",
    );
    assert!(
        matches!(error, DecoderError::InvalidParam(message) if message.contains("live allocation"))
    );
    assert_eq!(observation.drops(), 0);
    assert_eq!(observation.registered_candidate_count(), 0);
    assert!(
        crate::av1::tile_decode::observed_palette_allocation_labels()
            .iter()
            .all(|count| *count == 0)
    );
    assert_eq!(under_budget.accounting().aggregate_live, 0);
    drop(observation);
    drop(under_extra);
    drop(_all_candidates);

    // The over-capacity candidate must be physically dropped before retrying.
    // Use the plan's dynamic reservation plus one byte so the first forced
    // palette-map Vec fails its actual-capacity admission immediately.
    let Some((_payload, _sequence, frame, _group, plan)) = sample_decode_inputs() else {
        return;
    };
    let tile = plan
        .tiles
        .first()
        .expect("palette sample must contain a tile");
    let dynamic_reservation = super::TileDecoderMemoryPlan::for_frame(&frame, Some(tile))
        .expect("palette tile memory plan must be representable")
        .dynamic_bytes;
    let forced_extra = dynamic_reservation
        .checked_add(1)
        .expect("forced palette capacity must fit");
    let labels = [
        "native AV1 palette cached luma colors",
        "native AV1 palette luma colors",
        "native AV1 palette cached chroma colors",
        "native AV1 palette U/V colors",
        "native AV1 palette V colors",
        "native AV1 palette color map",
        "native AV1 nested luma palette colors",
        "native AV1 nested chroma palette colors",
    ];
    let mut retry_budget = DecodeBudget::new(Some(exact_bytes));
    for (expected_index, label) in labels.into_iter().enumerate() {
        let observation = crate::test_allocation_observer::Observation::begin(1, false);
        crate::av1::tile_decode::reset_observed_palette_allocation_labels();
        let extra =
            crate::test_allocation_observer::force_fresh_capacity_extra(label, forced_extra);
        let error = decode_sample(&mut retry_budget)
            .expect_err("forced palette overcapacity must fail transactionally");
        assert!(
            matches!(error, DecoderError::InvalidParam(ref message) if message.contains("capacity exceeds admitted budget")),
            "forced route {label} returned {error:?}"
        );
        assert_eq!(
            observation.drops(),
            1,
            "forced route {label} must drop its candidate"
        );
        let observed = crate::av1::tile_decode::observed_palette_allocation_labels();
        assert!(
            observed[expected_index] > 0,
            "forced route {label} did not reach its allocation hook: {observed:?}"
        );
        assert_eq!(retry_budget.accounting().aggregate_live, 0);
        drop(observation);
        drop(extra);

        let retry_cdf_count = decode_sample(&mut retry_budget)
            .expect("same budget must retry after forced palette failure");
        assert_eq!(retry_cdf_count, cdf_count);
        assert_eq!(retry_budget.accounting().aggregate_live, 0);
    }
}

#[test]
fn palette_scratch_owner_preserves_bytes_across_overflow_and_retry() {
    let owner = super::PaletteScratchLedger::default();
    owner.add(32).expect("initial scratch debit must succeed");
    let baseline = owner.bytes();
    assert!(owner.add(usize::MAX).is_err());
    assert_eq!(owner.bytes(), baseline);
    owner
        .add(16)
        .expect("same scratch owner must remain usable");
    assert_eq!(owner.bytes(), baseline + 16);
}
