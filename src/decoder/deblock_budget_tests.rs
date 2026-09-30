use super::{
    STRICT_CDEF_DIRECTION_BLOCK_LABEL, STRICT_CDEF_INDEX_LABEL, STRICT_CDEF_MASK_LABEL,
    STRICT_CDEF_ORIGINS_LABEL, STRICT_CDEF_SNAPSHOT_LABELS, STRICT_DEBLOCK_FILTER_GRID_LABEL,
    STRICT_DEBLOCK_HORIZONTAL_ORDER_LABEL, STRICT_RESTORATION_BOUNDARY_META_LABEL,
    STRICT_RESTORATION_BOUNDARY_SAMPLES_LABELS, STRICT_RESTORATION_SCRATCH_LABELS,
    STRICT_RESTORATION_SNAPSHOT_LABELS, StrictCdefDirectionBlockOwner, StrictCdefDirectionContext,
    StrictCdefIndexOwner, StrictCdefIndexPlan, StrictCdefScratchOwner, StrictCdefScratchPlan,
    StrictCdefSnapshotOwner, StrictCdefSnapshotPlan, StrictDeblockBoundaryOrderOwner,
    StrictDeblockBoundaryOrderPlan, StrictDeblockFilterGridOwner, StrictDeblockFilterGridPlan,
    StrictDeblockWorkspaceOwner, StrictRestorationBoundaryOwner, StrictRestorationBoundaryPlan,
    StrictRestorationExecutionScratchOwner, StrictRestorationExecutionScratchPlan,
    StrictRestorationSnapshotPlan, StrictRestorationWorkspaceOwner, apply_cdef_plane,
    apply_cdef_plane_from_source, sort_strict_deblock_horizontal_order,
    sort_strict_deblock_vertical_order, validate_strict_cdef_filter_endpoints,
};
use crate::av1::{
    BlockFilterState, BlockSize, CdefBlockIndex, CdefParams, CdefUnit, ColorConfig,
    ColorDescription, ColorRange, FrameBuffers, FrameDecodePlan, NativeFrameAllocation,
    PlaneBuffer, PlaneLayout, PostFilterState, PredictionMode, RestorationUnit, TransformBlock,
    TransformBoundary, TxMode, TxSize, TxType, alloc_coded_frame_buffers_with_budget,
};
use crate::container::DecodeBudget;

fn boundary_state(count: usize) -> PostFilterState {
    let transform_boundaries = (0..count)
        .map(|index| TransformBoundary {
            block: TransformBlock {
                plane: 0,
                x: index * 4,
                y: index * 8,
                tx_size: TxSize::Tx4x4,
            },
            tx_type: TxType::DctDct,
            non_zero_coefficients: 0,
            skip: false,
            is_inter: false,
            reference_frame: None,
            has_nonzero_mv: false,
            y_mode: PredictionMode::Dc,
            uv_mode: None,
        })
        .collect();
    PostFilterState {
        transform_boundaries,
        ..PostFilterState::default()
    }
}

#[test]
fn deblock_filter_grid_plan_checks_empty_and_overflow_dimensions() {
    assert!(StrictDeblockFilterGridPlan::for_frame(0, 8).is_err());
    assert!(StrictDeblockFilterGridPlan::for_frame(8, 0).is_err());
    assert!(StrictDeblockFilterGridPlan::for_frame(usize::MAX, usize::MAX).is_err());
}

#[test]
fn deblock_filter_grid_exact_admission_releases_before_retry() {
    let plan = StrictDeblockFilterGridPlan::for_frame(8, 8).unwrap();
    assert!(plan.requested_bytes > 0);
    let mut budget = DecodeBudget::new(Some(plan.requested_bytes));
    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let all_candidates = crate::test_allocation_observer::track_all_candidates();
    let owner = StrictDeblockFilterGridOwner::admit(plan, &mut budget).unwrap();
    assert_eq!(budget.accounting().frame_live, plan.requested_bytes);
    owner.release(&mut budget).unwrap();
    assert!(observation.release_drop_snapshots()[0] > 0);
    assert_eq!(budget.accounting().frame_live, 0);
    drop(all_candidates);
    drop(observation);
    let owner = StrictDeblockFilterGridOwner::admit(plan, &mut budget).unwrap();
    owner.release(&mut budget).unwrap();
    assert_eq!(budget.accounting().frame_live, 0);
}

#[test]
fn deblock_filter_grid_one_under_and_actual_overage_are_transactional() {
    let plan = StrictDeblockFilterGridPlan::for_frame(16, 9).unwrap();
    let mut under = DecodeBudget::new(Some(plan.requested_bytes - 1));
    assert!(StrictDeblockFilterGridOwner::admit(plan, &mut under).is_err());
    assert_eq!(under.accounting().frame_live, 0);

    let mut budget = DecodeBudget::new(Some(plan.requested_bytes));
    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let extra = crate::test_allocation_observer::force_fresh_capacity_extra(
        STRICT_DEBLOCK_FILTER_GRID_LABEL,
        1 << 20,
    );
    assert!(StrictDeblockFilterGridOwner::admit(plan, &mut budget).is_err());
    assert_eq!(observation.drops(), 1);
    drop(extra);
    drop(observation);
    let owner = StrictDeblockFilterGridOwner::admit(plan, &mut budget).unwrap();
    owner.release(&mut budget).unwrap();
    assert_eq!(budget.accounting().frame_live, 0);
}

#[test]
fn deblock_boundary_orders_are_shared_and_transactional() {
    let state = boundary_state(3);
    let plan = StrictDeblockBoundaryOrderPlan::for_count(3).unwrap();
    assert_eq!(plan.requested_bytes, 3 * 4 * std::mem::size_of::<usize>());

    let mut under = DecodeBudget::new(Some(plan.requested_bytes - 1));
    assert!(StrictDeblockBoundaryOrderOwner::admit(plan, &state, &mut under).is_err());
    assert_eq!(under.accounting().frame_live, 0);

    let mut budget = DecodeBudget::new(Some(plan.requested_bytes));
    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let all_candidates = crate::test_allocation_observer::track_all_candidates();
    let owner = StrictDeblockBoundaryOrderOwner::admit(plan, &state, &mut budget).unwrap();
    assert_eq!(observation.registered_candidate_count(), 4);
    owner.release(&mut budget).unwrap();
    assert_eq!(budget.accounting().frame_live, 0);
    assert_eq!(
        &observation.release_drop_masks()[0][..4],
        &[true, true, true, true]
    );
    drop(all_candidates);
    drop(observation);

    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let extra = crate::test_allocation_observer::force_fresh_capacity_extra(
        STRICT_DEBLOCK_HORIZONTAL_ORDER_LABEL,
        1 << 20,
    );
    assert!(StrictDeblockBoundaryOrderOwner::admit(plan, &state, &mut budget).is_err());
    assert_eq!(observation.drops(), 1);
    assert_eq!(budget.accounting().frame_live, 0);
    drop(extra);
    drop(observation);
    let owner = StrictDeblockBoundaryOrderOwner::admit(plan, &state, &mut budget).unwrap();
    owner.release(&mut budget).unwrap();
    assert_eq!(budget.accounting().frame_live, 0);
}

#[test]
fn deblock_boundary_order_sort_uses_no_heap_scratch_and_preserves_ties() {
    let mut state = boundary_state(4);
    for boundary in &mut state.transform_boundaries {
        boundary.block.x = 0;
        boundary.block.y = 0;
    }
    let boundaries = &state.transform_boundaries;
    let mut vertical = vec![3, 1, 0, 2];
    let mut horizontal = vertical.clone();
    let ((), vertical_requests) =
        crate::test_allocation_observer::count_allocation_requests(|| {
            sort_strict_deblock_vertical_order(&mut vertical, boundaries);
        });
    let ((), horizontal_requests) =
        crate::test_allocation_observer::count_allocation_requests(|| {
            sort_strict_deblock_horizontal_order(&mut horizontal, boundaries);
        });
    assert_eq!(vertical_requests, 0);
    assert_eq!(horizontal_requests, 0);
    assert_eq!(vertical, [0, 1, 2, 3]);
    assert_eq!(horizontal, [0, 1, 2, 3]);
}

#[test]
fn deblock_workspace_combined_under_limit_rejects_before_candidates() {
    let state = boundary_state(3);
    let grid_plan = StrictDeblockFilterGridPlan::for_frame(8, 8).unwrap();
    let order_plan = StrictDeblockBoundaryOrderPlan::for_count(3).unwrap();
    let requested = grid_plan
        .requested_bytes
        .checked_add(order_plan.requested_bytes)
        .unwrap();
    let mut budget = DecodeBudget::new(Some(requested - 1));
    let baseline = budget.accounting();
    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let all_candidates = crate::test_allocation_observer::track_all_candidates();
    assert!(
        StrictDeblockWorkspaceOwner::admit(grid_plan, order_plan, &state, &mut budget,).is_err()
    );
    assert_eq!(observation.registered_candidate_count(), 0);
    assert_eq!(observation.drops(), 0);
    assert_eq!(budget.accounting(), baseline);
    drop(all_candidates);
    drop(observation);
}

#[test]
fn strict_cdef_index_plan_checks_zero_and_overflow() {
    assert!(StrictCdefIndexPlan::for_visible_dimensions(0, 64).is_err());
    assert!(StrictCdefIndexPlan::for_visible_dimensions(64, 0).is_err());
    assert!(StrictCdefIndexPlan::for_visible_dimensions(usize::MAX, 64).is_err());
    assert!(StrictCdefIndexPlan::for_visible_dimensions(64, usize::MAX).is_err());
}

#[test]
fn strict_cdef_index_owner_is_transactional_and_preserves_table() {
    let mut state = PostFilterState::default();
    state.cdef_units.push(CdefUnit {
        x: 64,
        y: 0,
        index: 3,
    });
    state.cdef_blocks.push(CdefBlockIndex {
        x: 0,
        y: 64,
        index: 5,
    });
    let plan = StrictCdefIndexPlan::for_visible_dimensions(128, 128).unwrap();
    assert_eq!(plan.requested_bytes, 4);
    let mut under = DecodeBudget::new(Some(plan.requested_bytes - 1));
    assert!(StrictCdefIndexOwner::admit(plan, 3, &state, &mut under).is_err());
    assert_eq!(under.accounting().frame_live, 0);

    let mut budget = DecodeBudget::new(Some(plan.requested_bytes));
    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let all_candidates = crate::test_allocation_observer::track_all_candidates();
    let owner = StrictCdefIndexOwner::admit(plan, 3, &state, &mut budget).unwrap();
    assert_eq!(owner.indices, [u8::MAX, 3, 5, u8::MAX]);
    owner.release(&mut budget).unwrap();
    assert_eq!(budget.accounting().frame_live, 0);
    assert_eq!(observation.registered_candidate_count(), 1);
    assert!(observation.release_drop_masks()[0][0]);
    drop(all_candidates);
    drop(observation);

    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let extra = crate::test_allocation_observer::force_fresh_capacity_extra(
        STRICT_CDEF_INDEX_LABEL,
        1 << 20,
    );
    assert!(StrictCdefIndexOwner::admit(plan, 3, &state, &mut budget).is_err());
    assert_eq!(observation.drops(), 1);
    assert_eq!(budget.accounting().frame_live, 0);
    drop(extra);
    drop(observation);
    let owner = StrictCdefIndexOwner::admit(plan, 3, &state, &mut budget).unwrap();
    owner.release(&mut budget).unwrap();
    assert_eq!(budget.accounting().frame_live, 0);
}

fn cdef_filter_state(x: usize, y: usize, block_size: BlockSize) -> BlockFilterState {
    BlockFilterState {
        x,
        y,
        block_size,
        segment_id: 0,
        skip: false,
        is_inter: false,
        reference_frame: None,
        has_nonzero_mv: false,
        y_mode: PredictionMode::Dc,
        uv_mode: None,
        delta_lf: [0; 4],
    }
}

fn admit_cdef_index_owner(
    width: usize,
    height: usize,
    state: &PostFilterState,
    budget: &mut DecodeBudget,
) -> StrictCdefIndexOwner {
    let plan = StrictCdefIndexPlan::for_visible_dimensions(width, height).unwrap();
    StrictCdefIndexOwner::admit(plan, 3, state, budget).unwrap()
}

fn admit_cdef_scratch_owner(
    width: usize,
    height: usize,
    state: &PostFilterState,
    budget: &mut DecodeBudget,
) -> StrictCdefScratchOwner {
    let index_plan = StrictCdefIndexPlan::for_visible_dimensions(width, height).unwrap();
    let index_owner = StrictCdefIndexOwner::admit(index_plan, 3, state, budget).unwrap();
    let scratch_plan = StrictCdefScratchPlan::for_visible_dimensions(width, height).unwrap();
    StrictCdefScratchOwner::admit(
        scratch_plan,
        index_owner,
        index_plan.units_width,
        state,
        width,
        height,
        budget,
    )
    .unwrap()
}

fn admit_cdef_direction_owner(budget: &mut DecodeBudget) -> StrictCdefDirectionBlockOwner {
    let mut state = PostFilterState::default();
    state.cdef_units.push(CdefUnit {
        x: 0,
        y: 0,
        index: 0,
    });
    state
        .block_filter_states
        .push(cdef_filter_state(0, 0, BlockSize::Block8x8));
    let index_plan = StrictCdefIndexPlan::for_visible_dimensions(8, 8).unwrap();
    let index_owner = StrictCdefIndexOwner::admit(index_plan, 3, &state, budget).unwrap();
    let scratch_plan = StrictCdefScratchPlan::for_visible_dimensions(8, 8).unwrap();
    let scratch_owner = StrictCdefScratchOwner::admit(
        scratch_plan,
        index_owner,
        index_plan.units_width,
        &state,
        8,
        8,
        budget,
    )
    .unwrap();
    let samples = [0u16; 64];
    StrictCdefDirectionBlockOwner::admit(
        scratch_owner,
        StrictCdefDirectionContext {
            luma_samples: &samples,
            luma_width: 8,
            luma_height: 8,
            visible_width: 8,
            visible_height: 8,
            coeff_shift: 0,
        },
        budget,
    )
    .unwrap()
}

#[test]
fn strict_cdef_scratch_plan_checks_zero_and_overflow() {
    assert!(StrictCdefScratchPlan::for_visible_dimensions(0, 8).is_err());
    assert!(StrictCdefScratchPlan::for_visible_dimensions(8, 0).is_err());
    assert!(StrictCdefScratchPlan::for_visible_dimensions(usize::MAX, usize::MAX).is_err());
}

#[test]
fn strict_cdef_filter_endpoints_are_checked_before_admission() {
    let mut state = PostFilterState::default();
    state
        .block_filter_states
        .push(cdef_filter_state(usize::MAX - 3, 0, BlockSize::Block4x4));
    assert!(validate_strict_cdef_filter_endpoints(&state, usize::MAX, 64).is_err());

    state.block_filter_states[0] = cdef_filter_state(65, 0, BlockSize::Block8x8);
    assert!(validate_strict_cdef_filter_endpoints(&state, 64, 64).is_err());

    state.block_filter_states[0] = cdef_filter_state(64, 0, BlockSize::Block8x8);
    assert!(validate_strict_cdef_filter_endpoints(&state, 64, 64).is_ok());
}

#[test]
fn strict_cdef_scratch_preserves_mask_and_row_major_origin_parity() {
    let mut state = PostFilterState::default();
    state
        .block_filter_states
        .push(cdef_filter_state(0, 0, BlockSize::Block8x8));
    state
        .block_filter_states
        .push(cdef_filter_state(4, 0, BlockSize::Block8x8));
    state
        .block_filter_states
        .push(cdef_filter_state(8, 8, BlockSize::Block8x8));
    state.cdef_units.push(CdefUnit {
        x: 0,
        y: 0,
        index: 3,
    });
    let plan = StrictCdefScratchPlan::for_visible_dimensions(16, 16).unwrap();
    let index_plan = StrictCdefIndexPlan::for_visible_dimensions(16, 16).unwrap();
    let mut budget = DecodeBudget::new(Some(
        plan.requested_bytes
            .checked_add(index_plan.requested_bytes)
            .unwrap(),
    ));
    let index_owner = admit_cdef_index_owner(16, 16, &state, &mut budget);
    let owner = StrictCdefScratchOwner::admit(
        plan,
        index_owner,
        index_plan.units_width,
        &state,
        16,
        16,
        &mut budget,
    )
    .unwrap();
    assert_eq!(budget.accounting().frame_live, plan.origins_bytes);
    assert_eq!(owner.origins, [(0, 0, 3), (8, 0, 3), (8, 8, 3)]);
    owner.release(&mut budget).unwrap();
    assert_eq!(budget.accounting().frame_live, 0);
}

#[test]
fn strict_cdef_scratch_combined_preflight_and_individual_rollback_are_retryable() {
    let mut state = PostFilterState::default();
    state
        .block_filter_states
        .push(cdef_filter_state(0, 0, BlockSize::Block8x8));
    state.cdef_units.push(CdefUnit {
        x: 0,
        y: 0,
        index: 1,
    });
    let plan = StrictCdefScratchPlan::for_visible_dimensions(8, 8).unwrap();
    let index_plan = StrictCdefIndexPlan::for_visible_dimensions(8, 8).unwrap();
    let total_requested = index_plan
        .requested_bytes
        .checked_add(plan.requested_bytes)
        .unwrap();

    let mut under = DecodeBudget::new(Some(total_requested - 1));
    let index_owner = admit_cdef_index_owner(8, 8, &state, &mut under);
    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let all_candidates = crate::test_allocation_observer::track_all_candidates();
    assert!(
        StrictCdefScratchOwner::admit(
            plan,
            index_owner,
            index_plan.units_width,
            &state,
            8,
            8,
            &mut under,
        )
        .is_err()
    );
    assert_eq!(observation.registered_candidate_count(), 0);
    assert_eq!(under.accounting().frame_live, 0);
    drop(all_candidates);
    drop(observation);

    let mut budget = DecodeBudget::new(Some(total_requested));
    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let extra = crate::test_allocation_observer::force_fresh_capacity_extra(
        STRICT_CDEF_MASK_LABEL,
        1 << 20,
    );
    let index_owner = admit_cdef_index_owner(8, 8, &state, &mut budget);
    assert!(
        StrictCdefScratchOwner::admit(
            plan,
            index_owner,
            index_plan.units_width,
            &state,
            8,
            8,
            &mut budget,
        )
        .is_err()
    );
    assert_eq!(budget.accounting().frame_live, 0);
    assert_eq!(observation.drops(), 1);
    drop(extra);
    drop(observation);

    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let extra = crate::test_allocation_observer::force_fresh_capacity_extra(
        STRICT_CDEF_ORIGINS_LABEL,
        1 << 20,
    );
    let index_owner = admit_cdef_index_owner(8, 8, &state, &mut budget);
    assert!(
        StrictCdefScratchOwner::admit(
            plan,
            index_owner,
            index_plan.units_width,
            &state,
            8,
            8,
            &mut budget,
        )
        .is_err()
    );
    assert_eq!(budget.accounting().frame_live, 0);
    assert_eq!(observation.drops(), 1);
    drop(extra);
    drop(observation);

    let index_owner = admit_cdef_index_owner(8, 8, &state, &mut budget);
    let owner = StrictCdefScratchOwner::admit(
        plan,
        index_owner,
        index_plan.units_width,
        &state,
        8,
        8,
        &mut budget,
    )
    .unwrap();
    owner.release(&mut budget).unwrap();
    assert_eq!(budget.accounting().frame_live, 0);
}

#[test]
fn strict_cdef_direction_blocks_consume_origins_and_are_transactional() {
    let mut state = PostFilterState::default();
    state.cdef_units.push(CdefUnit {
        x: 0,
        y: 0,
        index: 1,
    });
    state
        .block_filter_states
        .push(cdef_filter_state(0, 0, BlockSize::Block8x8));
    let index_plan = StrictCdefIndexPlan::for_visible_dimensions(8, 8).unwrap();
    let scratch_plan = StrictCdefScratchPlan::for_visible_dimensions(8, 8).unwrap();
    let block_bytes = std::mem::size_of::<(usize, usize, usize, usize, i32)>();
    let total = index_plan
        .requested_bytes
        .checked_add(scratch_plan.requested_bytes)
        .and_then(|bytes| bytes.checked_add(block_bytes))
        .unwrap();
    let samples = [0u16; 64];
    let context = StrictCdefDirectionContext {
        luma_samples: &samples,
        luma_width: 8,
        luma_height: 8,
        visible_width: 8,
        visible_height: 8,
        coeff_shift: 0,
    };
    let mut budget = DecodeBudget::new(Some(total));
    let origins = admit_cdef_scratch_owner(8, 8, &state, &mut budget);
    let owner = StrictCdefDirectionBlockOwner::admit(origins, context, &mut budget).unwrap();
    assert_eq!(budget.accounting().frame_live, block_bytes);
    assert_eq!(owner.blocks.len(), 1);
    owner.release(&mut budget).unwrap();
    assert_eq!(budget.accounting().frame_live, 0);
}

#[test]
fn strict_cdef_direction_block_preflight_and_overcapacity_retry_leave_no_live_bytes() {
    let mut state = PostFilterState::default();
    state.cdef_units.push(CdefUnit {
        x: 0,
        y: 0,
        index: 1,
    });
    state
        .block_filter_states
        .push(cdef_filter_state(0, 0, BlockSize::Block8x8));
    let index_plan = StrictCdefIndexPlan::for_visible_dimensions(8, 8).unwrap();
    let scratch_plan = StrictCdefScratchPlan::for_visible_dimensions(8, 8).unwrap();
    let block_bytes = std::mem::size_of::<(usize, usize, usize, usize, i32)>();
    let total = index_plan
        .requested_bytes
        .checked_add(scratch_plan.requested_bytes)
        .and_then(|bytes| bytes.checked_add(block_bytes))
        .unwrap();
    let samples = [0u16; 64];
    let context = StrictCdefDirectionContext {
        luma_samples: &samples,
        luma_width: 8,
        luma_height: 8,
        visible_width: 8,
        visible_height: 8,
        coeff_shift: 0,
    };

    let mut under = DecodeBudget::new(Some(
        scratch_plan.origins_bytes.checked_add(block_bytes).unwrap() - 1,
    ));
    let origins = admit_cdef_scratch_owner(8, 8, &state, &mut under);
    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let all_candidates = crate::test_allocation_observer::track_all_candidates();
    assert!(StrictCdefDirectionBlockOwner::admit(origins, context, &mut under).is_err());
    assert_eq!(observation.registered_candidate_count(), 0);
    assert_eq!(under.accounting().frame_live, 0);
    drop(all_candidates);
    drop(observation);

    let mut budget = DecodeBudget::new(Some(total));
    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let extra = crate::test_allocation_observer::force_fresh_capacity_extra(
        STRICT_CDEF_DIRECTION_BLOCK_LABEL,
        1 << 20,
    );
    let origins = admit_cdef_scratch_owner(8, 8, &state, &mut budget);
    assert!(StrictCdefDirectionBlockOwner::admit(origins, context, &mut budget).is_err());
    assert_eq!(observation.drops(), 1);
    assert_eq!(budget.accounting().frame_live, 0);
    drop(extra);
    drop(observation);

    let origins = admit_cdef_scratch_owner(8, 8, &state, &mut budget);
    let owner = StrictCdefDirectionBlockOwner::admit(origins, context, &mut budget).unwrap();
    owner.release(&mut budget).unwrap();
    assert_eq!(budget.accounting().frame_live, 0);
}

#[cfg(not(target_family = "wasm"))]
#[test]
fn strict_cdef_parallel_direction_fill_matches_serial_without_result_allocations() {
    let width = 512;
    let height = 256;
    let samples = vec![0u16; width * height];
    let context = StrictCdefDirectionContext {
        luma_samples: &samples,
        luma_width: width,
        luma_height: height,
        visible_width: width,
        visible_height: height,
        coeff_shift: 0,
    };
    let origins: Vec<_> = (0..129)
        .map(|index| ((index % 16) * 8, (index / 16) * 8, index % 4))
        .collect();
    let mut serial = vec![(0, 0, 0, 0, 0); origins.len()];
    let mut parallel = vec![(0, 0, 0, 0, 0); origins.len()];
    super::fill_strict_cdef_direction_blocks_serial(&origins, &mut serial, context);
    let ((), requests) = crate::test_allocation_observer::count_allocation_requests(|| {
        super::fill_strict_cdef_direction_blocks_parallel(&origins, &mut parallel, context, 4);
    });
    assert_eq!(serial, parallel);
    assert_eq!(requests, 0);

    let mut cdef = CdefParams {
        enabled: true,
        damping: 3,
        ..CdefParams::default()
    };
    cdef.strengths[0].y_pri = 4;
    cdef.strengths[0].y_sec = 2;
    let mut serial_plane = samples.clone();
    let mut parallel_plane = samples.clone();
    super::apply_cdef_plane_from_source(
        &samples,
        &mut serial_plane,
        width,
        height,
        0,
        false,
        false,
        0,
        cdef,
        width,
        height,
        &serial,
    );
    super::apply_cdef_plane_from_source(
        &samples,
        &mut parallel_plane,
        width,
        height,
        0,
        false,
        false,
        0,
        cdef,
        width,
        height,
        &parallel,
    );
    assert_eq!(serial_plane, parallel_plane);
}

fn cdef_snapshot_frame(lengths: &[usize]) -> super::DecodedFrame {
    let planes = lengths
        .iter()
        .enumerate()
        .map(|(plane, &length)| PlaneBuffer {
            layout: PlaneLayout {
                plane: plane as u8,
                width: length,
                height: usize::from(length != 0),
                subsampling_x: 0,
                subsampling_y: 0,
                sample_count: length,
            },
            samples: (0..length).map(|sample| sample as u16).collect(),
        })
        .collect();
    super::DecodedFrame {
        width: 8,
        height: 8,
        render_width: 8,
        render_height: 8,
        bit_depth: 8,
        color_config: ColorConfig {
            high_bitdepth: false,
            twelve_bit: false,
            bit_depth: 8,
            monochrome: lengths.len() == 1,
            color_description: Some(ColorDescription {
                color_primaries: 1,
                transfer_characteristics: 13,
                matrix_coefficients: 1,
            }),
            color_range: ColorRange::Full,
            subsampling_x: false,
            subsampling_y: false,
            chroma_sample_position: None,
            separate_uv_delta_q: false,
        },
        color_information: None,
        alpha_premultiplied: false,
        buffers: FrameBuffers {
            width: 8,
            height: 8,
            planes,
        },
    }
}

#[test]
fn strict_cdef_snapshot_plan_checks_shapes_and_overflow_without_candidates() {
    assert!(StrictCdefSnapshotPlan::for_sample_lengths(&[usize::MAX]).is_err());
    assert!(StrictCdefSnapshotPlan::for_sample_lengths(&[1, 2, 3, 4, 5]).is_err());
    let empty = StrictCdefSnapshotPlan::for_sample_lengths(&[]).unwrap();
    assert_eq!(empty.requested_bytes, 0);
    assert_eq!(empty.plane_count, 0);
    let frame = cdef_snapshot_frame(&[4, 2, 1]);
    let plan = StrictCdefSnapshotPlan::for_frame(&frame, [true, true, true, false]).unwrap();
    assert_eq!(plan.plane_count, 3);
    assert_eq!(plan.sample_lengths, [4, 2, 1, 0]);
    assert_eq!(
        plan.requested_bytes,
        (4 + 2 + 1) * std::mem::size_of::<u16>()
    );
    let luma_only = StrictCdefSnapshotPlan::for_frame(&frame, [true, false, false, false]).unwrap();
    assert_eq!(luma_only.sample_lengths, [4, 0, 0, 0]);
    assert_eq!(luma_only.active_planes, [true, false, false, false]);
    assert_eq!(luma_only.requested_bytes, 4 * std::mem::size_of::<u16>());
}

#[test]
fn strict_cdef_active_planes_validate_used_strength_indices() {
    let blocks = [(0, 0, 0, 0, 0)];
    let mut cdef = CdefParams {
        bits: 1,
        ..CdefParams::default()
    };
    cdef.strengths[0].y_pri = 1;
    assert_eq!(
        super::strict_cdef_active_planes(&blocks, &cdef, 3).unwrap(),
        [true, false, false, false]
    );
    cdef.strengths[0].uv_sec = 1;
    assert_eq!(
        super::strict_cdef_active_planes(&blocks, &cdef, 3).unwrap(),
        [true, true, true, false]
    );
    assert!(super::strict_cdef_active_planes(&[(0, 0, 2, 0, 0)], &cdef, 3).is_err());
}

#[test]
fn strict_cdef_snapshot_admission_skips_inactive_planes() {
    let frame = cdef_snapshot_frame(&[4, 2, 1]);
    let plan = StrictCdefSnapshotPlan::for_frame(&frame, [false, true, false, false]).unwrap();
    assert_eq!(plan.sample_lengths, [0, 2, 0, 0]);
    let mut budget = DecodeBudget::new(None);
    let direction_owner = admit_cdef_direction_owner(&mut budget);
    let baseline = budget.accounting().frame_live;
    budget.set_max_live_bytes_for_test(baseline + plan.requested_bytes);
    let owner = StrictCdefSnapshotOwner::admit(direction_owner, plan, &frame, &mut budget).unwrap();
    assert!(owner.snapshots[0].is_none());
    assert_eq!(owner.snapshots[1].as_deref(), Some(&[0, 1][..]));
    owner.release(&mut budget).unwrap();
    assert_eq!(budget.accounting().frame_live, 0);
}

#[test]
fn strict_cdef_snapshot_rejects_invalid_layout_before_candidates() {
    let mut frame = cdef_snapshot_frame(&[8, 4]);
    frame.buffers.planes[1].layout.sample_count = 3;
    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let all_candidates = crate::test_allocation_observer::track_all_candidates();
    assert!(StrictCdefSnapshotPlan::for_frame(&frame, [true, true, false, false]).is_err());
    assert_eq!(observation.registered_candidate_count(), 0);
    drop(all_candidates);
    drop(observation);
}

#[test]
fn strict_cdef_snapshot_admission_copies_all_planes_and_releases_before_retry() {
    let frame = cdef_snapshot_frame(&[4, 2, 1, 3]);
    let plan = StrictCdefSnapshotPlan::for_frame(&frame, [true, true, true, true]).unwrap();
    let original: Vec<_> = frame
        .buffers
        .planes
        .iter()
        .map(|plane| (plane.samples.as_ptr(), plane.samples.capacity()))
        .collect();
    let mut budget = DecodeBudget::new(None);
    let direction_owner = admit_cdef_direction_owner(&mut budget);
    let baseline = budget.accounting().frame_live;
    budget.set_max_live_bytes_for_test(baseline + plan.requested_bytes);
    let owner = StrictCdefSnapshotOwner::admit(direction_owner, plan, &frame, &mut budget).unwrap();
    assert_eq!(
        budget.accounting().frame_live,
        baseline + plan.requested_bytes
    );
    for (index, snapshot) in owner.snapshots.iter().enumerate().take(plan.plane_count) {
        assert_eq!(
            snapshot.as_deref(),
            Some(frame.buffers.planes[index].samples.as_slice())
        );
    }
    assert!(
        frame
            .buffers
            .planes
            .iter()
            .zip(original)
            .all(|(plane, (pointer, capacity))| {
                plane.samples.as_ptr() == pointer && plane.samples.capacity() == capacity
            })
    );
    owner.release(&mut budget).unwrap();
    assert_eq!(budget.accounting().frame_live, 0);
    budget.set_max_live_bytes_for_test(usize::MAX);
    let direction_owner = admit_cdef_direction_owner(&mut budget);
    let baseline = budget.accounting().frame_live;
    budget.set_max_live_bytes_for_test(baseline + plan.requested_bytes);
    let owner = StrictCdefSnapshotOwner::admit(direction_owner, plan, &frame, &mut budget).unwrap();
    owner.release(&mut budget).unwrap();
    assert_eq!(budget.accounting().frame_live, 0);
}

#[test]
fn strict_cdef_snapshot_combined_under_and_each_plane_overcapacity_are_retryable() {
    let frame = cdef_snapshot_frame(&[8, 4, 2]);
    let plan = StrictCdefSnapshotPlan::for_frame(&frame, [true, true, true, false]).unwrap();
    let mut under = DecodeBudget::new(None);
    let direction_owner = admit_cdef_direction_owner(&mut under);
    let baseline = under.accounting().frame_live;
    under.set_max_live_bytes_for_test(baseline + plan.requested_bytes - 1);
    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let all_candidates = crate::test_allocation_observer::track_all_candidates();
    assert!(StrictCdefSnapshotOwner::admit(direction_owner, plan, &frame, &mut under).is_err());
    assert_eq!(observation.registered_candidate_count(), 0);
    assert_eq!(under.accounting().frame_live, 0);
    drop(all_candidates);
    drop(observation);

    for label in STRICT_CDEF_SNAPSHOT_LABELS.iter().take(3) {
        let mut budget = DecodeBudget::new(None);
        let direction_owner = admit_cdef_direction_owner(&mut budget);
        let baseline = budget.accounting().frame_live;
        budget.set_max_live_bytes_for_test(baseline + plan.requested_bytes);
        let observation = crate::test_allocation_observer::Observation::begin(1, false);
        let extra = crate::test_allocation_observer::force_fresh_capacity_extra(label, 1 << 20);
        assert!(
            StrictCdefSnapshotOwner::admit(direction_owner, plan, &frame, &mut budget,).is_err()
        );
        assert_eq!(observation.drops(), 1);
        assert_eq!(budget.accounting().frame_live, 0);
        drop(extra);
        drop(observation);
        budget.set_max_live_bytes_for_test(usize::MAX);
        let direction_owner = admit_cdef_direction_owner(&mut budget);
        let baseline = budget.accounting().frame_live;
        budget.set_max_live_bytes_for_test(baseline + plan.requested_bytes);
        let owner =
            StrictCdefSnapshotOwner::admit(direction_owner, plan, &frame, &mut budget).unwrap();
        owner.release(&mut budget).unwrap();
        assert_eq!(budget.accounting().frame_live, 0);
    }
}

fn restoration_boundary_frame(planes: &[(usize, usize, u8, u8)]) -> super::DecodedFrame {
    let mut frame = cdef_snapshot_frame(&[1]);
    frame.width = 128;
    frame.height = 128;
    frame.render_width = 128;
    frame.render_height = 128;
    frame.buffers.width = 128;
    frame.buffers.height = 128;
    frame.buffers.planes = planes
        .iter()
        .enumerate()
        .map(
            |(plane, &(width, height, subsampling_x, subsampling_y))| PlaneBuffer {
                layout: PlaneLayout {
                    plane: plane as u8,
                    width,
                    height,
                    subsampling_x,
                    subsampling_y,
                    sample_count: width * height,
                },
                samples: (0..width * height)
                    .map(|index| ((index * 13 + plane * 17) & 0x0fff) as u16)
                    .collect(),
            },
        )
        .collect();
    frame
}

fn large_restoration_boundary_frame() -> super::DecodedFrame {
    let mut frame =
        restoration_boundary_frame(&[(512, 512, 0, 0), (512, 512, 1, 1), (512, 512, 1, 1)]);
    frame.width = 512;
    frame.height = 512;
    frame.render_width = 512;
    frame.render_height = 512;
    frame.buffers.width = 512;
    frame.buffers.height = 512;
    frame
}

#[test]
fn strict_restoration_boundary_owner_matches_legacy_for_420_422_444() {
    let cases = [
        vec![(128, 128, 0, 0), (64, 64, 1, 1), (64, 64, 1, 1)],
        vec![(128, 128, 0, 0), (64, 128, 1, 0), (64, 128, 1, 0)],
        vec![(128, 128, 0, 0), (128, 128, 0, 0), (128, 128, 0, 0)],
    ];
    for case in cases {
        let frame = restoration_boundary_frame(&case);
        let legacy = super::capture_restoration_boundary_rows(&frame);
        let plan = StrictRestorationBoundaryPlan::for_frame(&frame).unwrap();
        assert!(plan.requested_bytes > 0);
        let mut budget = DecodeBudget::new(Some(plan.requested_bytes));
        let owner = StrictRestorationBoundaryOwner::admit(plan, &frame, &mut budget).unwrap();
        for (plane_index, legacy_plane) in legacy.iter().enumerate().take(case.len()) {
            let metadata = owner.metadata[plane_index].as_ref().unwrap();
            let samples = owner.samples[plane_index].as_ref().unwrap();
            assert_eq!(metadata.len(), legacy_plane.rows.len());
            for (entry, &(row, ref expected)) in metadata.iter().zip(&legacy_plane.rows) {
                assert_eq!(entry.row, row);
                assert_eq!(entry.len, expected.len());
                assert_eq!(&samples[entry.offset..entry.offset + entry.len], expected);
            }
        }
        owner.release(&mut budget).unwrap();
        assert_eq!(budget.accounting().frame_live, 0);
    }
}

#[test]
fn strict_restoration_boundary_owner_is_transactional_and_retryable() {
    let frame = restoration_boundary_frame(&[(128, 128, 0, 0), (64, 64, 1, 1)]);
    let plan = StrictRestorationBoundaryPlan::for_frame(&frame).unwrap();
    let mut under = DecodeBudget::new(Some(plan.requested_bytes - 1));
    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let all_candidates = crate::test_allocation_observer::track_all_candidates();
    assert!(StrictRestorationBoundaryOwner::admit(plan, &frame, &mut under).is_err());
    assert_eq!(observation.registered_candidate_count(), 0);
    assert_eq!(under.accounting().frame_live, 0);
    drop(all_candidates);
    drop(observation);

    for label in [
        STRICT_RESTORATION_BOUNDARY_META_LABEL,
        STRICT_RESTORATION_BOUNDARY_SAMPLES_LABELS[1],
    ] {
        let mut budget = DecodeBudget::new(Some(plan.requested_bytes));
        let observation = crate::test_allocation_observer::Observation::begin(1, false);
        let extra = crate::test_allocation_observer::force_fresh_capacity_extra(label, 1 << 20);
        assert!(StrictRestorationBoundaryOwner::admit(plan, &frame, &mut budget).is_err());
        assert_eq!(observation.drops(), 1);
        assert_eq!(budget.accounting().frame_live, 0);
        drop(extra);
        drop(observation);

        let observation = crate::test_allocation_observer::Observation::begin(1, false);
        let all_candidates = crate::test_allocation_observer::track_all_candidates();
        let owner = StrictRestorationBoundaryOwner::admit(plan, &frame, &mut budget).unwrap();
        owner.release(&mut budget).unwrap();
        assert!(observation.release_count() >= 4);
        assert!(
            observation.release_drop_masks()[..observation.release_count()]
                .iter()
                .all(|mask| mask.iter().any(|dropped| *dropped))
        );
        assert_eq!(budget.accounting().frame_live, 0);
        drop(all_candidates);
        drop(observation);
    }
}

#[test]
fn strict_restoration_boundary_plan_rejects_overflow_and_allows_empty_plane() {
    let empty = restoration_boundary_frame(&[(0, 0, 0, 0)]);
    let empty_plan = StrictRestorationBoundaryPlan::for_frame(&empty).unwrap();
    assert_eq!(empty_plan.requested_bytes, 0);
    assert_eq!(empty_plan.metadata_lengths, [0; 4]);
    assert_eq!(empty_plan.sample_lengths, [0; 4]);

    let mut malformed = cdef_snapshot_frame(&[0]);
    malformed.buffers.planes[0].layout.width = usize::MAX;
    malformed.buffers.planes[0].layout.height = 2;
    assert!(StrictRestorationBoundaryPlan::for_frame(&malformed).is_err());
}

#[test]
fn flat_restoration_boundaries_match_legacy_core_output() {
    let frame = restoration_boundary_frame(&[(128, 128, 0, 0)]);
    let state = PostFilterState {
        restoration_units: vec![RestorationUnit {
            x: 0,
            y: 64,
            plane: 0,
            restoration_type: 1,
            wiener: Some([[0, -3, 8], [0, 4, -7]]),
            sgrproj: None,
            sgrproj_index: None,
        }],
        ..PostFilterState::default()
    };
    let legacy_boundaries = super::capture_restoration_boundary_rows(&frame);
    let plan = StrictRestorationBoundaryPlan::for_frame(&frame).unwrap();
    let mut budget = DecodeBudget::new(Some(plan.requested_bytes));
    let owner = StrictRestorationBoundaryOwner::admit(plan, &frame, &mut budget).unwrap();
    let mut legacy_frame = frame.clone();
    let mut flat_frame = frame;
    super::apply_loop_restoration_stage_with_boundaries(
        &mut legacy_frame,
        &state,
        8,
        &[1, 2],
        Some(&legacy_boundaries),
    );
    super::apply_loop_restoration_stage_with_flat_boundaries(
        &mut flat_frame,
        &state,
        8,
        &[1, 2],
        &owner,
    );
    assert_eq!(flat_frame.buffers.planes, legacy_frame.buffers.planes);
    owner.release(&mut budget).unwrap();
    assert_eq!(budget.accounting().frame_live, 0);
}

#[test]
fn strict_restoration_execution_matches_legacy_without_replacing_planes() {
    let cases = [
        vec![(128, 128, 0, 0), (64, 64, 1, 1), (64, 64, 1, 1)],
        vec![(128, 128, 0, 0), (64, 128, 1, 0), (64, 128, 1, 0)],
        vec![(128, 128, 0, 0), (128, 128, 0, 0), (128, 128, 0, 0)],
    ];
    for case in cases {
        let mut source_frame = restoration_boundary_frame(&case);
        source_frame.bit_depth = 12;
        let state = PostFilterState {
            restoration_units: vec![
                RestorationUnit {
                    x: 0,
                    y: 64,
                    plane: 0,
                    restoration_type: 1,
                    wiener: Some([[0, -3, 8], [0, 4, -7]]),
                    sgrproj: None,
                    sgrproj_index: None,
                },
                RestorationUnit {
                    x: 0,
                    y: 32,
                    plane: 1,
                    restoration_type: 2,
                    wiener: None,
                    sgrproj: Some([12, 64]),
                    sgrproj_index: Some(0),
                },
                RestorationUnit {
                    x: 0,
                    y: 32,
                    plane: 2,
                    restoration_type: 2,
                    wiener: None,
                    sgrproj: Some([12, 64]),
                    sgrproj_index: Some(14),
                },
            ],
            ..PostFilterState::default()
        };
        let mut legacy_frame = source_frame.clone();
        let legacy_boundaries = super::capture_restoration_boundary_rows(&source_frame);
        super::apply_loop_restoration_stage_with_source(
            &mut legacy_frame,
            &state,
            64,
            &[1, 2],
            Some(super::RestorationBoundaryCollection::Legacy(
                &legacy_boundaries,
            )),
        );

        let boundary_plan = StrictRestorationBoundaryPlan::for_frame(&source_frame).unwrap();
        let snapshot_plan =
            StrictRestorationSnapshotPlan::for_frame(&source_frame, &state, &[1, 2]).unwrap();
        let execution_plan =
            StrictRestorationExecutionScratchPlan::for_frame(&source_frame, &state, &[1, 2], 64)
                .unwrap();
        let mut budget = DecodeBudget::new(None);
        let boundary =
            StrictRestorationBoundaryOwner::admit(boundary_plan, &source_frame, &mut budget)
                .unwrap();
        let workspace = StrictRestorationWorkspaceOwner::admit(
            boundary,
            snapshot_plan,
            &source_frame,
            &mut budget,
        )
        .unwrap();
        let mut owner = StrictRestorationExecutionScratchOwner::admit(
            workspace,
            execution_plan,
            &source_frame,
            &state,
            &[1, 2],
            64,
            &mut budget,
        )
        .unwrap();
        let pointers: Vec<_> = source_frame
            .buffers
            .planes
            .iter()
            .map(|plane| {
                (
                    plane.samples.as_ptr(),
                    plane.samples.len(),
                    plane.samples.capacity(),
                )
            })
            .collect();
        let mut strict_frame = source_frame;
        let (_, requests) = crate::test_allocation_observer::count_allocation_requests(|| {
            super::apply_loop_restoration_stage_with_strict_workspace(
                &mut strict_frame,
                &state,
                64,
                &[1, 2],
                &mut owner,
            );
        });
        assert_eq!(requests, 0);
        for (plane, &(pointer, len, capacity)) in strict_frame.buffers.planes.iter().zip(&pointers)
        {
            assert_eq!(plane.samples.as_ptr(), pointer);
            assert_eq!(plane.samples.len(), len);
            assert_eq!(plane.samples.capacity(), capacity);
        }
        assert_eq!(strict_frame.buffers.planes, legacy_frame.buffers.planes);
        owner.release(&mut budget).unwrap();
        assert_eq!(budget.accounting().frame_live, 0);
    }
}

#[test]
fn strict_restoration_parallel_matches_legacy_and_keeps_owner_live_until_join() {
    let mut source_frame = large_restoration_boundary_frame();
    source_frame.bit_depth = 12;
    let state = restoration_execution_state();
    let legacy_boundaries = super::capture_restoration_boundary_rows(&source_frame);
    let mut legacy_frame = source_frame.clone();
    super::apply_loop_restoration_stage_with_source(
        &mut legacy_frame,
        &state,
        64,
        &[1, 2],
        Some(super::RestorationBoundaryCollection::Legacy(
            &legacy_boundaries,
        )),
    );

    let boundary_plan = StrictRestorationBoundaryPlan::for_frame(&source_frame).unwrap();
    let snapshot_plan =
        StrictRestorationSnapshotPlan::for_frame(&source_frame, &state, &[1, 2]).unwrap();
    let execution_plan =
        StrictRestorationExecutionScratchPlan::for_frame(&source_frame, &state, &[1, 2], 64)
            .unwrap();
    let mut budget = DecodeBudget::new(None);
    let boundary =
        StrictRestorationBoundaryOwner::admit(boundary_plan, &source_frame, &mut budget).unwrap();
    let workspace =
        StrictRestorationWorkspaceOwner::admit(boundary, snapshot_plan, &source_frame, &mut budget)
            .unwrap();
    let mut owner = StrictRestorationExecutionScratchOwner::admit(
        workspace,
        execution_plan,
        &source_frame,
        &state,
        &[1, 2],
        64,
        &mut budget,
    )
    .unwrap();
    let owner_live = budget.accounting().frame_live;
    assert!(owner_live > 0);
    let mut strict_frame = source_frame;
    super::apply_loop_restoration_stage_with_strict_workspace(
        &mut strict_frame,
        &state,
        64,
        &[1, 2],
        &mut owner,
    );
    assert_eq!(budget.accounting().frame_live, owner_live);
    assert_eq!(strict_frame.buffers.planes, legacy_frame.buffers.planes);
    owner.release(&mut budget).unwrap();
    assert_eq!(budget.accounting().frame_live, 0);
}

#[test]
fn strict_restoration_patch_capacity_failures_retry_for_each_active_plane() {
    let mut frame = large_restoration_boundary_frame();
    frame.bit_depth = 12;
    let state = restoration_execution_state();
    let boundary_plan = StrictRestorationBoundaryPlan::for_frame(&frame).unwrap();
    let snapshot_plan = StrictRestorationSnapshotPlan::for_frame(&frame, &state, &[1, 2]).unwrap();
    let execution_plan =
        StrictRestorationExecutionScratchPlan::for_frame(&frame, &state, &[1, 2], 64).unwrap();
    for label in [
        STRICT_RESTORATION_SCRATCH_LABELS[0][5],
        STRICT_RESTORATION_SCRATCH_LABELS[0][6],
        STRICT_RESTORATION_SCRATCH_LABELS[1][5],
        STRICT_RESTORATION_SCRATCH_LABELS[1][6],
        STRICT_RESTORATION_SCRATCH_LABELS[2][5],
        STRICT_RESTORATION_SCRATCH_LABELS[2][6],
    ] {
        let mut budget = DecodeBudget::new(None);
        let boundary =
            StrictRestorationBoundaryOwner::admit(boundary_plan, &frame, &mut budget).unwrap();
        let workspace =
            StrictRestorationWorkspaceOwner::admit(boundary, snapshot_plan, &frame, &mut budget)
                .unwrap();
        let workspace_live = budget.accounting().frame_live;
        budget.set_max_live_bytes_for_test(workspace_live + execution_plan.requested_bytes);
        let observation = crate::test_allocation_observer::Observation::begin(1, false);
        let extra = crate::test_allocation_observer::force_fresh_capacity_extra(label, 1 << 20);
        assert!(
            StrictRestorationExecutionScratchOwner::admit(
                workspace,
                execution_plan,
                &frame,
                &state,
                &[1, 2],
                64,
                &mut budget,
            )
            .is_err()
        );
        assert_eq!(observation.drops(), 1);
        assert_eq!(budget.accounting().frame_live, 0);
        drop(extra);
        drop(observation);

        budget.set_max_live_bytes_for_test(usize::MAX);
        let boundary =
            StrictRestorationBoundaryOwner::admit(boundary_plan, &frame, &mut budget).unwrap();
        let workspace =
            StrictRestorationWorkspaceOwner::admit(boundary, snapshot_plan, &frame, &mut budget)
                .unwrap();
        let owner = StrictRestorationExecutionScratchOwner::admit(
            workspace,
            execution_plan,
            &frame,
            &state,
            &[1, 2],
            64,
            &mut budget,
        )
        .unwrap();
        owner.release(&mut budget).unwrap();
        assert_eq!(budget.accounting().frame_live, 0);
    }
}

fn restoration_state_for_planes(planes: &[usize]) -> PostFilterState {
    PostFilterState {
        restoration_units: planes
            .iter()
            .map(|&plane| RestorationUnit {
                x: 0,
                y: if plane == 0 { 64 } else { 32 },
                plane,
                restoration_type: 1,
                wiener: Some([[0, -3, 8], [0, 4, -7]]),
                sgrproj: None,
                sgrproj_index: None,
            })
            .collect(),
        ..PostFilterState::default()
    }
}

fn restoration_execution_state() -> PostFilterState {
    PostFilterState {
        restoration_units: vec![
            RestorationUnit {
                x: 0,
                y: 64,
                plane: 0,
                restoration_type: 1,
                wiener: Some([[0, -3, 8], [0, 4, -7]]),
                sgrproj: None,
                sgrproj_index: None,
            },
            RestorationUnit {
                x: 0,
                y: 32,
                plane: 1,
                restoration_type: 2,
                wiener: None,
                sgrproj: Some([12, 64]),
                sgrproj_index: Some(0),
            },
            RestorationUnit {
                x: 0,
                y: 32,
                plane: 2,
                restoration_type: 2,
                wiener: None,
                sgrproj: Some([12, 64]),
                sgrproj_index: Some(14),
            },
        ],
        ..PostFilterState::default()
    }
}

fn sentinel_native_frame_allocation(budget: &mut DecodeBudget) -> NativeFrameAllocation {
    let plan = FrameDecodePlan {
        width: 1,
        height: 1,
        upscaled_width: 1,
        render_width: 1,
        render_height: 1,
        bit_depth: 8,
        base_q_idx: 0,
        tx_mode: TxMode::Largest,
        superblock_size: 64,
        superblock_cols: 1,
        superblock_rows: 1,
        uses_cdef: false,
        uses_restoration: false,
        planes: vec![PlaneLayout {
            plane: 0,
            width: 1,
            height: 1,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: 1,
        }],
        tiles: Vec::new(),
    };
    let limits = crate::limits::NativeDecodeLimits::new(
        1 << 20,
        64,
        64,
        4096,
        1 << 20,
        1 << 20,
        1 << 20,
        8,
        8,
        1,
        1,
        1,
    );
    let (buffers, allocation) =
        alloc_coded_frame_buffers_with_budget(&plan, budget, &limits).unwrap();
    drop(buffers);
    allocation
}

fn assert_restoration_sources_unchanged(
    frame: &super::DecodedFrame,
    original: &[(usize, usize, usize, Vec<u16>)],
) {
    assert_eq!(frame.buffers.planes.len(), original.len());
    for (plane, &(pointer, len, capacity, ref samples)) in frame.buffers.planes.iter().zip(original)
    {
        assert_eq!(plane.samples.as_ptr() as usize, pointer);
        assert_eq!(plane.samples.len(), len);
        assert_eq!(plane.samples.capacity(), capacity);
        assert_eq!(plane.samples.as_slice(), samples.as_slice());
    }
}

#[test]
fn strict_restoration_snapshot_plan_tracks_only_active_planes() {
    let frame = restoration_boundary_frame(&[(128, 128, 0, 0), (64, 64, 1, 1), (64, 64, 1, 1)]);
    let state = restoration_state_for_planes(&[0, 2]);
    let plan = StrictRestorationSnapshotPlan::for_frame(&frame, &state, &[1, 2]).unwrap();
    assert_eq!(plan.active_planes, [true, false, true, false]);
    assert_eq!(plan.sample_lengths, [128 * 128, 0, 64 * 64, 0]);
    assert_eq!(
        plan.requested_bytes,
        (128 * 128 + 64 * 64) * std::mem::size_of::<u16>()
    );
}

#[test]
fn strict_restoration_snapshot_owner_is_transactional_and_preserves_source() {
    let frame = restoration_boundary_frame(&[(128, 128, 0, 0), (64, 64, 1, 1), (64, 64, 1, 1)]);
    let state = restoration_state_for_planes(&[0, 1, 2]);
    let plan = StrictRestorationSnapshotPlan::for_frame(&frame, &state, &[1, 2]).unwrap();
    let original: Vec<_> = frame
        .buffers
        .planes
        .iter()
        .map(|plane| {
            (
                plane.samples.as_ptr() as usize,
                plane.samples.len(),
                plane.samples.capacity(),
                plane.samples.clone(),
            )
        })
        .collect();
    let boundary_plan = StrictRestorationBoundaryPlan::for_frame(&frame).unwrap();
    let mut under = DecodeBudget::new(None);
    let mut sentinel = sentinel_native_frame_allocation(&mut under);
    let sentinel_baseline = under.accounting().frame_live;
    assert!(sentinel_baseline > 0);
    let boundary =
        StrictRestorationBoundaryOwner::admit(boundary_plan, &frame, &mut under).unwrap();
    let boundary_live = under.accounting().frame_live;
    under.set_max_live_bytes_for_test(boundary_live + plan.requested_bytes - 1);
    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let all_candidates = crate::test_allocation_observer::track_all_candidates();
    assert!(StrictRestorationWorkspaceOwner::admit(boundary, plan, &frame, &mut under).is_err());
    assert_eq!(observation.registered_candidate_count(), 0);
    assert_eq!(under.accounting().frame_live, sentinel_baseline);
    assert_restoration_sources_unchanged(&frame, &original);
    drop(all_candidates);
    drop(observation);

    under.set_max_live_bytes_for_test(usize::MAX);
    let boundary =
        StrictRestorationBoundaryOwner::admit(boundary_plan, &frame, &mut under).unwrap();
    let workspace =
        StrictRestorationWorkspaceOwner::admit(boundary, plan, &frame, &mut under).unwrap();
    workspace.release(&mut under).unwrap();
    assert_eq!(under.accounting().frame_live, sentinel_baseline);
    assert_restoration_sources_unchanged(&frame, &original);
    sentinel.release(&mut under).unwrap();
    assert_eq!(under.accounting().frame_live, 0);

    for &label in &[
        STRICT_RESTORATION_SNAPSHOT_LABELS[0],
        STRICT_RESTORATION_SNAPSHOT_LABELS[1],
        STRICT_RESTORATION_SNAPSHOT_LABELS[2],
    ] {
        let mut budget = DecodeBudget::new(None);
        let mut sentinel = sentinel_native_frame_allocation(&mut budget);
        let sentinel_baseline = budget.accounting().frame_live;
        let boundary =
            StrictRestorationBoundaryOwner::admit(boundary_plan, &frame, &mut budget).unwrap();
        let boundary_live = budget.accounting().frame_live;
        budget.set_max_live_bytes_for_test(boundary_live + plan.requested_bytes);
        let observation = crate::test_allocation_observer::Observation::begin(1, false);
        let extra = crate::test_allocation_observer::force_fresh_capacity_extra(label, 1 << 20);
        assert!(
            StrictRestorationWorkspaceOwner::admit(boundary, plan, &frame, &mut budget).is_err()
        );
        assert_eq!(budget.accounting().frame_live, sentinel_baseline);
        assert_eq!(observation.drops(), 1);
        assert_restoration_sources_unchanged(&frame, &original);
        drop(extra);
        drop(observation);
        budget.set_max_live_bytes_for_test(usize::MAX);
        let boundary =
            StrictRestorationBoundaryOwner::admit(boundary_plan, &frame, &mut budget).unwrap();
        let workspace =
            StrictRestorationWorkspaceOwner::admit(boundary, plan, &frame, &mut budget).unwrap();
        workspace.release(&mut budget).unwrap();
        assert_eq!(budget.accounting().frame_live, sentinel_baseline);
        assert_restoration_sources_unchanged(&frame, &original);
        sentinel.release(&mut budget).unwrap();
        assert_eq!(budget.accounting().frame_live, 0);
    }
}

#[test]
fn strict_restoration_execution_scratch_plan_is_checked_and_transactional() {
    let frame = restoration_boundary_frame(&[(128, 128, 0, 0), (64, 64, 1, 1), (64, 64, 1, 1)]);
    let state = restoration_execution_state();
    let plan =
        StrictRestorationExecutionScratchPlan::for_frame(&frame, &state, &[1, 2], 64).unwrap();
    assert!(plan.requested_bytes > 0);
    assert_eq!(plan.wiener_lengths[0], 4480);
    assert_eq!(plan.sgr_lengths[1], [2992; 4]);
    assert_eq!(plan.sgr_lengths[2], [2992, 2992, 0, 0]);
    assert_eq!(plan.patch_metadata_lengths, [6, 0, 0, 0]);
    assert_eq!(plan.patch_sample_lengths, [6 * 67, 0, 0, 0]);

    let mut malformed = frame.clone();
    malformed.buffers.planes[0].layout.width = usize::MAX;
    assert!(
        StrictRestorationExecutionScratchPlan::for_frame(&malformed, &state, &[1, 2], 64).is_err()
    );

    let boundary_plan = StrictRestorationBoundaryPlan::for_frame(&frame).unwrap();
    let snapshot_plan = StrictRestorationSnapshotPlan::for_frame(&frame, &state, &[1, 2]).unwrap();
    let mut stale_state = restoration_execution_state();
    stale_state.restoration_units[0].x = 1;
    let mut stale_budget = DecodeBudget::new(None);
    let stale_boundary =
        StrictRestorationBoundaryOwner::admit(boundary_plan, &frame, &mut stale_budget).unwrap();
    let stale_workspace = StrictRestorationWorkspaceOwner::admit(
        stale_boundary,
        snapshot_plan,
        &frame,
        &mut stale_budget,
    )
    .unwrap();
    assert!(
        StrictRestorationExecutionScratchOwner::admit(
            stale_workspace,
            plan,
            &frame,
            &stale_state,
            &[1, 2],
            64,
            &mut stale_budget,
        )
        .is_err()
    );
    assert_eq!(stale_budget.accounting().frame_live, 0);

    for short_snapshot in [false, true] {
        let mut snapshot_budget = DecodeBudget::new(None);
        let observation = crate::test_allocation_observer::Observation::begin(1, false);
        let all_candidates = crate::test_allocation_observer::track_all_candidates();
        let boundary =
            StrictRestorationBoundaryOwner::admit(boundary_plan, &frame, &mut snapshot_budget)
                .unwrap();
        let mut workspace = StrictRestorationWorkspaceOwner::admit(
            boundary,
            snapshot_plan,
            &frame,
            &mut snapshot_budget,
        )
        .unwrap();
        if short_snapshot {
            workspace.snapshots.snapshots[0]
                .as_mut()
                .unwrap()
                .truncate(1);
        } else {
            workspace.snapshots.snapshots[0] = None;
        }
        assert!(
            StrictRestorationExecutionScratchOwner::admit(
                workspace,
                plan,
                &frame,
                &state,
                &[1, 2],
                64,
                &mut snapshot_budget,
            )
            .is_err()
        );
        assert_eq!(snapshot_budget.accounting().frame_live, 0);
        let release_count = observation.release_count();
        assert!(release_count > 0);
        assert!(
            observation.release_drop_masks()[..release_count]
                .iter()
                .all(|mask| mask.iter().any(|dropped| *dropped))
        );
        drop(all_candidates);
        drop(observation);
    }

    let mut budget = DecodeBudget::new(None);
    let boundary =
        StrictRestorationBoundaryOwner::admit(boundary_plan, &frame, &mut budget).unwrap();
    let workspace =
        StrictRestorationWorkspaceOwner::admit(boundary, snapshot_plan, &frame, &mut budget)
            .unwrap();
    let workspace_live = budget.accounting().frame_live;
    budget.set_max_live_bytes_for_test(workspace_live + plan.requested_bytes - 1);
    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let all_candidates = crate::test_allocation_observer::track_all_candidates();
    assert!(
        StrictRestorationExecutionScratchOwner::admit(
            workspace,
            plan,
            &frame,
            &state,
            &[1, 2],
            64,
            &mut budget,
        )
        .is_err()
    );
    assert_eq!(observation.registered_candidate_count(), 0);
    assert_eq!(budget.accounting().frame_live, 0);
    drop(all_candidates);
    drop(observation);

    budget.set_max_live_bytes_for_test(usize::MAX);
    let boundary =
        StrictRestorationBoundaryOwner::admit(boundary_plan, &frame, &mut budget).unwrap();
    let workspace =
        StrictRestorationWorkspaceOwner::admit(boundary, snapshot_plan, &frame, &mut budget)
            .unwrap();
    let owner = StrictRestorationExecutionScratchOwner::admit(
        workspace,
        plan,
        &frame,
        &state,
        &[1, 2],
        64,
        &mut budget,
    )
    .unwrap();
    assert_eq!(
        owner.planes[0]
            .as_ref()
            .and_then(|plane| plane.wiener.as_ref())
            .map(Vec::len),
        Some(4480)
    );
    assert_eq!(
        owner.planes[1].as_ref().map(|plane| {
            plane
                .sgr
                .iter()
                .map(|scratch| scratch.as_ref().map(Vec::len))
                .collect::<Vec<_>>()
        }),
        Some(vec![Some(2992), Some(2992), Some(2992), Some(2992)])
    );
    assert_eq!(
        owner.planes[2].as_ref().map(|plane| {
            plane
                .sgr
                .iter()
                .map(|scratch| scratch.as_ref().map(Vec::len))
                .collect::<Vec<_>>()
        }),
        Some(vec![Some(2992), Some(2992), None, None])
    );
    owner.release(&mut budget).unwrap();
    assert_eq!(budget.accounting().frame_live, 0);
}

#[test]
fn strict_restoration_execution_scratch_actual_capacity_failures_retry() {
    let frame = restoration_boundary_frame(&[(128, 128, 0, 0), (64, 64, 1, 1), (64, 64, 1, 1)]);
    let state = restoration_execution_state();
    let plan =
        StrictRestorationExecutionScratchPlan::for_frame(&frame, &state, &[1, 2], 64).unwrap();
    let boundary_plan = StrictRestorationBoundaryPlan::for_frame(&frame).unwrap();
    let snapshot_plan = StrictRestorationSnapshotPlan::for_frame(&frame, &state, &[1, 2]).unwrap();
    for label in [
        STRICT_RESTORATION_SCRATCH_LABELS[0][0],
        STRICT_RESTORATION_SCRATCH_LABELS[1][1],
        STRICT_RESTORATION_SCRATCH_LABELS[1][2],
        STRICT_RESTORATION_SCRATCH_LABELS[1][3],
        STRICT_RESTORATION_SCRATCH_LABELS[1][4],
        STRICT_RESTORATION_SCRATCH_LABELS[2][1],
        STRICT_RESTORATION_SCRATCH_LABELS[2][2],
        STRICT_RESTORATION_SCRATCH_LABELS[0][5],
        STRICT_RESTORATION_SCRATCH_LABELS[0][6],
    ] {
        let mut budget = DecodeBudget::new(None);
        let boundary =
            StrictRestorationBoundaryOwner::admit(boundary_plan, &frame, &mut budget).unwrap();
        let workspace =
            StrictRestorationWorkspaceOwner::admit(boundary, snapshot_plan, &frame, &mut budget)
                .unwrap();
        let workspace_live = budget.accounting().frame_live;
        budget.set_max_live_bytes_for_test(workspace_live + plan.requested_bytes);
        let observation = crate::test_allocation_observer::Observation::begin(1, false);
        let extra = crate::test_allocation_observer::force_fresh_capacity_extra(label, 1 << 20);
        assert!(
            StrictRestorationExecutionScratchOwner::admit(
                workspace,
                plan,
                &frame,
                &state,
                &[1, 2],
                64,
                &mut budget,
            )
            .is_err()
        );
        assert_eq!(budget.accounting().frame_live, 0);
        assert_eq!(observation.drops(), 1);
        drop(extra);
        drop(observation);

        budget.set_max_live_bytes_for_test(usize::MAX);
        let boundary =
            StrictRestorationBoundaryOwner::admit(boundary_plan, &frame, &mut budget).unwrap();
        let workspace =
            StrictRestorationWorkspaceOwner::admit(boundary, snapshot_plan, &frame, &mut budget)
                .unwrap();
        let observation = crate::test_allocation_observer::Observation::begin(1, false);
        let all_candidates = crate::test_allocation_observer::track_all_candidates();
        let owner = StrictRestorationExecutionScratchOwner::admit(
            workspace,
            plan,
            &frame,
            &state,
            &[1, 2],
            64,
            &mut budget,
        )
        .unwrap();
        owner.release(&mut budget).unwrap();
        let release_count = observation.release_count();
        assert!(release_count >= 7);
        assert!(
            observation.release_drop_masks()[..release_count]
                .iter()
                .all(|mask| mask.iter().any(|dropped| *dropped))
        );
        assert_eq!(budget.accounting().frame_live, 0);
        drop(all_candidates);
        drop(observation);
    }
}

#[test]
fn strict_cdef_snapshot_release_drops_planes_before_releasing_tickets() {
    let mut frame = cdef_snapshot_frame(&[64, 32, 16, 8]);
    let plan = StrictCdefSnapshotPlan::for_frame(&frame, [true, true, true, true]).unwrap();
    let mut budget = DecodeBudget::new(None);
    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let all_candidates = crate::test_allocation_observer::track_all_candidates();
    let direction_owner = admit_cdef_direction_owner(&mut budget);
    let owner = StrictCdefSnapshotOwner::admit(direction_owner, plan, &frame, &mut budget).unwrap();
    let original = (
        frame.buffers.planes[0].samples.as_ptr(),
        frame.buffers.planes[0].samples.len(),
        frame.buffers.planes[0].samples.capacity(),
    );
    let mut cdef = CdefParams {
        enabled: true,
        ..CdefParams::default()
    };
    cdef.strengths[0].y_pri = 1;
    let source = owner.snapshots[0].as_deref().unwrap();
    let blocks = owner.blocks();
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                apply_cdef_plane_from_source(
                    source,
                    &mut frame.buffers.planes[0].samples,
                    64,
                    1,
                    0,
                    false,
                    false,
                    0,
                    cdef,
                    8,
                    8,
                    blocks,
                );
            })
            .join()
            .unwrap();
    });
    assert_eq!(
        (
            frame.buffers.planes[0].samples.as_ptr(),
            frame.buffers.planes[0].samples.len(),
            frame.buffers.planes[0].samples.capacity(),
        ),
        original
    );
    owner.release(&mut budget).unwrap();
    let release_masks = observation.release_drop_masks();
    let release_count = observation.release_count();
    assert!(release_count >= 5);
    assert!(
        release_masks[..release_count]
            .iter()
            .all(|mask| mask.iter().any(|dropped| *dropped))
    );
    assert_eq!(budget.accounting().frame_live, 0);
    drop(all_candidates);
    drop(observation);
}

#[test]
fn strict_cdef_source_core_matches_legacy_for_subsampling_and_edges() {
    let cases = [
        (1usize, true, true, 5usize, 4usize), // 4:2:0
        (1, true, false, 5, 7),               // 4:2:2
        (1, false, false, 9, 7),              // 4:4:4 chroma
        (0, false, false, 9, 7),              // 4:4:4 luma
    ];
    for &(plane_index, subsampling_x, subsampling_y, width, height) in &cases {
        let source: Vec<u16> = (0..width * height)
            .map(|index| ((index * 97 + 31) as u16) & 0x0fff)
            .collect();
        let mut cdef = CdefParams {
            enabled: true,
            damping: 3,
            bits: 0,
            ..CdefParams::default()
        };
        if plane_index == 0 {
            cdef.strengths[0].y_pri = 4;
            cdef.strengths[0].y_sec = 2;
        } else {
            cdef.strengths[0].uv_pri = 4;
            cdef.strengths[0].uv_sec = 2;
        }
        let blocks = [
            (0, 0, 0, 0, 0),
            (8, 0, 0, 3, 7),
            (0, 8, 0, 6, 11),
            (8, 8, 0, 9, 13),
        ];
        let mut legacy_plane = PlaneBuffer {
            layout: PlaneLayout {
                plane: plane_index as u8,
                width,
                height,
                subsampling_x: u8::from(subsampling_x),
                subsampling_y: u8::from(subsampling_y),
                sample_count: source.len(),
            },
            samples: source.clone(),
        };
        apply_cdef_plane(
            &mut legacy_plane,
            plane_index,
            subsampling_x,
            subsampling_y,
            4,
            cdef,
            9,
            7,
            &blocks,
        );
        let mut strict_output = source.clone();
        apply_cdef_plane_from_source(
            &source,
            &mut strict_output,
            width,
            height,
            plane_index,
            subsampling_x,
            subsampling_y,
            4,
            cdef,
            9,
            7,
            &blocks,
        );
        assert_eq!(strict_output, legacy_plane.samples);
    }

    // A luma-only strength table must not modify the chroma snapshot.
    let source = vec![123u16; 20];
    let mut strict_output = source.clone();
    let mut cdef = CdefParams {
        enabled: true,
        ..CdefParams::default()
    };
    cdef.strengths[0].y_pri = 4;
    apply_cdef_plane_from_source(
        &source,
        &mut strict_output,
        5,
        4,
        1,
        true,
        true,
        0,
        cdef,
        9,
        7,
        &[(0, 0, 0, 0, 0)],
    );
    assert_eq!(strict_output, source);
}
