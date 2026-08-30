use super::container_budget::{AllocationClass, AllocationToken, ParseContext};
use super::{DecodeBudget, MetaState};
use crate::limits::NativeDecodeLimits;

struct RestoreObserved<'a, L> {
    inner: &'a mut L,
    observation: &'a crate::test_allocation_observer::Observation,
    restores: usize,
}

impl<L: crate::allocation::AllocationLedger> crate::allocation::AllocationLedger
    for RestoreObserved<'_, L>
{
    type Checkpoint = L::Checkpoint;

    fn checkpoint(&self) -> Self::Checkpoint {
        self.inner.checkpoint()
    }

    fn restore(&mut self, checkpoint: Self::Checkpoint) {
        assert_eq!(
            self.observation.drops(),
            1,
            "candidate must be deallocated before restore"
        );
        self.restores += 1;
        self.inner.restore(checkpoint);
    }

    fn check(
        &self,
        class: AllocationClass,
        bytes: usize,
        label: &str,
    ) -> Result<(), crate::DecoderError> {
        self.inner.check(class, bytes, label)
    }

    fn charge(&mut self, class: AllocationClass, bytes: usize) -> Result<(), crate::DecoderError> {
        self.inner.charge(class, bytes)
    }

    fn release(&mut self, class: AllocationClass, bytes: usize) -> Result<(), crate::DecoderError> {
        self.inner.release(class, bytes)
    }
}

fn limits(max_live: Option<usize>) -> NativeDecodeLimits {
    let limits = NativeDecodeLimits::new(4096, 64, 64, 4096, 4096, 4096, 4096, 32, 32, 32, 8, 1);
    match max_live {
        Some(max_live) => limits
            .with_max_live_allocation_bytes(max_live)
            .expect("positive aggregate limit"),
        None => limits,
    }
}

#[test]
fn native_parse_handoff_retains_metadata_and_payload_owners_once() {
    let bounds = limits(None);
    let mut context = ParseContext::native_still(&bounds);
    let mut metadata = Vec::<u8>::new();
    let mut metadata_token = AllocationToken::new(AllocationClass::Metadata);
    context
        .try_reserve_class_with_token(
            &mut metadata,
            &mut metadata_token,
            8,
            AllocationClass::Metadata,
            "retained metadata",
        )
        .unwrap();
    let mut payload = Vec::<u8>::new();
    let mut payload_token = AllocationToken::new(AllocationClass::Payload);
    context
        .try_reserve_class_with_token(
            &mut payload,
            &mut payload_token,
            16,
            AllocationClass::Payload,
            "retained payload",
        )
        .unwrap();

    let budget = context.into_budget(8, 16, 0).unwrap();
    let accounting = budget.accounting();
    assert_eq!(accounting.metadata_live, 8);
    assert_eq!(accounting.metadata_retained, 8);
    assert_eq!(accounting.payload_live, 16);
    assert!(accounting.metadata_peak >= 8);
    assert!(accounting.payload_peak >= 16);
    assert_eq!(accounting.aggregate_live, 24);
    assert!(accounting.aggregate_peak >= 24);
}

#[test]
fn retained_owner_handoff_is_fixed_and_does_not_double_charge_icc() {
    let bounds = limits(Some(64));
    let mut context = ParseContext::native_still(&bounds);
    let mut metadata = Vec::<u8>::new();
    let mut metadata_token = AllocationToken::new(AllocationClass::Metadata);
    context
        .try_reserve_class_with_token(
            &mut metadata,
            &mut metadata_token,
            4,
            AllocationClass::Metadata,
            "rich metadata",
        )
        .unwrap();
    let mut icc = Vec::<u8>::new();
    let mut icc_token = AllocationToken::new(AllocationClass::Icc);
    context
        .try_reserve_class_with_token(
            &mut icc,
            &mut icc_token,
            4,
            AllocationClass::Icc,
            "ICC metadata subset",
        )
        .unwrap();
    let mut payload = Vec::<u8>::new();
    let mut payload_token = AllocationToken::new(AllocationClass::Payload);
    context
        .try_reserve_class_with_token(
            &mut payload,
            &mut payload_token,
            12,
            AllocationClass::Payload,
            "primary payload",
        )
        .unwrap();
    let handoff = super::container_budget::RetainedOwnerHandoff::new(4, 0, 4, 4, 4, 4);
    let budget = context.into_budget_with_handoff(8, 12, 4, handoff).unwrap();
    assert_eq!(budget.accounting().aggregate_live, 20);
    assert_eq!(budget.accounting().metadata_live, 8);
    assert_eq!(budget.accounting().icc_live, 4);
    assert_eq!(
        budget.retained_owner_handoff().unwrap().ticket_bytes(),
        [4, 0, 4, 4, 4, 4]
    );
}

#[test]
fn aggregate_peak_records_simultaneous_live_history() {
    let mut budget = DecodeBudget::new(Some(32));
    let mut metadata = Vec::<u8>::new();
    let mut metadata_token = AllocationToken::new(AllocationClass::Metadata);
    crate::allocation::replace_vec(
        &mut budget,
        &mut metadata,
        &mut metadata_token,
        8,
        AllocationClass::Metadata,
        "aggregate metadata",
    )
    .unwrap();
    let mut payload = Vec::<u8>::new();
    let mut payload_token = AllocationToken::new(AllocationClass::Payload);
    crate::allocation::replace_vec(
        &mut budget,
        &mut payload,
        &mut payload_token,
        16,
        AllocationClass::Payload,
        "aggregate payload",
    )
    .unwrap();
    assert_eq!(budget.accounting().aggregate_live, 24);
    assert_eq!(budget.accounting().aggregate_peak, 24);
    drop(metadata);
    crate::allocation::AllocationLedger::release(
        &mut budget,
        metadata_token.class,
        metadata_token.charged_capacity_bytes,
    )
    .unwrap();
    assert_eq!(budget.accounting().aggregate_live, 16);
    assert_eq!(budget.accounting().aggregate_peak, 24);
}

#[test]
fn replacement_candidate_overcapacity_is_dropped_before_retry() {
    let mut budget = DecodeBudget::new(Some(40));
    let mut other = Vec::<u8>::new();
    let mut other_token = AllocationToken::new(AllocationClass::Payload);
    crate::allocation::replace_vec(
        &mut budget,
        &mut other,
        &mut other_token,
        8,
        AllocationClass::Payload,
        "other owner",
    )
    .unwrap();
    let mut values = Vec::<u8>::new();
    let mut token = AllocationToken::new(AllocationClass::Metadata);
    crate::allocation::replace_vec(
        &mut budget,
        &mut values,
        &mut token,
        8,
        AllocationClass::Metadata,
        "old owner",
    )
    .unwrap();
    values.extend_from_slice(&[0x39; 8]);
    let checkpoint = budget.accounting();
    let pointer = values.as_ptr();
    let error = crate::allocation::replace_vec_with_maker(
        &mut budget,
        &mut values,
        &mut token,
        8,
        AllocationClass::Metadata,
        "overcapacity candidate",
        |_needed, label| {
            let candidate = Vec::with_capacity(32);
            crate::allocation::test_fresh_replacement(candidate, label)
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("live allocation"));
    assert_eq!(budget.accounting(), checkpoint);
    assert_eq!(values.as_ptr(), pointer);
    assert_eq!(values, [0x39; 8]);
    assert_eq!(token.charged_capacity_bytes, 8);

    crate::allocation::replace_vec(
        &mut budget,
        &mut values,
        &mut token,
        8,
        AllocationClass::Metadata,
        "same owner retry",
    )
    .unwrap();
    assert_eq!(values, [0x39; 8]);
    assert_eq!(budget.accounting().aggregate_live, 24);
}

#[test]
fn replacement_candidate_must_be_empty_and_sized_before_append() {
    let mut budget = DecodeBudget::new(Some(32));
    let mut values = Vec::<u8>::new();
    let mut token = AllocationToken::new(AllocationClass::Metadata);
    let before = budget.accounting();
    let error = crate::allocation::replace_vec_with_maker(
        &mut budget,
        &mut values,
        &mut token,
        2,
        AllocationClass::Metadata,
        "nonempty candidate",
        |_needed, label| crate::allocation::test_fresh_replacement(vec![0x55], label),
    )
    .unwrap_err();
    assert!(error.to_string().contains("fresh"));
    assert_eq!(budget.accounting(), before);
    assert!(values.is_empty());
    assert_eq!(token.charged_capacity_bytes, 0);
}

fn exercise_candidate_boundary<L>(ledger: &mut L, capacity: usize, nonempty: bool)
where
    L: crate::allocation::AllocationLedger,
    L::Checkpoint: PartialEq + std::fmt::Debug,
{
    let mut other = Vec::<u8>::new();
    let mut other_token = AllocationToken::new(AllocationClass::Payload);
    crate::allocation::replace_vec(
        ledger,
        &mut other,
        &mut other_token,
        8,
        AllocationClass::Payload,
        "candidate boundary other",
    )
    .unwrap();
    let mut values = Vec::<u8>::new();
    let mut token = AllocationToken::new(AllocationClass::Metadata);
    crate::allocation::replace_vec(
        ledger,
        &mut values,
        &mut token,
        8,
        AllocationClass::Metadata,
        "candidate boundary old",
    )
    .unwrap();
    values.extend_from_slice(&[0x39; 8]);
    let checkpoint = ledger.checkpoint();
    let pointer = values.as_ptr();
    let result = crate::allocation::replace_vec_with_maker(
        ledger,
        &mut values,
        &mut token,
        8,
        AllocationClass::Metadata,
        "candidate boundary",
        |_, label| {
            let mut candidate = Vec::<u8>::with_capacity(capacity);
            if nonempty {
                candidate.push(0xa5);
            }
            crate::allocation::test_fresh_replacement(candidate, label)
        },
    );
    assert!(result.is_err());
    assert_eq!(ledger.checkpoint(), checkpoint);
    assert_eq!(values.as_ptr(), pointer);
    assert_eq!(values, [0x39; 8]);
    assert_eq!(token.charged_capacity_bytes, 8);
    crate::allocation::replace_vec(
        ledger,
        &mut values,
        &mut token,
        8,
        AllocationClass::Metadata,
        "candidate boundary retry",
    )
    .unwrap();
    assert_eq!(values, [0x39; 8]);
}

#[test]
fn candidate_boundary_covers_both_native_adapters_and_short_candidates() {
    let bounds = limits(Some(40));
    for (capacity, nonempty) in [(32, false), (4, false), (16, true)] {
        let mut context = ParseContext::native_still(&bounds);
        exercise_candidate_boundary(&mut context, capacity, nonempty);
        let mut budget = DecodeBudget::new(Some(40));
        exercise_candidate_boundary(&mut budget, capacity, nonempty);
    }
}

fn exercise_observed_candidate_boundary<L>(ledger: &mut L, capacity: usize, nonempty: bool)
where
    L: crate::allocation::AllocationLedger,
    L::Checkpoint: PartialEq + std::fmt::Debug,
{
    let mut other = Vec::<u8>::new();
    let mut other_token = AllocationToken::new(AllocationClass::Payload);
    crate::allocation::replace_vec(
        ledger,
        &mut other,
        &mut other_token,
        8,
        AllocationClass::Payload,
        "observed candidate other",
    )
    .unwrap();
    let mut values = Vec::<u8>::new();
    let mut token = AllocationToken::new(AllocationClass::Metadata);
    crate::allocation::replace_vec(
        ledger,
        &mut values,
        &mut token,
        8,
        AllocationClass::Metadata,
        "observed candidate old",
    )
    .unwrap();
    values.extend_from_slice(&[0x39; 8]);
    let before = ledger.checkpoint();
    let pointer = values.as_ptr();
    let other_pointer = other.as_ptr();

    let observation = crate::test_allocation_observer::Observation::begin(capacity, false);
    {
        let mut observed = RestoreObserved {
            inner: ledger,
            observation: &observation,
            restores: 0,
        };
        let result = crate::allocation::replace_vec_with_maker(
            &mut observed,
            &mut values,
            &mut token,
            8,
            AllocationClass::Metadata,
            "observed candidate",
            |needed, label| {
                assert_eq!(needed, 16);
                let mut candidate = Vec::<u8>::new();
                candidate.try_reserve_exact(capacity).unwrap();
                assert_eq!(candidate.capacity(), capacity);
                observation.track(&candidate);
                if nonempty {
                    candidate.push(0xa5);
                }
                crate::allocation::test_fresh_replacement(candidate, label)
            },
        );
        assert!(
            matches!(result, Err(crate::DecoderError::InvalidParam(_))),
            "invalid or overcapacity candidate must reject before publication"
        );
        assert_eq!(observed.restores, 1);
        assert_eq!(observation.requests(), 1);
        assert_eq!(observation.drops(), 1);
        assert_eq!(observation.reallocations(), 0);
    }
    drop(observation);

    assert_eq!(ledger.checkpoint(), before);
    assert_eq!(values.as_ptr(), pointer);
    assert_eq!(values.capacity(), 8);
    assert_eq!(values, [0x39; 8]);
    assert_eq!(token.charged_capacity_bytes, 8);
    assert_eq!(other.as_ptr(), other_pointer);
    assert_eq!(other.capacity(), 8);
    assert_eq!(other_token.charged_capacity_bytes, 8);

    crate::allocation::replace_vec(
        ledger,
        &mut values,
        &mut token,
        8,
        AllocationClass::Metadata,
        "observed candidate same-state retry",
    )
    .unwrap();
    assert_eq!(values, [0x39; 8]);
    assert_eq!(token.charged_capacity_bytes, 16);
}

#[test]
fn candidate_boundary_observes_real_drop_before_restore_for_both_adapters() {
    let bounds = limits(Some(40));
    for (capacity, nonempty) in [(32, false), (4, false), (16, true)] {
        let mut context = ParseContext::native_still(&bounds);
        exercise_observed_candidate_boundary(&mut context, capacity, nonempty);
        let mut budget = DecodeBudget::new(Some(40));
        exercise_observed_candidate_boundary(&mut budget, capacity, nonempty);
    }
}

#[test]
fn aggregate_live_limit_rejects_before_candidate_allocation_and_retries() {
    let bounds = limits(Some(23));
    let mut context = ParseContext::native_still(&bounds);
    let mut metadata = Vec::<u8>::new();
    let mut metadata_token = AllocationToken::new(AllocationClass::Metadata);
    context
        .try_reserve_class_with_token(
            &mut metadata,
            &mut metadata_token,
            8,
            AllocationClass::Metadata,
            "live metadata",
        )
        .unwrap();
    let before = context.accounting();
    let mut payload = Vec::<u8>::new();
    let mut payload_token = AllocationToken::new(AllocationClass::Payload);
    let error = context
        .try_reserve_class_with_token(
            &mut payload,
            &mut payload_token,
            16,
            AllocationClass::Payload,
            "live payload",
        )
        .unwrap_err();
    assert!(error.to_string().contains("live allocation"));
    assert!(payload.is_empty());
    assert_eq!(context.accounting(), before);

    context
        .try_reserve_class_with_token(
            &mut payload,
            &mut payload_token,
            8,
            AllocationClass::Payload,
            "live payload retry",
        )
        .unwrap();
    let accounting = context.accounting();
    assert_eq!(payload.len(), 0);
    assert_eq!(accounting.metadata_live, 8);
    assert_eq!(accounting.payload_live, 8);
    assert_eq!(accounting.metadata_live + accounting.payload_live, 16);
}

#[test]
fn decode_budget_uses_the_same_replacement_engine() {
    let mut budget = DecodeBudget::new(Some(16));
    let mut values = Vec::<u8>::new();
    let mut token = AllocationToken::new(AllocationClass::Metadata);
    crate::allocation::replace_vec(
        &mut budget,
        &mut values,
        &mut token,
        8,
        AllocationClass::Metadata,
        "handoff metadata",
    )
    .unwrap();
    assert_eq!(values.capacity(), 8);
    assert_eq!(budget.accounting().metadata_live, 8);
}

fn boxed_iinf_test(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let size = u32::try_from(payload.len() + 8).expect("test box fits in u32");
    let mut output = Vec::with_capacity(payload.len() + 8);
    output.extend_from_slice(&size.to_be_bytes());
    output.extend_from_slice(kind);
    output.extend_from_slice(payload);
    output
}

fn iinf_name_fixture(name: &[u8]) -> Vec<u8> {
    iinf_names_fixture(&[name])
}

fn iinf_names_fixture(names: &[&[u8]]) -> Vec<u8> {
    let mut iinf = vec![0, 0, 0, 0];
    iinf.extend_from_slice(
        &u16::try_from(names.len())
            .expect("test iinf entry count fits in u16")
            .to_be_bytes(),
    );
    for (index, name) in names.iter().enumerate() {
        let mut infe = vec![2, 0, 0, 0];
        infe.extend_from_slice(
            &u16::try_from(index + 1)
                .expect("test item id fits in u16")
                .to_be_bytes(),
        );
        infe.extend_from_slice(&[0, 0]);
        infe.extend_from_slice(b"av01");
        infe.extend_from_slice(name);
        infe.push(0);
        iinf.extend_from_slice(&boxed_iinf_test(b"infe", &infe));
    }
    boxed_iinf_test(b"iinf", &iinf)
}

fn run_iinf_test(
    input: &[u8],
    state: &mut MetaState,
    context: &mut ParseContext<'_>,
) -> Result<(), crate::DecoderError> {
    super::parse_meta_children_with_context(input, input, 0, state, context, 0)
}

fn iinf_owned_bytes(names: &[&[u8]]) -> usize {
    names.len() * std::mem::size_of::<super::ItemInfo>()
        + names.len() * std::mem::size_of::<AllocationToken>()
        + names.iter().map(|name| name.len()).sum::<usize>()
}

#[test]
fn native_iinf_retirement_debits_only_after_each_owner_is_dropped() {
    let names: &[&[u8]] = &[&[b'a'; 17], &[b'b'; 29]];
    let input = iinf_names_fixture(names);
    let limit = limits(Some(iinf_owned_bytes(names)));
    let mut context = ParseContext::native_still(&limit);
    let mut state = MetaState::default();
    run_iinf_test(&input, &mut state, &mut context).unwrap();

    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    observation.track_raw_slot(
        0,
        state.item_infos.as_ptr().cast(),
        state.item_infos.capacity() * std::mem::size_of::<super::ItemInfo>(),
    );
    for (slot, info) in state.item_infos.iter().enumerate() {
        observation.track_raw_slot(slot + 1, info.item_name.as_ptr(), info.item_name.capacity());
    }
    let owners = state.item_name_owners.as_ref().unwrap();
    observation.track_raw_slot(
        names.len() + 1,
        owners.name_tokens.as_ptr().cast(),
        owners.name_tokens.capacity() * std::mem::size_of::<AllocationToken>(),
    );

    let bundle = super::OwnedItemInfos {
        infos: std::mem::take(&mut state.item_infos),
        outer_token: std::mem::take(&mut state.item_infos_token),
        name_owners: state.item_name_owners.take(),
    };
    bundle.retire(&mut context).unwrap();

    let drops = observation.registered_drops();
    assert_eq!(drops[0], 1, "iinf outer Vec must be dropped");
    assert_eq!(drops[1], 1, "first item name must be dropped");
    assert_eq!(drops[2], 1, "second item name must be dropped");
    assert_eq!(
        drops[names.len() + 1],
        1,
        "name sidecar Vec must be dropped"
    );
    let releases = observation.release_drop_snapshots();
    assert_eq!(observation.release_count(), names.len() + 2);
    assert!(releases[0] > names.len());
    assert!(releases[names.len()] > names.len());
    assert!(releases[names.len() + 1] >= names.len() + 2);
    let release_masks = observation.release_drop_masks();
    assert!(
        release_masks[names.len()][0],
        "outer Vec is dropped before its debit"
    );
    assert!(
        !release_masks[names.len()][names.len() + 1],
        "sidecar Vec remains live until after the outer debit"
    );
    assert!(
        release_masks[names.len() + 1][names.len() + 1],
        "sidecar Vec is dropped before its backing debit"
    );
    assert_eq!(context.accounting().metadata_live, 0);
    assert_eq!(context.accounting().aggregate_live, 0);
}

#[test]
fn native_iinf_sidecar_overcapacity_drops_candidate_before_restore_and_retries() {
    let old_names: &[&[u8]] = &[&[b'a'; 17]];
    let incoming_names: &[&[u8]] = &[&[b'b'; 19]];
    let old_input = iinf_names_fixture(old_names);
    let incoming = iinf_names_fixture(incoming_names);
    let probe_limits = limits(None);
    let mut old_probe_context = ParseContext::native_still(&probe_limits);
    let mut old_probe_state = MetaState::default();
    run_iinf_test(&old_input, &mut old_probe_state, &mut old_probe_context).unwrap();
    let old_bytes = old_probe_context.accounting().metadata_live;
    super::OwnedItemInfos {
        infos: old_probe_state.item_infos,
        outer_token: old_probe_state.item_infos_token,
        name_owners: old_probe_state.item_name_owners,
    }
    .retire(&mut old_probe_context)
    .unwrap();
    let mut incoming_probe_context = ParseContext::native_still(&probe_limits);
    let mut incoming_probe_state = MetaState::default();
    run_iinf_test(
        &incoming,
        &mut incoming_probe_state,
        &mut incoming_probe_context,
    )
    .unwrap();
    let incoming_bytes = incoming_probe_context.accounting().metadata_live;
    super::OwnedItemInfos {
        infos: incoming_probe_state.item_infos,
        outer_token: incoming_probe_state.item_infos_token,
        name_owners: incoming_probe_state.item_name_owners,
    }
    .retire(&mut incoming_probe_context)
    .unwrap();
    let limit = limits(Some(old_bytes + incoming_bytes));
    let mut context = ParseContext::native_still(&limit);
    let mut state = MetaState::default();
    run_iinf_test(&old_input, &mut state, &mut context).unwrap();
    let checkpoint = context.checkpoint();
    let pointer = state.item_infos.as_ptr();
    let name_pointer = state.item_infos[0].item_name.as_ptr();
    let old_name = state.item_infos[0].item_name.clone();
    let old_outer_bytes = state.item_infos_token.charged_capacity_bytes;
    let old_owners = format!("{:?}", state.item_name_owners);
    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let _all_candidates = crate::test_allocation_observer::track_all_candidates();
    let _excess = crate::test_allocation_observer::force_fresh_capacity_extra(
        "iinf item-name owner tickets",
        8,
    );
    let result = run_iinf_test(&incoming, &mut state, &mut context);
    assert!(matches!(result, Err(crate::DecoderError::InvalidParam(_))));
    assert_eq!(observation.drops(), 1);
    assert_eq!(context.checkpoint(), checkpoint);
    assert_eq!(state.item_infos.as_ptr(), pointer);
    assert_eq!(state.item_infos[0].item_name.as_ptr(), name_pointer);
    assert_eq!(state.item_infos[0].item_name, old_name);
    assert_eq!(
        state.item_infos_token.charged_capacity_bytes,
        old_outer_bytes
    );
    assert_eq!(format!("{:?}", state.item_name_owners), old_owners);
    assert_eq!(observation.reallocations(), 0);
    assert_eq!(observation.restore_count(), 2);
    let restore_snapshots = observation.restore_drop_snapshots();
    assert!(restore_snapshots[0] >= 1);
    assert!(restore_snapshots[1] >= 2);
    let restore_masks = observation.restore_drop_masks();
    assert!(
        !restore_masks[0][0] && restore_masks[0][1],
        "inner restore must retain the incoming outer while dropping its sidecar candidate"
    );
    assert!(
        restore_masks[1][0] && restore_masks[1][1],
        "parser restore must observe all incoming candidates already dropped"
    );
    drop(observation);
    drop(_all_candidates);
    drop(_excess);

    run_iinf_test(&incoming, &mut state, &mut context).unwrap();
    assert_eq!(state.item_infos[0].item_name.as_bytes(), incoming_names[0]);
    let bundle = super::OwnedItemInfos {
        infos: state.item_infos,
        outer_token: state.item_infos_token,
        name_owners: state.item_name_owners,
    };
    bundle.retire(&mut context).unwrap();
    assert_eq!(context.accounting().aggregate_live, 0);
}

#[test]
fn legacy_iinf_name_allocation_failure_keeps_the_historical_diagnostic() {
    let input = iinf_name_fixture(&[b'a'; 17]);
    let mut context = ParseContext::legacy();
    let mut state = MetaState::default();
    let observation = crate::test_allocation_observer::Observation::begin(17, true);
    let result =
        super::parse_meta_children_with_context(&input, &input, 0, &mut state, &mut context, 0);
    assert_eq!(observation.requests(), 1);
    assert!(matches!(
        result,
        Err(crate::DecoderError::InvalidParam(ref message))
            if message == "AVIF string allocation failed"
    ));
}

#[test]
fn legacy_iinf_replacement_keeps_untracked_collection_compatibility() {
    let old_input = iinf_names_fixture(&[&[b'a'; 17]]);
    let replacement = iinf_names_fixture(&[&[b'b'; 19], &[]]);
    let mut context = ParseContext::legacy();
    let mut state = MetaState::default();

    run_iinf_test(&old_input, &mut state, &mut context).unwrap();
    assert_eq!(state.item_infos.len(), 1);
    run_iinf_test(&replacement, &mut state, &mut context).unwrap();

    assert_eq!(state.item_infos.len(), 2);
    assert_eq!(state.item_infos[0].item_name, "bbbbbbbbbbbbbbbbbbb");
    assert!(state.item_infos[1].item_name.is_empty());
}
