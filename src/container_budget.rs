//! Parsing policy and accounting context for AVIF container readers.
//!
//! The legacy parser deliberately uses an unbounded compatibility policy. A
//! native still parse carries its caller-owned limits explicitly instead of
//! consulting global state or silently falling back to that legacy policy.

use crate::DecoderError;
use crate::limits::NativeDecodeLimits;

use crate::allocation::AllocationLedger;
pub(crate) use crate::allocation::{AllocationClass, AllocationTicket as AllocationToken};

#[derive(Debug, Clone, Copy)]
pub(crate) enum ParseMode<'a> {
    Legacy,
    NativeStill(&'a NativeDecodeLimits),
    NativeDerivedStill(&'a NativeDecodeLimits),
    NativeSequence(&'a NativeDecodeLimits),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct ParseAccounting {
    pub(crate) metadata_live: usize,
    pub(crate) metadata_peak: usize,
    pub(crate) metadata_retained: usize,
    pub(crate) icc_live: usize,
    pub(crate) icc_peak: usize,
    pub(crate) payload_live: usize,
    pub(crate) payload_peak: usize,
    pub(crate) frame_live: usize,
    pub(crate) frame_peak: usize,
    /// Aggregate live bytes across metadata (including ICC), payload, and frames.
    /// ICC is a metadata subset and is deliberately not added a second time.
    pub(crate) aggregate_live: usize,
    pub(crate) aggregate_peak: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RetainedOwnerKind {
    RichMetadata,
    OrderedProperties,
    SequenceTiming,
    PrimaryPayload,
    AlphaPayload,
    SequencePayload,
    IccSubset,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct RetainedOwnerTicket {
    pub(crate) kind: RetainedOwnerKind,
    pub(crate) token: AllocationToken,
}

/// Move-only authority for owners which survive parser handoff.
///
/// The inline slots cover the normal native projection without a registry
/// allocation.  An unusually large property set spills into a Vec, but that
/// storage is admitted through the same metadata ticket engine rather than
/// being hidden from the native budget.
#[derive(Debug)]
pub(crate) struct RetainedOwnerHandoff {
    tickets: [Option<RetainedOwnerTicket>; INLINE_OWNER_TICKETS],
    overflow_tickets: Vec<RetainedOwnerTicket>,
    storage_token: AllocationToken,
}

const INLINE_OWNER_TICKETS: usize = 64;

impl RetainedOwnerHandoff {
    #[cfg(test)]
    pub(crate) fn new(
        rich_metadata: usize,
        ordered_properties: usize,
        primary_payload: usize,
        alpha_payload: usize,
        sequence_payload: usize,
        icc_subset: usize,
    ) -> Self {
        let mut handoff = Self::empty();
        handoff.tickets[0] = Some(Self::ticket(
            RetainedOwnerKind::RichMetadata,
            AllocationClass::Metadata,
            rich_metadata,
        ));
        handoff.tickets[1] = Some(Self::ticket(
            RetainedOwnerKind::OrderedProperties,
            AllocationClass::Metadata,
            ordered_properties,
        ));
        handoff.tickets[2] = Some(Self::ticket(
            RetainedOwnerKind::PrimaryPayload,
            AllocationClass::Payload,
            primary_payload,
        ));
        handoff.tickets[3] = Some(Self::ticket(
            RetainedOwnerKind::AlphaPayload,
            AllocationClass::Payload,
            alpha_payload,
        ));
        handoff.tickets[4] = Some(Self::ticket(
            RetainedOwnerKind::SequencePayload,
            AllocationClass::Payload,
            sequence_payload,
        ));
        handoff.tickets[5] = Some(Self::ticket(
            RetainedOwnerKind::IccSubset,
            AllocationClass::Icc,
            icc_subset,
        ));
        handoff
    }

    fn empty() -> Self {
        Self {
            tickets: std::array::from_fn(|_| None),
            overflow_tickets: Vec::new(),
            storage_token: AllocationToken::new(AllocationClass::Metadata),
        }
    }

    fn push(
        &mut self,
        context: &mut ParseContext<'_>,
        kind: RetainedOwnerKind,
        token: AllocationToken,
    ) -> Result<(), DecoderError> {
        if let Some(slot) = self.tickets.iter_mut().find(|slot| slot.is_none()) {
            *slot = Some(RetainedOwnerTicket { kind, token });
            return Ok(());
        }
        context.try_reserve_with_token(
            &mut self.overflow_tickets,
            &mut self.storage_token,
            1,
            "retained owner ticket storage",
        )?;
        self.overflow_tickets
            .push(RetainedOwnerTicket { kind, token });
        Ok(())
    }

    pub(super) fn storage_bytes(&self) -> usize {
        self.storage_token.charged_capacity_bytes
    }

    #[cfg(test)]
    fn ticket(
        kind: RetainedOwnerKind,
        class: AllocationClass,
        bytes: usize,
    ) -> RetainedOwnerTicket {
        let mut token = AllocationToken::new(class);
        token.charged_capacity_bytes = bytes;
        RetainedOwnerTicket { kind, token }
    }

    fn sum(&self, kinds: impl Fn(RetainedOwnerKind) -> bool) -> Result<usize, DecoderError> {
        self.tickets
            .iter()
            .flatten()
            .chain(self.overflow_tickets.iter())
            .filter(|ticket| kinds(ticket.kind))
            .try_fold(0usize, |total, ticket| {
                total
                    .checked_add(ticket.token.charged_capacity_bytes)
                    .ok_or_else(|| {
                        DecoderError::InvalidParam("retained owner bytes overflow".to_string())
                    })
            })
    }

    fn metadata_bytes(&self) -> Result<usize, DecoderError> {
        self.sum(|kind| {
            matches!(
                kind,
                RetainedOwnerKind::RichMetadata
                    | RetainedOwnerKind::OrderedProperties
                    | RetainedOwnerKind::SequenceTiming
                    | RetainedOwnerKind::IccSubset
            )
        })?
        .checked_add(self.storage_bytes())
        .ok_or_else(|| DecoderError::InvalidParam("retained owner bytes overflow".to_string()))
    }

    fn payload_bytes(&self) -> Result<usize, DecoderError> {
        self.sum(|kind| {
            matches!(
                kind,
                RetainedOwnerKind::PrimaryPayload
                    | RetainedOwnerKind::AlphaPayload
                    | RetainedOwnerKind::SequencePayload
            )
        })
    }

    fn icc_bytes(&self) -> Result<usize, DecoderError> {
        self.sum(|kind| kind == RetainedOwnerKind::IccSubset)
    }

    #[cfg(test)]
    pub(crate) fn ticket_bytes(&self) -> [usize; 6] {
        [
            self.sum(|kind| kind == RetainedOwnerKind::RichMetadata)
                .unwrap_or(0),
            self.sum(|kind| kind == RetainedOwnerKind::OrderedProperties)
                .unwrap_or(0),
            self.sum(|kind| kind == RetainedOwnerKind::PrimaryPayload)
                .unwrap_or(0),
            self.sum(|kind| kind == RetainedOwnerKind::AlphaPayload)
                .unwrap_or(0),
            self.sum(|kind| kind == RetainedOwnerKind::SequencePayload)
                .unwrap_or(0),
            self.sum(|kind| kind == RetainedOwnerKind::IccSubset)
                .unwrap_or(0),
        ]
    }
}

/// Move-only accounting carried from a bounded parse into its decoder.
///
/// It is intentionally not `Clone`: the same retained owner must never be
/// represented by two independent live counters.  Legacy parsing keeps its
/// historical unbounded policy and uses a budget without a native ceiling.
#[derive(Debug)]
pub(crate) struct DecodeBudget {
    accounting: ParseAccounting,
    max_live_bytes: Option<usize>,
    retained_owners: Option<RetainedOwnerHandoff>,
}

impl DecodeBudget {
    pub(crate) const fn new(max_live_bytes: Option<usize>) -> Self {
        Self {
            accounting: ParseAccounting {
                metadata_live: 0,
                metadata_peak: 0,
                metadata_retained: 0,
                icc_live: 0,
                icc_peak: 0,
                payload_live: 0,
                payload_peak: 0,
                frame_live: 0,
                frame_peak: 0,
                aggregate_live: 0,
                aggregate_peak: 0,
            },
            max_live_bytes,
            retained_owners: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn accounting(&self) -> ParseAccounting {
        self.accounting
    }

    pub(crate) const fn accounting_snapshot(&self) -> ParseAccounting {
        self.accounting
    }

    /// Returns the aggregate live ownership at the current transaction
    /// point.  This deliberately excludes no class: ICC is already included
    /// in metadata and is not counted a second time.
    pub(crate) const fn aggregate_live_bytes(&self) -> usize {
        self.accounting.aggregate_live
    }

    /// Applies a sequence commit accounting snapshot after all fallible
    /// validation has completed. The decoder holds exclusive access to this
    /// budget while a prepared commit token is alive, so the debug assertion
    /// is the complete invariant check and the assignment itself cannot
    /// partially release a ticket.
    pub(crate) fn apply_prevalidated_sequence_commit(
        &mut self,
        before: ParseAccounting,
        after: ParseAccounting,
    ) {
        debug_assert_eq!(self.accounting, before);
        self.accounting = after;
    }

    #[cfg(test)]
    pub(crate) fn retained_owner_handoff(&self) -> Option<&RetainedOwnerHandoff> {
        self.retained_owners.as_ref()
    }

    fn checkpoint(&self) -> ParseAccounting {
        self.accounting
    }

    fn rollback(&mut self, checkpoint: ParseAccounting) {
        self.accounting = checkpoint;
    }

    fn charge(&mut self, class: AllocationClass, bytes: usize) -> Result<(), DecoderError> {
        match class {
            AllocationClass::Metadata => {
                self.accounting.metadata_live = self
                    .accounting
                    .metadata_live
                    .checked_add(bytes)
                    .ok_or_else(|| {
                        DecoderError::InvalidParam("metadata budget overflows".to_string())
                    })?;
                self.accounting.metadata_peak = self
                    .accounting
                    .metadata_peak
                    .max(self.accounting.metadata_live);
            }
            AllocationClass::Icc => {
                self.accounting.metadata_live = self
                    .accounting
                    .metadata_live
                    .checked_add(bytes)
                    .ok_or_else(|| {
                        DecoderError::InvalidParam("metadata budget overflows".to_string())
                    })?;
                self.accounting.metadata_peak = self
                    .accounting
                    .metadata_peak
                    .max(self.accounting.metadata_live);
                self.accounting.icc_live =
                    self.accounting.icc_live.checked_add(bytes).ok_or_else(|| {
                        DecoderError::InvalidParam("ICC budget overflows".to_string())
                    })?;
                self.accounting.icc_peak = self.accounting.icc_peak.max(self.accounting.icc_live);
            }
            AllocationClass::Payload => {
                self.accounting.payload_live = self
                    .accounting
                    .payload_live
                    .checked_add(bytes)
                    .ok_or_else(|| {
                        DecoderError::InvalidParam("payload budget overflows".to_string())
                    })?;
                self.accounting.payload_peak = self
                    .accounting
                    .payload_peak
                    .max(self.accounting.payload_live);
            }
            AllocationClass::Frame => {
                self.accounting.frame_live = self
                    .accounting
                    .frame_live
                    .checked_add(bytes)
                    .ok_or_else(|| {
                        DecoderError::InvalidParam("frame budget overflows".to_string())
                    })?;
                self.accounting.frame_peak =
                    self.accounting.frame_peak.max(self.accounting.frame_live);
            }
        }
        self.refresh_aggregate()?;
        Ok(())
    }

    fn release(&mut self, class: AllocationClass, bytes: usize) -> Result<(), DecoderError> {
        match class {
            AllocationClass::Metadata => {
                self.accounting.metadata_live = self
                    .accounting
                    .metadata_live
                    .checked_sub(bytes)
                    .ok_or_else(|| {
                        DecoderError::InvalidParam("metadata ownership underflows".to_string())
                    })?;
            }
            AllocationClass::Icc => {
                self.accounting.metadata_live = self
                    .accounting
                    .metadata_live
                    .checked_sub(bytes)
                    .ok_or_else(|| {
                        DecoderError::InvalidParam("metadata ownership underflows".to_string())
                    })?;
                self.accounting.icc_live =
                    self.accounting.icc_live.checked_sub(bytes).ok_or_else(|| {
                        DecoderError::InvalidParam("ICC ownership underflows".to_string())
                    })?;
            }
            AllocationClass::Payload => {
                self.accounting.payload_live = self
                    .accounting
                    .payload_live
                    .checked_sub(bytes)
                    .ok_or_else(|| {
                        DecoderError::InvalidParam("payload ownership underflows".to_string())
                    })?;
            }
            AllocationClass::Frame => {
                self.accounting.frame_live = self
                    .accounting
                    .frame_live
                    .checked_sub(bytes)
                    .ok_or_else(|| {
                        DecoderError::InvalidParam("frame ownership underflows".to_string())
                    })?;
            }
        }
        self.refresh_aggregate()?;
        Ok(())
    }

    pub(crate) fn validate_release(
        &self,
        class: AllocationClass,
        bytes: usize,
        label: &str,
    ) -> Result<(), DecoderError> {
        let available = match class {
            AllocationClass::Metadata | AllocationClass::Icc => self.accounting.metadata_live,
            AllocationClass::Payload => self.accounting.payload_live,
            AllocationClass::Frame => self.accounting.frame_live,
        };
        if bytes > available {
            return Err(DecoderError::InvalidParam(format!(
                "native AVIF {label} ownership underflows"
            )));
        }
        if class == AllocationClass::Icc && bytes > self.accounting.icc_live {
            return Err(DecoderError::InvalidParam(format!(
                "native AVIF {label} ICC ownership underflows"
            )));
        }
        Ok(())
    }

    fn refresh_aggregate(&mut self) -> Result<(), DecoderError> {
        let aggregate = self
            .accounting
            .metadata_live
            .checked_add(self.accounting.payload_live)
            .and_then(|bytes| bytes.checked_add(self.accounting.frame_live))
            .ok_or_else(|| {
                DecoderError::InvalidParam("native AVIF aggregate budget overflows".to_string())
            })?;
        self.accounting.aggregate_live = aggregate;
        self.accounting.aggregate_peak = self.accounting.aggregate_peak.max(aggregate);
        Ok(())
    }

    #[cfg(test)]
    fn retain_owners(
        self,
        metadata_bytes: usize,
        payload_bytes: usize,
        icc_bytes: usize,
    ) -> Result<Self, DecoderError> {
        let handoff = RetainedOwnerHandoff::new(metadata_bytes, 0, payload_bytes, 0, 0, icc_bytes);
        self.retain_owners_with_handoff(metadata_bytes, payload_bytes, icc_bytes, handoff)
    }

    fn retain_owners_with_handoff(
        mut self,
        metadata_bytes: usize,
        payload_bytes: usize,
        icc_bytes: usize,
        handoff: RetainedOwnerHandoff,
    ) -> Result<Self, DecoderError> {
        if icc_bytes > metadata_bytes {
            return Err(DecoderError::InvalidParam(
                "native AVIF retained ICC exceeds retained metadata".to_string(),
            ));
        }
        if metadata_bytes > self.accounting.metadata_peak
            || payload_bytes > self.accounting.payload_peak
        {
            return Err(DecoderError::InvalidParam(
                "native AVIF retained ownership exceeds parser peak".to_string(),
            ));
        }
        if handoff.metadata_bytes()? != metadata_bytes
            || handoff.payload_bytes()? != payload_bytes
            || handoff.icc_bytes()? != icc_bytes
        {
            return Err(DecoderError::InvalidParam(format!(
                "native AVIF retained owner tickets do not match retained bytes (handoff metadata={}, payload={}, ICC={}; expected metadata={}, payload={}, ICC={})",
                handoff.metadata_bytes()?,
                handoff.payload_bytes()?,
                handoff.icc_bytes()?,
                metadata_bytes,
                payload_bytes,
                icc_bytes,
            )));
        }
        if self.accounting.metadata_live != metadata_bytes
            || self.accounting.payload_live != payload_bytes
            || self.accounting.icc_live != icc_bytes
        {
            return Err(DecoderError::InvalidParam(format!(
                "native AVIF parser accounting does not match retained owner tickets (accounting metadata={}, payload={}, ICC={}; expected metadata={}, payload={}, ICC={})",
                self.accounting.metadata_live,
                self.accounting.payload_live,
                self.accounting.icc_live,
                metadata_bytes,
                payload_bytes,
                icc_bytes,
            )));
        }
        self.accounting.metadata_retained = metadata_bytes;
        self.retained_owners = Some(handoff);
        self.refresh_aggregate()?;
        self.check_live(0)?;
        Ok(self)
    }

    fn check_live(&self, additional: usize) -> Result<(), DecoderError> {
        let current = self
            .accounting
            .metadata_live
            .checked_add(self.accounting.payload_live)
            .and_then(|bytes| bytes.checked_add(self.accounting.frame_live))
            .and_then(|bytes| bytes.checked_add(additional))
            .ok_or_else(|| {
                DecoderError::InvalidParam("native AVIF live budget overflows".to_string())
            })?;
        if self.max_live_bytes.is_some_and(|limit| current > limit) {
            let limit = self.max_live_bytes.expect("checked above");
            return Err(DecoderError::InvalidParam(format!(
                "native AVIF live allocation {current} exceeds limit {limit}"
            )));
        }
        Ok(())
    }

    pub(crate) fn check_additional_frame(&self, additional: usize) -> Result<(), DecoderError> {
        self.check_live(additional)
    }

    /// Admits an already-created strict-native owner into the live ledger.
    ///
    /// This is used for transactional sequence-state clones.  The owner is
    /// created by `Clone` rather than by one of the vector replacement
    /// helpers, so it needs the same checked live-budget admission and a
    /// move-only ticket for the later rollback/commit path.
    pub(crate) fn reserve_existing_bytes(
        &mut self,
        class: AllocationClass,
        bytes: usize,
        label: &str,
    ) -> Result<AllocationToken, DecoderError> {
        self.check_live(bytes).map_err(|error| match error {
            DecoderError::InvalidParam(message) => {
                DecoderError::InvalidParam(format!("{label}: {message}"))
            }
            other => other,
        })?;
        let checkpoint = self.checkpoint();
        if let Err(error) = self.charge(class, bytes) {
            self.rollback(checkpoint);
            return Err(error);
        }
        let mut token = AllocationToken::new(class);
        token.charged_capacity_bytes = bytes;
        Ok(token)
    }

    #[cfg(test)]
    pub(crate) fn set_max_live_bytes_for_test(&mut self, max_live_bytes: usize) {
        self.max_live_bytes = Some(max_live_bytes);
    }

    /// Allocates/replaces one strict-native frame owner while preserving the
    /// parser handoff accounting. This is private to the bounded decoder;
    /// parser-owned vectors continue to use `ParseContext`.
    pub(crate) fn try_reserve_class_with_token<T>(
        &mut self,
        values: &mut Vec<T>,
        token: &mut AllocationToken,
        additional: usize,
        class: AllocationClass,
        label: &str,
    ) -> Result<(), DecoderError> {
        crate::allocation::replace_vec(self, values, token, additional, class, label)
    }

    pub(crate) fn release_token(
        &mut self,
        token: &mut AllocationToken,
    ) -> Result<(), DecoderError> {
        #[cfg(test)]
        crate::test_allocation_observer::record_token_release(token);
        let bytes = token.charged_capacity_bytes;
        if bytes != 0 {
            self.release(token.class, bytes)?;
            token.charged_capacity_bytes = 0;
        }
        Ok(())
    }

    pub(crate) fn release_class_bytes(
        &mut self,
        class: AllocationClass,
        bytes: usize,
    ) -> Result<(), DecoderError> {
        if bytes != 0 {
            self.release(class, bytes)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct ParseCounts {
    iinf_entries: usize,
    iloc_entries: usize,
    ipma_entries: usize,
    properties: usize,
    associations: usize,
}

#[derive(Debug)]
pub(crate) struct ParseContext<'a> {
    mode: ParseMode<'a>,
    budget: DecodeBudget,
    counts: ParseCounts,
    native_owners: Option<RetainedOwnerHandoff>,
}

impl<'a> ParseContext<'a> {
    pub(crate) const fn legacy() -> Self {
        Self {
            mode: ParseMode::Legacy,
            budget: DecodeBudget::new(None),
            counts: ParseCounts {
                iinf_entries: 0,
                iloc_entries: 0,
                ipma_entries: 0,
                properties: 0,
                associations: 0,
            },
            native_owners: None,
        }
    }

    pub(crate) const fn native_still(limits: &'a NativeDecodeLimits) -> Self {
        Self {
            mode: ParseMode::NativeStill(limits),
            budget: DecodeBudget::new(limits.max_live_allocation_bytes()),
            counts: ParseCounts {
                iinf_entries: 0,
                iloc_entries: 0,
                ipma_entries: 0,
                properties: 0,
                associations: 0,
            },
            native_owners: None,
        }
    }

    pub(crate) const fn native_sequence(limits: &'a NativeDecodeLimits) -> Self {
        Self {
            mode: ParseMode::NativeSequence(limits),
            budget: DecodeBudget::new(limits.max_live_allocation_bytes()),
            counts: ParseCounts {
                iinf_entries: 0,
                iloc_entries: 0,
                ipma_entries: 0,
                properties: 0,
                associations: 0,
            },
            native_owners: None,
        }
    }

    pub(crate) const fn native_derived_still(limits: &'a NativeDecodeLimits) -> Self {
        Self {
            mode: ParseMode::NativeDerivedStill(limits),
            budget: DecodeBudget::new(limits.max_live_allocation_bytes()),
            counts: ParseCounts {
                iinf_entries: 0,
                iloc_entries: 0,
                ipma_entries: 0,
                properties: 0,
                associations: 0,
            },
            native_owners: None,
        }
    }

    pub(crate) const fn is_native_still(&self) -> bool {
        matches!(self.mode, ParseMode::NativeStill(_))
    }

    pub(crate) const fn is_native_derived_still(&self) -> bool {
        matches!(self.mode, ParseMode::NativeDerivedStill(_))
    }

    pub(crate) const fn is_native_still_like(&self) -> bool {
        self.is_native_still() || self.is_native_derived_still()
    }

    pub(crate) const fn is_native_sequence(&self) -> bool {
        matches!(self.mode, ParseMode::NativeSequence(_))
    }

    pub(crate) const fn is_native(&self) -> bool {
        matches!(
            self.mode,
            ParseMode::NativeStill(_)
                | ParseMode::NativeDerivedStill(_)
                | ParseMode::NativeSequence(_)
        )
    }

    pub(crate) const fn limits(&self) -> Option<&'a NativeDecodeLimits> {
        match self.mode {
            ParseMode::Legacy => None,
            ParseMode::NativeStill(limits)
            | ParseMode::NativeDerivedStill(limits)
            | ParseMode::NativeSequence(limits) => Some(limits),
        }
    }

    pub(crate) fn check_input_len(&self, length: usize) -> Result<(), DecoderError> {
        if let Some(limits) = self.limits() {
            limits.check_input_len(length)?;
        }
        Ok(())
    }

    pub(crate) fn reject_native(&self, description: &str) -> Result<(), DecoderError> {
        if self.is_native_still_like() {
            return Err(DecoderError::Unsupported(description.to_string()));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn accounting(&self) -> ParseAccounting {
        self.budget.accounting()
    }

    #[cfg(test)]
    pub(crate) fn into_budget(
        self,
        metadata_bytes: usize,
        payload_bytes: usize,
        icc_bytes: usize,
    ) -> Result<DecodeBudget, DecoderError> {
        self.budget
            .retain_owners(metadata_bytes, payload_bytes, icc_bytes)
    }

    pub(crate) fn into_budget_with_handoff(
        self,
        metadata_bytes: usize,
        payload_bytes: usize,
        icc_bytes: usize,
        handoff: RetainedOwnerHandoff,
    ) -> Result<DecodeBudget, DecoderError> {
        self.budget
            .retain_owners_with_handoff(metadata_bytes, payload_bytes, icc_bytes, handoff)
    }

    pub(crate) fn checkpoint(&self) -> ParseAccounting {
        self.budget.checkpoint()
    }

    pub(crate) fn admit_iinf_entries(&mut self, additional: usize) -> Result<(), DecoderError> {
        let next = self.next_item_count(self.counts.iinf_entries, additional, "iinf entry")?;
        self.counts.iinf_entries = next;
        Ok(())
    }

    pub(crate) fn admit_iloc_entries(&mut self, additional: usize) -> Result<(), DecoderError> {
        let next = self.next_item_count(self.counts.iloc_entries, additional, "iloc item")?;
        self.counts.iloc_entries = next;
        Ok(())
    }

    pub(crate) fn admit_ipma(
        &mut self,
        entries: usize,
        associations: usize,
    ) -> Result<(), DecoderError> {
        let Some(limits) = self.limits() else {
            return Ok(());
        };
        let next_entries = self
            .counts
            .ipma_entries
            .checked_add(entries)
            .ok_or_else(|| DecoderError::InvalidParam("ipma entry count overflows".to_string()))?;
        let next_associations = self
            .counts
            .associations
            .checked_add(associations)
            .ok_or_else(|| {
                DecoderError::InvalidParam("ipma association count overflows".to_string())
            })?;
        limits.check_count(
            next_entries,
            limits.max_items(),
            "property association entry",
        )?;
        limits.check_count(
            next_associations,
            limits.max_properties(),
            "property association",
        )?;
        self.counts.ipma_entries = next_entries;
        self.counts.associations = next_associations;
        Ok(())
    }

    pub(crate) fn admit_property(&mut self) -> Result<(), DecoderError> {
        self.admit_property_count(1)
    }

    fn next_item_count(
        &self,
        current: usize,
        additional: usize,
        label: &str,
    ) -> Result<usize, DecoderError> {
        let Some(limits) = self.limits() else {
            return Ok(current);
        };
        let next = current
            .checked_add(additional)
            .ok_or_else(|| DecoderError::InvalidParam(format!("{label} count overflows")))?;
        limits.check_count(next, limits.max_items(), label)?;
        Ok(next)
    }

    fn admit_property_count(&mut self, additional: usize) -> Result<(), DecoderError> {
        let Some(limits) = self.limits() else {
            return Ok(());
        };
        let next = self
            .counts
            .properties
            .checked_add(additional)
            .ok_or_else(|| DecoderError::InvalidParam("property count overflows".to_string()))?;
        limits.check_count(next, limits.max_properties(), "property")?;
        self.counts.properties = next;
        Ok(())
    }

    pub(crate) fn rollback(&mut self, checkpoint: ParseAccounting) {
        self.budget.rollback(checkpoint);
        self.native_owners = None;
    }

    pub(crate) fn retain_native_token(
        &mut self,
        kind: RetainedOwnerKind,
        token: AllocationToken,
    ) -> Result<(), DecoderError> {
        if !self.is_native() || token.charged_capacity_bytes == 0 {
            return Ok(());
        }
        let mut owners = self
            .native_owners
            .take()
            .unwrap_or_else(RetainedOwnerHandoff::empty);
        let result = owners.push(self, kind, token);
        self.native_owners = Some(owners);
        result
    }

    pub(crate) fn take_native_owners(&mut self) -> Option<RetainedOwnerHandoff> {
        self.native_owners.take()
    }

    pub(crate) fn release_class_bytes(
        &mut self,
        class: AllocationClass,
        bytes: usize,
    ) -> Result<(), DecoderError> {
        if bytes != 0 {
            self.release(class, bytes)?;
        }
        Ok(())
    }

    pub(crate) fn release_token(
        &mut self,
        token: &mut AllocationToken,
    ) -> Result<(), DecoderError> {
        #[cfg(test)]
        crate::test_allocation_observer::record_token_release(token);
        let bytes = token.charged_capacity_bytes;
        if bytes != 0 {
            self.release(token.class, bytes)?;
            token.charged_capacity_bytes = 0;
        }
        Ok(())
    }

    pub(crate) fn validate_release(
        &self,
        class: AllocationClass,
        bytes: usize,
        label: &str,
    ) -> Result<(), DecoderError> {
        let available = match class {
            AllocationClass::Metadata | AllocationClass::Icc => {
                self.budget.accounting.metadata_live
            }
            AllocationClass::Payload => self.budget.accounting.payload_live,
            AllocationClass::Frame => self.budget.accounting.frame_live,
        };
        if bytes > available {
            return Err(DecoderError::InvalidParam(format!(
                "native AVIF {label} ownership underflows"
            )));
        }
        if class == AllocationClass::Icc && bytes > self.budget.accounting.icc_live {
            return Err(DecoderError::InvalidParam(format!(
                "native AVIF {label} ICC ownership underflows"
            )));
        }
        Ok(())
    }

    fn check_bytes(
        &self,
        class: AllocationClass,
        bytes: usize,
        label: &str,
    ) -> Result<(), DecoderError> {
        let accounting = self.budget.accounting;
        self.budget.check_live(bytes)?;
        let metadata = match class {
            AllocationClass::Metadata | AllocationClass::Icc => {
                accounting.metadata_live.checked_add(bytes).ok_or_else(|| {
                    DecoderError::InvalidParam("metadata budget overflows".to_string())
                })?
            }
            AllocationClass::Payload => {
                accounting.payload_live.checked_add(bytes).ok_or_else(|| {
                    DecoderError::InvalidParam("payload budget overflows".to_string())
                })?
            }
            AllocationClass::Frame => return self.budget.check_live(bytes),
        };
        if let Some(limits) = self.limits() {
            let limit = match class {
                AllocationClass::Metadata | AllocationClass::Icc => limits.max_metadata_bytes(),
                AllocationClass::Payload => limits.max_input_bytes(),
                AllocationClass::Frame => unreachable!("native frame storage returned above"),
            };
            if metadata > limit {
                let class_name = match class {
                    AllocationClass::Metadata | AllocationClass::Icc => "metadata",
                    AllocationClass::Payload => "payload",
                    AllocationClass::Frame => unreachable!("native frame storage returned above"),
                };
                return Err(DecoderError::InvalidParam(format!(
                    "native AVIF {class_name} {label} allocation {metadata} exceeds limit {limit}"
                )));
            }
            if class == AllocationClass::Icc {
                let icc = accounting.icc_live.checked_add(bytes).ok_or_else(|| {
                    DecoderError::InvalidParam("ICC budget overflows".to_string())
                })?;
                if icc > limits.max_icc_bytes() {
                    return Err(DecoderError::InvalidParam(format!(
                        "native AVIF ICC allocation {icc} exceeds limit {}",
                        limits.max_icc_bytes()
                    )));
                }
            }
        }
        Ok(())
    }

    fn charge(&mut self, class: AllocationClass, bytes: usize) -> Result<(), DecoderError> {
        self.budget.charge(class, bytes)
    }

    fn release(&mut self, class: AllocationClass, bytes: usize) -> Result<(), DecoderError> {
        self.budget.release(class, bytes)
    }

    pub(crate) fn try_reserve<T>(
        &mut self,
        values: &mut Vec<T>,
        additional: usize,
        label: &str,
    ) -> Result<(), DecoderError> {
        self.require_fresh_tokenless(values, label)?;
        let mut token = AllocationToken::new(AllocationClass::Metadata);
        self.try_reserve_with_token(values, &mut token, additional, label)
    }

    pub(crate) fn try_reserve_with_token<T>(
        &mut self,
        values: &mut Vec<T>,
        token: &mut AllocationToken,
        additional: usize,
        label: &str,
    ) -> Result<(), DecoderError> {
        let class = token.class;
        self.try_reserve_class_with_token(values, token, additional, class, label)
    }

    pub(crate) fn try_reserve_class<T>(
        &mut self,
        values: &mut Vec<T>,
        additional: usize,
        class: AllocationClass,
        label: &str,
    ) -> Result<(), DecoderError> {
        self.require_fresh_tokenless(values, label)?;
        let mut token = AllocationToken::new(class);
        self.try_reserve_class_with_token(values, &mut token, additional, class, label)
    }

    fn require_fresh_tokenless<T>(&self, values: &Vec<T>, label: &str) -> Result<(), DecoderError> {
        if self.is_native() && values.capacity() != 0 {
            return Err(DecoderError::InvalidParam(format!(
                "native AVIF {label} existing storage requires an owner token"
            )));
        }
        Ok(())
    }

    pub(crate) fn try_reserve_class_with_token<T>(
        &mut self,
        values: &mut Vec<T>,
        token: &mut AllocationToken,
        additional: usize,
        class: AllocationClass,
        label: &str,
    ) -> Result<(), DecoderError> {
        if !self.is_native() {
            return values.try_reserve(additional).map_err(|_| {
                DecoderError::InvalidParam(format!("AVIF {label} allocation failed"))
            });
        }
        crate::allocation::replace_vec(self, values, token, additional, class, label)
    }

    /// Checks a grouped native reservation before any member allocation is
    /// attempted. Sequence timing owns two vectors for one logical table;
    /// checking their combined request keeps an under-limit parse
    /// allocation-free instead of reserving the first vector and failing on
    /// the second one.
    pub(crate) fn check_additional_class_bytes(
        &self,
        class: AllocationClass,
        bytes: usize,
        label: &str,
    ) -> Result<(), DecoderError> {
        if self.is_native() {
            self.check_bytes(class, bytes, label)?;
        }
        Ok(())
    }
}

impl AllocationLedger for DecodeBudget {
    type Checkpoint = ParseAccounting;

    fn checkpoint(&self) -> Self::Checkpoint {
        self.accounting
    }

    fn restore(&mut self, checkpoint: Self::Checkpoint) {
        #[cfg(test)]
        crate::test_allocation_observer::record_restore();
        self.accounting = checkpoint;
    }

    fn check(
        &self,
        _class: AllocationClass,
        bytes: usize,
        _label: &str,
    ) -> Result<(), DecoderError> {
        self.check_live(bytes)
    }

    fn charge(&mut self, class: AllocationClass, bytes: usize) -> Result<(), DecoderError> {
        DecodeBudget::charge(self, class, bytes)
    }

    fn release(&mut self, class: AllocationClass, bytes: usize) -> Result<(), DecoderError> {
        DecodeBudget::release(self, class, bytes)
    }
}

impl AllocationLedger for ParseContext<'_> {
    type Checkpoint = ParseAccounting;

    fn checkpoint(&self) -> Self::Checkpoint {
        self.budget.accounting
    }

    fn restore(&mut self, checkpoint: Self::Checkpoint) {
        #[cfg(test)]
        crate::test_allocation_observer::record_restore();
        self.budget.accounting = checkpoint;
    }

    fn check(&self, class: AllocationClass, bytes: usize, label: &str) -> Result<(), DecoderError> {
        self.check_bytes(class, bytes, label)
    }

    fn charge(&mut self, class: AllocationClass, bytes: usize) -> Result<(), DecoderError> {
        ParseContext::charge(self, class, bytes)
    }

    fn release(&mut self, class: AllocationClass, bytes: usize) -> Result<(), DecoderError> {
        ParseContext::release(self, class, bytes)
    }
}
