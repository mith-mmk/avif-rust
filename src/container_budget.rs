//! Parsing policy and accounting context for AVIF container readers.
//!
//! The legacy parser deliberately uses an unbounded compatibility policy. A
//! native still parse carries its caller-owned limits explicitly instead of
//! consulting global state or silently falling back to that legacy policy.

use crate::DecoderError;
use crate::limits::NativeDecodeLimits;

#[derive(Debug, Clone, Copy)]
pub(crate) enum ParseMode<'a> {
    Legacy,
    NativeStill(&'a NativeDecodeLimits),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AllocationClass {
    Metadata,
    Icc,
    Payload,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct ParseCounts {
    iinf_entries: usize,
    iloc_entries: usize,
    ipma_entries: usize,
    properties: usize,
    associations: usize,
}

/// Per-vector ownership token. It is intentionally move-only: a token is the
/// sole accounting authority for one vector allocation and cannot be copied to
/// accidentally charge the same owner twice.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct AllocationToken {
    class: AllocationClass,
    charged_capacity_bytes: usize,
}

impl AllocationToken {
    pub(crate) const fn new(class: AllocationClass) -> Self {
        Self {
            class,
            charged_capacity_bytes: 0,
        }
    }
}

impl Default for AllocationToken {
    fn default() -> Self {
        Self::new(AllocationClass::Metadata)
    }
}

#[derive(Debug)]
pub(crate) struct ParseContext<'a> {
    mode: ParseMode<'a>,
    accounting: ParseAccounting,
    counts: ParseCounts,
}

impl<'a> ParseContext<'a> {
    pub(crate) const fn legacy() -> Self {
        Self {
            mode: ParseMode::Legacy,
            accounting: ParseAccounting {
                metadata_live: 0,
                metadata_peak: 0,
                metadata_retained: 0,
                icc_live: 0,
                icc_peak: 0,
                payload_live: 0,
                payload_peak: 0,
            },
            counts: ParseCounts {
                iinf_entries: 0,
                iloc_entries: 0,
                ipma_entries: 0,
                properties: 0,
                associations: 0,
            },
        }
    }

    pub(crate) const fn native_still(limits: &'a NativeDecodeLimits) -> Self {
        Self {
            mode: ParseMode::NativeStill(limits),
            accounting: ParseAccounting {
                metadata_live: 0,
                metadata_peak: 0,
                metadata_retained: 0,
                icc_live: 0,
                icc_peak: 0,
                payload_live: 0,
                payload_peak: 0,
            },
            counts: ParseCounts {
                iinf_entries: 0,
                iloc_entries: 0,
                ipma_entries: 0,
                properties: 0,
                associations: 0,
            },
        }
    }

    pub(crate) const fn is_native_still(&self) -> bool {
        matches!(self.mode, ParseMode::NativeStill(_))
    }

    pub(crate) const fn limits(&self) -> Option<&'a NativeDecodeLimits> {
        match self.mode {
            ParseMode::Legacy => None,
            ParseMode::NativeStill(limits) => Some(limits),
        }
    }

    pub(crate) fn check_input_len(&self, length: usize) -> Result<(), DecoderError> {
        if let Some(limits) = self.limits() {
            limits.check_input_len(length)?;
        }
        Ok(())
    }

    pub(crate) fn reject_native(&self, description: &str) -> Result<(), DecoderError> {
        if self.is_native_still() {
            return Err(DecoderError::Unsupported(description.to_string()));
        }
        Ok(())
    }

    pub(crate) fn accounting(&self) -> ParseAccounting {
        self.accounting
    }

    pub(crate) fn retain_accounting(&mut self) {
        self.accounting.metadata_retained = self.accounting.metadata_live;
    }

    pub(crate) fn checkpoint(&self) -> ParseAccounting {
        self.accounting
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
        self.accounting = checkpoint;
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
        let bytes = token.charged_capacity_bytes;
        if bytes != 0 {
            self.release(token.class, bytes)?;
            token.charged_capacity_bytes = 0;
        }
        Ok(())
    }

    fn check_bytes(
        &self,
        class: AllocationClass,
        bytes: usize,
        label: &str,
    ) -> Result<(), DecoderError> {
        let accounting = self.accounting;
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
        };
        if let Some(limits) = self.limits() {
            let limit = match class {
                AllocationClass::Metadata | AllocationClass::Icc => limits.max_metadata_bytes(),
                AllocationClass::Payload => limits.max_input_bytes(),
            };
            if metadata > limit {
                let class_name = match class {
                    AllocationClass::Metadata | AllocationClass::Icc => "metadata",
                    AllocationClass::Payload => "payload",
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
        }
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
        }
        Ok(())
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
        if self.is_native_still() && values.capacity() != 0 {
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
        if !self.is_native_still() {
            return values.try_reserve(additional).map_err(|_| {
                DecoderError::InvalidParam(format!("AVIF {label} allocation failed"))
            });
        }
        if token.class != class {
            return Err(DecoderError::InvalidParam(format!(
                "native AVIF {label} allocation class does not match its owner token"
            )));
        }
        let len = values.len();
        let needed = len.checked_add(additional).ok_or_else(|| {
            DecoderError::InvalidParam(format!("native AVIF {label} length overflows"))
        })?;
        let old_capacity = values.capacity();
        let old_bytes = old_capacity
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| {
                DecoderError::InvalidParam(format!("native AVIF {label} size overflows"))
            })?;

        // A zero token denotes explicit adoption of caller-owned storage. It
        // is charged before the spare-capacity shortcut, while a nonzero token
        // must continue to describe the exact capacity of its Vec owner.
        let old_was_tracked = old_bytes != 0 && token.charged_capacity_bytes == old_bytes;
        if token.charged_capacity_bytes != 0 && !old_was_tracked {
            return Err(DecoderError::InvalidParam(format!(
                "native AVIF {label} owner token capacity is stale"
            )));
        }
        let accounting_before = self.accounting;
        if !old_was_tracked && old_bytes != 0 {
            self.check_bytes(class, old_bytes, label)?;
            if let Err(error) = self.charge(class, old_bytes) {
                self.accounting = accounting_before;
                return Err(error);
            }
        }
        if needed <= old_capacity {
            token.charged_capacity_bytes = old_bytes;
            return Ok(());
        }
        let requested_bytes = needed
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| {
                if !old_was_tracked && old_bytes != 0 {
                    self.accounting = accounting_before;
                }
                DecoderError::InvalidParam(format!("native AVIF {label} size overflows"))
            })?;
        // Reject the common growth case before asking the allocator. The
        // actual allocator capacity is reconciled after allocation below,
        // but the requested whole replacement must already fit alongside the
        // live old owner.
        if let Err(error) = self.check_bytes(class, requested_bytes, label) {
            self.accounting = accounting_before;
            return Err(error);
        }
        let mut replacement = Vec::new();
        if let Err(error) = replacement.try_reserve_exact(needed).map_err(|_| {
            DecoderError::InvalidParam(format!("native AVIF {label} allocation failed"))
        }) {
            self.accounting = accounting_before;
            return Err(error);
        }
        let actual_bytes = replacement
            .capacity()
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| {
                DecoderError::InvalidParam(format!("native AVIF {label} size overflows"))
            })?;
        if let Err(error) = self.check_bytes(class, actual_bytes, label) {
            self.accounting = accounting_before;
            return Err(error);
        }
        if let Err(error) = self.charge(class, actual_bytes) {
            self.accounting = accounting_before;
            return Err(error);
        }

        // All fallible checks have completed. Moving the elements cannot
        // allocate because `replacement` has room for `len + additional`.
        replacement.append(values);
        let old_values = std::mem::replace(values, replacement);
        drop(old_values);
        self.release(class, old_bytes)?;
        token.charged_capacity_bytes = actual_bytes;
        Ok(())
    }
}
