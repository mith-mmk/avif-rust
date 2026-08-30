//! Private allocation primitives shared by bounded native parsing.
//!
//! Policy and class ceilings stay in `container_budget`; this module owns the
//! move-only allocation ticket and the fallible fresh replacement operation.
//! Keeping those operations here prevents native callers from growing a
//! second, subtly different ownership implementation.

use crate::DecoderError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AllocationClass {
    Metadata,
    Icc,
    Payload,
    /// Native decoded frame storage, kept separate from compressed payloads.
    Frame,
}

/// Accounting authority for exactly one vector owner.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct AllocationTicket {
    pub(crate) class: AllocationClass,
    pub(crate) charged_capacity_bytes: usize,
}

/// Ledger operations needed by the shared native replacement transaction.
pub(crate) trait AllocationLedger {
    type Checkpoint: Copy;

    fn checkpoint(&self) -> Self::Checkpoint;
    fn restore(&mut self, checkpoint: Self::Checkpoint);
    fn check(&self, class: AllocationClass, bytes: usize, label: &str) -> Result<(), DecoderError>;
    fn charge(&mut self, class: AllocationClass, bytes: usize) -> Result<(), DecoderError>;
    fn release(&mut self, class: AllocationClass, bytes: usize) -> Result<(), DecoderError>;
}

impl AllocationTicket {
    pub(crate) const fn new(class: AllocationClass) -> Self {
        Self {
            class,
            charged_capacity_bytes: 0,
        }
    }
}

impl Default for AllocationTicket {
    fn default() -> Self {
        Self::new(AllocationClass::Metadata)
    }
}

pub(crate) fn capacity_bytes<T>(capacity: usize, label: &str) -> Result<usize, DecoderError> {
    capacity
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| DecoderError::InvalidParam(format!("native AVIF {label} size overflows")))
}

/// A newly allocated vector together with its checked actual capacity.
///
/// Candidate ownership stays in this value until all budget checks complete,
/// so callers can drop it before restoring their accounting on failure.
pub(crate) struct FreshReplacement<T> {
    values: Vec<T>,
    actual_bytes: usize,
}

impl<T> FreshReplacement<T> {
    pub(crate) const fn actual_bytes(&self) -> usize {
        self.actual_bytes
    }

    pub(crate) fn append(&mut self, values: &mut Vec<T>) {
        self.values.append(values);
    }

    pub(crate) fn values_mut(&mut self) -> &mut Vec<T> {
        &mut self.values
    }

    pub(crate) fn into_vec(self) -> Vec<T> {
        self.values
    }
}

/// Admits a fresh, independently owned allocation into a ledger.  Unlike a
/// replacement this leaves an existing owner untouched, which is required by
/// crop paths that must copy from the coded storage before retiring it.
pub(crate) fn admit_fresh_with<T, L, F>(
    ledger: &mut L,
    needed: usize,
    class: AllocationClass,
    label: &str,
    make: F,
) -> Result<(FreshReplacement<T>, AllocationTicket), DecoderError>
where
    L: AllocationLedger,
    F: FnOnce(usize, &str) -> Result<FreshReplacement<T>, DecoderError>,
{
    admit_fresh_inner(ledger, needed, class, label, make)
}

fn admit_fresh_inner<T, L, F>(
    ledger: &mut L,
    needed: usize,
    class: AllocationClass,
    label: &str,
    make: F,
) -> Result<(FreshReplacement<T>, AllocationTicket), DecoderError>
where
    L: AllocationLedger,
    F: FnOnce(usize, &str) -> Result<FreshReplacement<T>, DecoderError>,
{
    let checkpoint = ledger.checkpoint();
    let requested_bytes = match capacity_bytes::<T>(needed, label) {
        Ok(bytes) => bytes,
        Err(error) => {
            ledger.restore(checkpoint);
            return Err(error);
        }
    };
    if let Err(error) = ledger.check(class, requested_bytes, label) {
        ledger.restore(checkpoint);
        return Err(error);
    }
    let candidate = match make(needed, label) {
        Ok(candidate) => candidate,
        Err(error) => {
            ledger.restore(checkpoint);
            return Err(error);
        }
    };
    if !candidate.values.is_empty() || candidate.values.capacity() < needed {
        drop(candidate);
        ledger.restore(checkpoint);
        return Err(DecoderError::InvalidParam(format!(
            "native AVIF {label} allocation candidate is not fresh or is too small"
        )));
    }
    let actual_bytes = candidate.actual_bytes();
    if let Err(error) = ledger.check(class, actual_bytes, label) {
        drop(candidate);
        ledger.restore(checkpoint);
        return Err(error);
    }
    if let Err(error) = ledger.charge(class, actual_bytes) {
        drop(candidate);
        ledger.restore(checkpoint);
        return Err(error);
    }
    let mut ticket = AllocationTicket::new(class);
    ticket.charged_capacity_bytes = actual_bytes;
    Ok((candidate, ticket))
}

/// Creates a replacement without modifying the old vector.
pub(crate) fn fresh_replacement<T>(
    needed: usize,
    label: &str,
) -> Result<FreshReplacement<T>, DecoderError> {
    let mut replacement = Vec::new();
    replacement.try_reserve_exact(needed).map_err(|_| {
        DecoderError::InvalidParam(format!("native AVIF {label} allocation failed"))
    })?;
    #[cfg(test)]
    {
        let extra = crate::test_allocation_observer::fresh_capacity_extra(label);
        let track_all = crate::test_allocation_observer::track_all_candidates_enabled();
        if extra != 0 {
            replacement.try_reserve_exact(extra).map_err(|_| {
                DecoderError::InvalidParam(format!("native AVIF {label} allocation failed"))
            })?;
            // Keep the forced-capacity candidate in the fixed registry as
            // well as the single-owner slot.  The registry is what
            // `record_restore` snapshots, allowing tests to prove that the
            // real candidate deallocation preceded checkpoint restoration.
            crate::test_allocation_observer::track_candidate_slot(&replacement);
            crate::test_allocation_observer::track_candidate(&replacement);
        } else if track_all {
            crate::test_allocation_observer::track_candidate_slot(&replacement);
        }
    }
    let actual_bytes = capacity_bytes::<T>(replacement.capacity(), label)?;
    Ok(FreshReplacement {
        values: replacement,
        actual_bytes,
    })
}

/// Executes one fallible fresh-replacement transaction for a native owner.
///
/// The old owner remains untouched until requested and actual capacities have
/// both been admitted.  Every candidate failure path drops the new allocation
/// before restoring the ledger checkpoint; this ordering is part of the
/// ownership contract and is shared by all native ledgers.
pub(crate) fn replace_vec<T, L: AllocationLedger>(
    ledger: &mut L,
    values: &mut Vec<T>,
    token: &mut AllocationTicket,
    additional: usize,
    class: AllocationClass,
    label: &str,
) -> Result<(), DecoderError> {
    replace_vec_inner(
        ledger,
        values,
        token,
        additional,
        class,
        label,
        fresh_replacement::<T>,
    )
}

/// Test-only access to the production transaction with an observed candidate
/// maker.  The transaction, accounting checks, and drop ordering remain the
/// same as the normal allocator path.
pub(crate) fn replace_vec_with_maker<T, L, F>(
    ledger: &mut L,
    values: &mut Vec<T>,
    token: &mut AllocationTicket,
    additional: usize,
    class: AllocationClass,
    label: &str,
    make: F,
) -> Result<(), DecoderError>
where
    L: AllocationLedger,
    F: FnOnce(usize, &str) -> Result<FreshReplacement<T>, DecoderError>,
{
    replace_vec_inner(ledger, values, token, additional, class, label, make)
}

#[cfg(test)]
pub(crate) fn test_fresh_replacement<T>(
    values: Vec<T>,
    label: &str,
) -> Result<FreshReplacement<T>, DecoderError> {
    let actual_bytes = capacity_bytes::<T>(values.capacity(), label)?;
    Ok(FreshReplacement {
        values,
        actual_bytes,
    })
}

fn replace_vec_inner<T, L, F>(
    ledger: &mut L,
    values: &mut Vec<T>,
    token: &mut AllocationTicket,
    additional: usize,
    class: AllocationClass,
    label: &str,
    make: F,
) -> Result<(), DecoderError>
where
    L: AllocationLedger,
    F: FnOnce(usize, &str) -> Result<FreshReplacement<T>, DecoderError>,
{
    if token.class != class {
        return Err(DecoderError::InvalidParam(format!(
            "native AVIF {label} allocation class does not match its owner token"
        )));
    }
    let len = values.len();
    let needed = len.checked_add(additional).ok_or_else(|| {
        DecoderError::InvalidParam(format!("native AVIF {label} length overflows"))
    })?;
    let old_bytes = capacity_bytes::<T>(values.capacity(), label)?;
    let old_was_tracked = old_bytes != 0 && token.charged_capacity_bytes == old_bytes;
    if token.charged_capacity_bytes != 0 && !old_was_tracked {
        return Err(DecoderError::InvalidParam(format!(
            "native AVIF {label} owner token capacity is stale"
        )));
    }
    let checkpoint = ledger.checkpoint();
    if !old_was_tracked && old_bytes != 0 {
        if let Err(error) = ledger.check(class, old_bytes, label) {
            ledger.restore(checkpoint);
            return Err(error);
        }
        if let Err(error) = ledger.charge(class, old_bytes) {
            ledger.restore(checkpoint);
            return Err(error);
        }
    }
    if needed <= values.capacity() {
        token.charged_capacity_bytes = old_bytes;
        return Ok(());
    }
    let requested_bytes = match capacity_bytes::<T>(needed, label) {
        Ok(bytes) => bytes,
        Err(error) => {
            ledger.restore(checkpoint);
            return Err(error);
        }
    };
    if let Err(error) = ledger.check(class, requested_bytes, label) {
        ledger.restore(checkpoint);
        return Err(error);
    }
    let mut replacement = match make(needed, label) {
        Ok(replacement) => replacement,
        Err(error) => {
            ledger.restore(checkpoint);
            return Err(error);
        }
    };
    if !replacement.values.is_empty() || replacement.values.capacity() < needed {
        drop(replacement);
        ledger.restore(checkpoint);
        return Err(DecoderError::InvalidParam(format!(
            "native AVIF {label} replacement candidate is not fresh or is too small"
        )));
    }
    let actual_bytes = replacement.actual_bytes();
    if let Err(error) = ledger.check(class, actual_bytes, label) {
        drop(replacement);
        ledger.restore(checkpoint);
        return Err(error);
    }
    if let Err(error) = ledger.charge(class, actual_bytes) {
        drop(replacement);
        ledger.restore(checkpoint);
        return Err(error);
    }

    replacement.append(values);
    let old_values = std::mem::replace(values, replacement.into_vec());
    drop(old_values);
    ledger.release(class, old_bytes)?;
    token.charged_capacity_bytes = actual_bytes;
    Ok(())
}
