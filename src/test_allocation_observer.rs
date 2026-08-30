//! Test-only observation of real candidate allocation and destruction.
//!
//! The production allocation engine remains responsible for all admission and
//! rollback decisions. This module only records the candidate allocation that a
//! test creates and verifies that it is really deallocated before accounting is
//! restored.

#![cfg(test)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::marker::PhantomData;
use std::rc::Rc;

const MAX_REGISTERED_OWNERS: usize = 16;
const MAX_RELEASES: usize = 32;
const MAX_PHASES: usize = 4;

pub(crate) struct Probe;

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static REQUEST_SIZE: Cell<usize> = const { Cell::new(0) };
    static REQUESTS: Cell<usize> = const { Cell::new(0) };
    static COUNT_ALL_REQUESTS: Cell<bool> = const { Cell::new(false) };
    static PHASE: Cell<usize> = const { Cell::new(0) };
    static PHASE_REQUESTS: Cell<[usize; MAX_PHASES]> = const { Cell::new([0; MAX_PHASES]) };
    static DENY: Cell<bool> = const { Cell::new(false) };
    static OWNER_POINTER: Cell<usize> = const { Cell::new(0) };
    static OWNER_BYTES: Cell<usize> = const { Cell::new(0) };
    static OWNER_DROPS: Cell<usize> = const { Cell::new(0) };
    static OWNER_REALLOCS: Cell<usize> = const { Cell::new(0) };
    static REGISTERED_POINTERS: Cell<[usize; MAX_REGISTERED_OWNERS]> =
        const { Cell::new([0; MAX_REGISTERED_OWNERS]) };
    static REGISTERED_BYTES: Cell<[usize; MAX_REGISTERED_OWNERS]> =
        const { Cell::new([0; MAX_REGISTERED_OWNERS]) };
    static REGISTERED_DROPS: Cell<[usize; MAX_REGISTERED_OWNERS]> =
        const { Cell::new([0; MAX_REGISTERED_OWNERS]) };
    static RELEASE_COUNT: Cell<usize> = const { Cell::new(0) };
    static RELEASE_DROP_SNAPSHOTS: Cell<[usize; MAX_RELEASES]> =
        const { Cell::new([0; MAX_RELEASES]) };
    static RELEASE_DROP_MASKS: Cell<[[bool; MAX_REGISTERED_OWNERS]; MAX_RELEASES]> =
        const { Cell::new([[false; MAX_REGISTERED_OWNERS]; MAX_RELEASES]) };
    static RESTORE_COUNT: Cell<usize> = const { Cell::new(0) };
    static RESTORE_DROP_SNAPSHOTS: Cell<[usize; MAX_RELEASES]> =
        const { Cell::new([0; MAX_RELEASES]) };
    static RESTORE_DROP_MASKS: Cell<[[bool; MAX_REGISTERED_OWNERS]; MAX_RELEASES]> =
        const { Cell::new([[false; MAX_REGISTERED_OWNERS]; MAX_RELEASES]) };
    static TRACK_ALL_CANDIDATES: Cell<bool> = const { Cell::new(false) };
    static FRESH_EXTRA_LABEL: Cell<Option<&'static str>> = const { Cell::new(None) };
    static FRESH_EXTRA_BYTES: Cell<usize> = const { Cell::new(0) };
}

fn allocation(size: usize) -> bool {
    ACTIVE
        .try_with(|active| {
            if active.get() && COUNT_ALL_REQUESTS.get() {
                REQUESTS.set(REQUESTS.get() + 1);
                let phase = PHASE.get();
                if phase < MAX_PHASES {
                    let mut phase_requests = PHASE_REQUESTS.get();
                    phase_requests[phase] += 1;
                    PHASE_REQUESTS.set(phase_requests);
                }
                false
            } else if active.get() && REQUEST_SIZE.get() == size {
                REQUESTS.set(REQUESTS.get() + 1);
                DENY.replace(false)
            } else {
                false
            }
        })
        .unwrap_or(false)
}

fn deallocation(pointer: *mut u8, bytes: usize) {
    let _ = ACTIVE.try_with(|active| {
        if active.get() && OWNER_POINTER.get() == pointer as usize && OWNER_BYTES.get() == bytes {
            OWNER_DROPS.set(OWNER_DROPS.get() + 1);
            OWNER_POINTER.set(0);
        }
        if active.get() {
            let pointers = REGISTERED_POINTERS.get();
            let sizes = REGISTERED_BYTES.get();
            let mut drops = REGISTERED_DROPS.get();
            for index in 0..MAX_REGISTERED_OWNERS {
                if pointers[index] == pointer as usize && sizes[index] == bytes {
                    drops[index] += 1;
                }
            }
            REGISTERED_DROPS.set(drops);
        }
    });
}

unsafe impl GlobalAlloc for Probe {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if allocation(layout.size()) {
            return std::ptr::null_mut();
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if allocation(layout.size()) {
            return std::ptr::null_mut();
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        deallocation(pointer, layout.size());
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if allocation(size) {
            return std::ptr::null_mut();
        }
        let _ = ACTIVE.try_with(|active| {
            if active.get() && OWNER_POINTER.get() == pointer as usize {
                OWNER_REALLOCS.set(OWNER_REALLOCS.get() + 1);
            }
        });
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static TEST_ALLOCATOR: Probe = Probe;

/// Observes only one test candidate and never dereferences its address.
pub(crate) struct Observation {
    _thread_bound: PhantomData<Rc<()>>,
}

pub(crate) struct FreshCapacityExtraGuard;

pub(crate) struct TrackAllCandidatesGuard;

pub(crate) fn force_fresh_capacity_extra(
    label: &'static str,
    extra: usize,
) -> FreshCapacityExtraGuard {
    assert!(extra > 0);
    assert!(FRESH_EXTRA_LABEL.get().is_none());
    FRESH_EXTRA_LABEL.set(Some(label));
    FRESH_EXTRA_BYTES.set(extra);
    FreshCapacityExtraGuard
}

pub(crate) fn fresh_capacity_extra(label: &str) -> usize {
    if FRESH_EXTRA_LABEL.get() == Some(label) {
        FRESH_EXTRA_BYTES.get()
    } else {
        0
    }
}

pub(crate) fn track_all_candidates() -> TrackAllCandidatesGuard {
    assert!(!TRACK_ALL_CANDIDATES.get());
    TRACK_ALL_CANDIDATES.set(true);
    TrackAllCandidatesGuard
}

pub(crate) fn track_all_candidates_enabled() -> bool {
    TRACK_ALL_CANDIDATES.get()
}

pub(crate) fn track_candidate<T>(candidate: &Vec<T>) {
    if ACTIVE.get() {
        let bytes = candidate
            .capacity()
            .checked_mul(std::mem::size_of::<T>())
            .unwrap();
        assert!(candidate.is_empty());
        assert!(bytes > 0);
        assert_eq!(OWNER_POINTER.get(), 0);
        OWNER_POINTER.set(candidate.as_ptr() as usize);
        OWNER_BYTES.set(bytes);
    }
}

pub(crate) fn track_candidate_slot<T>(candidate: &Vec<T>) {
    if ACTIVE.get() {
        let bytes = candidate
            .capacity()
            .checked_mul(std::mem::size_of::<T>())
            .unwrap();
        assert!(candidate.is_empty());
        assert!(bytes > 0);
        let mut pointers = REGISTERED_POINTERS.get();
        let mut sizes = REGISTERED_BYTES.get();
        let slot = pointers
            .iter()
            .position(|pointer| *pointer == 0)
            .expect("candidate observer slots exhausted");
        pointers[slot] = candidate.as_ptr() as usize;
        sizes[slot] = bytes;
        REGISTERED_POINTERS.set(pointers);
        REGISTERED_BYTES.set(sizes);
    }
}

/// Runs a small borrowed selector under the test allocator and returns how
/// many heap allocation requests it made.  The callback must not retain the
/// returned observation state, so this remains a test-only check rather than
/// a production allocator policy.
pub(crate) fn count_allocation_requests<F, R>(operation: F) -> (R, usize)
where
    F: FnOnce() -> R,
{
    assert!(!ACTIVE.get(), "candidate observations may not nest");
    REQUESTS.set(0);
    COUNT_ALL_REQUESTS.set(true);
    PHASE.set(0);
    PHASE_REQUESTS.set([0; MAX_PHASES]);
    ACTIVE.set(true);
    let result = operation();
    let requests = REQUESTS.get();
    ACTIVE.set(false);
    COUNT_ALL_REQUESTS.set(false);
    (result, requests)
}

pub(crate) struct PhaseGuard {
    previous: usize,
}

pub(crate) fn begin_phase(phase: usize) -> PhaseGuard {
    assert!(phase < MAX_PHASES);
    let previous = PHASE.replace(phase);
    PhaseGuard { previous }
}

pub(crate) fn phase_allocation_requests() -> [usize; MAX_PHASES] {
    PHASE_REQUESTS.get()
}

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        PHASE.set(self.previous);
    }
}

impl Observation {
    pub(crate) fn begin(bytes: usize, deny: bool) -> Self {
        assert!(!ACTIVE.get(), "candidate observations may not nest");
        assert!(bytes > 0);
        REQUEST_SIZE.set(bytes);
        REQUESTS.set(0);
        COUNT_ALL_REQUESTS.set(false);
        DENY.set(deny);
        OWNER_POINTER.set(0);
        OWNER_BYTES.set(0);
        OWNER_DROPS.set(0);
        OWNER_REALLOCS.set(0);
        REGISTERED_POINTERS.set([0; MAX_REGISTERED_OWNERS]);
        REGISTERED_BYTES.set([0; MAX_REGISTERED_OWNERS]);
        REGISTERED_DROPS.set([0; MAX_REGISTERED_OWNERS]);
        RELEASE_COUNT.set(0);
        RELEASE_DROP_SNAPSHOTS.set([0; MAX_RELEASES]);
        RELEASE_DROP_MASKS.set([[false; MAX_REGISTERED_OWNERS]; MAX_RELEASES]);
        RESTORE_COUNT.set(0);
        RESTORE_DROP_SNAPSHOTS.set([0; MAX_RELEASES]);
        RESTORE_DROP_MASKS.set([[false; MAX_REGISTERED_OWNERS]; MAX_RELEASES]);
        TRACK_ALL_CANDIDATES.set(false);
        ACTIVE.set(true);
        Self {
            _thread_bound: PhantomData,
        }
    }

    pub(crate) fn track<T>(&self, candidate: &Vec<T>) {
        let bytes = candidate
            .capacity()
            .checked_mul(std::mem::size_of::<T>())
            .unwrap();
        assert!(candidate.is_empty());
        assert!(bytes > 0);
        assert_eq!(OWNER_POINTER.get(), 0);
        OWNER_POINTER.set(candidate.as_ptr() as usize);
        OWNER_BYTES.set(bytes);
    }

    pub(crate) fn track_raw_slot(&self, slot: usize, pointer: *const u8, bytes: usize) {
        assert!(slot < MAX_REGISTERED_OWNERS);
        assert!(bytes > 0);
        let mut pointers = REGISTERED_POINTERS.get();
        let mut sizes = REGISTERED_BYTES.get();
        assert_eq!(pointers[slot], 0);
        pointers[slot] = pointer as usize;
        sizes[slot] = bytes;
        REGISTERED_POINTERS.set(pointers);
        REGISTERED_BYTES.set(sizes);
    }

    pub(crate) fn requests(&self) -> usize {
        REQUESTS.get()
    }

    pub(crate) fn drops(&self) -> usize {
        OWNER_DROPS.get()
    }

    pub(crate) fn reallocations(&self) -> usize {
        OWNER_REALLOCS.get()
    }

    pub(crate) fn registered_drops(&self) -> [usize; MAX_REGISTERED_OWNERS] {
        REGISTERED_DROPS.get()
    }

    pub(crate) fn registered_candidate_count(&self) -> usize {
        REGISTERED_POINTERS
            .get()
            .into_iter()
            .filter(|pointer| *pointer != 0)
            .count()
    }

    pub(crate) fn release_drop_snapshots(&self) -> [usize; MAX_RELEASES] {
        RELEASE_DROP_SNAPSHOTS.get()
    }

    pub(crate) fn release_drop_masks(&self) -> [[bool; MAX_REGISTERED_OWNERS]; MAX_RELEASES] {
        RELEASE_DROP_MASKS.get()
    }

    pub(crate) fn release_count(&self) -> usize {
        RELEASE_COUNT.get()
    }

    pub(crate) fn restore_count(&self) -> usize {
        RESTORE_COUNT.get()
    }

    pub(crate) fn restore_drop_snapshots(&self) -> [usize; MAX_RELEASES] {
        RESTORE_DROP_SNAPSHOTS.get()
    }

    pub(crate) fn restore_drop_masks(&self) -> [[bool; MAX_REGISTERED_OWNERS]; MAX_RELEASES] {
        RESTORE_DROP_MASKS.get()
    }
}

pub(crate) fn record_restore() {
    let _ = ACTIVE.try_with(|active| {
        if !active.get() {
            return;
        }
        let index = RESTORE_COUNT.get();
        if index < MAX_RELEASES {
            let drops = REGISTERED_DROPS.get();
            let total = drops.into_iter().sum();
            let mut snapshots = RESTORE_DROP_SNAPSHOTS.get();
            snapshots[index] = total;
            RESTORE_DROP_SNAPSHOTS.set(snapshots);
            let mut masks = RESTORE_DROP_MASKS.get();
            masks[index] = drops.map(|count| count != 0);
            RESTORE_DROP_MASKS.set(masks);
        }
        RESTORE_COUNT.set(index + 1);
    });
}

pub(crate) fn record_token_release(_token: &crate::allocation::AllocationTicket) {
    let _ = ACTIVE.try_with(|active| {
        if !active.get() {
            return;
        }
        let index = RELEASE_COUNT.get();
        if index < MAX_RELEASES {
            let drops = REGISTERED_DROPS.get();
            let total = drops.into_iter().sum();
            let mut snapshots = RELEASE_DROP_SNAPSHOTS.get();
            snapshots[index] = total;
            RELEASE_DROP_SNAPSHOTS.set(snapshots);
            let mut masks = RELEASE_DROP_MASKS.get();
            masks[index] = drops.map(|count| count != 0);
            RELEASE_DROP_MASKS.set(masks);
        }
        RELEASE_COUNT.set(index + 1);
    });
}

impl Drop for Observation {
    fn drop(&mut self) {
        ACTIVE.set(false);
        DENY.set(false);
        COUNT_ALL_REQUESTS.set(false);
        OWNER_POINTER.set(0);
        REGISTERED_POINTERS.set([0; MAX_REGISTERED_OWNERS]);
        REGISTERED_BYTES.set([0; MAX_REGISTERED_OWNERS]);
        TRACK_ALL_CANDIDATES.set(false);
    }
}

impl Drop for FreshCapacityExtraGuard {
    fn drop(&mut self) {
        FRESH_EXTRA_LABEL.set(None);
        FRESH_EXTRA_BYTES.set(0);
    }
}

impl Drop for TrackAllCandidatesGuard {
    fn drop(&mut self) {
        TRACK_ALL_CANDIDATES.set(false);
    }
}
