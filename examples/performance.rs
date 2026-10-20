use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::time::Instant;
struct Counter;
static ACTIVE: AtomicBool = AtomicBool::new(false);
static COUNT: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
fn record(size: usize) {
    if ACTIVE.load(Relaxed) {
        COUNT.fetch_add(1, Relaxed);
        BYTES.fetch_add(size as u64, Relaxed);
    }
}
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(size);
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: Counter = Counter;
fn hash(values: impl IntoIterator<Item = u8>) -> u64 {
    values.into_iter().fold(14695981039346656037, |h, v| {
        (h ^ u64::from(v)).wrapping_mul(1099511628211)
    })
}
fn measure<T>(iterations: usize, mut run: impl FnMut() -> T, checksum: impl Fn(&T) -> u64) {
    assert!(iterations >= 3);
    let expected = checksum(&run()); // warm-up, outside measurement
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let started = Instant::now();
        let result = std::hint::black_box(run());
        times.push(started.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(checksum(&result), expected);
        drop(result); // output destruction and checksum excluded from timings
    }
    COUNT.store(0, Relaxed);
    BYTES.store(0, Relaxed);
    ACTIVE.store(true, Relaxed);
    let result = run();
    ACTIVE.store(false, Relaxed);
    let allocations = COUNT.load(Relaxed);
    let bytes = BYTES.load(Relaxed);
    assert_eq!(checksum(&result), expected);
    times.sort_by(f64::total_cmp);
    let rss = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .map(str::to_owned)
        });
    println!(
        "iterations={iterations} median_ms={:.6} min_ms={:.6} max_ms={:.6} allocations={allocations} requested_bytes={bytes} checksum={expected:016x} peak_rss={}",
        times[iterations / 2],
        times[0],
        times[iterations - 1],
        rss.as_deref().unwrap_or("unavailable")
    );
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let data = std::fs::read(&args[1]).unwrap();
    let sequence = args[2] == "sequence";
    let iterations = args.get(3).map_or(7, |s| s.parse().unwrap());
    let run = || {
        if sequence {
            avif_rust::decode_sequence_frames_bytes(&data).unwrap()
        } else {
            vec![avif_rust::decode_frame_bytes(&data).unwrap()]
        }
    };
    let info = run();
    println!(
        "input={} bytes={} dimensions={}x{} bit_depth={} frames={} mode={}",
        args[1],
        data.len(),
        info[0].width,
        info[0].height,
        info[0].bit_depth,
        info.len(),
        args[2]
    );
    drop(info);
    measure(iterations, run, |frames| {
        hash(
            frames
                .iter()
                .flat_map(|f| f.buffers.planes.iter())
                .flat_map(|p| p.samples.iter())
                .flat_map(|s| s.to_le_bytes()),
        )
    });
}
