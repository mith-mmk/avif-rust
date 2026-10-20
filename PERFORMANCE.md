# Performance and memory, 2026-10-20

Legacy decoding built the same luma-coordinate deblock metadata grid independently for each plane. One immutable grid now serves all plane workers. Strict decoder admission and filter arithmetic are unchanged.

The allocation counters identified avoidable temporary storage; these are allocation reductions rather than a demonstrated general throughput improvement. Detailed before/after raw trials are in `performance/*.json`. Baseline source: `5bd0c64d4c2a00c3c4912b9a26abfc49c6330326`. `performance/environment.json` records Rust, CPU, flags, source fingerprints and measurement scope. All measured inputs are 8-bit; 10/12-bit correctness continues to be covered by regression tests, not a performance claim.

## Measurement

Release builds, default features, generic Rust target, no custom RUSTFLAGS. Five fresh processes per version, alternating before/after order, one warm-up then nine timed API calls per process. Report median of the five process medians. No builds or other benchmarks ran concurrently with the recorded trials. Input preparation, checksum computation and output destruction are outside timings. The encoder uses deterministic procedural pixels, speed10, jobs0 with threading disabled, equal primary/alpha quality. The decoder materializes all native sequence frames; this does not measure streaming memory.

Allocation counters run in a separate call with timing disabled. Counts include alloc, alloc_zeroed and realloc; cumulative requested bytes count the full realloc target, not just growth. This is not peak/live heap. The allocator wrapper is present with counters disabled during timings in both binaries. Peak RSS is Linux process VmHWM, including setup/warm-up/allocator retention; sub-page and process-noise differences are not proof of an RSS improvement. The comparator requires identical output hashes across every trial. Decoder hashes every native u16 sample; encoder hashes every encoded byte.

## Reproduce

Keep before/after release binaries separate. From this repository, create the baseline source without changing the work branch:

```sh
mkdir -p /tmp/decode-baseline
git archive 5bd0c64d4c2a00c3c4912b9a26abfc49c6330326 | tar -x -C /tmp/decode-baseline
mkdir -p /tmp/decode-baseline/examples
cp examples/performance.rs /tmp/decode-baseline/examples/
CARGO_TARGET_DIR=/tmp/decode-build cargo build --locked --release --manifest-path /tmp/decode-baseline/Cargo.toml --example performance
cp /tmp/decode-build/release/examples/performance /tmp/avif-decode-before
CARGO_TARGET_DIR=/tmp/decode-build cargo build --locked --release --example performance
cp /tmp/decode-build/release/examples/performance /tmp/avif-decode-after
```

Then run the comparator with the workload arguments recorded in each JSON file, e.g.:

```sh
python3 scripts/compare-performance.py --before /tmp/avif-decode-before --after /tmp/avif-decode-after --output /tmp/decode.json tests/avis-subsampled-palette.avif sequence 9
```

The mandatory full sequence oracle still compares 368,640 native samples against independent reference output. The strict-budget path already shared its grid and receives no new allocation. Remaining candidates include restoration boundary workspace reuse; larger filter rewrites require separate profiling and exact oracle coverage.

## Results

| Workload | Median ms, before → after | Allocation calls | Requested bytes | Peak RSS KiB |
|---|---:|---:|---:|---:|
| decode-alpha | 2.356 → 2.335 | 2,756 → 2,754 | 1,952,009 → 1,886,473 | 3,256 → 3,296 |
| decode-sequence | 27.374 → 27.757 | 32,223 → 32,101 | 21,549,820 → 20,550,396 | 4,116 → 4,104 |
| decode-still | 123.894 → 123.428 | 27,058 → 27,056 | 52,553,475 → 49,313,523 | 31,000 → 30,672 |

## Validation

Release all-features suite: 915 passed, 0 failed, 9 existing ignored. Strict all-target/all-feature Clippy and format pass. Default mandatory AVIS tests7/7 and filter/strict-budget tests45/45 pass, including all368,640 independent reference samples.
