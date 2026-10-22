# Native pixel conformance validation

Base: main `070ad6509bad18f9ab76732d543659ef6d7f9129`. This branch is independent
of open PR #3 (`d3ea4e602763951626d678291cc9c325e2c475b2`). Investigation:
[5f2f38c](https://github.com/mith-mmk/avif-rust/tree/5f2f38c236051271a63798ab99662c9426573f80/investigation/native-pixel-20261022).

## Product changes

- Scale 6/8/14-tap deblock flatness thresholds by bit depth; disable deblock
  when both frame luma levels are zero, regardless of deltas.
- Reject edges outside the visible frame, but allow reconstructed padding taps
  and process the complete four-lane filtering unit. Perpendicular filtering
  can subsequently read those lanes.
- Retain complete 64-sample transform backing until the final crop. CfL uses
  only actually reconstructed transforms, rounded from the coded MI extent;
  when a chroma block extends farther, repeat its last subsampled luma group.
  Allocated capacity alone does not establish available reconstructed luma.
- Update UV smooth contexts at chroma owners over the shared footprint; use
  external chroma neighbours. Keep 4x4-MI deblock metadata padded to the coded
  even-MI extent, with bottom/right chroma owners and two-dimensional bounds.
- Apply CDEF to the full luma dimensions; subsample dimensions only for chroma.
- In high-bitdepth SGR restoration, normalize the variance inputs but retain
  the original-precision box sum in the affine offset.

Both legacy and strict paths use the corrected geometry. Strict admission
accounts for actual plane capacities and padded metadata, preserving exact/
one-under budget checks and rollback. This follow-up adds no allocations.
There are no runtime probes, trace hooks, reference dependencies or API changes.

## Regression review and correction

The earlier report at PR commit `a1f700df3d027cc651f5bc7ace79e4acabb20726`
incorrectly described broader residuals as pre-existing. Its aggregate counts
hid new wrong pixels. This report supersedes that classification. Compare
**individual samples**, including originally correct samples and error increases.

| Case / first differing stage | main | Previous PR | Current |
| --- | ---: | ---: | ---: |
| fox 8-bit 422 / reconstruction | 0 | 768 (max 2) | 0 |
| kimono.rotate90 / reconstruction | 60,958 | 637, including 548 newly wrong | 0 |
| synthetic 85x73 12-bit 422 / deblock | 234 | 10, including 2 newly wrong | 0 |

Coordinates below are zero-based native plane coordinates:

- Fox U(592,368): reference/main **123**, previous PR **122**, current **123**.
  CfL read unpopulated allocation beyond the last reconstructed transform.
  Clamp at the transform availability boundary, repeating the subsampled average.
  A unit vector distinguishes repeating `[100,200,300,300]` from reading zero
  padding or repeating an individual luma pixel. Full-transform cases remain
  covered by the original partial-edge tests and the seven-image comparison.
- Kimono U(440,353): reference/main **138**, previous PR **136**, current **138**.
  The same CfL availability correction removes all 637 raw errors.
- Synthetic V(40,71): reference/main **1370**, previous PR **1414**, current **1370**;
  V(40,72): **1534**, **1490**, **1534** respectively. Before the horizontal
  six-tap edge, reference p2..q2 was `[1371,1370,1370,1534,1534,1534]`; native
  used `[1371,1370,1370,1534,1512,1512]`. Skipped padding lanes of an earlier
  vertical edge changed the horizontal mask. Filtering the full four-lane unit
  fixes this; the new 12-bit 422 oracle fixture catches the regression.
- Default-filter cases additionally exposed existing CDEF luma-size and
  high-bitdepth SGR-offset defects. Correcting upstream samples alone exposed
  new wrong output positions downstream. Fixing both operations, with separate
  10-bit 420 CDEF and 12-bit gray SGR oracle fixtures, removes those populations.

## References and conditions

Linux x86-64; FFmpeg 7.1.5. Native planar samples, no RGB conversion, resizing
or range conversion. Copy visible rows using each decoder's stride; preserve
plane order, subsampling and bit depth. 8-bit samples are bytes, 10/12-bit
samples are little-endian u16. Separate primary colour and alpha. Film grain
is disabled on all seven attached inputs.

- libaom 3.12.1, Debian `3.12.1-1+deb13u1`; C-only source build commit
  `10aece4157eb79315da205f39e19bf6ab3ee30d0`, single-thread, private diagnostic
  dumps before deblock and after deblock/CDEF/restoration. Final output was
  also cross-checked against the installed FFmpeg decoder.
- dav1d 1.5.1, Debian `1.5.1-1`, source commit
  `42b2b24fb8819f1ed3643aa9cf2a62f03868e3aa`; single-thread, grain disabled,
  `inloop_filters` masks 0/1/3/7 expose the same stages.
- Native stage dumps were inserted only into a disposable copy of product
  source. The main control is unchanged main plus the same dump boundaries.
  Private inputs, extracted bitstreams and raw dumps are not committed.

Both independent references agree at **all four stages of all 73 core cases**.
Native strict and legacy agree at every stage and final output on all 73.
The two alpha-bearing core inputs also match both references (plum: 16,384
samples; abc: 131,072). Results are sample counts, not differing byte counts:

| Set | Cases | Exact at all four stages | Remaining |
| --- | ---: | ---: | --- |
| Original attached images | 7 | 7 | none |
| Existing comparable corpus streams | 18 | 18 | none |
| 85x73 synthetic, CDEF/restoration disabled | 24 | 24 | none |
| 93x77 synthetic, default filters | 24 | 23 | 8-bit 420 lossy, described below |

All 19 corpus downloads were SHA-256 checked against `scripts/test_corpus.csv`.
The grid item is excluded from elementary-stream comparison; existing grid tests
cover its API behaviour. Each synthetic set covers 8/10/12-bit, 400/420/422/444,
lossy/lossless. All lossless cases are exact. The machine-readable
`native-pixel-validation.json` records input hashes, main/previous PR controls,
current errors, per-stage reference/strict agreement and individual regressions.

## Remaining differences: demonstrated existing, not assumed

The default-filter synthetic `yuv420p-lossy.avif` is exact after reconstruction;
24 samples (max 1) differ after deblock and 25 (max 1) after CDEF/restoration.
Main has 10/37/1645/1645 differences at those stages. **Every remaining wrong
sample has the same position and value as main**, and no sample's absolute
error increases. For example, after deblock, V(23,8) is 145 in main/current
versus reference 146; final V(24,7) is 146 versus 147. The residual begins at
deblock. The deeper mask/metadata cause is not established here.

An additional 92 deterministic CfL fixture-search patterns were compared at
final output against main and both independent decoders. 89 are exact. Three
85x73 forced-32-partition, 8-bit noise-pattern cases retain 4/10/8 differences
(420/422/444, max 1). Stage checks on those three show reconstruction exact and
first differences after deblock; all remaining values equal main, with zero
newly incorrect samples and zero increased errors at every checked stage.
`native-pixel-exploratory.json` records hashes and per-sample classifications.
These exploratory patterns are not bundled fixtures or an exhaustive corpus.

Across the audited cases there are **zero newly incorrect samples and zero
increased absolute errors versus main**. This is a bounded regression result,
not general AV1 conformance, inter-frame coverage, or proof for all bitstreams.
Do not regenerate oracle expectations from native output to hide differences.

## Tests and reproduction

Eight bundled MIT arithmetic-pattern fixture/oracle pairs compare every native
sample and strict/legacy equality without external executables. All eight pairs
regenerate with identical SHA-256 hashes using the versions above. See
`tests/FIXTURES.md` and `scripts/generate_native_pixel_fixtures.py` for generation,
including `--default-filters`. Unit tests cover padding, CfL availability, shared
contexts, high-bitdepth filter vectors and strict allocation admission/rollback.

Local release default suite: **870 passed, 0 failed, 9 ignored**. All features:
**882 passed, 0 failed, 9 ignored**. `cargo fmt --check` and
`cargo clippy --all-targets -- -D warnings` pass. Hosted CI is linked in PR #4.

To independently classify already-normalized dumps from the same stage:

```sh
python scripts/compare_native_pixel_stages.py --bit-depth 12 \
  reference.yuv main.yuv candidate.yuv
```

The helper rejects unequal lengths and reports main/current counts, maximum
and total error, newly incorrect samples, increased errors, changed residuals,
and first differing sample indices. Establish libaom/dav1d equality first.
Stage extraction requires private instrumentation at the named boundaries;
no instrumentation is enabled in product builds. Final bundled tests need none.

PR #3 remains independent and unchanged. A non-mutating merge-tree check found
conflicts in `.gitattributes`, `src/av1/tile_decode/tests/tile_decode_reconstruction.rs`,
`src/decoder.rs`, `src/decoder/deblock_budget_tests.rs`, and `tests/FIXTURES.md`.
That branch needs its own reconciliation and validation before any later merge.
