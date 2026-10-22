# PR #3 integration with native pixel fixes

The integration merges main `94876c382ea517fa91f6444b19789f0d34fe7ae7`
(PR #4) into PR #3 `d3ea4e602763951626d678291cc9c325e2c475b2` without
rebasing or rewriting history. This record supersedes pre-integration validation
claims for the combined tree; historical measurements retain their original pins.

## Semantic conflict resolution

- Preserve binary attributes for every YUV oracle and native AVIF fixture.
- Retain both PRs' reconstruction/CfL and deblock-budget regression tests.
- Retain PR #3's one immutable deblock metadata grid shared by plane workers.
  Its allocation, population and lookup all use PR #4's padded coded-MI
  dimensions: `ceil(visible/8)*2` cells per axis, four pixels per cell. Keeping
  the old shared allocation `ceil(visible/4)` would disagree at partial edges.
- Preserve PR #4's checked strict admission, 64-sample transform backing,
  actual CfL transform availability, complete four-lane edge filtering,
  bitdepth-scaled flatness, zero-level gating and SGR precision correction.
- Preserve PR #3's logical reference/mode filtering, current/previous strength
  selection, normative delta math, inter prediction precision, masks, OBMC,
  sub8 chroma and warp corrections. The common luma CDEF fix is retained once.
- Combine AVIS oracle provenance with the eight native synthetic fixture records.
  No oracle was replaced, tolerance widened, test ignored or CI gate removed.

## Native sample audit

Baseline is merged PR #4 main, not the older `070ad65` main. Input SHA-256 and
extracted elementary bitstream equality were checked before reusing the
independently agreed libaom/dav1d stage oracles from the PR #4 investigation.
libaom 3.12.1 (`10aece4157eb79315da205f39e19bf6ab3ee30d0`) and dav1d 1.5.1
(`42b2b24fb8819f1ed3643aa9cf2a62f03868e3aa`) agree at all four core stages.
Measurements use visible native planar samples and their strides, with no RGB,
range or depth conversion. Stage hooks exist only in a disposable source copy.

- Original seven images: exact reconstruction/deblock/CDEF/restoration against
  both references, in strict and legacy modes; alpha also exact.
- Eighteen comparable public corpus streams: exact at all four stages, including
  fox 8-bit 422, all odd-dimension variants and kimono. Grid is excluded from
  the elementary-stream audit and covered by existing grid API regressions.
- Forty-eight synthetic format/depth/filter combinations: 47 exact at all stages.
  The remaining default-filter 8-bit 420 has 24 deblock / 25 final errors,
  maximum 1, at exactly the same positions and values as PR #4 main.
  The 12-bit 422 ±44 regression fixture remains exact.
- Across all 73 core cases, strict and legacy match at every stage and final
  output. Newly incorrect samples and increased absolute errors versus main:
  **zero**. All remaining erroneous samples retain their exact main value.
- Ninety-two additional deterministic patterns: strict/legacy agree; zero newly
  incorrect or increased-error samples. The same three 420/422/444 patterns
  retain 4/10/8 errors, maximum 1, unchanged from main. Their earlier stage
  attribution is recorded in `native-pixel-exploratory.json`.

`pr3-main-integration.json` records per-case hashes and sample comparisons.
The existing deblock residuals remain bounded known limitations, not a general
AV1 conformance claim. Private inputs, bitstreams and raw dumps are not committed.

## AVIS, allocation and regression checks

The mandatory seven AVIS tests pass, including exact comparison of 368,640
samples over 60 displayed frames. Fresh FFmpeg decodes with both libaom and
dav1d equal the checked-in sequence oracle, SHA-256
`3c97d5771e916c2cc1c54df81f1c5ac9fd6a505b02953509af7a6d998171c8bc`.
The eight bundled native fixture/oracle pairs also pass unchanged.

The separate allocation-counting call in `examples/performance.rs` confirms
that sharing is retained. Output hashes agree with main:

| Existing tracked input | Allocation calls main / combined | Requested bytes main / combined |
| --- | ---: | ---: |
| 900x900 WML2Viewer | 27,045 / 27,043 | 54,493,667 / 51,224,851 |
| 128x128 plum alpha | 2,756 / 2,754 | 1,952,009 / 1,886,473 |

Counts include the full requested realloc size and are not live/peak memory.
These are allocation checks, not new throughput or RSS measurements; concurrent
validation was running, so timing output is deliberately not reported. Historical
isolated timing methodology/results remain in `PERFORMANCE.md`.

Local release default: **911 passed, 0 failed, 9 existing ignored**.
All features: **923 passed, 0 failed, 9 existing ignored**.
Formatting and `cargo clippy --all-targets --all-features -- -D warnings` pass.
Hosted CI is recorded in PR #3 after normal commit/push. Additional Hadrianus/
NERO hardware tests are omitted as requested; hosted OS CI is a distinct check.
