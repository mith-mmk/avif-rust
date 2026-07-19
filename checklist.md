# Pure Rust AVIF/AV1 decoder implementation checklist

## Current vertical slice

The first completion target is an exact, usable still-image decoder for the
profile exercised by `samples/WML2Viewer.avif`:

- 8-bit AV1 intra key frame
- 4:4:4 identity GBR
- one frame and one tile group
- native decoded planes plus SDR RGBA8/RGBA16 output

`WML2Viewer.avif` is a regression fixture, not the whole implementation
target. Its raw native planes now match the filter-disabled AOM oracle exactly;
the current priority is the normative post-filter pipeline. No format expansion
or additional architecture refactor is promoted above this vertical slice
until its final filtered planes match the reference oracle.

## Publication readiness

- [x] Declare Rust 1.88 as the crate MSRV and document it in both READMEs.
- [x] Include the complete MIT license text in the published crate archive.
- [x] Restrict the published archive to the library sources, English and
      Japanese READMEs, and the license, in addition to Cargo-generated
      metadata and lockfile files.

The integration tests, this checklist, fuzz targets, oracle scripts, and local
fixtures remain available in the repository but are intentionally excluded
from the consumer-facing crate archive.

### other avif sample images

- https://github.com/link-u/avif-sample-images
- https://colinbendell.github.io/webperf/animated-gif-decode/avif.html

### External 8-bit YUV444/420/422 compatibility gate (2026-07-16)

- [x] Resolve primary-item `ipma` associations from the ordered `ipco`
      property table; reject missing, duplicate, zero and out-of-range
      singleton properties as `Bitstream`.
- [x] Run the public composition preflight before AV1 header parsing and keep
      RGBA ICC and active film-grain rejection fail-closed; `clap`/`imir` and
      grid composition now compose.
- [x] Keep tile/block/residual probes diagnostic-only; probe failure no longer
      changes image decode success.
- [x] Validate AV1 entropy termination from the decoder state and trailing
      pattern without disabling validation.
- [x] Use the AV1 `Tx32x32` default scan for the coded coefficient area of
      `Tx64x32` and `Tx32x64` rectangular transforms.
- [x] Convert all four external 8-bit YUV444 samples and verify PNG
      signature/IHDR dimensions: `1204x800`, `1204x799`, `1203x800` and
      `1203x799`.
- [x] Convert the external 8-bit YUV420, monochrome and YUV422 samples and
      verify their `1204x800` PNG dimensions.
- [x] Add pixel-level RGBA oracle coverage for the new subsampling paths.
- [x] Add a reproducible libaom adaptive-quantization still-image sample and
      compare its decoded dimensions and RGBA output with FFmpeg.
- [x] Add native-plane oracle coverage for the new subsampling paths.
- [x] Convert all 25 external samples, including the 12-bit, grid, alpha,
      ICC, subsampling and transform variants, with no partial PNG output.

The reproducible parent-root gate is:

```powershell
pwsh -File test/avif_external_compat.ps1 -DownloadMissing
```

The gate result is 25 successes, 0 expected failures, 0 unexpected results
and 0 partial PNGs. The converted PNGs remain ignored under
`test/images/external/converted/avif/` for visual review; the AVIF sample files
remain ignored as well.

## Guardrails

- AVIF and AV1 decoding remain pure Rust. FFmpeg/libaom are test or fixture
  oracles only.
- Keep the existing callback API, `wml2` AVIF feature and public exports.
- Keep `test_data/` ignored. Commit only reproducible scripts, manifests and
  schema tests; generated AVIF/reference bytes stay local.
- Keep temporary files under ignored `.test*` directories and remove them.
- Unsupported AV1/AVIF tools must return explicit `Unsupported` rather than a
  partially decoded image presented as valid output.
- [x] Reject unknown essential `ipco` properties before public decode instead
      of silently accepting a potentially incomplete interpretation.
- `avif/` is an independently managed nested repository; inspect and test it
  directly.

## 1. Oracle gate and integration boundary

- [x] Keep reusable exact-plane and RGBA max-error assertions.
- [x] Keep manifest schema, duplicate-ID, dimensions, bit-depth, plane-count
      and overflow validation.
- [x] Add a local fixture generator for AVIF, native planes, RGBA8/RGBA16,
      source hash and `test_data/oracles.csv`.
- [x] Add `AVIF_REQUIRE_ORACLES=1` strict mode so conformance runs fail when
      the manifest is absent, while normal parser/safety tests remain runnable
      without local test data.
- [x] Make strict mode reject a header-only/zero-entry manifest and require
      the approved fixture IDs; manifest absence is already a strict failure.
- [ ] Add the future `-RegisterInStrictManifest` path to
      `generate_oracles.ps1`; the default `WML2Viewer` output remains
      diagnostic-only, while filter-disabled generation must opt in explicitly.
- [x] Recompute source hashes and compare them with the source manifest using
      `scripts/verify_oracle_sources.ps1`.
- [x] Generate the first filter-disabled 8-bit 4:4:4 identity-GBR fixture.
- [x] Generate exact residual, partition and directional fixtures in that same
      profile.
- [x] Generate the palette fixture in that same profile and assert that the
      bitstream actually contains decoded palette blocks and color maps.
- [ ] Generate the `WML2Viewer.avif` native-plane and RGBA fixtures.
- [ ] Make diagnostic fixture generation opt-in for strict registration
      (for example, `generate_oracles.ps1 -RegisterInStrictManifest`), and keep
      the default `WML2Viewer` generation out of `oracles.csv`.
- [ ] Add one reproducible recipe/bootstrap for `BlackLossless` and the five
      filter-disabled fixtures from a fresh clone (`scripts/bootstrap_oracles.ps1`).
- [x] Make the strict oracle command part of the documented AVIF validation
      gate.
- [x] Capture and assert `wml2` draw callback bytes and dimensions in the
      AVIF integration test.
- [x] Assert callback order as `init -> draw -> terminate` on the
      `filter-disabled-gbr` public fixture together with callback bytes and
      dimensions.

## 2. Raw block reconstruction: exact pre-filter checkpoint

The following work is one atomic correctness track. Each item must pass an
exact native-plane fixture before the next feature is enabled.

- [x] Align partition recursion and block traversal with the filter-disabled
      partition and directional reference streams.
- [x] Route first-leaf traversal through the first child of vertical,
      horizontal, extended and four-way partitions.
- [x] Align transform-block placement for luma and chroma, including clipped
      frame edges and per-plane coordinates.
- [x] Verify transform-size selection and transform partition traversal for
      every transform used by the filter-disabled fixture set.
- [x] Verify entropy/CDF update state across blocks in the current vertical
      slice: partition, mode, txb skip,
      DC sign, EOB, coefficient base/range, signs and Golomb extension.
- [x] Verify palette-mode filter-intra exclusion, entropy termination and
      complete coded-tile coverage through the aligned MI edge for
      `WML2Viewer.avif` and the strict filter-disabled fixtures.
- [ ] Validate all 19 rectangular and square transform sizes, including
      32x32/64x64 paths, against standalone normative vectors.
- [x] Verify the default zig-zag scan used by the supported 2-D
      `ADST_DCT`/`DCT_ADST` classes with known vectors.
- [x] Apply AOM's inverse `1/sqrt(2)` input normalization when a rectangular
      transform has an odd log2 aspect ratio; pin the Tx4x8 ADST_DCT path with
      a fixed residual vector.
- [x] Match AOM inverse-transform `round_shift` semantics for negative values,
      including negative half-step anchors shared by `round2`.
- [x] Preserve the AOM `mrow`/`mcol` entropy scan for 1-D directional classes
      and transpose square ADST, 1-D, identity and lossy 4x4/8x8 `DctDct`
      coefficient storage at the inverse-transform boundary; preserve the
      existing 4x4 storage for coded-lossless WHT and cover both paths with
      decoder vectors.
- [x] Pin the dense 4x4 `ADST_ADST` block feeding the former first
      `WML2Viewer` mismatch against AOM prediction, dequant and output vectors.
- [x] Pin the 8x8 `DctDct` residual inside a palette block against AOM palette
      prediction, dequant input and all 64 output samples.
- [x] Pin a lossy 4x4 `DctDct` block against AOM prediction, dequant input and
      output while keeping the strict lossless fixture exact.
- [x] Route non-zero angle deltas on vertical/horizontal modes through the
      directional predictor and use AOM's zone-specific top-right/bottom-left
      edge lengths.
- [x] Track top-right and bottom-left reconstruction availability as exact
      sample lengths, including partial transform-edge coverage, and pad each
      directional edge from its last available sample.
- [x] Verify coefficient context selection with known vectors for the enabled
      transform sizes and plane types.
- [x] Verify palette and filter-intra syntax gating for the supported profile.
- [ ] Support all 19 AV1 transform sizes, including rectangular transform
      placement and per-plane coordinates.
- [x] Derive intra chroma transform types from the signalled UV mode and transform
      set for the supported sub-32 transform sizes.
- [x] Match AOM's staged 32-point inverse DCT and sample-count-based dequant
      shift for large rectangular transforms; pin the `Tx32x16` chroma DC
      reconstruction used by `WML2Viewer.avif`.
- [x] Replace the approximate 32x32 fixed-basis/DC shortcut with the staged
      32-point row/column transform and pin the `WML2Viewer` luma DC block;
      use `bit_depth + 8` for the row-stage range through 12-bit paths.
- [x] Match the two-stage rounding for DC-only `Tx64x64` and remove the
      square-transform `1/8` output scaling from the 64-point stage reused by
      rectangular transforms.
- [x] Port the normative staged 64-point inverse DCT, including intermediate
      range clamps and rounding, and pin the `WML2Viewer` `Tx16x64` path.
- [x] Route `Tx64x64 DctDct` through the staged row/column core and transpose
      its lossy coefficient storage at the inverse-transform boundary.
- [ ] Validate the remaining chroma transform derivation and 32x32/64x64
      transform paths against normative reference vectors.
- [x] Reject partial tile groups before accepting a still-image decode.
- [x] Require entropy termination/trailing-bit validation before accepting a
      decoded frame.
- [x] Interleave Y/U/V residual traversal inside each 64x64 coding unit for
      blocks larger than 64x64, matching AV1 syntax order instead of decoding
      one complete 128x128 plane at a time.
- [x] Use rounded restoration-unit counts at the frame edge so a remainder of
      at most half a unit is merged into the preceding unit instead of reading
      nonexistent restoration coefficients.
- [x] Replace diagnostic-only sample assertions with exact plane assertions
      for the generated filter-disabled fixtures.
- [x] Complete the filter-disabled fixture set with exact Y/U/V plane matches.
- [x] Complete the `WML2Viewer.avif` raw reconstruction comparison with exact
      Y/U/V equality against the filter-disabled AOM oracle.
- [x] Add a private, ignored `AOM_PREFILTER_ORACLE` diagnostic that compares
      all native planes with an AOM build whose deblock, CDEF and restoration
      stages are disabled.
- [x] Read AV1 angle deltas for every block size at or above `BLOCK_8X8`,
      including the rectangular `4x16` and `16x4` forms.
- [x] Record the true AOM pre-filter result (no mismatch in any plane) and keep
      this diagnostic fixture out of the passing strict manifest.

Raw reconstruction is the fixed next correctness track for any remaining
bitstream mismatch. Keep palette-mode filter-intra gating, all 19 rectangular
and square TxSizes, chroma TxType derivation, 32/64-point transforms, entropy
termination and complete tile coverage explicit in the vectors before moving
the public decode gate to final filtered pixels.

Completed prerequisites retained as stable code:

- [x] Tile decoder responsibilities are split into syntax, entropy,
      reconstruction, diagnostics and public API modules.
- [x] Existing private tests moved with their implementation modules.
- [x] Size/class-specific coefficient CDFs and neighbour txb state are wired.
- [x] DC-sign contexts, coefficient state propagation and scan helpers have
      known-vector coverage.
- [x] Partition-aware top-right/bottom-left availability is derived from live
      reconstructed-MI coverage.
- [x] CFL-allowed UV mode CDFs, joint alpha syntax and 4:4:4 CFL prediction are
      wired for the current non-subsampled reconstruction path.
- [x] Existing transform dispatch and 8-bit reference anchors are retained.

## 3. Reconstruction filters

Raw reconstruction now passes the filter-disabled fixtures and the complete
`WML2Viewer` pre-filter oracle. Post-filter implementation is the current main
work. The supported 8-bit 4:4:4 path is now public; the integrated oracle and
broader format gate remain. Keep filter metadata in private reconstruction state; do not change the
public decoded-frame shape.

- [x] Collect CDEF indices within each tile during block traversal.
- [x] Aggregate CDEF indices into frame-private post-filter state during full/prefix traversal; retain the state for the filter-application stage.
- [x] Retain transform boundaries and restoration unit type/coefficients in
      the frame-level filter state used by every stage.
- [ ] Retain transform-boundary skip/mode state and consume it when deriving
      normative deblock levels and edge lengths.
- [ ] Integrate frame-level post-filter state and apply filters in fixed order:
      deblock -> CDEF -> loop restoration; verify each stage with plane oracles.
- [x] Preserve aligned coded-plane storage through the filter pipeline and crop
      only after filtering; deblock eligibility is bounded by the visible frame
      while retaining the coded stride for neighbor taps.
- [ ] Implement deblocking in normative order with boundary and strength
      vectors.
- [x] Implement the scalar 4-tap deblock edge kernel and limit/HEV mask.
- [x] Connect deblock stage traversal to retained transform boundaries for
      4:4:4 scalar frames.
- [x] Implement the scalar CDEF constrain, direction search and 8x8 block
      kernel.
- [x] Collect tile-local CDEF indices during traversal.
- [ ] Integrate frame-level CDEF state and apply each decoded per-block index
      after deblock; verify all planes against a CDEF oracle.
- [x] Map each luma CDEF block to its subsampled chroma-plane origin and block
      extent; add a generated non-lossless 4:2:0 CDEF sample against FFmpeg
      (native YUV average error <=`2`, maximum <=`32`; RGB average error
      `1.217`, maximum `179`).
- [ ] Match normative CDEF frame preparation and block semantics: luma
      direction/variance is shared with chroma, luma primary strength uses the
      directional-variance adjustment, secondary strength `3` maps to `4`,
      and frame-edge sentinel/clipping behavior remains to be made exact. The
      scalar direction table now follows AOM's axis order; 8-bit chroma damping
      applies the normative one-step reduction.
- [x] Generate a local diagnostic oracle with deblock enabled and restoration
      disabled; keep the CDEF result non-authoritative until all plane values
      are exact and add a reproducible bootstrap recipe before release.
- [x] Add a generated lossless 8-bit 4:4:4 CDEF sample with an explicit
      BT.709/range oracle; current RGB error is average `0.216`, maximum `9`.
- [x] Implement loop restoration and restoration-unit boundary handling for
      the 8-bit 4:4:4 diagnostic path, including AOM stripe halos and the
      150%-sized final restoration unit.
- [x] Implement the scalar Wiener restoration kernel with transmitted
      vertical/horizontal axis order, AOM's 3/11-bit intermediate rounding,
      chroma 5-tap outer-zero rule and vertical halo.
- [x] Retain the SGRPROJ parameter index and projection coefficients in
      frame-private restoration state.
- [x] Implement SGRPROJ restoration and verify stripe/boundary behavior
      against the WML2Viewer AOM restoration oracle.
- [x] Enable public 8-bit 4:4:4 filtered decode and verify the WML2Viewer
      image and `wml2` callback path through deblock, CDEF and restoration.
- [ ] Verify the complete reconstruction/filter order against plane oracles.
- [ ] Enable the `WML2Viewer.avif` final oracle only after the required filters
      are active and exact.

## 4. Format and composition backlog

Do not start these items until the current vertical slice reaches exact plane
and RGBA gates.

- [x] Public 8-bit monochrome decode and luma-to-RGB conversion.
- [x] Public 8-bit 4:2:0 decode with subsampled chroma conversion.
- [x] Public 8-bit 4:2:2 decode with subsampled chroma conversion.
- [x] Honour AV1 4:2:0 chroma sample positions (unknown/vertical/colocated)
      during RGBA conversion, with generated three-position FFmpeg oracle
      coverage.
- [x] Validate 4:2:0 chroma sample position consistency between the AVIF
      `av1C` property and the AV1 sequence header before public decode.
- [x] Public alpha auxiliary composition for the 8-bit 4:4:4 sample, including
      optional native `DecodedFrame.buffers.planes[3]` alpha output and RGBA
      conversion coverage.
- [x] Keep the limited SDR BT.601/BT.709 colour-conversion core.
- [x] Verify the SDR BT.601/BT.709 conversion core with generated BT.709 and
      BT.470 BG AVIF plane/RGBA fixtures against the FFmpeg oracle.
- [x] Accept H.273 GBR (`matrix_coefficients=3`) through the native GBR
      identity plane order for 8/16-bit output and film-grain handling; a
      synthetic metadata vector covers the path because FFmpeg does not emit
      this matrix value directly.
- [x] Verify 4:4:4 non-identity colour paths with a real AVIF plane/RGBA
      fixture (`fox.profile1.8bpc.yuv444.avif`); the external oracle reports
      average RGB error `0.00192` and maximum channel error `5`.
- [x] Add a generated 8-bit 4:4:4 SMPTE 240M (`matrix_coefficients=7`)
      AVIF sample and compare its RGBA output against FFmpeg.
- [x] Decode BT.2020 constant-luminance (`matrix_coefficients=10`) with the
      sign-dependent chroma branches from H.273. A focused vector covers both
      branches, and a generated libaom sample exercises AVIF metadata parsing;
      this FFmpeg build can encode matrix 10 but cannot convert it to RGBA for
      an independent pixel oracle.
- [x] Decode SMPTE ST 2085 Y'DzDx (`matrix_coefficients=11`) with the
      H.273 76--78 inverse equations. A focused vector covers the centered
      chroma scaling and a generated libaom sample exercises metadata parsing;
      the available external decoders do not expose a stable RGBA oracle for
      this matrix.
- [x] Decode BT.2100 ICtCp (`matrix_coefficients=14`) by reversing the
      normative LMS'/ICtCp matrices. A generated FFmpeg/libaom PQ sample now
      exercises the path against an independent zscale RGB oracle.
- [x] Decode chromaticity-derived non-constant/constant luminance matrices
      (`matrix_coefficients=12/13`) by deriving KR/KB from the H.273 primary
      chromaticities. BT.2020 vectors cover the derived coefficients, unknown
      primaries remain fail-closed, and generated libaom samples exercise both
      metadata paths.
- [x] 10-bit quantisation/reconstruction with dedicated AOM quantizer tables;
      12-bit parsing/output is covered, while film grain is applied for the
      non-overlap still-image path.
- [x] Parse and apply AV1 quantisation-matrix levels 0 through 14 using the
      normative libaom inverse tables; level 15 remains the flat identity
      matrix. Generated level-0 and level-1 samples match FFmpeg.
- [x] Grid image-cell composition for still images, including ordered `dimg`
      references and RGBA/native-plane placement; verified with
      `sofa_grid1x5_420.avif`. Native planes also apply aligned `clap`/`imir`/
      `irot` geometry for 4:4:4 and 4:2:0, and alpha grids now compose an
      optional plane 3; 4:2:2 quarter-turns swap chroma subsampling axes in
      the native plane metadata.
- [x] `clap` clean-aperture crop and `imir` horizontal mirror composition.
- [x] `irot` rotation composition, including an official 8-bit alpha fixture;
      `kimono.rotate90.avif` now decodes and converts with the expected
      dimensions; its strict RGB error is approximately `2.10` average and
      remains just above the `2.0` promotion threshold.
- [x] Multiple tile-group OBU composition for one still frame. Payload
      assembly validates duplicates/holes and merges tile IDs in order; a
      generated libaom sample is rewritten into separate FrameHeader and
      single-tile TileGroup OBUs and decoded against the original pixels.
- [x] Accept a primary-item AV1 sequence by selecting the first frame's OBU
      and tile groups up to the next frame boundary; generated two-frame AVIF
      coverage verifies the existing still-image API returns frame one without
      mixing later tiles.
- [x] AV1 horizontal super-resolution output resize with the normative 8-tap
      phase filter; coded planes are reconstructed at reduced width and
      expanded before colour conversion.
- [x] Film grain for still images when `overlap_flag == 0`, including the
      normative Gaussian sequence, AR synthesis and chroma scaling paths.
- [x] Film grain overlap blending (`overlap_flag == 1`) for luma and
      subsampled chroma planes.
- [x] Decode CICP PQ (transfer 16) and HLG (transfer 18) through a bounded
      SDR tone map for the existing RGBA8/16 output API.
- [x] Decode CICP gamma 2.2/2.8 (transfer 4/5) and linear (transfer 8) curves
      into the existing sRGB RGBA8/16 output API; generated gamma 2.2 AVIF is
      checked against FFmpeg after an explicit zscale transfer conversion.
- [x] Decode CICP SMPTE ST 428-1 (transfer 17) into the existing sRGB
      RGBA8/16 output API; a generated sample is checked against FFmpeg with
      the same explicit transfer conversion.
- [x] Accept CICP IEC 61966-2-4 and BT.1361 extended transfer metadata
      (transfers 11/12) in the display-referred RGBA path; a generated
      BT.1361 sample verifies that metadata no longer fails as `Unsupported`.
- [x] Decode CICP logarithmic and log-sqrt transfer curves (transfers 9/10)
      into the existing sRGB RGBA output; generated FFmpeg samples cover both
      metadata paths.
- [x] Apply ICC matrix-shaper profiles with identity/gamma and lookup-table
      (`curv`) tone curves, with bounded table sizes and interpolation checks.
- [x] Apply ICC matrix-shaper parametric (`para`) tone curves for function
      types 0 through 4, including malformed-coefficient checks.
- [x] Apply bounded ICC `A2B0` `mft1`/`mft2` RGB-to-XYZ LUT profiles with
      trilinear CLUT interpolation, table-bound checks and synthetic coverage
      for both LUT precisions.
- [x] Apply bounded ICC `A2B0` `mAB` RGB-to-XYZ pipelines with A/CLUT/B and
      A/CLUT/M/matrix/B combinations, embedded `curv` curves, PCS-Lab output,
      and malformed offset/curve pairing checks.
- [x] Fall back to ICC `A2B1`/`A2B2` RGB-to-PCS LUT tags when a profile omits
      `A2B0`; synthetic `mft1` coverage keeps alpha and RGB conversion checks.
- [x] Accept ICC `mBA` multi-process LUT tags in the bounded RGB pipeline by
      applying the reverse B/matrix/M/CLUT/A stage order; a synthetic profile
      covers the path and alpha preservation.
- [ ] Add display-specific HDR gamut/tone calibration and the remaining ICC
      display-conversion profile forms beyond the bounded `mBA`/PCS-Lab path.

## 5. Safety, performance and release gate

### Decode benchmark checkpoint (2026-07-18)

The release benchmark uses `AVIF_BENCH_ITERS=11 cargo bench --bench decode`
against `samples/WML2Viewer.avif`. The current optimized run measured
`295.40/292.58 ms` (native/RGBA, 7 iterations on 2026-07-18); the restoration
stage now writes directly into the destination plane instead of cloning the
whole plane for every restoration chunk. A follow-up 11-iteration run on the
same sample measured `288.35/295.41 ms` (native/RGBA), so the ICC LUT path did
not regress the no-profile decode benchmark. The earlier 348.14/346.93 ms run
remains the previous checkpoint for comparison. The deblock stage now indexes block filter
state on an 8-pixel grid instead of linearly scanning every block for each
edge. A post-mAB 11-iteration validation measured `294.64/295.22 ms`
(native/RGBA); this remains within the host's observed scheduling variance and
is not claimed as a speedup. After replacing per-block residual-unit `Vec`
construction with a fixed-state iterator, three follow-up runs measured
`285.84/295.79`, `272.72/275.69` and `270.29/278.15 ms` (native/RGBA).
This is a promising decode-path improvement, but retain the baseline until a
quieter host confirms it. The reconstruction hot path also passes
decoded coefficient slices directly instead of cloning a coefficient `Vec`
for every transform. Rectangular inverse transforms now reuse fixed-size
stack scratch buffers instead of allocating a `Vec` for every row and column;
intra prediction similarly reuses fixed-size edge scratch instead of allocating
two `Vec`s for every non-DC transform block;
compare medians on the same host because this benchmark is sensitive to local
scheduling. Keep the image and plane oracle gates green
when changing this path. Super-resolution uses the scalar 64-phase resize
kernel and allocates the expanded row once per plane; the no-superres path
remains allocation-free.

The qmatrix parser checkpoint was remeasured on the same host with
`361.93/377.38 ms` (native/RGBA, 11 iterations); this is a validation run only,
not a replacement release baseline because local scheduling variance is larger
than the change in this checkpoint.

The 4:2:2 native-rotation validation run measured `287.95/289.71 ms`
(native/RGBA, 7 iterations on 2026-07-18); retain the release baseline above
because this path does not affect the WML2Viewer hot loop and host scheduling
variance remains significant.

The HDR transfer-map validation run measured `298.35/300.26 ms` (native/RGBA,
11 iterations on 2026-07-18); retain the release baseline above because the
PQ/HLG branch is not exercised by `samples/WML2Viewer.avif` and the host has
shown scheduling variance across adjacent runs.

The SDR transfer-curve validation run measured `306.92/313.03 ms` (native/RGBA,
11 iterations on 2026-07-18); the WML2Viewer sample uses transfer 13, so the
new gamma/linear conversion loop is not entered. Retain the release baseline
because this host remains scheduling-sensitive.

The subsequent transfer metadata validation run measured `292.67/297.65 ms`
(native/RGBA, 11 iterations on 2026-07-18); this is a validation checkpoint,
not a claimed speedup because adjacent host runs vary materially.

The deblock plane-lookup optimization validation runs measured `295.23/298.45`
and `294.26/299.74 ms` (native/RGBA, 11 iterations on 2026-07-18). It removes
the per-boundary scan over unrelated planes while preserving the release
baseline until a quieter host produces a stable delta.

The RGBA conversion hot path now iterates row-major and has a no-alpha identity
plane fast path, avoiding per-pixel quotient/remainder and optional alpha work.
Validation runs measured `317.06/315.91` and `310.73/315.96 ms` (native/RGBA,
11 iterations on 2026-07-18); keep this as a verified optimization checkpoint,
not a stable speedup claim, because the host remains scheduling-sensitive.

The full-resolution 8-bit YUV444 path now has a guarded direct-sample/f32
conversion loop for non-alpha YUV matrices. On
`fox.profile1.8bpc.yuv444.avif`, the same-host 5-iteration check moved from
`394.43/430.72 ms` to `384.54/390.38 ms` (native/RGBA); the existing pixel
oracle remained at average error `0.00192`, maximum `5`.
The matrix coefficients are now hoisted out of that inner loop; an 11-iteration
recheck measured `383.43/396.32 ms` on the same host, with the same oracle.

The current five-iteration release-sample recheck measured
`277.93/275.37 ms` (native/RGBA) on 2026-07-18. This confirms the optimized
path remains within the previous scheduling range; it is not treated as a
stable speedup claim without a longer quiet-host run.

A subsequent five-iteration recheck measured `270.24/273.09 ms`
(native/RGBA) on the same host after the GBR and post-filter coverage changes;
this remains a no-regression checkpoint rather than a stable speedup claim.

After the primary-item sequence boundary change, a five-iteration recheck
measured `270.08/275.89 ms` (native/RGBA) on 2026-07-18; this is within the
same host-sensitive range and is recorded as a no-regression checkpoint.

After the ICC mBA pipeline addition, a quiet five-iteration recheck measured
`269.89/271.23 ms` (native/RGBA) on 2026-07-18; this is also recorded as a
no-regression checkpoint.

After reusing a fixed 64x64 prediction scratch buffer for DC, palette and
intra-block-copy reconstruction, the five-iteration recheck measured
`274.52/276.26 ms`; a 15-iteration follow-up measured `293.95/287.20 ms`
(native/RGBA). Host scheduling variance remains larger than the observed delta,
so this is recorded as a no-regression allocation-reduction checkpoint rather
than a stable speedup claim.

After mapping CDEF blocks to subsampled chroma coordinates, the seven-iteration
WML2Viewer recheck measured `303.72/282.87 ms` (native/RGBA). The sample is
4:4:4, so this confirms no regression in the existing hot loop; the new 4:2:0
coverage is validated separately by the generated CDEF oracle.

After routing DC, straight horizontal/vertical, smooth and Paeth intra
prediction through caller-owned output buffers, two seven-iteration rechecks
measured `283.52/284.93` and `273.72/279.85 ms` (native/RGBA). Directional
zone output now also writes directly to the caller-owned buffer; two subsequent
rechecks measured `298.45/315.11` and `346.08/364.89 ms`. These runs remain
allocation-reduction/no-regression checkpoints because host scheduling variance
is larger than a stable speedup claim.

The directional edge construction was then moved to fixed stack scratch,
including corner filtering and upsampled samples. A serial seven-iteration
recheck measured `285.10/291.76 ms` (native/RGBA); this is recorded as a
no-regression/allocation-reduction checkpoint, not a stable speedup claim.

Rectangular inverse-transform row/column intermediates now use a bounded
64x64 stack scratch array instead of allocating an intermediate Vec per
transform. A serial seven-iteration recheck measured `314.53/331.84 ms`
(native/RGBA); host variance makes this a no-regression checkpoint only.

Transform reconstruction now writes the clipped prediction-plus-residual
block into a TileDecoder-owned 64x64 scratch buffer instead of allocating a
second output Vec for every transform block. A serial seven-iteration
recheck measured `300.72/312.80 ms` (native/RGBA); this is recorded as an
allocation-reduction checkpoint, not a stable speedup claim until a quieter
host confirms the delta. The allocating compatibility wrapper and exact
oracle paths remain unchanged.

The dequantized coefficient block now uses a second TileDecoder-owned 64x64
`i32` scratch buffer, covering both ordinary and qmatrix dequantization while
preserving the allocating quantization APIs. The next seven-iteration recheck
measured `283.85/293.76 ms` (native/RGBA); host scheduling remains variable,
so this is recorded as an allocation-reduction checkpoint rather than a
stable speedup claim. Quantization unit tests and the strict plane/RGBA gates
remain green.

The full qmatrix-table checkpoint measured `384.70/381.61 ms` on a subsequent
11-iteration run; retain the earlier release baseline until repeated runs on a
quiet host are available.

After enabling delta-q parsing and block-local quantizer lookup, the same
11-iteration WML2Viewer benchmark measured `269.35/276.46 ms` (native/RGBA).
The sample does not signal delta-q, so this is a no-regression checkpoint rather
than a delta-q speedup claim.

After enabling delta-lf parsing and block-local deblock deltas, the same
11-iteration WML2Viewer benchmark measured `282.64/291.07 ms` (native/RGBA).
The sample does not signal delta-lf, so this is a no-regression checkpoint
rather than a delta-lf speedup claim.

The 8-bit identity-GBR RGBA fast path now bypasses the intermediate RGBA16
buffer when no ICC profile is present. The same 11-iteration benchmark measured
`274.15/279.31 ms` (native/RGBA); the improvement is recorded as a checkpoint
because host scheduling remains variable.

The reconstruction path now appends decoded luma transforms directly into the
block result, removing one intermediate `Vec` allocation per residual unit.
Two 11-iteration WML2Viewer checks measured `290.64/300.47 ms` and
`299.49/302.78 ms` (native/RGBA); retain this as an allocation-reduction
checkpoint until a quieter host confirms a stable speedup.

After carrying segment IDs into the post-filter state for segmentation ALT_LF,
two further 11-iteration checks measured `276.25/281.12 ms` and
`269.85/275.31 ms` (native/RGBA). The WML2Viewer sample does not signal
segmentation ALT_LF, so these remain hot-path validation checkpoints rather
than an ALT_LF speedup claim.

Normal still-image decoding now skips retaining diagnostic luma-block and
transform vectors; the public prefix/diagnostic APIs continue to collect them.
The no-diagnostics path has a regression test asserting empty diagnostic output
while reconstructed planes are populated. Two 11-iteration WML2Viewer checks
measured `318.76/331.27 ms` and `324.77/310.12 ms` (native/RGBA); host variance
is larger than the observed delta, so this is recorded as an allocation-
reduction/no-regression checkpoint rather than a stable speedup claim.

The external FFmpeg conformance suite now includes 4:4:4 non-identity RGBA and
native-plane checks. The current run passes 47 tests with 2 intentionally
ignored samples. AV1 intrabc decoding now searches the block-geometry-specific
neighbor MV candidates; a successful high-complexity external intrabc fixture
is still required before marking the full tool supported.

### Unsupported-syntax audit checkpoint (2026-07-19)

The external compatibility manifest now exercises the formerly labelled
`unsupported/` 8-bit YUV420, monochrome and YUV422 samples as successful
decodes (25 successes, no partial PNGs). The remaining AV1 `Unsupported`
branches were reviewed against the frame/tile parser and grouped as follows:

- `show_existing_frame`, inter-frame reference tools and inherited inter-frame
  film grain still require reference-frame storage and are intentionally
  rejected before reconstruction.
- No-op segmentation signalling, still-image `ALT_Q` deltas and the
  pre-skip `SKIP` feature are parsed; multi-segment map/CDF state is decoded
  for these still-image-safe features, including segment-level loop-filter
  deltas. Segmentation reference and `GLOBALMV` features remain fail-closed.
- Frame syntax tests cover positive, negative and zero-valued `ALT_Q` deltas
  (including signed-value alignment), multi-segment values, pre-skip `SKIP`,
  and rejection of reference/global-motion features.
- `iloc` construction methods beyond 0/1, malformed partial tile groups and
  invalid rectangular-transform combinations remain fail-closed container or
  syntax errors rather than partial images.
- Axis-swapping 4:2:2 geometry is covered for native single-frame rotation;
  grid cases still require a sample with chroma-aligned cell boundaries before
  they can be promoted.

This audit keeps the unsupported boundary explicit while the next expansion
targets segmentation/reference-frame state instead of silently ignoring a
signalled tool.

- [x] Add malformed/truncated coverage for the currently supported container,
      OBU, entropy and metadata paths.
- [ ] Add malformed/truncated cases for each future syntax path as it becomes
      supported.
- [x] Audit frame dimensions, item offsets/extents and frame-buffer allocations
      for overflow and resource limits.
- [ ] Audit post-filter scratch-buffer sizing once normative filters are
      enabled.
- [x] Reject malformed qmatrix levels and other unsupported AV1 tools before
      public decode returns an image.
- [x] Keep public decode fail-closed (`Unsupported`) whenever an unimplemented
      filter is active; qmatrix levels 0 through 14 use normative inverse
      matrices and level 15 remains the flat identity matrix.
- [x] Validate primary-item info/location, `ipma` property indices and
      essential-property flags.
- [x] Resolve `iloc` construction methods 0 and 1, including meta `idat`
      payloads, with missing-source and extent-boundary regression tests.
- [x] Validate `ispe` dimensions against the AV1 frame dimensions.
- [x] Validate SDR `nclx` colour description and range against the AV1
      sequence header, while accepting CICP value `2` as unspecified.
- [x] Gate the AVIF-enabled `avif_decode` integration target with
      `required-features = ["avif"]`.
- [x] Add an explicit feature-off test target and verify the AVIF-disabled
      library with `cargo test -p wml2 --lib --no-default-features`.
- [ ] Gate the remaining parent `wml2` integration targets by their own
      features so `cargo test -p wml2 --no-default-features` compiles. The
      current failures are JPEG/PNG/TIFF/WebP/EXIF tests, not AVIF targets.
- [x] Keep container, OBU, frame-header and entropy fuzz targets.
- [x] Optimise a per-transform coefficient allocation after exact-plane
      conformance passes; retain the public allocating wrapper for compatibility.
- [x] Keep rectangular inverse-transform intermediates in bounded stack scratch;
      retain the allocating public transform wrapper for compatibility.
- [x] Reuse fixed-size intra-prediction edge scratch in the reconstruction hot
      path; retain the public allocating edge-reader wrapper for compatibility.
- [x] Reuse a fixed 64x64 prediction scratch buffer for DC, palette and
      intra-block-copy reconstruction paths while preserving scalar conformance.
- [x] Route DC, straight horizontal/vertical, smooth and Paeth intra prediction
      through caller-owned output buffers while retaining allocating wrappers.
- [x] Route directional zone 1/2/3 output through caller-owned buffers; retain
      fixed stack scratch for edge extension, corner filtering and upsampling;
      allocating edge helpers are test-only.
- [x] Sort references to retained transform boundaries for each deblock pass
      instead of cloning every boundary record twice.
- [x] Reuse a caller-owned 8x8 CDEF output buffer during frame filtering while
      retaining the allocating kernel wrapper for compatibility.
- [x] Replace per-block residual-unit ordering allocations with a fixed-state
      iterator and verify the 64x64 interleaving order against the existing
      unit test and decode benchmark.
- [x] Collect decoded luma blocks directly during recursive partition traversal
      instead of allocating and concatenating a temporary `Vec` per partition;
      two same-condition runs measured `274.56/273.46` and `270.81/273.23 ms`
      (native/RGBA, 7 iterations) on 2026-07-18. Keep the result as a host-
      sensitive checkpoint until a quieter host confirms the delta.
- [x] Reserve luma decoded-transform storage from block/transform geometry to
      avoid growth reallocations in the reconstruction loop.
- [x] Skip diagnostic luma-block/transform retention in normal still-image
      decoding while preserving the public diagnostic prefix APIs and regression
      coverage for both paths.
- [x] Parse AV1 superblock delta-q syntax, carry the tile-local `CurrentQIndex`
      into block quantization and verify a libaom-generated delta-q AVIF sample
      against FFmpeg.
- [x] Implement delta loop-filter block syntax, carry the tile-local four-plane
      delta state into block filter metadata, apply it at deblock boundaries,
      and verify a libaom-generated `delta_lf_present` AVIF sample against
      FFmpeg, including a truncated-input regression.
- [x] Correct the nested crate repository URL to the independently maintained
      `avif-rust` repository and verify it from a fresh clone.
- [ ] Add SIMD/parallel paths only with scalar equivalence and Wasm fallback
      tests.

Required validation after every implementation step:

Run these commands from the parent workspace root (`wml2/`):

```powershell
cargo fmt --all -- --check
cargo test -p avif-rust
cargo check --manifest-path avif/fuzz/Cargo.toml --bins
cargo test -p wml2 --test avif_decode --no-default-features --features avif
cargo check -p wml2 --no-default-features
cargo test -p wml2 --lib --no-default-features
cargo check -p wml2 --target wasm32-unknown-unknown --no-default-features --features avif
$avifPath = (Resolve-Path -LiteralPath avif).Path.Replace('\', '/')
git -c "safe.directory=$avifPath" -C avif diff --check
```

Strict fixture validation additionally requires:

```powershell
$env:AVIF_REQUIRE_ORACLES = '1'
cargo test -p avif-rust --test oracle_fixtures
pwsh -NoProfile -ExecutionPolicy Bypass -File avif/scripts/verify_oracle_sources.ps1
```

Release additionally requires `cargo test --workspace`, exact supported-stream
planes, RGBA8/RGBA16 maximum error 1, native and Wasm checks, AVIF-enabled and
feature-off `wml2` tests, and explicit errors for unsupported tools and
colour-management paths.

The current structural gate is green for the supported feature-on workspace,
the AVIF-disabled `wml2` library, fuzz-bin compilation, AVIF-enabled `wml2`
tests, and the Wasm check. The full feature-off integration run is not yet a
success criterion because unrelated JPEG/PNG/TIFF/WebP/EXIF integration
targets are not gated by their own features. The public 8-bit 4:4:4 WML2Viewer
sample now decodes through
deblock -> CDEF -> loop restoration and passes the FFmpeg/RGBA threshold gate.
The permanent integrated plane oracle and broader format coverage remain
release work.

`decode`, `image_from_bytes`, `decode_frame_bytes` and the `wml2` callback now
accept the implemented 8-bit 4:4:4 filter and film-grain paths. They
remain fail-closed with `Unsupported` for unsupported bit depth, malformed
qmatrix levels and other unavailable AV1 tools. Prefilter diagnostics stay
private or test-only and do not define the public decoded-frame contract.

## Diagnostic history

The local planes produced by the ordinary FFmpeg path are final filtered
output. They remain useful for end-to-end investigation, but must not be
described or registered as a pre-filter oracle. The retained RGB measurements
(`68.77400905349795` before the private filter pipeline and
`68.7115646090535` after it) compare against that final output and are
diagnostic only.

The authoritative raw-reconstruction checkpoint uses
`decoder::prefilter_diagnostic_tests::reports_wml2viewer_against_aom_prefilter_oracle`
with `AOM_PREFILTER_ORACLE` set to planar 8-bit GBR emitted by an AOM build in
which deblock, CDEF and restoration are disabled. The 900x900 fixture now
reports:

- plane 0: `first=None`, `mismatches=0`;
- plane 1: `first=None`, `mismatches=0`;
- plane 2: `first=None`, `mismatches=0`.

The `reports_wml2viewer_raw_against_generated_final_planes` test compares raw
Rust planes with final filtered FFmpeg planes and therefore does not define the
raw acceptance gate. The filter-disabled strict fixtures remain the stable,
reproducible conformance set; generating the diagnostic `WML2Viewer` fixture
does not register it in the strict manifest.

The former first luma difference in the `(632,16)` `Block4x8` was caused by an
upstream dense 4x4 `ADST_ADST` transform. Its AOM prediction, dequant input and
output now match through an explicit square-ADST storage transpose. The former
32x32 DC-only mismatch also matches after routing it through the staged
32-point transform. The following palette-block 8x8 `DctDct` mismatch matches
after a size-specific square storage transpose. The clipped right-edge
`(896,112)` difference was caused by decoding all luma transforms of the
preceding `(768,0)` `Block128x128` before chroma; AV1 interleaves Y/U/V inside
each 64x64 coding unit. A subsequent `(32,128)` difference came from treating
the four-pixel right-edge remainder as a new restoration unit at `x=896`; AV1
rounds the unit count to nearest and merges that remainder. Correcting both
advanced the investigation to `(523,142)`. Exact partial bottom-left padding
then fixed the `Tx16x4 DctAdst` block at `(448,152)`, while two-stage DC-only
rounding fixed the `Tx64x64` block at `(192,192)`. Decode-order transform
hashes agreed through index 2492; the former first differing transform was
index 2493, the `Tx16x64 DctDct` block at `(176,256)`. Its prediction already
agreed with AOM, while a direct cosine-basis 64-point core could not reproduce
AOM's intermediate rounding. The staged DCT64 now fixes that block and
advances the first raster-order luma mismatch from `(176,277)` to `(541,320)`.
The `(541,320)` mismatch was a non-DC `Tx64x64` block still using the old
fixed-basis square path plus untransposed coefficient storage. Reusing the
staged core for both dimensions and transposing lossy Tx64 storage advanced
the next luma investigation to `(560,448)`. The remaining differences were
resolved by using AV1's eight-pixel-aligned MI dimensions, traversing the full
coded edge area, retaining coded padding until reconstruction completes, and
mapping square Tx16/Tx32 plus rectangular Tx64 coefficient storage at the
inverse-transform boundary. Post-filter pixels can now be implemented without
masking a raw-plane difference.
Lossy 4x4 `DctDct` also uses the square transpose, while
coded-lossless 4x4 WHT explicitly retains its original storage so the exact
`filter-disabled-directional` oracle stays green.

The current post-filter diagnostic compares each private stage with FFmpeg's
final `gbrp` output. A dedicated AOM oracle with deblock enabled and
restoration disabled is generated locally. Deblock traversal now consumes the
intra reference delta, chroma-direction levels, block-sized chroma Tx extents,
and AOM's point-local neighboring transform dimensions. The Rust deblock stage
uses the visible frame dimensions for edge eligibility while retaining coded-
plane padding for neighbor reads. It now filters only boundaries belonging to
the current plane and derives point-local dimensions at each 4-pixel lane.
The WML2Viewer coded-plane deblock oracle now reports `mismatches=0` and
`average_abs=0` for planes 0, 1 and 2.

Against the corresponding AOM deblock-plus-CDEF oracle, the Rust CDEF stage
now receives an exact deblock input. A separate coded-plane diagnostic that
replaces only the deblock input with complete AOM coded deblock planes already
reports exact CDEF equality (`0` mismatches, zero average error on planes
0/1/2). The remaining work is a reproducible bootstrap and a permanent
integrated coded-buffer stage oracle.

These are diagnostic checkpoints only; normative deblock/CDEF derivation and
the complete filter-order gate remain unfinished. The current private
pipeline reports a final-filter RGB average absolute error of about
`0.00043` against FFmpeg. The AOM restoration oracle now matches the Rust
restoration stage exactly: all three planes report `mismatches=0` and
`average_abs=0`.
The remaining release work is the reproducible integrated filter-order oracle
and broader format coverage. The public WML2Viewer image/callback gate now
passes for the supported 8-bit 4:4:4 path; no diagnostic generation may change
the strict manifest.
