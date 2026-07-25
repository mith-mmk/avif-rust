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
- [x] Convert all 30 external samples, including the 12-bit, grid, alpha,
      ICC, subsampling, transform and five libavif official variants, with no
      partial PNG output.

The reproducible parent-root gate is:

```powershell
pwsh -File test/avif_external_compat.ps1 -DownloadMissing
```

OneDrive の作業領域で一時フォルダを作れない場合は、`.test*` の作業ルートを
明示できる（例: `-WorkRoot C:\temp\pycache\.test-avif-script`）。

The gate result is 30 successes, 0 expected failures, 0 unexpected results
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
- [x] Add the future `-RegisterInStrictManifest` path to
      `generate_oracles.ps1`; the default `WML2Viewer` output remains
      diagnostic-only, while filter-disabled generation must opt in explicitly.
- [x] Recompute source hashes and compare them with the source manifest using
      `scripts/verify_oracle_sources.ps1`.
- [x] Generate the first filter-disabled 8-bit 4:4:4 identity-GBR fixture.
- [x] Generate exact residual, partition and directional fixtures in that same
      profile.
- [x] Generate the palette fixture in that same profile and assert that the
      bitstream actually contains decoded palette blocks and color maps.
- [x] Generate the `WML2Viewer.avif` native-plane and RGBA fixtures and make
      the fixture a required entry in the strict oracle manifest.
- [x] Make diagnostic fixture generation opt-in for strict registration
      (for example, `generate_oracles.ps1 -RegisterInStrictManifest`), and keep
      the default `WML2Viewer` generation out of `oracles.csv`.
- [x] Add one reproducible recipe/bootstrap for `BlackLossless` and the five
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
- [x] Enable the `WML2Viewer.avif` final oracle after the required filters are
      active and exact; the strict generated plane/RGBA oracle passes with the
      coded-stride/visible-bounds restoration path.

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
- [x] Public alpha auxiliary composition for 8-bit 4:4:4 and 4:2:0 samples,
      including optional native `DecodedFrame.buffers.planes[3]` alpha output
      and RGBA conversion coverage.
- [x] Honour the AVIF `prem` property by unpremultiplying RGB channels at the
      RGBA8/RGBA16 API boundary while preserving the encoded native planes;
      zero-alpha and 1x1 identity GBR vectors cover rounding and channel order.
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
- [x] Keep the original luma source available to chroma film-grain synthesis
      without cloning the full luma plane; generated libaom film-grain output
      is covered against FFmpeg RGBA output.
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
- [x] Compose linear-affine forward ICC `mft1`/`mft2`/`mAB` alternates in the
      Gain Map path when curves and matrix offsets are identity-safe; reject
      non-linear, non-affine, PCS-Lab and reverse-direction profiles.
- [ ] Add display-specific HDR gamut/tone calibration and the remaining ICC
      display-conversion profile forms beyond the bounded `mBA`/PCS-Lab path.
      PQ/HLG RGBA conversion now maps supported P3/BT.2020 primaries through
      linear BT.709 before the existing bounded SDR shoulder; display policy
      and the remaining ICC forms are still open.

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

For the generated 900x900 YUV420 film-grain sample, two 11-iteration checks
after removing the full-luma source clone measured `122.78/138.12 ms` and
`123.03/141.19 ms` (native/RGBA). This is a film-grain-path validation
checkpoint; no stable percentage improvement is claimed without a pre-change
same-host run.

After carrying segment IDs into the post-filter state for segmentation ALT_LF,
two further 11-iteration checks measured `276.25/281.12 ms` and
`269.85/275.31 ms` (native/RGBA). The WML2Viewer sample does not signal
segmentation ALT_LF, so these remain hot-path validation checkpoints rather
than an ALT_LF speedup claim.

CDEF now reuses the retained luma source for the luma plane instead of cloning
that plane a second time; chroma planes still receive independent source
snapshots because each filter writes in place. Two 11-iteration WML2Viewer
checks measured `325.44/338.17 ms` and `313.36/314.13 ms` (native/RGBA), so
this remains an allocation-reduction/no-regression checkpoint under the
host's scheduling variance.

Normal still-image decoding now skips retaining diagnostic luma-block and
transform vectors; the public prefix/diagnostic APIs continue to collect them.
The no-diagnostics path has a regression test asserting empty diagnostic output
while reconstructed planes are populated. Two 11-iteration WML2Viewer checks
measured `318.76/331.27 ms` and `324.77/310.12 ms` (native/RGBA); host variance
is larger than the observed delta, so this is recorded as an allocation-
reduction/no-regression checkpoint rather than a stable speedup claim.

The external FFmpeg conformance suite now includes 4:4:4 non-identity RGBA and
native-plane checks. The current run passes 48 tests with 2 intentionally
ignored samples. AV1 intrabc decoding now searches the block-geometry-specific
neighbor MV candidates; a successful high-complexity external intrabc fixture
is still required before marking the full tool supported.

### Unsupported-syntax audit checkpoint (2026-07-19)

The external compatibility manifest now exercises the formerly labelled
`unsupported/` 8-bit YUV420, monochrome and YUV422 samples plus nine libavif
official samples as successful decodes (34 successes at this audit point, no
partial PNGs). The
remaining AV1 `Unsupported` branches were reviewed against the frame/tile
parser and grouped as follows:

- `show_existing_frame` and the remaining inter-frame reference tools still
  require reference-frame reconstruction; inherited inter-frame film grain is
  now resolved from stored reference metadata and no longer rejected for that
  reason.
- No-op segmentation signalling, still-image `ALT_Q` deltas and the
  pre-skip `SKIP` feature are parsed; multi-segment map/CDF state is decoded
  for these still-image-safe features, including segment-level loop-filter
  deltas. Still-image headers now also consume segmentation reference-frame
  and `GLOBALMV` feature values; actual inter-frame reference state remains
  fail-closed.
- Frame syntax tests cover positive, negative and zero-valued `ALT_Q` deltas
  (including signed-value alignment), multi-segment values, pre-skip `SKIP`,
  and reference/global-motion feature consumption for still headers.
- AV1 inverse-signed literals now use the normative two's-complement
  `bits + 1` representation for frame delta-q and signed segmentation data.
  Intra `TX_MODE_SELECT` also consumes the transmitted transform-size symbol
  for `skip_txfm` blocks, preserving transform-context parity at frame edges.
- `iloc` construction method 2 (`item_offset`) now resolves the indexed
  `iloc` item reference, including explicit extent indexes, recursive-cycle
  and extent-boundary checks. Malformed partial tile groups and invalid
  rectangular-transform combinations remain fail-closed container or syntax
  errors rather than partial images.
- Axis-swapping 4:2:2 geometry is covered for native single-frame rotation;
  grid cases still require a sample with chroma-aligned cell boundaries before
  they can be promoted.

This audit keeps the unsupported boundary explicit while the next expansion
targets actual inter-frame reference storage instead of silently ignoring a
signalled tool.

The reference-state checkpoint now has explicit eight-slot storage and
`show_existing_frame` prefix parsing. Refresh flags replace only the selected
slots, missing slots fail closed, and no public decode path uses the slots yet;
decoded hidden Key/IntraOnly frames are now wired through the limited public
sequence probe; promotion still requires an actual `show_existing_frame`
fixture because inter/MV frames remain fail-closed. AVIS container parsing now
exposes each AV1 track sample independently from `moov` sample tables; the
current FFmpeg sample yields eight independently framed OBU payloads while the
still-image API continues to select the primary item. The next fixture must
exercise reference state across those samples rather than another single-item
still sample.

The AVIS sample-table path is now covered by both a generated eight-frame
sequence and the external `star-8bpc.avifs` five-sample sequence. Each track
sample is parsed as an independent OBU stream, while the public still-image
API continues to decode only the primary item; later inter/MV samples remain
an explicit fail-closed boundary until reference-backed reconstruction lands.

On 2026-07-20, edge-partition CDF restriction stopped cloning a heap `Vec` for
each boundary partition and deblock edge de-duplication now uses sorted
adjacent keys instead of a per-frame `HashSet`. CDEF chroma source storage and
palette cache merging also reuse stack/caller-owned storage. The optimized
decode benchmark remained host-noisy (roughly 300--325 ms native and
295--322 ms RGBA in seven-iteration runs), so this checkpoint is recorded as
allocation reduction/no-regression rather than a stable speedup claim.

On 2026-07-21, the native post-filter path now processes CDEF and loop-
restoration planes concurrently after entropy reconstruction has completed.
Wasm keeps the sequential path. A three-plane restoration regression test
checks that per-plane source snapshots and chroma Wiener-center handling stay
unchanged. Two 11-iteration optimized `WML2Viewer.avif` runs measured
`261.14/259.43 ms` and `261.85/261.06 ms` (native/RGBA), versus the prior
`288.35/295.41 ms` checkpoint; this is recorded as a repeatable local
speedup, not a cross-machine guarantee.

The external unsupported-sample audit now also enumerates every top-level
`.avif`/`.avifs` file in the fixture directory instead of relying only on the
hard-coded 25-sample list. Each discovered file must produce non-empty
dimensions and exactly `width * height * 4` RGBA bytes, so adding a new
unsupported fixture automatically becomes a complete-output regression gate.

The segmentation map reader now follows the normative symbol consumption even
when `last_active_segment == 0`: segment 0 is still decoded from the full
8-symbol CDF before negative deinterleaving. A focused regression test pins the
entropy position and prevents this single-segment edge from desynchronizing
following block syntax.

On 2026-07-20, `alpha_noispe.avif` was promoted from the unsupported boundary
after fixing intra `TX_MODE_SELECT` symbol consumption for skipped blocks and
updating the transform context from the transmitted size. The reduced-still
8-bit YUV444 CDEF/SGRPROJ stream now passes strict entropy validation and
decodes as an 80x80 image with no partial output.

The decode benchmark now accepts `AVIF_BENCH_SAMPLES` as a semicolon-separated
list while retaining `AVIF_BENCH_SAMPLE` for one-off runs. Each sample is
warmed twice and reports independent `decode_frame_bytes` and RGBA medians,
so unsupported-syntax expansion can be measured against more than one image
without changing the decoder API. A fresh 10-iteration WML2Viewer recheck on
the current host measured `327.02 ms` (native) and `347.69 ms` (RGBA); this is
a reference measurement, not a stable cross-run speedup claim.

The expanded `ffmpeg_conformance` fixture set was rerun on 2026-07-20: 49
tests passed, 2 diagnostic tests remained ignored, and no test failed. This
includes the 8-bit 4:2:0/4:2:2/4:4:4, monochrome, 10-bit, 12-bit, alpha,
sequence, grid, transform and colour-management samples. `alpha_noispe` is
now part of the supported external conversion gate.

On 2026-07-20, normal reconstruction stopped allocating a transform-geometry
`Vec` for every plane block and now traverses the same clipped geometry through
an allocation-free iterator; the public diagnostic planning API remains
`Vec`-based. The iterator is covered at full-frame and clipped-edge cases. An
11-iteration multi-sample recheck measured `377.63/389.04 ms` (native/RGBA) for
`WML2Viewer.avif` and `4.12/4.38 ms` for the 128x128 alpha sample. These are
recorded as allocation-reduction/no-regression measurements until a quieter
same-host baseline confirms a stable speedup.

The post-fix release benchmark was rerun with the multi-sample harness at
three iterations: `WML2Viewer.avif` measured `306.37/305.48 ms` (native/RGBA)
and the 128x128 alpha sample measured `2.93/3.23 ms`. The short run is kept as
a fresh reference point only; it is not treated as a stable speedup claim.

The AVIS sequence path now inspects a shared Sequence Header and a later
sample as separate OBU parts instead of allocating a concatenated payload for
every Key/IntraOnly/show-existing sample. The OBU helper is covered by a
split-input regression test. A five-iteration release run measured
`star-8bpc.avifs` at `3.91/4.34 ms` (native/RGBA); this is a same-host AVIS
checkpoint, not a cross-machine percentage claim.

On 2026-07-20, CDEF source handling now swaps each plane's decoded samples into
caller-owned scratch storage and reuses the previous output allocation for the
next plane. This removes the per-plane source `clone` while preserving the
unfiltered samples before writing filtered blocks. A five-iteration optimized
`WML2Viewer.avif` recheck measured `317.32/331.61 ms` (native/RGBA); host noise
still makes this an allocation-reduction/no-regression checkpoint rather than a
stable speedup claim.

The generated conformance set now also covers a two-stream 10-bit AVIF with a
10-bit grayscale alpha item. Native Y/U/V/alpha planes are compared against
FFmpeg's 10-bit outputs, and the public RGBA path checks the down-converted
alpha channel. This closes a previously unexercised high-bit-depth auxiliary
item combination while keeping the existing external alpha fixtures intact.

The generated conformance set now includes a 128x128 libaom AV1 IntrABC
sample with CDEF and restoration disabled. Native Y/U/V planes match FFmpeg's
`yuv444p` output exactly, and the public RGBA path completes successfully.

On 2026-07-22, fractional inter prediction now precomputes the fixed-point
source coordinates once per block and reuses the bilinear sampler across rows
and columns. This removes per-pixel division and closure construction from the
hot path while preserving the existing scalar output. The path is bounded by
the AV1 128-pixel maximum block dimension and fails closed on coordinate
overflow. Reconstruction tests, the public inter-frame oracle, and the full
unsupported-sample gate all pass; an 11-iteration release recheck measured
`2.83/3.19 ms` for `star-8bpc.avifs` and `186.09/192.54 ms` for
`WML2Viewer.avif` (native/RGBA); a second same-host run measured
`2.89/3.32 ms` and `178.66/175.31 ms`, respectively. These are same-host
reference measurements, not a cross-machine speedup claim.

This proves the currently supported low-complexity IntrABC syntax path while
the high-complexity reference/transform cases remain an explicit follow-up.

On 2026-07-22, inter-frame film grain inheritance was promoted from the
unsupported boundary. Reference slots now retain the decoded grain parameters;
an inter frame with `update_parameters=0` reuses the selected slot and applies
the current frame's random seed. A generated libaom AVIS fixture exercises a
Key followed by Inter samples with `film-grain-test=1`, and all generated
frames decode at complete 64x64 dimensions. Missing or non-grain reference
slots remain explicit `Unsupported` errors.

The post-change 11-iteration release benchmark remains in the same range:
`2.88/3.19 ms` for `star-8bpc.avifs` and `185.89/187.32 ms` for
`WML2Viewer.avif` (native/RGBA). This is a same-host reference checkpoint,
not a cross-machine percentage claim.

The conformance set now also generates a 256x256 YUV444 IntrABC sample. Its
native planes match FFmpeg exactly and the RGBA path completes, extending the
IntrABC regression beyond a single 128x128 coding unit while keeping the
high-complexity reference/transform cases explicitly tracked.

Native AVIS batch decoding now uses the existing worker pool only when the
sequence's total pixel work reaches 256K pixels; small sequences stay on the
caller thread to avoid thread startup/join overhead. A 15-iteration optimized
run of a generated four-frame 64x64 sequence measured `0.134 ms` native and
`0.190 ms` RGBA; this is a small-sequence checkpoint, not a stable percentage
claim.

On 2026-07-20, AVIS samples gained a header-level classifier for Key, Inter,
IntraOnly, Switch and show-existing frames. The generated and external sequence
fixtures now assert a Key first sample followed by real Inter samples, keeping
the unsupported inter/MV boundary observable instead of treating later track
payloads as opaque. Reference slots now share one decoded frame through `Arc`
when multiple refresh flags target the same frame, avoiding up to eight full
frame clones while reference-backed reconstruction is being expanded.

The generated AVIS audit now includes a 64x64, 60-frame libaom sequence with
lagged alternate references. Its 60 track samples include both Inter frame
OBUs and `show_existing_frame` headers, and the test asserts those kinds while
still decoding the primary Key sample through the public still-image API.

`decode_sequence_frame_bytes` now exposes an indexed AVIS sample path for
Key/IntraOnly and `show_existing_frame` samples. It prepends the shared
sequence header when later track samples omit it, refreshes only the signalled
reference slots, and returns an explicit `Unsupported` for Inter/Switch
samples. A generated four-frame all-Key AVIS fixture decodes every index and
checks identical native planes; the external `star-8bpc.avifs` fixture asserts
the Inter boundary without accepting partial output.

`decode_sequence_frames_bytes` now exposes the same supported AVIS boundary as
one batch, preserving reference-slot order and rejecting an Inter/Switch sample
without returning earlier frames. The generated all-Key fixture checks the
four-frame batch result, and the external Inter fixture pins the fail-closed
batch behavior.

The callback `decode` path now emits supported multi-frame AVIS sequences after
the complete batch has decoded: `init` reports `animation: true`, followed by
one `draw` call per Key/IntraOnly/show-existing sample and a final `terminate`.
Inter/Switch sequences still fail before callback initialization, so callers
cannot observe a partial animation. The generated four-frame fixture records
the callback dimensions, animation flag, frame count and identical RGBA bytes.

AVIS batch/index decoding now classifies the requested sample range before
reconstructing any frame. A later Inter/Switch sample therefore exits without
spending decode work on earlier Key/IntraOnly samples, while indexed requests
still stop at their requested frame. The existing external fail-closed test
continues to assert that no partial output is returned.

The progressive `draw_points_idat_progressive.avif` sample now has a pixel
oracle as well: FFmpeg rejects its progressive `idat` layout, so the test uses
ImageMagick's AVIF decoder and records average RGB error `0.2222`, maximum `1`.
This closes the previous dimensions-only audit gap without weakening the
existing FFmpeg gates.

Loop restoration now also swaps a reusable output buffer across enabled planes
instead of allocating a fresh `source.clone()` for each plane. A seven-iteration
optimized `WML2Viewer.avif` recheck measured `302.50/306.33 ms` (native/RGBA),
which is retained as an allocation-reduction/no-regression measurement rather
than a stable speedup claim.

A seven-iteration two-sample optimized recheck measured
`304.41/294.45 ms` (native/RGBA) for the workspace `samples/WML2Viewer.avif`
and `299.80/299.15 ms` for the nested `test_data` copy. These remain host-
specific reference points, not a stable cross-run speedup claim.

After the indexed AVIS path, a seven-iteration release benchmark on the
workspace `WML2Viewer.avif` measured `303.14/305.33 ms` (native/RGBA). This is
within the existing host variance and records a no-regression checkpoint; the
reference-slot API does not add work to the still-image hot path.

The CDEF stage now skips the directional filter for blocks whose effective
primary and secondary strengths are both zero, since the source block has
already been copied to the output. A seven-iteration release recheck measured
`302.06/300.98 ms` (native/RGBA); this remains a host-specific no-regression
checkpoint rather than a stable speedup claim.

The CDEF fast path now also returns before source-scratch setup when every
active CDEF index has zero primary and secondary strengths. The follow-up
seven-iteration release recheck measured `290.11/282.26 ms` (native/RGBA);
this is recorded as another host-specific measurement, not a stable speedup
claim.

The 8-bit RGBA conversion path now reuses the bounded f32 YUV coefficients
for subsampled YUV420/YUV422 as well as direct YUV444, while retaining the
normative chroma sample lookup. A focused 4:2:0 test stays within one 16-bit
code value of the scalar path. A seven-iteration release recheck measured
`330.04/346.55 ms` for the external YUV420 sample,
`367.79/384.69 ms` for YUV422, and `296.36/299.07 ms` for WML2Viewer
(native/RGBA); host variance prevents a stable end-to-end speedup claim.
A five-iteration release recheck after the AVIS callback work measured
`302.80/296.62 ms` for `samples/WML2Viewer.avif`; this remains a no-regression
reference point because the callback branch is inactive for still samples.

The same bounded f32 YUV coefficients now cover the RGBA16 path for 10/12-bit
YUV420/YUV422/YUV444 samples, while non-YUV matrices retain the scalar path.
The high-bit-depth coefficient check stays within two 16-bit code values of
the scalar conversion. Five-iteration release checkpoints measured
`368.54/402.55 ms` for the external 10-bit YUV444 sample and
`409.36/412.21 ms` for the 12-bit YUV444 sample; these are host-specific
no-regression references, not a stable speedup claim.
A generated 64x64 10-bit YUV420 and YUV422 AVIF now exercise the subsampled
high-bit path against FFmpeg as well: YUV420 average/max RGB error `1.4538/38`,
YUV422 `1.1789/10`.
The f32 conversion loop now receives precomputed f32 range constants instead
of converting the f64 range fields for every pixel. A five-iteration recheck
measured `340.57/358.29 ms` for `WML2Viewer.avif` and `417.75/437.84 ms` for
the external 10-bit YUV444 sample; these are host-specific checkpoints only.

High-bit-depth SDR `to_rgba8` now writes directly to an 8-bit output buffer
for monochrome, identity GBR and YUV matrices, avoiding the intermediate
RGBA16 allocation. The 10-bit and 12-bit external tests compare this path to
the scalar RGBA16 conversion with a maximum difference of one code value.
A follow-up five-iteration run measured `407.33/429.17 ms` (10-bit) and
`399.25/425.33 ms` (12-bit), retained as local no-regression checkpoints.

The 8-bit direct YUV444 conversion now keeps a full-resolution auxiliary alpha
plane on the same allocation-free loop instead of falling back to per-pixel
subsampling lookups. A focused RGBA16 equivalence test and the external
128x128 alpha oracle both pass (`alpha_max=0`). A five-iteration WML2Viewer
recheck measured `305.51/306.03 ms` (native/RGBA); this remains a local
no-regression checkpoint because the sample itself has no auxiliary alpha.

The same IntrABC fixture is now generated and checked in 4:2:0 as well as
4:4:4. The 4:2:0 luma plane is exact and chroma remains within the existing
oracle tolerance (maximum error 4), covering the subsampled block geometry
without widening the unsupported high-complexity boundary.

The IntrABC fixture also covers 4:2:2. Its luma plane is exact and the
subsampled chroma planes remain within maximum error 2 against FFmpeg, closing
the three supported chroma layouts for the current low-complexity IntrABC
path.

The generated IntrABC fixture now also enables CDEF while keeping restoration
disabled. The 4:4:4 native planes remain exact against FFmpeg, confirming that
the CDEF signalling path does not regress the supported IntrABC decode boundary.

The remaining official samples that had still lived under `unsupported/` are
now covered by a direct public RGBA gate: both grid `dimg` ordering variants,
all three `draw_points`/`idat` forms, extended `pixi`, non-essential `clap` +
`irot` + `imir`, `clop` + `irot` + `imir`, and the non-rotated alpha sample.
`draw_points_idat_progressive` is decode/dimension-only because the installed
FFmpeg build does not implement its progressive input form. `extended_pixi`
uses a native-plane oracle because its explicit vertical chroma position is
upsampled differently by the installed FFmpeg/ImageMagick RGB paths; its AV1
planes and metadata dimensions are exact. The official sample set now reports
native/public success without partial output; the remaining RGB difference is
kept as an explicit oracle limitation until a chroma-siting-compatible RGB
reference is available.

The unsupported-directory audit is now a permanent 25-sample gate. Every
official sample in `test/images/external/avif/unsupported` must produce a
non-partial RGBA buffer with the expected post-container dimensions, including
the sequence primary item, alpha, grid, crop/mirror/rotate, 10/12-bit and
progressive-IDAT samples. The gate completed in 55.7 seconds on the current
host; its cost is intentional because it exercises the full public decode path
instead of only checking headers.

On 2026-07-20, coefficient decoding began returning its owned level buffer to
the decoder scratch after reconstruction; only the public diagnostic luma
capture retains it. The scratch-transfer regression test confirms that two
consecutive coefficient decodes reuse the same allocation. A fresh 10-iteration
single-sample benchmark measured `319.37/324.12 ms` (native/RGBA) for
`WML2Viewer.avif`, versus the earlier `327.02/347.69 ms` reference. This is a
local checkpoint; cross-run speedup claims still require a quieter repeated
baseline.

Zero-coefficient transforms now bypass dequantization, inverse-transform
dispatch, and temporary residual allocation by copying the prediction directly
into the reconstructed block. The existing zero-transform regression covers
this path. A fresh optimized 10-iteration run measured `316.69/313.85 ms`
(native/RGBA) for `WML2Viewer.avif`; this remains a same-host checkpoint rather
than a cross-run speedup claim.

The 4x4 and 8x8 inverse-transform paths now write into the tile decoder's
reusable residual scratch instead of allocating a `Vec` per non-zero block.
The allocating and scratch-backed paths share an equivalence test. A fresh
optimized 10-iteration run measured `308.45/308.11 ms` (native/RGBA) for
`WML2Viewer.avif`; this is recorded as a local performance checkpoint pending
another same-host baseline.

The reusable residual path now includes 16x16 transforms, eliminating the
remaining small-transform temporary buffer. The 16x16 output is covered by
the same allocating-vs-scratch equivalence test. A follow-up optimized
10-iteration run measured `306.09/307.88 ms` (native/RGBA) for
`WML2Viewer.avif`; this remains a local checkpoint under host scheduling
variance.

The reusable residual path now also covers the staged 32x32 and 64x64 DCT
dispatches; sparse 32/64 output is checked against the allocating path. A
follow-up optimized 10-iteration run measured `305.46/303.99 ms`
(native/RGBA) for `WML2Viewer.avif`. The result is retained as a local
checkpoint, not as a cross-machine speedup claim.

Rectangular transform dispatches now also write directly into the reusable
residual scratch; the equivalence test covers all 19 square and rectangular
transform sizes. A follow-up optimized 10-iteration run measured
`311.86/313.61 ms` (native/RGBA) for `WML2Viewer.avif`, so this change is
recorded as an allocation-reduction/no-regression checkpoint until a quieter
same-host baseline confirms a stable speedup.

- [x] Add malformed/truncated coverage for the currently supported container,
      OBU, entropy and metadata paths.
- [ ] Add malformed/truncated cases for each future syntax path as it becomes
      supported.
- [x] Audit frame dimensions, item offsets/extents and frame-buffer allocations
      for overflow and resource limits.
- [ ] Audit post-filter scratch-buffer sizing once normative filters are
      enabled. (The Wiener and SGRPROJ 64x64 paths are now exercised with
      their full halo scratch requirements; remaining work is an end-to-end
      large-frame allocation audit.)
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
- [x] Gate the remaining parent `wml2` integration targets by their own
      features so `cargo test -p wml2 --no-default-features --no-run`
      compiles and `cargo test -p wml2 --lib --no-default-features` passes.
      Codec-specific doctests remain intentionally tied to the default feature
      set.
- [x] Keep container, OBU, frame-header and entropy fuzz targets.
- [x] Optimise a per-transform coefficient allocation after exact-plane
      conformance passes; retain the public allocating wrapper for compatibility.
- [x] Return decoded coefficient storage to the TileDecoder scratch between
      ordinary transforms while retaining owned vectors for diagnostic luma
      capture; cover the transfer with a pointer-reuse regression test.
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
- [x] Add direct 8-bit SDR YUV/monochrome-to-RGBA8 paths for 4:2:0, 4:2:2
      and 4:4:4 images, avoiding the intermediate RGBA16 allocation while
      preserving the existing high-precision, HDR and ICC paths. The 3x3
      subsampled and monochrome conversions are regression-tested byte-for-byte
      against the RGBA16 path; two 7-iteration FFmpeg-sample checks measured
      `371.92/414.43 ms` and `422.41/451.56 ms` (native/RGBA), so this remains
      an allocation-reduction/no-regression checkpoint under host scheduling
      variance. A monochrome 7-iteration check measured `74.56/78.10 ms`
      (native/RGBA) on the same host.
- [x] Add a native scoped-thread row path for large RGBA8/RGBA16 and identity
      conversions, retaining the sequential path below the 256K-pixel
      threshold and on Wasm. The 640x512 row-partition test compares the
      parallel-capable path with a scalar reference; a generated 10-bit 4:2:0
      sample now exercises the high-bit subsampled conversion after the change.
      The 2026-07-20 11-iteration release check measured `321.29/308.04 ms`
      (native/RGBA) for `WML2Viewer.avif` and `342.29/366.87 ms` for the
      external YUV420 sample; retain these as host-specific no-regression
      checkpoints, not stable speedup claims.
- [x] Decode independent native grid cells through a scoped-thread path when
      the aggregate cell area is at least 256K pixels, preserving source order
      and retaining sequential decoding for small grids and Wasm. The public
      1x5 grid FFmpeg pixel and native-plane oracles pass after this change;
      a 7-iteration release check measured `14.92/57.23 ms` (native/RGBA) for
      `sofa_grid1x5_420.avif`; retain this as a host-specific checkpoint rather
      than a stable speedup claim.
- [x] Share the bounded grid-cell worker path with the public RGBA grid decoder
      and alpha-grid composition, capping native work at eight workers while
      preserving cell order and sequential Wasm fallback. The same 1x5 grid
      pixel/native-plane and alpha-grid composition oracles pass; a later
      7-iteration check measured `14.82/17.20 ms` (native/RGBA), retained as a
      host-specific no-regression checkpoint rather than a stable speedup claim.

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

On 2026-07-20, `alpha_noispe.avif` gained a permanent FFmpeg pixel oracle:
the 80x80 RGB comparison reports average absolute error `0.0016` and maximum
error `2`. Chroma reconstruction also avoids the former four-sample blend for
the integer-position 4:2:0/4:2:2/4:4:0 cases while preserving the normative
4:2:0 vertical half-sample rounding; focused tests cover all edge coordinates
and the 4:2:2 mapping. A five-iteration YUV420 release recheck measured
`353.13/339.75 ms` (native/RGBA), so this is recorded as an allocation/load
reduction and no-regression checkpoint rather than a stable speedup claim.

On 2026-07-20, normal transform reconstruction now reuses the non-zero
coefficient count returned by the reconstruction scratch path when recording
post-filter boundaries, avoiding a second full-transform scan. The focused
reconstruction suite passed, and an 11-iteration release recheck measured
`307.32/301.79 ms` (native/RGBA) for `WML2Viewer.avif` and
`345.94/358.04 ms` for the external YUV420 sample. These are host-specific
allocation/scan-reduction checkpoints, not a stable speedup claim.

The generated conformance set now includes an 8-bit 4:4:4 YCgCo (`matrix_coefficients=8`)
sample. FFmpeg's RGBA conversion is unavailable for this matrix in the current
toolchain, so the test uses an independent native `yuv444p` oracle for all
three planes. The RGBA8/RGBA16 YCgCo path now uses a bounded f32 lifting
transform, with a scalar-equivalence unit vector; this removes the previous
fallback to the slower f64 matrix path while keeping the native plane gate.

AVIS track parsing now accepts repeated, byte-identical `av01` sample
descriptions in `stsd`/`stsc` instead of rejecting every sample-description
index other than 1. Differing descriptions are accepted when the changed
sample carries its own Sequence Header, while changed samples without one
remain fail-closed; focused container tests cover the safety gate.

AVIS batch decode now parallelizes sequences made entirely of Key/IntraOnly
samples on native hosts with an eight-worker cap while preserving sample order.
show-existing and Inter/Switch samples keep the reference-dependent sequential
path, and Wasm remains sequential. Generated all-key callback and batch tests
cover the optimized path. Indexed Key/IntraOnly requests now decode only the
requested sample instead of replaying earlier independent samples.

The official `abc_color_irot_alpha_NOirot.avif` fixture now also compares the
auxiliary alpha plane after the primary image's quarter-turn transform. This
keeps alpha orientation coverage separate from the existing RGB and rotated
alpha checks.

AVIS sample decoding now builds a minimal per-sample `AvifInfo` containing only
the payload, dimensions, color metadata, alpha premultiplication flag, and
applicable `av1C` bytes. This avoids cloning the parent image's complete track
and auxiliary metadata for every parallel or indexed sample without changing
the public decode behavior.

AVIS classification now retains the per-sample frame kind and Sequence Header
presence from one OBU scan. The batch/indexed decode paths reuse that metadata
when composing sample payloads, avoiding a second parse of each sample. A
seven-iteration optimized run on a generated 16-frame 1024x1024 all-Key AVIS
sample measured `153.06 ms` native frame decode and `166.45 ms` RGBA decode;
this is a host-specific AVIS checkpoint and not a cross-machine speedup claim.

The AVIS reference regression now covers a batch containing a decoded Key frame
followed by a `show_existing_frame` sample, verifying that the supported
reference-slot path returns both frames without partial output.

Common AVIS Sequence Header OBU bytes are now encoded once per sequence and
reused when composing samples that omit their own header. A follow-up
seven-iteration optimized run on the same generated 16-frame 1024x1024 sample
measured `151.03 ms` native and `165.27 ms` RGBA (previous checkpoint:
`153.06/166.45 ms`); host noise is significant, so this remains a local
allocation/reuse checkpoint rather than a stable percentage claim.

The `avio` AVIF brand is now accepted in `ftyp` validation, covering the
intra-only image-item profile used by newer AVIF writers. A regression test
mutates the checked-in still-image sample to an `avio` major brand and verifies
metadata parsing plus full-frame decode; `avis` remains the sequence brand that
activates track-sample extraction.

AVIF 1.2 Sample Transform (`sato`) support now decodes the official
`weld_sato_12B_8B_q0.avif` shape: postfix constants, input references, unary
operators, and binary arithmetic/bitwise operators are validated with a
fail-closed stack parser, and the still-frame API returns the transformed
16-bit native planes. The sample is an optional external fixture so CI stays
reproducible without committing a binary; set `AVIF_SATO_SAMPLE` to run the
oracle locally. A three-iteration optimized checkpoint measured
`485.32 ms` native / `500.93 ms` RGBA for the 1024x684 sato sample and
`309.34/309.18 ms` for `WML2Viewer.avif`; these are host-specific decode
checkpoints, not stable cross-machine speedup claims. Inter/Switch reference
reconstruction remains explicitly unsupported.

The sato evaluator now reuses one postfix stack allocation across all output
samples and accepts the file-declared `dimg` input order instead of requiring
the primary item to be first. A five-iteration optimized recheck of the
official sample measured `396.10 ms` native / `445.70 ms` RGBA on this host;
the result is a local allocation-reuse checkpoint rather than a stable
cross-machine speedup claim.

The skip reconstruction branch now consumes its transform iterator directly
after the duplicate boundary scan was removed, avoiding the remaining
iterator clone on that hot path. A seven-iteration release recheck measured
`158.64/161.48 ms` (native/RGBA) for `WML2Viewer.avif`; this remains a
same-host reference point.

Sato-derived color output now also carries the primary item's auxiliary alpha
plane (including an alpha grid) when present, preserving the existing
native-plane and RGBA alpha contract instead of rejecting the derived color
path outright.

The container preflight now accepts a `sato` item as the primary item (not
only as an alternate derived item). The optional official-fixture regression
rewrites `pitm` to the Sample Transform item and verifies native 16-bit and
complete RGBA decode through the public APIs.

The sato evaluator now accepts the 64-bit intermediate expression width. Signed
64-bit big-endian constants are parsed and evaluated with an i128 stack, while
each unary/binary result is clamped to the declared intermediate range before
the final output clamp. Parser and saturation vectors cover the newly promoted
width; no public AVIF API shape changes.

The optional external conformance set now also audits seven official Gain Map
fixtures (including unsupported `tmap` metadata versions). Each file must
still produce a complete base RGBA image with exact dimensions; HDR gain-map
tone mapping remains intentionally outside the current RGBA API and is
tracked separately from AV1 decode support.

The `prem` item property is now honoured at the public RGBA boundary. RGB
channels are unpremultiplied with bounded integer rounding for RGBA8/RGBA16,
zero-alpha pixels are cleared, and native decoded planes remain unchanged.

ICC matrix-shaper, LUT, and mAB/mBA profile application now reuse the native
RGBA16 row-chunk scheduler used by AV1 colour conversion. Large ICC-bearing
frames therefore avoid a single-threaded per-pixel post-processing pass while
Wasm remains sequential; a 1,048,593-pixel equivalence vector covers the
parallel chunk boundary and preserves alpha/output ordering.

Layered-image item properties `a1op`, `lsel`, and `a1lx` are now parsed with
strict payload and reserved-bit validation. The existing still-image policy
accepts only the default operating point/layer (`a1op=0`, `lsel=0`); non-default
selectors return explicit `Unsupported` instead of being silently ignored,
while `a1lx` indexing metadata is retained for future layer-range decode.

Tone-map (`tmap`) primary items now resolve their first referenced base `av01`
item for complete SDR/native decode, preserving the fallback-image contract.
The public `parse_gain_map_metadata` API now parses the ISO 21496 descriptor,
including common/per-field rationals, multichannel flags, and strict version,
denominator, range, and gamma validation. Gain-map application and alternate
HDR rendering remain intentionally un-applied; missing or malformed base
references fail closed rather than passing the `tmap` payload into the AV1
decoder.

A synthetic `meta`/`iinf`/`iloc`/`idat` fixture now exercises the public parser
through a real `tmap` item payload, in addition to the descriptor-level vectors.

The coded-frame allocation path now builds only the aligned plane layouts it
needs instead of cloning the complete frame decode plan (including tile
vectors) for every frame. The crop stage also returns without copying when
the coded layout already matches the visible layout; a pointer-preservation
regression test covers that fast path while the existing padded-edge test
continues to cover the copying path.

Alpha auxiliary composition now uses the native row-chunk scheduler for large
RGBA frames (up to eight workers), while Wasm and small images retain the
sequential path to avoid thread overhead. A subsampled alpha regression vector
verifies that row chunking preserves the one-pass output and alpha coordinates.

Sequence alpha composition now borrows the decoded auxiliary frame while
cloning only its single plane for each output frame. This removes a full
`DecodedFrame` clone per animation sample without changing the public alpha
plane contract; the existing alpha and sequence integration tests cover the
regression path.

A five-iteration optimized recheck of `test_data/images/WML2Viewer.avif`
measured `240.02 ms` native frame decode and `240.79 ms` RGBA decode on this
host. This is a local baseline for the allocation changes, not a
cross-machine speedup claim.

RGBA16 identity-plane conversion and PQ/HLG transfer application now share the
same native row-chunk scheduler; Wasm remains sequential. A 640x512 transfer
equivalence vector covers chunk boundaries. A three-iteration multi-sample
checkpoint measured `244.83/245.21 ms` for `WML2Viewer.avif`,
`396.91/407.49 ms` for the external 12-bit YUV444 sample, and
`3.18/3.67 ms` for the 128x128 alpha sample (native/RGBA); these are local
measurements, not cross-machine speedup claims.

The official `extended_pixi.avif` sample now parses the AVIF extended `pixi`
channel descriptors instead of silently discarding the flagged payload. The
decoder exposes unsigned-integer channel format and per-channel subsampling
type/location metadata, rejects reserved channel fields, and keeps the 4x4
YUV420 sample's complete RGBA decode as a regression gate. Descriptor-level
vectors cover truncation and unsupported channel/component combinations.

The same five-iteration release benchmark after the parser change measured
`241.88 ms` native / `250.11 ms` RGBA for `WML2Viewer.avif`; the 4x4 extended
`pixi` sample measured `0.0187/0.0194 ms`. This confirms that the metadata
extension does not add a measurable cost to the normal large-frame decode;
the numbers remain host-specific checkpoints rather than a cross-machine
speedup claim.

The YUV422 RGBA8/RGBA16 conversion path now has a dedicated row traversal for
full-resolution luma plus horizontally subsampled chroma, including odd-width
frames and auxiliary alpha. The new 10-bit alpha vector asserts RGBA8 output
against the RGBA16 conversion, and the 4:2:2 external conformance samples
remain green. A seven-iteration release benchmark measured
`53.7754/56.5244 ms` native/RGBA for
`fox.profile2.8bpc.yuv422.avif`; this is a same-host checkpoint rather than a
portable speedup claim. The checklist work-item rate remains `157/171`
(`91.81%`) because this closes a hot-path optimization gap, not a normative
filter item.

The external conformance audit now recursively covers every sample under the
official `supported/`, `unsupported/`, and `gainmap/` directories instead of
only a hand-maintained subset. The dynamic gate passes all three classes with
complete RGBA output; the animated auxiliary variant additionally verifies
that the alpha plane is retained on all five frames and audio/depth variants
do not synthesize one. This expands regression discovery without changing the
explicit malformed/unsupported fail-closed cases. The checklist rate remains
`157/171` (`91.81%`).

On 2026-07-24, the entropy hot path was measured again after the fixed
probability `read_bool` shortcut. Marking `read_raw_bit` and `renormalize` as
always-inline reduced two 100-iteration release medians for
`samples/WML2Viewer.avif` to `98.6355/100.1637` and `101.0170/101.4118 ms`
(native/RGBA), compared with the clean no-inline run at `108.3963/109.7268
ms`. The small `star-8bpc.avifs` sample remained at `1.3396/1.6800 ms`, so
the change is retained as a hot-path optimization with no small-sample
regression.

The unsupported-sample regression gate now recursively enumerates every
`.avif`/`.avifs` file below `test/images/external/avif/unsupported`, including
nested fixture directories. Optional external samples and FFmpeg oracles now
continue to the remaining cases when unavailable instead of aborting the whole
matrix. The complete-output and dimension checks remain unchanged. A fresh
five-iteration benchmark measured `243.25/241.89 ms` for `WML2Viewer.avif`
and `0.0102/0.0105 ms` for `extended_pixi.avif` (native/RGBA); these are local
checkpoints and show no regression from the audit-only changes.

The gain-map boundary now exposes `decode_gain_map_frame_bytes`, which locates
the second `dimg` input of a `tmap` item, validates that it is an AV1 item with
matching dimensions and required `ispe`/`pixi`/`av1C` metadata, and decodes its
native frame alongside `GainMapMetadata`. `DecodedFrame::to_rgba16_with_gain_map`
now applies the ISO 21496 log2 gain formula for the base colour space, keeps
alpha unchanged, and returns the base image through an exact fast path when the
selected headroom is zero. Alternate-colour-space composition remains
fail-closed unless the decoded base and gain-map colour configurations are
identical, in which case no colour-space conversion is required. The optional
gain-map conformance test now checks the decoded item
whenever the external fixture set is available, and synthetic vectors cover
headroom, gain, alpha, equivalent alternate colour space, and rejection
behavior.

Gain-map composition now converts its per-channel rational coefficients once
before walking the RGBA pixels, avoiding repeated metadata division in the
post-decode loop. A fresh five-iteration `WML2Viewer.avif` release benchmark
measured `248.24/243.45 ms` (native/RGBA); this path does not change ordinary
decode output and remains a host-specific checkpoint.

The common 8-bit sample-to-RGBA16 conversion now uses exact byte replication
instead of a per-sample divide for luma, chroma, and auxiliary alpha values.
The existing official unsupported-sample gate and a focused scaling vector
cover the path. A fresh five-iteration `WML2Viewer.avif` release benchmark
measured `242.95/244.90 ms` (native/RGBA), retained as a host-specific
checkpoint rather than a cross-machine speedup claim.

The generated high-bit-depth subsampling set now covers 12-bit YUV420 and
YUV422 in addition to the 10-bit variants. Both native bit-depth assertions
and FFmpeg RGBA oracles pass; the 12-bit samples measured average/max RGB
errors `1.4553/38` (420) and `1.2152/10` (422). This closes the previously
untested 12-bit subsampled decode boundary.

The same generated matrix now includes a 12-bit monochrome (`gray12le`) AVIF;
native bit depth and public grayscale RGBA output pass the FFmpeg oracle with
average/max RGB error `0.2839/1`. This covers the high-bit-depth
single-plane path in addition to the 4:2:0 and 4:2:2 cases.

An attempted generated 4:4:0 (`yuv440p`) matrix was rejected as a false
coverage signal: the available libaom/FFmpeg encoder normalizes that request
to `yuv444p` before writing AVIF. No 4:4:0 fixture is promoted until a source
that preserves `subsampling_x=0, subsampling_y=1` is available.

The normal reconstruction path no longer walks each transform geometry once
just to pre-register a zero-valued filter boundary and then walks it again to
decode the block. Skip, all-zero, and non-zero transform paths now register
their boundary at the point where they are processed, preserving post-filter
state while removing the duplicate iterator/filter pass. The 11-iteration
release benchmark measured `167.49/168.37 ms` (native/RGBA) for
`WML2Viewer.avif`; this is a same-host optimization checkpoint, not a
cross-machine speedup claim.

Monochrome AVIFs carrying a valid GRAY/XYZ ICC profile now apply the profile's
media white point and tone curve before converting luma to sRGB, while
preserving alpha. Matrix-shaper and GRAY profiles from the ICC `spac` colour
space class are accepted alongside display/output classes. Synthetic profile
vectors plus a real external ICC AVIF class-mutation sample cover the transform;
malformed profiles remain fail-closed. Inter/Switch motion-vector frames are
still explicitly unsupported.

A fresh five-iteration release benchmark measured `170.24 ms` native decode and
`174.90 ms` RGBA conversion for `WML2Viewer.avif`; this is a same-host
checkpoint and the GRAY ICC branch is bypassed for ordinary RGB samples.

After the ICC device-class expansion, a second five-iteration release run
measured `168.30/169.74 ms` (native/RGBA) for the same sample. This remains a
same-host no-regression checkpoint rather than a cross-machine speedup claim.

The 8-bit RGBA conversion path now maps native 8-bit alpha and monochrome
samples directly to `u8`, avoiding an intermediate 16-bit scale/divide while
preserving the exact existing output. Two five-iteration checks measured
`2.01/2.13 ms` and `1.92/2.11 ms` (native/RGBA) for the external alpha sample,
and `56.40/56.31 ms` and `56.02/56.66 ms` for the external monochrome sample;
these are same-host checkpoints, not cross-machine speedup claims.

ICC matrix-shaper and GRAY profiles from the input-device `scnr` class are now
accepted alongside display/output/colour-space classes because they use the
same validated RGB/XYZ or GRAY/XYZ transform contracts. The real embedded ICC
sample is mutated to both `spac` and `scnr` in the external regression gate and
must produce byte-identical RGBA output. High-bit-depth identity GBR RGBA8
conversion now uses the native row-chunk scheduler; its channel-order and
alpha mapping are pinned by a focused 10-bit regression vector.

High-bit-depth identity GBR RGBA8 conversion now precomputes the exact
sample-to-byte mapping for 10/12-bit sources, removing three per-pixel integer
divides while retaining the previous rounded output for every table entry. A
generated 512x512 10-bit identity-GBR sample measured `20.11/20.57 ms` over
five iterations (native/RGBA) after the table path; this is a same-host
checkpoint, not a cross-machine speedup claim. The same generated sample is now part of
the FFmpeg RGBA oracle suite with native 10-bit and identity-matrix assertions.

SATO inputs may now be `grid` derived image items in addition to direct `av01`
items. Grid cells are composed through the existing native grid decoder before
the sample-transform expression runs; malformed or dimension-inconsistent
inputs remain fail-closed. This covers the AVIF 1.2 grid-input form without
weakening the same-dimensions requirement for sample-transform operands.

Inter/Switch frame headers now consume the AV1 reference-index signalling,
`frame_size_with_refs` probe, high-precision-MV, interpolation, motion-mode,
reference-frame-MV, `reference_select`, skip-mode and warped-motion flags
before reaching the existing fail-closed decode boundary. The decoder now
retains eight reference slots with decoded-frame geometry and resolves
reference-derived dimensions against those slots. Global-motion signalling
and reference-backed motion reconstruction remain the next implementation
step; no Inter/Switch frame is emitted partially.

The inter-frame path now retains primary reference-frame IDs and decoded motion
vectors on the MI grid. Single-reference NEWMV/NEARMV and compound modes consume
the normative DRL index symbols with candidate-count bounds, then use the
nearest same-reference neighbours as predictors. The generated libaom inter
sequence test now decodes its actual second sample and checks complete native
plane dimensions. The external `star-8bpc.avifs` inter oracle improved from an
average RGB error of about `52` to `39` on this host; strict entropy validation
still remains fail-closed while motion-mode/compound predictor parity is
completed. A five-iteration release benchmark measured `2.74 ms` native and
`3.15 ms` RGBA for the 159x159 AVIS sample.

Compound inter reconstruction now applies the independently decoded secondary
motion vector instead of reusing the primary vector, and both primary and
secondary reference/MV grids are retained for later candidate prediction.
Intra blocks clear stale inter candidates from the MI grid. A focused dual-MV
prediction vector and the existing external inter oracle cover this path. A
five-iteration release benchmark measured `2.74 ms` native and `3.13 ms` RGBA
for the same 159x159 sample; strict entropy validation and warped/global-motion
parity remain future work.

On 2026-07-22, sequence-level AV1 frame-ID signalling was promoted from the
unsupported header boundary. The parser now retains the normative
`delta_frame_id_length`/`frame_id_length` values, consumes `current_frame_id`
before inter-frame sizing, and carries the ID through reference-slot state.
Synthetic sequence-header and current-ID vectors cover enabled and disabled
signalling; referenced slots are now rejected when their frame ID falls
outside the normative age window. Previous-frame monotonicity checks remain a
follow-up once the broader inter reference lifecycle is complete. A
seven-iteration release benchmark
after the header change measured `169.95/174.24 ms` for `WML2Viewer.avif` and
`2.6765/3.0943 ms` for `star-8bpc.avifs` (native/RGBA), a same-host checkpoint.

On 2026-07-22, the warped-motion header boundary was hardened to consume the
full AV1 global-motion model syntax (identity, translation, rotzoom and affine
matrix parameters, including signed reference subexp values). Identity models
are retained on `FrameHeader`; non-identity models still fail closed until
reference-backed warped reconstruction is wired into block prediction. Two
synthetic parser vectors cover the all-identity and truncated-model cases.
The 35-sample external compatibility gate remains green with zero partial PNGs.

On 2026-07-22, non-identity global motion was connected to inter-block
prediction. Reference slots now retain the prior frame's global matrices so
delta-coded models can be resolved; GLOBALMV and GLOBAL_GLOBALMV select the
translation/rotzoom/affine motion vector at the block centre, including AV1
high-precision and integer-MV rounding. Translation and affine unit vectors
cover the matrix-to-MV path. Global-motion parameters are consumed for every
inter frame, including frames whose `allow_warped_motion` flag is false, while
the existing inter oracle remains green.

The post-filter state merge now moves the first tile's vectors directly when
the destination is empty, avoiding duplicate origin scans and allocations on
the common single-tile path. A same-host seven-iteration release benchmark
measured `88.20/86.88 ms` (native/RGBA) for `WML2Viewer.avif` and
`1.30/1.62 ms` for `star-8bpc.avifs`; these are local checkpoints, not
cross-machine speedup claims.

On 2026-07-22, fractional inter prediction now uses the AV1 regular 8-tap
sub-pel kernel in both the in-bounds and edge-extension paths instead of the
previous bilinear approximation. A focused fractional-motion vector pins the
filter output, and the generated two-frame inter fixture now compares its
second frame against FFmpeg (average RGB error `42.44` on this host). The
external inter oracle remains a quality gate at average error `64` while
compound weighting, switchable filter selection, and warped/OBMC prediction
are completed.

The inter predictor now retains the frame's fixed interpolation filter and
the per-block switchable/dual-filter symbols. Regular, smooth, sharp and
bilinear kernels are selected on the horizontal/vertical axes instead of
being consumed and discarded. Filter-symbol mapping has a focused unit test;
the 401-test lib suite and external unsupported gate remain green. A fresh
five-iteration release benchmark measured `92.56/87.17 ms` for
`WML2Viewer.avif` and `1.24/1.55 ms` for `star-8bpc.avifs` (native/RGBA), a
same-host checkpoint.

Compound inter blocks now retain reference order hints from the eight-slot
state. When `compound_idx=0` selects distance-weighted blending, the primary
prediction weight is derived with AV1 order-hint wraparound; average blending
remains the explicit `compound_idx=1` path. A focused reconstruction vector
covers weighted and average output, while the inter oracle and complete
unsupported-sample gate remain green. Wedge/diff-weighted masked compound and
warped/OBMC modes remain the next unsupported boundary.

On 2026-07-23, the all-bilinear fractional inter path now uses a dedicated
2x2 two-tap reconstruction fast path, avoiding the generic eight-tap
intermediate buffer while preserving the AV1 bilinear rounding rule. A focused
4x4 ramp vector covers half-pel output. The five-iteration release benchmark
measured `83.40/83.60 ms` for `WML2Viewer.avif` and `1.37/1.72 ms` for
`star-8bpc.avifs` (native/RGBA), a same-host checkpoint.

On 2026-07-23, OBMC now leaves blocks without an eligible same-reference
causal neighbour on their ordinary inter predictor instead of blending a
synthetic frame-edge sample. Neighbour interpolation filters are retained in
the MI grid, chroma 4x4/8x4/4x8 above blending follows AOM's skip rule, and
only the intersecting overlap strip is predicted (avoiding a full-block
temporary for each neighbour). The generated 128x128 OBMC oracle improved
from average RGB error `35.63` to `29.65` (max `255`); the seven-iteration
`WML2Viewer.avif` benchmark measured `82.42/85.18 ms` (native/RGBA), a
no-regression host checkpoint.

On 2026-07-23, the generated OBMC regression matrix added a second 4:2:0
sample with libaom's switchable interpolation filter enabled. Both ordinary
and dual-filter OBMC samples decode successfully and remain within the
existing FFmpeg oracle threshold (`29.66` average RGB absolute error,
`255` max), covering the neighbour-filter metadata path without changing
the baseline output.

On 2026-07-24, the generated LOCALWARP regression matrix added a native
4:4:4 YUV sample in addition to the existing 4:2:0 fixture. Both complete
128x128 inter frames pass the FFmpeg oracle (`33.18` average RGB error for
4:2:0 and `38.01` for 4:4:4; max `255`), exercising the warp path without
subsampling as well as the existing chroma-plane geometry.

On 2026-07-23, AVIS sequence decoding now honors `primary_ref_frame ==
PRIMARY_REF_NONE` by starting the frame from the default CDF context instead of
carrying the previous sample's adapted CDFs. The generated two-frame Inter
sample and all 407 library tests plus 81 FFmpeg conformance tests remain green.
Strict tile entropy validation is still intentionally diagnostic-only for
Inter/Switch samples until motion/reconstruction parity closes the remaining
bit-consumption boundary.

On 2026-07-23, 4:2:0 YUV-to-RGBA conversion now has a dedicated no-alpha fast
path that reuses each chroma row pair and preserves the AV1 unknown/vertical
versus colocated sample-position rules. The external
`fox.profile0.8bpc.yuv420.avif` benchmark improved from `51.06/57.05 ms` to
`49.76/51.32 ms` (native/RGBA, seven iterations); `WML2Viewer.avif` remained
within host variance at `85.52/90.26 ms`.

On 2026-07-23, CDEF plane application now skips the source clone and filter
thread when the selected plane has no configured primary or secondary strength.
The parser's AV1 secondary-strength value `3 -> 4` mapping is covered by a
focused unit test. The 4:2:0 CDEF oracle now reports native plane error per
plane (Y average/max `0/0`, U `0.0374/2`, V `0.0212/3`); the remaining RGB
average/max `1.2174/179` is therefore a color-conversion/chroma-position
boundary rather than a native CDEF mismatch.

On 2026-07-23, CDEF direction search now uses the normative large-value
sentinel for partial frame-edge 8x8 blocks instead of clamping missing samples
to the last visible pixel. A focused 5x5 edge vector covers the distinction;
the 405-test lib suite, 4:4:4/4:2:0 CDEF oracles and unsupported-sample audit
remain green. A five-iteration benchmark measured `82.8311/85.2262 ms` for
`WML2Viewer.avif` and `48.7963/51.5426 ms` for the external 4:2:0 sample
(native/RGBA), recorded as a same-host no-regression checkpoint. The remaining
4:2:0 CDEF RGB gap is average `1.2174`, maximum `179`.

On 2026-07-23, AVIS reference refresh now returns before allocating metadata
and cloning planes when a frame signals no refresh flags. A focused reference
slot regression covers the no-op path; the 409-test lib suite and dynamic
unsupported-sample audit remain green. An 11-iteration `star-8bpc.avifs`
recheck measured `1.3492/1.6836 ms` (native/RGBA), which is recorded as a
small-sequence no-regression checkpoint because the fixture refreshes most
frames and does not isolate the fast path.

On 2026-07-23, OBMC now collects causal top/left neighbors that use the same
reference frame from the decoded motion grid and predicts their overlap before
applying the dimension-specific AOM masks. Blocks without a usable same-reference
neighbor retain the boundary-only fallback. The generated 128x128 OBMC sample
remains complete and matches the FFmpeg oracle threshold (average RGB absolute
error `35.64599609375`, max `255`). The 404-test lib suite and 83-test FFmpeg
conformance suite remain green. A five-iteration release benchmark measured
`83.9505/84.7796 ms` for `WML2Viewer.avif` (native/RGBA); different-reference
neighbor parity and a dedicated fixture that forces the new neighbor path remain
follow-ups.

On 2026-07-23, AVIS inter/switch reconstruction now runs the normal deblock,
CDEF and restoration pipeline before refreshing reference slots and before
returning the decoded inter frame. This keeps motion compensation on the
post-filtered reference image instead of the raw reconstruction buffer. The
generated 64x64 inter fixture tightened its RGB oracle from average error `64`
to `48` and measured `42.561767578125`; the external five-sample sequence
remains a broad `64` average-error gate because it contains compound and warped
blocks. Entropy validation remains a separate follow-up for inter samples whose
current CDF/trailing-bit state is not yet stable. A five-iteration release
benchmark measured `81.1587/85.3834 ms` for `WML2Viewer.avif` and
`1.3135/1.6506 ms` for `star-8bpc.avifs` (native/RGBA); this is a same-host
no-regression checkpoint because the still-image hot path is unchanged.

On 2026-07-23, inter-intra prediction stopped allocating a temporary `Vec`
for every transform block and now uses a TileDecoder-owned 64x64 scratch
buffer. The existing inter-intra oracle remains green, and a rotated
global-motion AVIS fixture now exercises a non-trivial transformed reference
sample at complete 256x256 dimensions (average RGB error `31.98`, max `255`).
The optimization is recorded as an allocation-reduction checkpoint; the
remaining motion-mode oracle work still includes strict OBMC/warped parity.
A ten-iteration release recheck measured `91.29/94.79 ms` for
`WML2Viewer.avif` (native/RGBA), which remains within host variance rather
than a stable end-to-end speedup claim.

On 2026-07-23, CDEF and loop-restoration output snapshots now clone the
existing source buffer directly instead of zero-initializing and then copying
the same samples. This removes the redundant zero-fill pass while preserving
the source snapshot required by filter taps. The focused post-filter tests and
403-test lib suite remain green; a 30-iteration release recheck measured
`92.44/96.20 ms` for `WML2Viewer.avif` (native/RGBA), so this is recorded as
an allocation-reduction/no-regression checkpoint rather than a stable speedup
claim under the current host variance.

On 2026-07-23, the OBMC approximation now uses the normative AOM overlap
length (`min(block_dimension, 64) / 2`) and the official 1/2/4/8/16/32/64
sample mask tables instead of a linear blend. A generated libaom OBMC AVIS
fixture decodes at complete 128x128 dimensions with average RGB error `35.63`
(max `255`); the mask table has focused vectors, while neighbor-prediction
parity remains an explicit follow-up. A ten-iteration release recheck of the
ordinary `WML2Viewer.avif` sample measured `82.92/84.37 ms` (native/RGBA),
which is retained as a host-specific reference rather than a stable speedup
claim because OBMC is inactive for that still image.
The generated 128x128 OBMC sequence measured `0.79/1.02 ms` (native/RGBA)
over ten iterations; this small-sample result is also host-specific.

On 2026-07-23, the generated sample matrix gained a moving-crop AVIS with
libaom global motion enabled. The second Inter sample decodes at complete
128x128 dimensions and matches the FFmpeg RGBA oracle with average RGB error
`35.72` (max `255`); this keeps global-motion/reference-backed reconstruction
in the regression loop while strict entropy and broader motion-mode parity
remain explicit follow-up boundaries.

On 2026-07-23, LOCALWARP now has the AV1 warped-motion 193-phase filter bank
and two-pass horizontal/vertical reconstruction path, including shear setup,
signed model reduction, bit-depth rounding, and edge clamping. The generated
libaom LOCALWARP oracle remains diagnostic while block-center alignment is
refined: this path measured average RGB absolute error `40.74` and max `255`
(the prior regular sub-pel path measured `40.75`). The filter-bank unit vector,
403-test lib suite, and 80-test FFmpeg conformance suite are green. A
ten-iteration release benchmark measured `82.88/84.37 ms` for
`WML2Viewer.avif` (native/RGBA); the next precision/speed boundary is further
reuse of block-local intermediate rows and broader LOCALWARP fixtures.

On 2026-07-23, compound-mask geometry now uses the complete block-relative
coordinates for both integer and fractional inter prediction, including
inter-intra wedge blending; this avoids restarting the A64 mask at each
transform tile. LOCALWARP diagnostics also retain the causal above-left
neighbor and reconstruction averages the available vertical deltas while
using the above-right horizontal delta. The 400-test lib suite and 77-test
FFmpeg conformance suite remain green. The five-iteration release benchmark
measured `83.85/89.07 ms` for `WML2Viewer.avif` and `1.28/1.64 ms` for
`star-8bpc.avifs` (native/RGBA), a same-host checkpoint. Full least-squares
LOCALWARP reconstruction remains the next unsupported prediction boundary.

On 2026-07-23, LOCALWARP now follows the AV1 causal sample search (up to eight
deduplicated above/left/top-left/top-right candidates), retains candidate
block geometry, and derives the local affine model through the normative 2x2
least-squares equations and warp-parameter clamps. A generated libaom sample
with `enable-warped-motion=1` is now an FFmpeg oracle test; it measured average
RGB absolute error `40.75` and max `255`. The 402-test lib suite and 78-test
FFmpeg conformance suite remain green. The ten-iteration release benchmark
measured `88.73/89.34 ms` for `WML2Viewer.avif` and `1.30/1.74 ms` for
`star-8bpc.avifs` (native/RGBA), a same-host checkpoint. The AV1-specific
warped 2D filter kernel remains the next precision boundary; the affine model
currently projects through the existing regular sub-pel sampler.

On 2026-07-23, the sequence inter-intra flag and normative inter-intra CDFs
are retained. Inter-intra mode and optional wedge syntax are consumed and
blended with DC/vertical/horizontal/smooth intra prediction; the generated
libaom inter-intra sequence is now an FFmpeg RGB oracle test. The 400-test lib
suite and 77-test FFmpeg conformance suite remain green. The five-iteration
release benchmark measured `85.25/87.19 ms` for `WML2Viewer.avif` and
`1.35/1.76 ms` for `star-8bpc.avifs` (native/RGBA), a same-host checkpoint.
LOCALWARP still uses the observable translation fallback pending local warp
model reconstruction.

On 2026-07-23, LOCALWARP blocks now retain causal above/left/above-right MV
neighbors and apply a bounded affine MV tilt through the existing regular
sub-pel sampler. This is an incremental local-warp reconstruction path; the
full AV1 least-squares sample selection remains a follow-up. The 400-test lib
suite and 77-test FFmpeg conformance suite remain green. A five-iteration
release benchmark measured `83.72/91.74 ms` for `WML2Viewer.avif` and
`1.31/1.65 ms` for `star-8bpc.avifs` (native/RGBA), a same-host checkpoint.

On 2026-07-23, switchable motion-mode symbols are now retained in block
diagnostics instead of being discarded. OBMC blocks apply the decoded
prediction through the reconstructed top/left overlap edges, with a focused
edge-only reconstruction vector; LOCALWARP remains observable and uses the
existing translation fallback until local least-squares warp reconstruction
is implemented. A generated wedge-enabled libaom sequence is now an FFmpeg
RGB oracle test. The 400-test lib suite and 76-test FFmpeg conformance suite
remain green. The five-iteration release benchmark measured `89.31/86.40 ms`
for `WML2Viewer.avif` and `1.38/1.68 ms` for `star-8bpc.avifs`
(native/RGBA), a same-host checkpoint.

On 2026-07-23, wedge compound prediction now reconstructs the AV1 master
mask from the dimension-specific HGTW/HLTW/HEQW codebooks, direction tables,
and sign-flip rules. The luma 8–32 pixel block shapes use the generated A64
mask during prediction; unsupported shapes continue to fail closed to the
existing average path. A focused 8×8 mask vector covers orientation and
weighted output, and the 404-test lib suite plus 75-test FFmpeg conformance
suite remain green. Warped motion and OBMC remain the next unsupported
prediction boundary.

The post-wedge five-iteration release benchmark measured `87.23/89.47 ms`
for `WML2Viewer.avif` and `1.35/1.70 ms` for `star-8bpc.avifs`
(native/RGBA), a same-host checkpoint.

On 2026-07-23, masked compound syntax now consumes the AV1 compound-type
group with the normative compound-type and wedge-index CDFs. Difference-weighted
compound builds the A64 luma mask (`38 + abs(pred0-pred1)/16`), including the
inverse mask flag and high-bit-depth scaling; wedge syntax is retained for the
next mask-table reconstruction step. A generated libaom AVIS sample with
distance weighting and wedge disabled now decodes completely and stays within
the existing inter oracle threshold. The five-iteration release benchmark
measured `84.99/86.32 ms` for `WML2Viewer.avif` and `1.36/1.67 ms` for
`star-8bpc.avifs` (native/RGBA), a same-host checkpoint.

On 2026-07-24, AVIS block filter metadata now retains whether a block is inter,
which reference frame it uses, and whether its motion vector is non-zero.
Deblocking uses these fields to select the AV1 reference and motion-mode loop
filter deltas instead of applying the intra-frame delta to every block. The
generated inter sample remains within the existing FFmpeg gate (RGB average
error `42.5653`, max `255`), affine global-motion coverage remains green
(`34.5424` average), and the optimized five-iteration WML2Viewer checkpoint
measured `89.0557/89.6886 ms` (native/RGBA).

On 2026-07-24, AVIS `show_existing_frame` dispatch now reuses the frame-slot
index captured during sequence-sample classification instead of reparsing the
sample OBU stream during reconstruction. The indexed and batch show-existing
tests, generated all-key AVIS sequence gate, and focused library suite remain
green; the full frame-header parser stays fail-closed because sequence
dispatch resolves this prefix before coded-frame parsing.

On 2026-07-24, the entropy regression suite now pins the cumulative-CDF update
direction and the count/rate transition with an exact three-symbol vector. The
generated two-frame Inter sample remains green, while strict tile trailing-bit
validation stays diagnostic-only for Inter/Switch samples until the remaining
motion/CDF bit-consumption boundary is resolved.

On 2026-07-24, filter-intra prediction now writes directly into the tile
decoder's reusable prediction scratch instead of allocating and copying a
temporary block `Vec`. A generated 256x256 YUV444 libaom sample with
filter-intra enabled exercises modes 0/1/2/3 and passes the native-plane
FFmpeg oracle (maximum error `2`). A seven-iteration release checkpoint for
that sample measured `5.4131/6.0750 ms` (native/RGBA); the ordinary
`WML2Viewer.avif` recheck measured `83.4133/82.4604 ms`, so this remains a
local allocation-reduction/no-regression result rather than a cross-machine
speedup claim.

On 2026-07-24, fractional inter prediction now clips the regular and bilinear
filter outputs to the declared AV1 bit-depth range instead of the full `u16`
range. A focused high-contrast vector catches the 8-bit regular-kernel
overshoot, and both generated LOCALWARP samples assert that every decoded
plane stays within its declared range. The 416-test lib suite, 86-test FFmpeg
conformance suite, 14-test container suite, and WML2 AVIF integration tests
remain green. A three-iteration release recheck measured
`88.0658/84.9364 ms` for `WML2Viewer.avif` (native/RGBA); this is a correctness
fix with no stable end-to-end speedup claim.

On 2026-07-24, loop-restoration Wiener and SGRPROJ kernels now accept
caller-owned scratch buffers. The decoder reuses those buffers per plane across
restoration stripes, removing repeated intermediate `Vec` allocations while
preserving the allocating wrappers and their regression vectors. The full
416-test library suite, 86-test FFmpeg conformance suite, 14-test container
suite, and WML2 AVIF feature-on/off tests remain green. An 11-iteration release
recheck measured `86.0879/87.8933 ms` (native/RGBA) for `WML2Viewer.avif` on
this host; this is recorded as allocation reduction and a local checkpoint,
not a cross-machine speedup claim.

On 2026-07-24, the generated AVIS matrix now includes a 10-bit YUV420 Inter
sample. The test checks Inter classification, complete 64x64 native planes,
the declared 10-bit range, and an FFmpeg RGBA oracle (average RGB error
`44.85`, max `255`). This extends reference-dependent coverage beyond the
existing 8-bit Inter sample; strict Inter/Switch entropy trailing-bit
validation remains diagnostic-only until the motion/CDF bit-consumption
boundary is resolved.

The same generated Inter matrix now also covers a 12-bit YUV420 sample, with
complete native planes and declared-range checks plus an FFmpeg RGBA oracle
(average RGB error `41.69`, max `255`). A 12-bit LOCALWARP sample likewise
passes the complete-frame gate and RGBA oracle (average RGB error `31.54`,
max `255`), extending the motion-compensation coverage through the highest
currently supported AV1 bit depth.

The strict Inter entropy probe was also narrowed to a reproducible boundary:
the generated 8-bit Inter tile reaches its final decoded block (the last block
is an 8x32 OBMC block), then the optional trailing-bit validator rejects the
range-decoder position before any partial frame is returned. This is a
bit-consumption/termination discrepancy, not an unhandled block syntax branch;
strict Inter/Switch acceptance therefore remains disabled until the entropy
position is reconciled with the motion/CDF traversal. The diagnostic probe and
all temporary logging are kept out of the normal decode path.

The post-checkpoint release bench for `samples/WML2Viewer.avif` (10
iterations) measured `84.2075 ms` for native `decode_frame_bytes` and
`90.1377 ms` through RGBA conversion on this host. This is a local reference
for the existing allocation/reuse work, not a cross-machine speedup claim.

On 2026-07-24, the generated filter-intra oracle was generalized to compare
native planes at the declared bit depth, and now covers both the existing
8-bit YUV444 sample and a generated 10-bit YUV444 sample. Both pass the
FFmpeg raw-plane gate; the new 10-bit sample also asserts the native frame
bit depth and 10-bit sample range.

On 2026-07-24, the fixed-probability AV1 `read_bool` path now uses the raw
midpoint-bit decoder directly instead of rebuilding and updating a temporary
two-symbol CDF for every boolean. Entropy tests and the generated filter-intra
gates remain green. A 100-iteration release checkpoint for
`samples/WML2Viewer.avif` measured `107.2613/107.5944 ms` (native/RGBA),
versus the prior `108.6493/109.2108 ms` run on this host; retain this as a
local checkpoint because system noise prevents treating it as a portable
speedup claim.

On 2026-07-24, filter-intra coverage now includes generated 10-bit lossy and
12-bit lossless YUV444 samples. The 12-bit fixture uses libaom lossless
encoding so the native Y/U/V oracle is exact rather than conflating chroma
quantisation error with decoder behavior; all three planes match exactly.
The 10-bit fixture keeps the bounded lossy oracle (average error <=2 and
maximum error <=16 per plane).

On 2026-07-24, rectangular transforms with a 32-point stage now support the
AV1 Identity stage for `Tx8x32`, `Tx16x32`, `Tx32x8`, and `Tx32x16`. The
implementation uses the AOM `identity32` scale (`input * 4`) and keeps ADST
rectangular stages fail-closed. Fixed 32-point vectors plus allocating/in-place
dispatch parity tests cover the new path; the full library and FFmpeg
conformance suites remain green.

On 2026-07-24, reduced inter transform-set 3 now reads the AV1 two-symbol
Identity/DCT CDF for luma blocks whose square-up transform dimension is 32
(`Tx8x32`, `Tx16x32`, `Tx32x8`, `Tx32x16`, and `Tx32x32`). The CDF rows follow
the AV1 square-size contexts, while chroma continues to derive the luma type
without consuming a second symbol. A generated libaom AVIS with
`enable-flip-idtx=1`, 128x128 inter content, and loop filters disabled now
decodes all samples through the public sequence API; the full library and
FFmpeg conformance suites remain green. Inter set 1/2, flipADST, and strict
entropy/oracle registration remain open follow-up items.

On 2026-07-24, Inter transform-set 1 and set 2 now use the AV1 default CDF
tables and complete symbol maps, including the FLIPADST variants. The inverse
transform dispatcher and coefficient-context selection cover vertical and
horizontal ADST/FLIPADST forms; 32-point non-DCT stages remain fail-closed.
The generated 128x128 libaom Inter fixture explicitly enables the full
transform set and decodes completely. The 12-bit Inter and LOCALWARP fixtures
pass the FFmpeg RGBA oracle with average RGB errors `38.51` and `32.54`
respectively (maximum `255`). The full library suite is green (`418 passed`,
`5 ignored`) and FFmpeg conformance is green (`90 passed`, `2 ignored`).

The 7-iteration release decode checkpoint for `samples/WML2Viewer.avif`
measured `86.6327/85.2226 ms` for native/RGBA on this host. This is a local
reference for the transform and allocation work, not a cross-machine speedup
claim. Strict entropy/oracle registration, post-filter coverage, HDR edge
cases, and non-DCT transforms with 32-point stages remain follow-up items.

On 2026-07-24, CDEF and loop-restoration now avoid spawning one worker per
plane for small frames; the existing plane-parallel path is retained once the
combined plane sample count reaches `128 * 1024`. A focused threshold test,
the full library suite (`419 passed`, `5 ignored`), and FFmpeg conformance
(`90 passed`, `2 ignored`) remain green. The seven-iteration release bench
measured `86.2827/88.56 ms` (native/RGBA) for `samples/WML2Viewer.avif`; this
is a local scheduling checkpoint, not a portable speedup claim.

On 2026-07-24, LOCALWARP prediction now reuses a caller-owned `15 * 8` i64
intermediate scratch buffer across the 8x8 warped blocks in a prediction
pass. This removes the repeated stack initialization from the hot loop while
preserving the allocating reference output; a dedicated reuse-parity unit
test and the three generated LOCALWARP FFmpeg gates remain green. The full
library suite is green (`420 passed`, `5 ignored`) and FFmpeg conformance is
green (`90 passed`, `2 ignored`). The seven-iteration release bench for
`samples/WML2Viewer.avif` measured `84.6383/83.8039 ms` (native/RGBA) on this
host; this is a local allocation checkpoint, not a portable speedup claim.

On 2026-07-24, gain-map composition now supports an alternate image encoded
with a different supported CICP RGB primary set. Linear RGB conversion uses
the same AV1 primary chromaticity table as the derived-matrix path, converts
the base into the alternate math space before applying the log2 gain, and
converts the result back to the base space. BT.709/BT.2020 matrix round-trip
and alternate-space composition tests are covered; ICC-backed alternate
conversion remains fail-closed. The full library suite is green (`421 passed`,
`5 ignored`) and FFmpeg conformance is green (`90 passed`, `2 ignored`). The
seven-iteration release bench for `samples/WML2Viewer.avif` measured
`80.5375/83.1505 ms` (native/RGBA) on this host; this is a local checkpoint,
not a portable speedup claim.

On 2026-07-24, ISO 21496 gain-map items with dimensions different from the
base image are now accepted. Composition resamples the decoded map to the
base RGBA16 dimensions with a bounded bilinear pass; constant-map parity and
the existing alternate-colour-space path are covered by unit tests. The
external gain-map matrix now also includes the official big-map fixture when
present, and validates composed output dimensions (the current workspace has
no external gain-map directory, so that optional audit is skipped here). The
full library suite is green (`422 passed`, `5 ignored`) and FFmpeg conformance
is green (`90 passed`, `2 ignored`). The seven-iteration release bench for
`samples/WML2Viewer.avif` measured `81.3502/88.7201 ms` (native/RGBA) on this
host; this is a local checkpoint, not a portable speedup claim.

On 2026-07-24, ISO 21496 gain-map items carried by an AVIF `grid` item are now
parsed through the existing grid compositor before gain-map resampling and
composition. The external audit covers the official grid gain-map fixtures
`color_nogrid_alpha_nogrid_gainmap_grid.avif`,
`color_grid_alpha_grid_gainmap_nogrid.avif`, and
`color_grid_gainmap_different_grid.avif` when they are available (the current
workspace has no external gain-map directory, so this optional audit skips).
The full library suite is green (`422 passed`, `5 ignored`) and FFmpeg
conformance is green (`91 passed`, `2 ignored`). The seven-iteration release
bench for `samples/WML2Viewer.avif` measured `86.1501/82.3831 ms`
(native/RGBA) on this host; this is a local checkpoint, not a portable speedup
claim.

On 2026-07-24, `tmap` gain-map metadata now follows the ISO 21496-1 layout:
the explicit metadata version byte is consumed, channel/headroom rationals use
per-field denominators, and the legacy draft common-denominator/backward flags
are no longer inferred from the reserved flag bits. Supported writer version 0
rejects non-zero trailing bytes, while newer writer versions may carry extra
metadata; unsupported version/minimum fields remain fail-closed at the
gain-map API while ordinary base decode is preserved. The official libavif
grid fixtures and writer-version boundary fixtures are covered through the
`.test*` sample directory, with complete base decode and composed grid checks.
The full library suite is green (`422 passed`, `5 ignored`) and FFmpeg
conformance is green (`91 passed`, `2 ignored`). The seven-iteration release
bench measured `81.1717/82.9439 ms` (native/RGBA) for
`samples/WML2Viewer.avif`; this is a local checkpoint, not a portable speedup
claim.

On 2026-07-24, official libavif grid fixtures now cover two previously
unsupported container cases: edge cells whose coded dimensions exceed the
declared output rectangle, and per-cell alpha auxiliaries (including the
reverse `auxl` association direction). A shared grid tile that decodes as
monochrome is normalized to the color grid's plane configuration before native
composition, while RGBA composition clips the coded edge cells to the visible
80x80 output. The external conformance test compares both fixtures with the
FFmpeg RGBA oracle and native-plane dimensions; the private grid-alpha map has
a round-trip unit test. The full library suite is green (`424 passed`,
`5 ignored`) and FFmpeg conformance is green (`92 passed`, `2 ignored`). The
seven-iteration release bench measured `83.8378/83.3717 ms` (native/RGBA) for
`samples/WML2Viewer.avif`; this is a local regression checkpoint, not a
portable speedup claim.

On 2026-07-24, animated AVIF container parsing now filters `moov` children to
actual `trak` boxes before reading track metadata. The official
`colors-animated-8bpc.avif` sample is covered by an optional metadata
regression test and exposes all five samples, including `show-existing`; full
animated frame decode remains a separate AV1 tile-entropy compatibility gap.
Still-image decode no longer allocates CDF snapshots unless the AVIS sequence
path needs them for the next predicted sample. The 429-test library suite and
the animated metadata test pass. Two seven-iteration local release benches
measured `83.7375/81.9339 ms` and `79.1575/80.6369 ms` (native/RGBA), so this
change is recorded as an allocation checkpoint rather than a portable speedup
claim.

The same official animated sample was replayed through the public frame API as
an AV1 tile-entropy audit. Its first key sample still fails closed at strict
tile termination before a frame is accepted, so animated frame decode remains
an explicit unsupported boundary; no partial frame is exposed by the API.

On 2026-07-24, generated AVIS coverage now decodes every frame in a 60-sample
Inter/show-existing sequence and verifies indexed show-existing lookup through
the public sequence API. Rectangular Identity dispatch remains supported for
the 32-point stage shapes (`Tx8x32`, `Tx16x32`, `Tx32x8`, `Tx32x16`); 64-point
non-DCT rectangular requests now return an explicit `Unsupported` error
instead of reaching an internal `unreachable!`. The 425-test library suite,
15-test container suite, and FFmpeg conformance suite remain green. A fresh
seven-iteration release benchmark measured `81.6585/82.0603 ms`
(native/RGBA) for `samples/WML2Viewer.avif`; this is a same-host checkpoint,
not a portable speedup claim.

On 2026-07-24, AV1 inter transform-set 3 now reconstructs 32x32 `Identity`,
`VerticalDct`, and `HorizontalDct` blocks through the staged 32-point inverse
kernels, with allocating and in-place dispatch covered by fixed-vector tests.
The 32-point ADST/FlipADST cases remain fail-closed, and 64-point non-DCT
rectangular requests retain their explicit `Unsupported` boundary. The
426-test library suite, 15-test container suite, and 92-pass/2-ignored FFmpeg
conformance suite remain green. A fresh seven-iteration release benchmark
measured `79.4775/80.5235 ms` (native/RGBA) for `samples/WML2Viewer.avif`;
this is a same-host checkpoint, not a portable speedup claim.

On 2026-07-24, the official libavif
`colors-animated-12bpc-keyframes-0-2-3.avif` sample exposed an integer
overflow in the SGRPROJ high-bit-depth restoration variance calculation.
Restoration now uses AOM's highbd box-sum normalization, 64-bit accumulators,
and bit-depth-specific output clamps for SGRPROJ and Wiener restoration; the
8-bit paths keep their previous fixed-point arithmetic. The sample now decodes
all sequence frames, and an optional external-sample test checks its 12-bit
range. The 434-test library suite is green (`429 passed`, `5 ignored`), the
16-test container suite and 94-test FFmpeg conformance suite are green (`92
passed`, `2 ignored`). The expanded parent external gate reports 38
successes, 0 expected failures, 0 unexpected results, and 0 partial PNGs.
The
the 11-iteration release benchmark measured `82.6978/84.1873 ms`
(native/RGBA) for `samples/WML2Viewer.avif`. This is a same-host
no-regression checkpoint, not a portable speedup claim.

On 2026-07-24, official libavif HDR, sample-transform, and ISO 21496 gain-map
samples were added to the external manifest. Gain-map composition now treats
an unspecified CICP primary set (`2`) on the scalar gain-map item as inheriting
the peer image's primaries, instead of attempting an invalid chromaticity
conversion. The audit covers HDR P3/Rec.2020/sRGB, `weld_sato_12B_8B_q0`,
different-size and grid gain maps, and supported/unsupported writer-version
metadata. Deblocking now exits before building boundary lookup tables when all
frame, segmentation, and block deltas are zero. The 434-test library suite,
17-test container suite, and 95-test FFmpeg conformance suite remain green;
the external gate reports 61 successes, 1 expected failure, 0 unexpected
results, and 0 partial PNGs. The malformed `poc_b_506387278.avif` is retained
as one expected fail-closed rejection. A seven-iteration release benchmark measured
`89.5729/86.8459 ms` for `WML2Viewer.avif`; a 20-iteration filter-disabled
directional fixture measured `0.3857/0.3903 ms`. These are same-host
checkpoints, not portable speedup claims.

The CDEF stage now exits after resolving the per-64x64 indices when every
selected index maps to zero strength, avoiding direction analysis, filtered
block bookkeeping, and source/output cloning for frames whose header contains
unused non-zero strength slots. A seven-iteration release benchmark measured
`88.0570/84.0579 ms` (native/RGBA) for `WML2Viewer.avif`; this is a same-host
checkpoint and not a portable speedup claim.

Implementation-rate snapshot (2026-07-24): the checklist contains 153 of 170
completed items (`90.00%`), but this is a historical work-item metric rather
than a feature-completeness claim. The external manifest has 65 successful
decodes and 2 explicit fail-closed cases out of 67 entries (`97.01%` successful
decode coverage, `100%` expected-behavior coverage). At the transform API
boundary all 19 AV1 transform sizes have a DCT path (`19/19`, `100%` size
coverage); counting every 16-way size/class pair conservatively gives
`169/304` (`55.59%`) because the remaining non-DCT classes are intentionally
restricted or not yet implemented for larger sizes.

On 2026-07-24, the official `grpl/altr` gain-map preference vectors were
audited. The parser now honors entity ordering when the alternate group
contains both the primary item and the `tmap` item: a gain map is exposed only
when it is the preferred entity, while the official `wrongaltr` sample remains
fully decodable through its base-image fallback. The audit also adds HDR
`seine_hdr_srgb`/`seine_hdr_rec2020`, zero-gamma and duplicate-ICC malformed
gain-map samples, with explicit fail-closed expectations. The 437-test library
suite is green (`432 passed`, `5 ignored`), the 17-test container suite and
96-test FFmpeg conformance suite are green (`94 passed`, `2 ignored`). The
parent external gate reports 65 successes, 2 expected failures, 0 unexpected
results, and 0 partial PNGs; the expected failures are the malformed nclx
range sample and duplicate-ICC association sample.

The CDEF index fast path and its regression vector are now included in the
438-test library suite (`433 passed`, `5 ignored`); the 96-test FFmpeg suite
still passes (`94 passed`, `2 ignored`).

The external manifest now includes the latest libavif WCG/HDR text-colour
matrix (`colors_wcg_hdr_rec2020.avif` and six `colors_text_*` variants) plus
the five-frame `colors-animated-8bpc.avif` sequence. The animation metadata
tests use the checked external directory by default, so the sequence now
checks all five frame kinds, including `show-existing`, without requiring a
special environment variable. The malformed `paris_icc_exif_xmp.avif` ICC
association is registered as a second explicit fail-closed case. The expanded
gate reports 73 successful decodes, 3 expected failures, 0 unexpected results,
and 0 partial PNGs out of 76 manifest entries (`96.05%` successful decode
coverage, `100%` expected-behavior coverage). The checklist work-item rate is
unchanged at 153/170 (`90.00%`); this remains a historical task metric, not a
feature-completeness percentage.

The container metadata boundary now accepts distinct ICC (`prof`/`rICC`) and
`nclx` colour properties on the same item while still rejecting repeated
descriptions of the same colour-property family. The effective public colour
description prefers `nclx`, matching the FFmpeg oracle for the official
`paris_icc_exif_xmp.avif` sample. That sample and the official
`seine_sdr_gainmap_srgb_icc.avif` gain-map sample are now supported; focused
tests cover complete output, ICC gain-map composition, and the Paris RGB
oracle (`average RGB error 0.0024`, `max 6`). Matrix-shaper ICC alternate
gain-map composition now uses a linear 3x3 profile transform, while LUT
alternates remain fail-closed. The refreshed 76-entry external
gate reports 75 successes, 1 explicit fail-closed case, 0 unexpected results,
and 0 partial PNGs (`98.68%` successful decode coverage and `100%`
expected-behavior coverage). The full AVIF suites remain green: 442 library
tests (`437 passed`, `5 ignored`), 17 container tests, and 97 FFmpeg tests
(`95 passed`, `2 ignored`).

The generated AVIS Inter fixture also remains green with complete native
planes and an FFmpeg RGB oracle (average error `43.20`, max `255` under the
existing filtered-frame tolerance). The public support matrix therefore
lists tested Inter samples as supported; Switch and LUT-backed ICC alternate
gain-map conversion remain explicitly fail-closed.

The latest local decode checkpoint is `85.925 ms` for native frame decode and
`82.706 ms` for RGBA conversion on `samples/WML2Viewer.avif` (7 iterations,
same host). This is a local regression checkpoint, not a portable speedup
claim.

On 2026-07-25, strict oracle registration was verified with a custom output
directory under `.test*`; temporary oracle files are now nested below that
output and are removed without deleting the generated manifest or fixtures.
The 8-bit identity GBR RGBA path also separates alpha and no-alpha loops to
avoid an optional-plane lookup for every pixel; both paths have regression
coverage. The current checklist work-item snapshot is 156/170 (`91.76%`),
while the external sample and benchmark percentages above remain separate
validation metrics.

On 2026-07-25, the 8-bit YUV420 RGBA8 conversion path now uses the same
dedicated chroma-row traversal as the RGBA16 path when no auxiliary alpha is
present, avoiding a generic per-pixel chroma-layout dispatch. The existing
RGBA8/RGBA16 equivalence and scalar fast-path tolerance vectors remain green.
A 15-iteration release checkpoint measured `48.7819/51.9314 ms` (native/RGBA)
for `fox.profile0.8bpc.yuv420.avif`; `WML2Viewer.avif` measured
`94.1903/97.9325 ms` in the same run. These are same-host measurements and
do not establish a portable end-to-end speedup.

The strict Inter entropy audit on 2026-07-25 was reproduced with the generated
64x64 libaom sequence used by the FFmpeg oracle. Forcing the internal terminal
validator on the Inter sample reaches the final decoded blocks but reports
`tell=11778`, `max_bits=1942`, and the first non-zero candidate at bit `11765`.
Because the decoded frame still passes the public relaxed path, this is a
decoder-state/CDF consumption discrepancy rather than a safe two-bit terminal
offset correction. Inter and Switch therefore keep terminal validation in the
diagnostic-only path until the arithmetic/CDF state is reconciled; the public
decoder continues to fail closed for genuinely unsupported prediction tools.
The checklist work-item snapshot remains `156/170` (`91.76%`). The refreshed
external gate remains `75` successful decodes and `1` explicit fail-closed case
out of `76` entries (`98.68%` successful decode coverage and `100%`
expected-behavior coverage).

The 4:2:0 fast conversion path now keeps the dedicated RGBA8/RGBA16 traversal
when an auxiliary alpha plane is present, and selects chroma interpolation once
per output row instead of once per pixel. The new alpha-bearing regression
vector matches the scalar RGBA16 reference. A 15-iteration release checkpoint
measured `46.8158/49.6258 ms` (native/RGBA) for
`fox.profile0.8bpc.yuv420.avif`; `WML2Viewer.avif` measured
`87.5215/94.6123 ms` in the same run. These same-host values are follow-up
regression checkpoints, not a portable speedup claim.

The FFmpeg conformance suite now also generates an 8-bit 4:2:0 colour stream
with a separate full-resolution alpha stream. The fixture verifies native
4:2:0 chroma layouts, the auxiliary plane, and RGBA alpha against FFmpeg; the
98-test suite remains green. A 15-iteration release checkpoint for this small
alpha sample measured `0.9211/1.0831 ms` (native/RGBA), while the same run gave
`78.3252/78.8205 ms` for `WML2Viewer.avif`. These are host-specific
checkpoints, not portable end-to-end speedup claims.

The 4:2:0 converter now selects the alpha and no-alpha pixel writers outside
the row loop, so the common no-alpha path does not execute an optional-plane
branch for every pixel. The equivalence and generated 4:2:0-alpha tests remain
green. A follow-up 15-iteration release run measured `0.9581/1.1116 ms` for
the generated alpha sample and `77.6818/78.2171 ms` for `WML2Viewer.avif`;
these values are host-specific regression checkpoints.

On 2026-07-25, Switch-frame header parsing now follows the AV1 S-frame rules:
the refresh mask is inferred as `0xff`, and error-resilient reference order
hints are consumed before the frame-size and tile syntax. A real six-sample
SVT-AV1 fixture (including a Switch sample at index 3) now classifies and
decodes every sample completely, with an FFmpeg RGBA oracle on the Switch
frame. The FFmpeg conformance suite is `97 passed, 2 ignored` out of 99 tests;
the work-item implementation snapshot remains `156/170` (`91.76%`).

A fresh five-iteration optimized decode benchmark on the same host measured
`81.5013 ms` for native `decode_frame_bytes` and `82.6117 ms` for
`image_from_bytes` on `samples/WML2Viewer.avif`. This is a host-specific
regression checkpoint; the earlier 15-iteration `77.6818/78.2171 ms` run is
retained as the lower-noise comparison point.

The AVIF 1.2 layered-image boundary now accepts the normative `lsel=0xffff`
progressive-selection value as equivalent to the decoder's existing default
output policy, while concrete non-default layer IDs remain fail-closed. The
container regression covers both behaviors; the work-item snapshot remains
`156/170` (`91.76%`).

The YUV444 SDR converter now selects alpha and no-alpha row writers outside the
pixel loop, removing the optional-plane branch from the common no-alpha path.
A fresh five-iteration optimized run measured `82.6459/79.0358 ms` for native
`decode_frame_bytes`/`image_from_bytes` on `samples/WML2Viewer.avif`. The RGBA
value is close to the prior `82.6117 ms` checkpoint and the native value remains
host-noisy; these are regression checkpoints rather than portable speedup
claims.

The reconstruction hot path now reuses its already-known quantized non-zero
count and skips a second full dequantized-coefficient zero scan before inverse
transform dispatch. A regression vector covers all 19 AV1 square and
rectangular `TxSize` variants, including the case where quantization produces
an all-zero dequantized buffer. An 11-iteration optimized run measured
`79.8479/81.8986 ms` for `WML2Viewer.avif` native/RGBA; this is a same-host
checkpoint and not a portable speedup claim.

The FFmpeg conformance suite now generates an 8-bit 4:2:2 color-plus-alpha
AVIF sample and checks both the native horizontal-subsampling layout and the
full-resolution alpha plane against FFmpeg. With this fixture included, the
suite completes `98 passed, 2 ignored` tests; the work-item snapshot remains
`156/170` (`91.76%`).

Gain-map ICC composition now accepts linear-affine `mft1` and `mft2` LUT
profiles when their input/output tables are identity and the CLUT is affine;
non-linear LUTs and `mAB`/`mBA` pipelines remain fail-closed. The nested
library suite completes `442 passed, 5 ignored`, the FFmpeg conformance suite
completes `98 passed, 2 ignored`, and the external compatibility gate reports
`75` successful decodes plus `1` expected failure with no partial PNGs. The
implementation checklist remains `156/170` (`91.76%`), while the external
sample gate provides `75/76` (`98.68%`) successful decode coverage and `100%`
expected-behavior coverage. An 11-iteration optimized run measured
`81.8133/81.6517 ms` for native/RGBA decoding of `WML2Viewer.avif`; this is a
same-host regression checkpoint, not a portable speedup claim.

The linear ICC gain-map boundary now also accepts forward `mAB` profiles when
all embedded curves are identity, the CLUT is affine, the optional matrix has
no offset, and the PCS is XYZ. Reverse `mBA`, PCS-Lab, non-linear curves and
non-affine CLUTs remain fail-closed. The nested library suite is `444 passed,
5 ignored`, FFmpeg conformance remains `98 passed, 2 ignored`, and the latest
11-iteration optimized checkpoint is `79.5907/81.733 ms` native/RGBA for
`WML2Viewer.avif`. The checklist snapshot is now `157/171` (`91.81%`);
external compatibility coverage remains `75/76` successful decodes
(`98.68%`) with `100%` expected behavior.

Single-reference GLOBALMV reconstruction now applies signalled rotzoom and
affine matrices through the existing two-pass warped filter instead of using
only the block-centre translation vector; translation and compound global
motion retain the regular predictor path. The generated translation and affine
global-motion FFmpeg samples remain green, with the affine sample improving to
`30.57` average RGB absolute error (max `255`). The nested library suite is
`446 passed, 5 ignored`; the checklist snapshot remains `157/171` (`91.81%`)
and the external compatibility gate remains `75/76` successful decodes
(`98.68%`) with `100%` expected behavior.

GLOBAL_GLOBALMV now preserves the two independently signalled affine/rotzoom
models: each reference prediction is reconstructed through the warped filter
into the reusable compound scratch before the existing mask/weight blend.
Translation-only and unsupported warp-model fallbacks remain on the regular
inter predictor. The generated affine-global matrix now covers both 8-bit and
10-bit samples; the new 10-bit oracle reports `39.68` average RGB absolute
error (max `255`). The external compatibility snapshot remains `75/76`
successful decodes (`98.68%`) with `100%` expected behavior.
The nested library suite remains `446 passed, 5 ignored`; the expanded FFmpeg
matrix is now `99 passed, 2 ignored`.

The local-warp sample matrix now also includes a generated 10-bit 4:2:0
sequence fixture, exercising the warped prediction and high-bit-depth conversion
paths together. Entropy boolean/literal/symbol helpers are marked for release
inlining; the change is allocation-free and remains a no-regression checkpoint
until a quieter repeated benchmark proves a portable speedup.

The Inter/Switch AVIS path was rechecked with strict entropy termination enabled
against the generated 8-bit inter sample. Reconstruction reaches the complete
frame, but the AV1 trailing-padding oracle still rejects the tile at bit 11765
(`tell=11778`, `padding_end=13720`). The production path therefore keeps the
existing fail-closed boundary (`validate_entropy=false` for predicted frames)
instead of weakening the entropy oracle. A power-of-two `read_uniform` fast path
now dispatches directly to literal-bit reads and has an equivalence regression
test. Two seven-iteration optimized checkpoints measured
`80.8061/82.4734 ms` and `83.7614/87.5665 ms` native/RGBA for
`samples/WML2Viewer.avif`; this remains host-noisy and is not claimed as a
portable speedup. The implementation snapshot remains `157/171` (`91.81%`).

The official animated sequence boundary is now covered by public batch and
indexed-frame tests: `colors-animated-8bpc.avif` decodes all five samples
(including Key/Inter/show-existing), and
`colors-animated-12bpc-keyframes-0-2-3.avif` decodes all five 12-bit samples.
Both batch results match their indexed lookups, so the earlier animated
sequence audit is no longer an unverified Unsupported gap. The checklist
work-item rate remains `157/171` (`91.81%`) because this closes a validation
hole rather than a still-unchecked normative filter item.

The same checkpoint's seven-iteration release recheck measured
`82.6161/85.0707 ms` and `82.0771/82.4811 ms` for native/RGBA decoding of
`samples/WML2Viewer.avif`. These remain same-host no-regression checkpoints;
the spread confirms that a portable speedup claim requires a quieter harness.

TileDecoder now reserves frame-scale capacity for CDEF units, transform
boundaries, and block filter metadata before reconstruction. The full library
suite remains `448 passed, 5 ignored`, and the expanded FFmpeg suite is
`102 passed, 2 ignored`; seven-iteration WML2Viewer rechecks measured
`82.4952/85.1529 ms` and `82.2884/85.6741 ms` native/RGBA. The allocation
change is retained as a no-regression optimization checkpoint, not a portable
speedup claim.

The CDEF frame index table now stores the three-bit syntax values in a compact
`u8` grid with an explicit absent-unit sentinel, removing `Option<usize>`
unpacking from the per-8x8 filtering loop while preserving the default index-0
behavior for missing units. The active-index fast path has a sentinel
regression vector. The current suites are `449 passed, 5 ignored` in the
library, `17 passed` in container tests, and `104 passed, 2 ignored` in FFmpeg
conformance. A 15-iteration optimized WML2Viewer checkpoint measured
`81.9244/82.9463 ms` native/RGBA; this is a same-host allocation/dispatch
checkpoint, not a portable end-to-end speedup claim. The implementation-rate
snapshot is `157/171` (`91.81%`); the remaining 14 items are intentionally
open normative/filter-oracle work.

Large-frame CDEF direction analysis now partitions the immutable luma 8x8
origins across scoped standard-library workers once the frame exceeds the
existing post-filter threshold and has at least 512 candidate blocks. Results
are joined in chunk order, so the per-plane filter output remains deterministic;
small frames and Wasm retain the serial path. The library suite is `449 passed,
5 ignored`, container tests are `17 passed`, and FFmpeg conformance is `104
passed, 2 ignored`. A 15-iteration optimized WML2Viewer checkpoint measured
`77.2317/80.3228 ms` native/RGBA, versus the prior `81.9244/82.9463 ms` local
checkpoint; this is a same-host parallelism result, not a portable speedup
claim. The implementation-rate snapshot remains `157/171` (`91.81%`) because
the 14 open items are normative transform/filter-oracle boundaries rather than
performance checklist items.

A follow-up 15-iteration run with the CDEF worker cap fixed at eight measured
`79.1971/79.8544 ms` native/RGBA for `samples/WML2Viewer.avif`. The variation
from the earlier `77.2317/80.3228 ms` run is retained as host noise; both runs
remain below the pre-parallel local `81.9244/82.9463 ms` checkpoint, without a
portable speedup claim.

The loop-restoration source now preserves AOM's internal-stripe boundary
semantics: deblocked context rows are captured before CDEF, substituted only
while the corresponding processing stripe is filtered, and restored before the
next stripe. A deterministic regression test covers the scoped substitution and
restoration. On a local WML2Viewer strict-oracle probe, this reduced the
remaining plane mismatch from 2,488 samples with a whole-frame substitution to
13 samples; exact WML2Viewer oracle parity remains open pending the remaining
normative post-filter investigation. The implementation-rate snapshot remains
`157/171` (`91.81%`).

HDR PQ/HLG display conversion now maps supported P3 and BT.2020 primaries
through linear BT.709 before the existing bounded SDR shoulder; ICtCp keeps its
existing display-reference path. The new primary-matrix and HDR transfer tests,
the full library suite (`452 passed`, `5 ignored`), and FFmpeg conformance
(`104 passed`, `2 ignored`) are green. Two 11-iteration optimized
WML2Viewer checkpoints measured `82.7718/78.7947 ms` and
`78.3757/79.6544 ms` native/RGBA; because the host spread is noisy, this is retained
as a regression checkpoint rather than a portable speedup claim. The current
checklist snapshot is `158/171` (`92.40%`); the remaining 13 items are
normative transform/filter-oracle, display-profile, malformed-future-syntax,
and scratch-sizing work.

The diagnostic `WML2Viewer` native-plane/RGBA fixture is now generated by
`scripts/bootstrap_oracles.ps1` and required by the strict oracle manifest,
alongside the six filter-disabled fixtures. A fresh bootstrap generated all
seven fixtures, source-hash verification passed, and the strict oracle test
passed with `12` tests. The checklist snapshot is now `159/171` (`92.98%`),
with 12 normative/filter-oracle or future-syntax items remaining.

Deblock traversal now runs independently per plane through scoped workers on
large frames, matching the existing CDEF and restoration parallel boundaries;
small frames and Wasm remain serial. The post-filter subset passed `43` tests,
the full library suite passed `453` tests with `5` ignored, and FFmpeg
conformance passed `104` tests with `2` ignored. Two 11-iteration optimized
WML2Viewer checkpoints measured `78.6205/78.3933 ms` and
`80.3829/77.6366 ms` native/RGBA; the variation is retained as a same-host
no-regression checkpoint rather than a portable speedup claim.
