# Native pixel conformance fix validation

Base: main `070ad6509bad18f9ab76732d543659ef6d7f9129`. This independent branch
does not include the AVIS changes in open PR #3 (`d3ea4e602763951626d678291cc9c325e2c475b2`).
Investigation: [5f2f38c](https://github.com/mith-mmk/avif-rust/tree/5f2f38c236051271a63798ab99662c9426573f80/investigation/native-pixel-20261022).

## Changes

- Scale deblock flatness thresholds by bit depth for 6/8/14-tap filtering.
- Reject edges outside the visible frame while allowing their taps to read
  reconstructed coded padding. Preserve visible lane bounds.
- Disable deblock when both frame luma levels are zero, regardless of deltas.
- Keep complete 64-sample transform extents until post-filter crop so CfL uses
  the reconstructed luma outside the visible/8-aligned extent. This adds at most
  63 rows/columns per luma axis. Both allocation paths share checked geometry;
  strict decode admits actual capacities and preserves transactional failure.
- Update UV smooth contexts only at chroma owners, over the shared chroma
  footprint; query external chroma neighbours rather than internal luma MIs.
- Use 4x4-MI deblock metadata, padded to the coded even-MI extent. Select the
  bottom/right owner for subsampled chroma and check both lookup coordinates.
  Strict deblock workspace admission includes the larger metadata allocation.

There are no runtime probe switches, trace hooks, reference decoder dependencies,
or public API changes in the product implementation.

## Independent references and private seven-image verification

Linux x86-64; FFmpeg 7.1.5. Native planar samples only, no RGB conversion,
rescaling or range conversion. 8-bit samples are bytes; 10/12-bit samples are
little-endian u16. Visible rows are copied using each decoder's stride. Primary
colour and alpha are kept separate. All seven inputs have film grain disabled.

References:

- Installed libaom 3.12.1 (Debian `3.12.1-1+deb13u1`).
- C-only libaom v3.12.1 source `10aece4157eb79315da205f39e19bf6ab3ee30d0`,
  single-thread decoder, diagnostic dumps immediately before deblock, after
  deblock, after CDEF and after restoration. Its final output also matches the
  installed decoder. Diagnostic modifications are confined to a private build.
- Installed dav1d 1.5.1 (Debian `1.5.1-1`), one thread, grain disabled;
  `inloop_filters` masks 0/1/3/7 expose the corresponding stages. Source reference
  tag 1.5.1 is `42b2b24fb8819f1ed3643aa9cf2a62f03868e3aa`.

The seven attached images (fox 8-bit 420, fox 10/12-bit 444, odd-size fox 8-bit
444, plum alpha, Rec.2020 HDR, Paris ICC/Exif/XMP) have **zero differences at all
four stages against both references**, in both legacy and strict paths, and zero final strict/legacy differences.
The 16,384 alpha samples also match both references.
Stage dumps were inserted only in a disposable copy of this product patch.
The private attached inputs, extracted bitstreams and raw frames are not committed.
See the machine-readable summary in `native-pixel-validation.json`.

## Required regressions

Five bundled arithmetic-pattern fixtures compare every final native sample
against libaom/dav1d-agreed oracles and require strict/legacy equality. They run
without external executables in `cargo test`. Unit tests additionally cover
high-bitdepth flatness, visible-edge padding, shared UV contexts, partial-edge
CfL, padded metadata, and strict exact/one-under admission and rollback.
Fixture regeneration and licensing are in `tests/FIXTURES.md`.

Local default release suite: 869 passed, 0 failed, 9 explicitly ignored.
All-features release suite: 881 passed, 0 failed, 9 explicitly ignored.
`cargo clippy --all-targets -- -D warnings` passed. Hosted CI
and additional feature checks are recorded in the PR after completion.

## Broader coverage and remaining limits

This fixes the identified seven-image problem; it does not establish general
AV1 conformance. Broader tests intentionally retain and report remaining errors:

- All 19 entries of `scripts/test_corpus.csv` were downloaded privately and
  SHA-256 verified. The 18 single-image elementary primary streams decoded;
  all had matching independent references and exact strict/legacy agreement.
  Nine match the reference exactly. Nine retain smaller pre-existing mismatch
  populations (odd-height/subsampled fox and kimono). The grid fixture is not
  comparable through the elementary-primary wrapper; grid behaviour is covered
  by the existing strict grid tests, not counted as reference agreement here.
- 24 arithmetic-pattern cases at 93x77 cover 8/10/12-bit, 400/420/422/444,
  lossy/lossless with default filters. All 12 lossless cases match. Lossy cases
  have existing differences, predominantly first appearing at CDEF; reconstruction
  matches and 444 10-bit has a two-byte residual after deblock. One aggregate
  byte mismatch count increases by one (gray12), so this is **not** a claim of
  zero regressions on arbitrary nonconforming inputs. Stage attribution, not
  aggregate count alone, distinguishes the corrected operations.
- Another 24 cases at 85x73 disable CDEF/restoration. All lossless cases and
  five lossy cases match; seven high-bitdepth lossy cases retain 3–10 differing
  bytes (gray12: four). These remain outside this six-cause patch. The generator
  reproduces them; their before/after counts are retained in the JSON.
- Existing PR #3's inter/AVIS correctness work remains independent. No merge,
  release, changes to that branch, or publication of user input images occurred.

The broader remaining differences warrant follow-up before claiming full native
conformance. Do not weaken exact oracle assertions or regenerate expected output
from this decoder to make those cases pass.
