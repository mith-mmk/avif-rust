# AVIF external-image issues

This file records the external AVIF verification state. The fixtures remain
outside the repository under `test/images/external/avif`.

## Resolved in the current work tree

- `hq720.avif`: multi-tile YUV420 reconstruction now keeps tile-local
  prediction edges and the WML2 chroma sample-position policy. Native Y/U/V
  comparison passes; RGBA8 comparison is within average error 2 and maximum
  channel error 64.
- `seine_hdr_rec2020.avif` and `seine_hdr_srgb.avif`: 10-bit native plane
  comparisons no longer show the right-bottom block displacement. RGBA8
  display conversion is covered separately with the wider HDR display
  tolerance required by the transfer conversion.
- AVIS container handling: color/alpha tracks are identified through
  `tref/auxl` and the auxiliary type, depth/audio tracks are ignored, and
  `mdhd`/`stts` durations are exposed to the callback. Alpha count and frame
  dimensions are checked before the callback is entered.
- WML2 callback ordering and animation storage: `init -> next -> draw` is
  covered, durations are forwarded, and the alpha track is synchronized per
  color frame. The external five-frame cases are recognized, retained as five
  full-canvas layers, and survive a WML2 APNG encode/decode round trip.
- AVIS Inter reconstruction: single-reference MV stacks now receive the
  normative adjacent fallback candidates, inter-intra neighbours are excluded
  from local-warp projection samples, compound contexts/masks follow the
  decoded neighbour state, and inter prediction keeps AV1's 16-phase Q4 sample
  coordinates. `colors-animated-8bpc*.avif` and `star-8bpc.avif` now pass the
  five-frame FFmpeg RGBA oracle, including per-frame alpha.
- Skipped intra blocks retain transform-size signalling. This also restores
  strict entropy termination and FFmpeg parity for `alpha_noispe.avif`.

## Open

No open issue remains for the external images covered by this work item. The
full AVIF package suite passes with `470` library tests, `17` container tests,
and `107` FFmpeg conformance tests; five diagnostics and two conformance probes
remain intentionally ignored.
