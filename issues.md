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
  color frame. The external five-frame cases are recognized and retained as
  five full-canvas layers.

## Open

### AV1 Inter reconstruction for external color AVIS

`colors-animated-8bpc.avif` and `star-8bpc.avif` still fail the strict FFmpeg
RGBA oracle on Inter frames. For `star-8bpc.avif`, frame 1 is currently about
40.8 average RGB error (maximum 240); the decoded image remains close to the
key-frame prediction instead of the rotated star. The mismatch is present
before RGBA conversion and is not caused by alpha composition or duration
handling.

The frame headers show a valid hidden-reference sequence followed by a
`show_frame` Inter frame. `PRIMARY_REF_NONE` was checked against the AV1
specification and must use the default CDF context with the current frame's
quantizer context; changing it to a previous CDF or a q=0 context does not
fix the stream and can fail entropy validation. The remaining work is to
compare the Inter block syntax/residual reconstruction against a normative
decoder and then add a targeted regression for the first failing block.

The same strict suite must still be run for the alpha, depth, audio, and
`star-8bpc.avif` variants after this Inter reconstruction issue is fixed.
