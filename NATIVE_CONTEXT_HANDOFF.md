# Native context handoff: first integration checkpoint

This checkpoint is an additive prerequisite for the adopted shared-buffer and
avif2avif preservation designs. It does not implement the complete buffer model,
preservation preflight, transcode writer, or optional encode feature.

`NativeDecodedFrame::into_parts()` returns `(DecodedFrame,
NativeAvifInformation)`. The existing frame and context owners are moved together.
This keeps accounting, primary item type, ordered primary properties and rich
metadata available without another parse/decode or a pixel conversion. The three
new tests check real native fixtures, sample/property allocation addresses, the
old consuming projections and existing public struct literals/signatures.

Existing `into_frame()` and `into_frame_and_rich()` retain their previous behavior.
In particular, the latter still projects the context to `RichAvifInfo`; callers
needing the outer information can now choose the new method explicitly. All
public fields, legacy pixel/decode functions, manifests and dependencies remain
unchanged. There is no new encoder dependency.

The context is the complete **existing NativeAvifInformation**, not an opaque copy
of every original file property. For example, an unknown property's recorded
fourcc does not provide its payload. Retention is not proof that the writer can
preserve it. The later preflight must reject unsupported preservation by default,
and require explicit, individually reported conversion/drop permissions. The
native path must not fall back through RGBA, ICC application, unpremultiplication
or precision reduction.

## Executed verification

Baseline: `ecc37b220daf37b24134aacac0bc12723aadc357` (avif-rust0.0.8).
Linux x86_64, Rust/Cargo1.91.0; dependencies resolved offline with Cargo.lock.

```sh
cargo test --locked --offline --release --test native_context --test native_hookup --test native_boundary --test native_limits --test native_phase_boundary --test native_prefix
cargo clippy --locked --offline --test native_context
cargo rustdoc --locked --offline --lib -- -D warnings
cargo check --locked --offline --target wasm32-unknown-unknown
cargo fmt --all -- --check
git diff --check
cargo tree --locked --offline --no-default-features --depth 1
```

- New `native_context` tests:3 passed,0 failed; no missing-fixture early returns.
- Selected integration targets: harness reports33 passed,0 failed,0 ignored.
  `native_prefix::normal_prefix_resumes_with_the_same_public_frame_geometry`
  returns early because the optional parent `samples/WML2Viewer.avif` is absent.
  That test body is **unexecuted**, despite its harness pass status. The new
  handoff tests use the checked-in `tests/` fixtures and did execute native decode.
- Ordinary Clippy, rustdoc, wasm32/default/no-default compile, fmt and diff checks
  passed. Clippy reports three existing warnings. The additional `-D warnings`
  gate fails on two `collapsible_else_if` diagnostics in
  `src/av1/tile_decode/block_syntax.rs` and `src/container.rs`, and one
  `search_is_some` in `src/container.rs`. The same strict failure was reproduced
  on the unmodified baseline; these paths have no diff in this checkpoint.
  No source changes were made to suppress these inherited diagnostics.
  WASM runtime,
  MSRV1.88, other operating systems, full codec/oracle corpus and new color/alpha
  conversion compatibility baselines were not executed for this checkpoint.
- Default and no-default dependency trees still contain only `bin-rs` beneath
  avif-rust; encoder/backend dependencies were not added.
- Allocation address assertions apply only to the owned handoff boundary; they
  are not a whole decode/encode zero-copy or performance claim.

## Source access and remaining decisions

The adopted documents are `avif_avifenc_unified_buffer_spec_2026-10-16.md`
and `avif2avif_contract_2026-10-16.md`. Their complete551/170 lines were read
through Library. The official local materialization attempt returned
`download failed` for both; no local spec bytes were obtained and materialization
is not claimed successful. This source-access limitation is separate from the
tested code checkpoint.

The shared crate's name, location and distribution remain undecided. Its final
dependency direction must be `avif -> shared`, `avif --encode-> avifenc -> shared`;
avifenc must not depend on avif. No codec-default feature reorganization, shared
crate, new repository, crate publication or release is included here. WML2
follow-up comes later; WML2 PR13 and separate JPEG2000 work remain untouched.
