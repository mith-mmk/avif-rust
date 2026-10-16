# Shared pixel model integration

The codec-independent `avif-image-buffer` package lives in
`crates/avif-image-buffer`. It is unpublished (`publish = false`). Consumers
must use the same immutable Git source/version to share Rust type identity.
No repository or crate publication is part of this work.

`avif_rust::buffer` reexports that package. Existing RGBA8/RGBA16/native types
keep their fields and signatures. Their additive `view_ex` adapters borrow
storage with explicit depth/role/color context. Full-range RGBA16 is never
reinterpreted as native12-bit data.

`decode_native_ex` uses the bounded strict decoder once and returns `NativeImageEx`:
shared native pixels plus separate `AvifSourceContext`. Original render/coded
dimensions, AV1 color configuration, alpha association and the complete existing
`NativeAvifInformation` remain available. Sample allocations move; ICC metadata
may be copied. No ICC/display, precision or alpha conversion occurs. Unsupported
native component roles reject rather than falling back through RGBA.

Source context is not a promise that the writer can preserve every property or
unknown payload. The later preservation preflight must validate this separately.
`NativeDecodedFrame::into_parts` and older projections remain available.

WML2 PR13 and separate JPEG2000 work are unchanged. The final dependency direction
is avif -> shared and optional avif --encode-> avifenc -> shared, with no reverse
encoder dependency on avif. Shared crate placement does not change decoder defaults.
