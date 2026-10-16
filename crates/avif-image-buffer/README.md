# avif-image-buffer

An unpublished, codec-independent pixel model shared by avif-rust and
avifenc-rust. This is not a wmlcore/paintcore extraction. It depends only on
the Rust standard library and forbids unsafe code.

- Packed Gray/RGBA and native planar monochrome, identity GBR or YUV are tagged
  separately. Native color roles are not inferred from sample values.
- Numeric U16, U8 and explicit U16LeBytes storage have independent8/10/12/16-bit
  sample precision. U8 permits8 bits only. Samples use their low valid bits.
- Plane offsets/strides are bytes. Numeric U16 requires even offsets/strides;
  LE bytes support odd/unaligned layouts without casts. Final-row padding is
  not required; padding is not validated as sample data or exported compactly.
- Immutable views borrow samples and metadata. Validation owns a descriptor
  snapshot; no validity flag is cached on mutable legacy structs. Owned
  constructors adopt sample vectors, while compact export copies logical
  samples and metadata with fallible allocation.
- Straight/premultiplied/absent alpha and unknown multiplication domains remain
  distinct. ICC bytes and raw CICP identifiers are held without applying them.
  AVIF property order, ICC property type and source graphs stay in AVIF context.
- Exact and explicitly selected Nearest tie rules scale full-range packed
  samples/alpha. They do not implement native limited-range, RGB/YUV, ICC,
  premultiplication or geometry conversions. Unsupported conversions reject.

```rust
use avif_image_buffer::*;
let samples = [0u16, 1, 512, 1023];
let view = ImageViewEx::Packed(PackedImageViewEx {
    plane: PlaneViewEx {
        samples: SamplesRef::U16(&samples),
        layout: PlaneLayoutEx {
            width: 4, height: 1, valid_bits: 10,
            offset_bytes: 0, row_stride_bytes: 8,
        },
    },
    channels: PackedChannels::Gray,
    alpha: AlphaMode::Absent,
    color: ColorMetadataRef::default(),
});
view.validate()?;
let owned = view.to_owned_compact()?;
assert_eq!(owned.view()?.width(), 4);
# Ok::<(), BufferError>(())
```

`cargo test -p avif-image-buffer` checks precision/storage combinations,
odd/padded byte layouts, no final padding, native odd dimensions and alpha,
allocation adoption, overflow rejection, explicit scaling and opaque metadata.
The inline fixed plane descriptors avoid allocating a view; this is not a claim
that codecs perform no allocations/copies internally.
