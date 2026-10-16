use avif_image_buffer::*;

fn plane<'a>(
    samples: SamplesRef<'a>,
    w: usize,
    h: usize,
    bits: u8,
    stride: usize,
) -> PlaneViewEx<'a> {
    PlaneViewEx {
        samples,
        layout: PlaneLayoutEx {
            width: w,
            height: h,
            valid_bits: bits,
            offset_bytes: 0,
            row_stride_bytes: stride,
        },
    }
}
fn gray(p: PlaneViewEx<'_>) -> PackedImageViewEx<'_> {
    PackedImageViewEx {
        plane: p,
        channels: PackedChannels::Gray,
        alpha: AlphaMode::Absent,
        color: ColorMetadataRef::default(),
    }
}
fn output(bits: u8, storage: SampleStorage) -> PackedOutput {
    PackedOutput {
        channels: PackedChannels::Gray,
        storage,
        valid_bits: bits,
        row_stride_bytes: None,
        precision: PrecisionPolicy::Exact,
    }
}

#[test]
fn storage_width_and_valid_bits_are_independent_and_range_checked() {
    for bits in [8, 10, 12, 16] {
        let max = sample_max(bits).unwrap() as u16;
        let words = [0, 1, max / 2, max];
        let bytes: Vec<_> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
        for samples in [SamplesRef::U16(&words), SamplesRef::U16LeBytes(&bytes)] {
            let p = plane(samples, 4, 1, bits, 8);
            p.validate(1, 0).unwrap();
            let compact = ImageViewEx::Packed(gray(p)).to_owned_compact().unwrap();
            let ImageViewEx::Packed(view) = compact.view().unwrap() else {
                panic!()
            };
            for (x, value) in words.iter().enumerate() {
                assert_eq!(view.plane.sample(x, 0, 0, 1).unwrap(), *value);
            }
        }
        if bits < 16 {
            let bad = [max + 1];
            assert!(
                plane(SamplesRef::U16(&bad), 1, 1, bits, 2)
                    .validate(1, 0)
                    .is_err()
            );
        }
    }
    assert!(
        plane(SamplesRef::U8(&[1]), 1, 1, 10, 1)
            .validate(1, 0)
            .is_err()
    );
    assert!(
        plane(SamplesRef::U16(&[1]), 1, 1, 9, 2)
            .validate(1, 0)
            .is_err()
    );
}

#[test]
fn odd_le_stride_offset_and_missing_final_padding_are_supported() {
    // Offset1, stride5, two12-bit samples/row, no padding after final row.
    let bytes = [0xee, 1, 0, 2, 0, 0xff, 3, 0, 4, 0];
    let mut p = plane(SamplesRef::U16LeBytes(&bytes), 2, 2, 12, 5);
    p.layout.offset_bytes = 1;
    p.validate(1, 0).unwrap();
    let compact = ImageViewEx::Packed(gray(p)).to_owned_compact().unwrap();
    let ImageViewEx::Packed(v) = compact.view().unwrap() else {
        panic!()
    };
    assert_eq!(
        v.plane.samples,
        SamplesRef::U16LeBytes(&[1, 0, 2, 0, 3, 0, 4, 0])
    );
    assert_eq!(bytes[0], 0xee);
    p.layout.row_stride_bytes = 3;
    assert!(p.validate(1, 0).is_err());
    let mut numeric = plane(SamplesRef::U16(&[1, 2]), 1, 2, 12, 3);
    assert!(numeric.validate(1, 0).is_err());
    numeric.layout.row_stride_bytes = 2;
    numeric.layout.offset_bytes = 1;
    assert!(numeric.validate(1, 0).is_err());
}

#[test]
fn native_roles_ceil_dimensions_and_separate_alpha_depth_are_checked() {
    for model in [
        NativeModel::Monochrome,
        NativeModel::IdentityGbr,
        NativeModel::Yuv444,
        NativeModel::Yuv422,
        NativeModel::Yuv420,
    ] {
        for bits in [8, 10, 12, 16] {
            for alpha in [false, true] {
                let (w, h) = (5usize, 3usize);
                let primary = vec![1u16; w * h];
                let (sx, sy) = model.chroma_subsampling().unwrap_or((0, 0));
                let (cw, ch) = (w.div_ceil(1 << sx), h.div_ceil(1 << sy));
                let chroma = vec![2u16; cw * ch];
                let a = vec![255u16; w * h];
                let desc = NativeDescriptor {
                    width: w,
                    height: h,
                    model,
                    alpha: if alpha {
                        AlphaMode::Straight
                    } else {
                        AlphaMode::Absent
                    },
                    range: Some(ColorRange::Limited),
                    chroma_location: Some(0),
                };
                let mut planes = [None; 4];
                planes[0] = Some(plane(SamplesRef::U16(&primary), w, h, bits, w * 2));
                if model != NativeModel::Monochrome {
                    planes[1] = Some(plane(SamplesRef::U16(&chroma), cw, ch, bits, cw * 2));
                    planes[2] = planes[1];
                }
                if alpha {
                    planes[3] = Some(plane(SamplesRef::U16(&a), w, h, 8, w * 2));
                }
                let n = NativePlanarViewEx {
                    descriptor: desc,
                    planes,
                    color: ColorMetadataRef::default(),
                };
                ImageViewEx::NativePlanar(n).validate().unwrap();
                let owned = ImageViewEx::NativePlanar(n).to_owned_compact().unwrap();
                assert_eq!(owned.view().unwrap().width(), w);
                let mut bad = n;
                bad.planes[0].as_mut().unwrap().layout.width = 4;
                assert!(ImageViewEx::NativePlanar(bad).validate().is_err());
                let mut bad = n;
                bad.descriptor.alpha = if alpha {
                    AlphaMode::Absent
                } else {
                    AlphaMode::Straight
                };
                assert!(ImageViewEx::NativePlanar(bad).validate().is_err());
            }
        }
    }
}

#[test]
fn single_allocation_bindings_preserve_native_values_without_copying() {
    let words = vec![1u16, 2, 999, 3, 4, 999, 5, 6];
    let address = words.as_ptr();
    let l = PlaneLayoutEx {
        width: 2,
        height: 1,
        valid_bits: 10,
        offset_bytes: 0,
        row_stride_bytes: 4,
    };
    let descriptor = NativeDescriptor {
        width: 2,
        height: 1,
        model: NativeModel::Yuv444,
        alpha: AlphaMode::Absent,
        range: Some(ColorRange::Full),
        chroma_location: None,
    };
    let planes = [
        Some(PlaneBinding {
            storage_index: 0,
            layout: l,
        }),
        Some(PlaneBinding {
            storage_index: 0,
            layout: PlaneLayoutEx {
                offset_bytes: 6,
                ..l
            },
        }),
        Some(PlaneBinding {
            storage_index: 0,
            layout: PlaneLayoutEx {
                offset_bytes: 12,
                ..l
            },
        }),
        None,
    ];
    let image = ImageBufferEx::native(
        descriptor,
        vec![OwnedSamples::U16(words)],
        planes,
        ColorMetadata::default(),
    )
    .unwrap();
    let ImageViewEx::NativePlanar(v) = image.view().unwrap() else {
        panic!()
    };
    let SamplesRef::U16(samples) = v.planes[0].unwrap().samples else {
        panic!()
    };
    assert_eq!(samples.as_ptr(), address);
    assert_eq!(v.planes[1].unwrap().sample(0, 0, 0, 1).unwrap(), 3);
    let compact = image.view().unwrap().to_owned_compact().unwrap();
    let ImageViewEx::NativePlanar(v) = compact.view().unwrap() else {
        panic!()
    };
    assert_eq!(v.planes[2].unwrap().samples, SamplesRef::U16(&[5, 6]));
}

#[test]
fn exact_scaling_and_explicit_nearest_use_normalized_values() {
    for v in 0..=255u16 {
        let widened = scale_full_range(v, 8, 16, PrecisionPolicy::Exact).unwrap();
        assert_eq!(widened, v * 257);
        assert_eq!(
            scale_full_range(widened, 16, 8, PrecisionPolicy::Exact).unwrap(),
            v
        );
    }
    assert!(scale_full_range(1, 8, 10, PrecisionPolicy::Exact).is_err());
    assert_eq!(
        scale_full_range(85, 8, 10, PrecisionPolicy::Exact).unwrap(),
        341
    );
    for v in [0u16, 1, 273, 4369, 32768, 65535] {
        let exact = scale_full_range(v, 16, 12, PrecisionPolicy::Exact);
        assert_eq!(exact.is_ok(), u32::from(v) * 4095 % 65535 == 0);
        let rounded = scale_full_range(
            v,
            16,
            12,
            PrecisionPolicy::Nearest {
                ties: TieRule::AwayFromZero,
            },
        )
        .unwrap();
        assert_eq!(rounded, ((u32::from(v) * 4095 + 32767) / 65535) as u16);
    }
}

#[test]
fn packed_conversion_preserves_hidden_rgb_icc_and_alpha_association() {
    let values = [1u16, 2048, 4095, 0, 4095, 3, 2, 4095];
    let icc = b"opaque profile, never applied";
    let p = PackedImageViewEx {
        plane: plane(SamplesRef::U16(&values), 2, 1, 12, 16),
        channels: PackedChannels::Rgba,
        alpha: AlphaMode::Premultiplied(AlphaDomain::Unknown),
        color: ColorMetadataRef {
            icc: Some(icc),
            cicp: Some(Cicp {
                primaries: 65000,
                transfer: 2,
                matrix: 0,
            }),
        },
    };
    let mut o = output(12, SampleStorage::U16LeBytes);
    o.channels = PackedChannels::Rgba;
    o.row_stride_bytes = Some(17);
    let owned = convert_packed(p, o).unwrap();
    let ImageViewEx::Packed(v) = owned.view().unwrap() else {
        panic!()
    };
    assert_eq!(v.alpha, p.alpha);
    assert_eq!(v.color, p.color);
    assert_eq!(values[0], 1);
    assert_eq!(v.plane.sample(0, 0, 0, 4).unwrap(), 1);
    let SamplesRef::U16LeBytes(bytes) = v.plane.samples else {
        panic!()
    };
    assert_eq!(bytes[16], 0);
    let mut bad = o;
    bad.channels = PackedChannels::Gray;
    assert!(convert_packed(p, bad).is_err());
}

#[test]
fn gray_expansion_is_not_a_color_transform_and_has_opaque_alpha() {
    let p = gray(plane(SamplesRef::U8(&[1, 255]), 2, 1, 8, 2));
    let mut o = output(16, SampleStorage::U16);
    o.channels = PackedChannels::Rgba;
    let image = convert_packed(p, o).unwrap();
    let ImageViewEx::Packed(v) = image.view().unwrap() else {
        panic!()
    };
    assert_eq!(
        v.plane.samples,
        SamplesRef::U16(&[257, 257, 257, 65535, 65535, 65535, 65535, 65535])
    );
}

#[test]
fn invalid_descriptors_never_panic_and_validation_does_not_cache_mutable_flags() {
    for width in [0, 1, usize::MAX, usize::MAX / 2 + 1] {
        for offset in [0, 1, usize::MAX] {
            for stride in [0, 1, 2, usize::MAX] {
                let p = PlaneViewEx {
                    samples: SamplesRef::U16LeBytes(&[0; 8]),
                    layout: PlaneLayoutEx {
                        width,
                        height: 2,
                        valid_bits: 12,
                        offset_bytes: offset,
                        row_stride_bytes: stride,
                    },
                };
                let _ = ImageViewEx::Packed(gray(p)).validate();
            }
        }
    }
    let values = [1023u16];
    let mut original = ImageViewEx::Packed(gray(plane(SamplesRef::U16(&values), 1, 1, 10, 2)));
    let frozen = original.validate().unwrap();
    if let ImageViewEx::Packed(ref mut p) = original {
        p.plane.layout.valid_bits = 8;
    }
    assert!(original.validate().is_err());
    frozen.view().validate().unwrap();
}

#[test]
fn padding_is_not_sample_data_and_compact_export_does_not_leak_it() {
    let data = [1u16, 65535, 2];
    let p = gray(plane(SamplesRef::U16(&data), 1, 2, 10, 4));
    let image = ImageViewEx::Packed(p).to_owned_compact().unwrap();
    let ImageViewEx::Packed(v) = image.view().unwrap() else {
        panic!()
    };
    assert_eq!(v.plane.samples, SamplesRef::U16(&[1, 2]));
}
