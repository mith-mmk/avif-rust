use avif_rust::{NativeDecodeLimits, buffer::*, decode_frame_bytes_strict_with_limits};

fn limits(data: &[u8]) -> NativeDecodeLimits {
    NativeDecodeLimits::new(
        data.len() * 2,
        8192,
        8192,
        1 << 28,
        1 << 30,
        1 << 24,
        1 << 24,
        128,
        512,
        128,
        16,
        1,
    )
}

#[test]
fn native_owners_and_complete_avif_context_survive_shared_model_adoption() {
    for data in [
        include_bytes!("WML2Viewer.avif").as_slice(),
        include_bytes!("plum-blossom-small.profile1.8bpc.yuv444.alpha-full.avif").as_slice(),
    ] {
        let native = decode_frame_bytes_strict_with_limits(data, &limits(data)).unwrap();
        let expected = native.information().clone();
        let config = native.frame().color_config;
        let dimensions = (native.frame().render_width, native.frame().render_height);
        let old: Vec<_> = native
            .frame()
            .buffers
            .planes
            .iter()
            .filter(|p| !p.samples.is_empty())
            .map(|p| (p.layout.plane, p.samples.as_ptr(), p.samples.clone()))
            .collect();
        native.view_ex().unwrap().validate().unwrap();
        let image = native.into_image_ex().unwrap();
        assert_eq!(image.context().information(), &expected);
        assert_eq!(image.context().color_config(), &config);
        assert_eq!(image.context().render_dimensions(), dimensions);
        let ImageViewEx::NativePlanar(view) = image.view().unwrap() else {
            panic!()
        };
        for (role, address, samples) in old {
            let p = view.planes[usize::from(role)].unwrap();
            let SamplesRef::U16(actual) = p.samples else {
                panic!()
            };
            assert_eq!(actual.as_ptr(), address);
            assert_eq!(actual, samples);
        }
        assert_eq!(
            view.color.icc,
            expected.color_information().icc_profile.as_deref()
        );
    }
}

#[test]
fn legacy_rgba16_remains_full_range16_and_old_fields_stay_mutable() {
    let mut old = avif_rust::Rgba16ImageBuffer {
        width: 1,
        height: 1,
        rgba: vec![1, 2, 4095, 0],
    };
    let ImageViewEx::Packed(v) = old.view_ex(ColorMetadataRef::default()).unwrap() else {
        panic!()
    };
    assert_eq!(v.plane.layout.valid_bits, 16);
    assert_eq!(v.plane.samples, SamplesRef::U16(&old.rgba));
    old.rgba.push(1);
    assert!(old.view_ex(ColorMetadataRef::default()).is_err());
}

#[test]
fn malformed_native_context_is_rejected_without_guessing_planes_or_depth() {
    let frame = avif_rust::av1::FrameBuffers {
        width: 1,
        height: 1,
        planes: vec![avif_rust::av1::PlaneBuffer {
            layout: avif_rust::av1::PlaneLayout {
                plane: 0,
                width: 1,
                height: 1,
                subsampling_x: 0,
                subsampling_y: 0,
                sample_count: 1,
            },
            samples: vec![1024],
        }],
    };
    let d = NativeDescriptor {
        width: 1,
        height: 1,
        model: NativeModel::Monochrome,
        alpha: AlphaMode::Absent,
        range: None,
        chroma_location: None,
    };
    assert!(frame.view_ex(d, 10, ColorMetadataRef::default()).is_err());
    frame.view_ex(d, 12, ColorMetadataRef::default()).unwrap();
    assert!(
        frame
            .view_ex(
                NativeDescriptor {
                    model: NativeModel::Yuv444,
                    ..d
                },
                12,
                ColorMetadataRef::default()
            )
            .is_err()
    );
}
