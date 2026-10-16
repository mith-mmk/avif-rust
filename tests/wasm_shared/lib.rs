use avif_rust::{buffer::*, encoder as enc, transcode::*, *};
use wasm_bindgen::prelude::*;
fn limits() -> NativeDecodeLimits {
    NativeDecodeLimits::new(
        1 << 20,
        1024,
        1024,
        1 << 20,
        1 << 26,
        1 << 20,
        1 << 20,
        128,
        512,
        128,
        16,
        1,
    )
}
#[wasm_bindgen]
pub fn verify_shared() -> u32 {
    let mut cases = 0;
    for bits in [8, 10, 12] {
        let max = sample_max(bits).unwrap() as u16;
        // Asymmetric RGB with hidden color at alpha zero and partial alpha.
        let rgba = [max, 1, 0, 0, 1, max, 1, max / 2, 0, 1, max, max];
        let color = ColorMetadataRef {
            icc: Some(b"opaque ICC"),
            cicp: Some(Cicp {
                primaries: 1,
                transfer: 13,
                matrix: 0,
            }),
        };
        let view = ImageViewEx::Packed(PackedImageViewEx {
            plane: PlaneViewEx {
                samples: SamplesRef::U16(&rgba),
                layout: PlaneLayoutEx {
                    width: 3,
                    height: 1,
                    valid_bits: bits,
                    offset_bytes: 0,
                    row_stride_bytes: 24,
                },
            },
            channels: PackedChannels::Rgba,
            alpha: AlphaMode::Straight,
            color,
        });
        let native = enc::NativeEncodeOptions {
            bit_depth: bits,
            subsampling: enc::NativeSubsampling::Cs444,
            encoder: enc::EncoderOptions {
                lossless: true,
                color_information: None,
                ..Default::default()
            },
            color_information: enc::NativeColorInformation {
                nclx: Some(enc::NclxColorInformation {
                    color_primaries: 1,
                    transfer_characteristics: 13,
                    matrix_coefficients: 0,
                    full_range_flag: true,
                }),
                icc_profile: Some(b"opaque ICC".to_vec()),
                icc_color_type: Some(*b"rICC"),
            },
        };
        let options = EncodeOptionsEx {
            native: native.clone(),
            precision: PrecisionPolicy::Exact,
            color: enc::ColorPolicyEx::Preserve,
        };
        assert!(!preflight_ex(view, &options).unwrap().encoded);
        let encoded = encode_ex(view, &options).unwrap();
        let image = decode_native_ex(&encoded.bytes, &limits()).unwrap();
        let ImageViewEx::NativePlanar(decoded) = image.view().unwrap() else {
            panic!()
        };
        for x in 0..3 {
            for (role, channel) in [(0, 1), (1, 2), (2, 0), (3, 3)] {
                assert_eq!(
                    decoded.planes[role].unwrap().sample(x, 0, 0, 1).unwrap(),
                    rgba[x * 4 + channel]
                );
            }
        }
        assert_eq!(
            image
                .context()
                .information()
                .color_information()
                .icc_color_type,
            Some(*b"rICC")
        );
        let output =
            transcode_native(&image, &native, &PreservationPolicy::default(), &limits()).unwrap();
        assert!(output.report.encoded && output.report.container_signalling_verified);
        assert!(!output.report.pixel_roundtrip_verified);
        let second = decode_native_ex(&output.bytes, &limits()).unwrap();
        assert_eq!(image.image(), second.image());
        let bad = NativeDecodeLimits::new(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert!(decode_native_ex(&encoded.bytes, &bad).is_err());
        cases += 1;
    }
    // Byte-only LE access: odd offset/stride, no padding after final row.
    let bytes = [0xee, 1, 0, 2, 0, 0xff, 3, 0, 4, 0];
    let plane = PlaneViewEx {
        samples: SamplesRef::U16LeBytes(&bytes),
        layout: PlaneLayoutEx {
            width: 2,
            height: 2,
            valid_bits: 12,
            offset_bytes: 1,
            row_stride_bytes: 5,
        },
    };
    plane.validate(1, 0).unwrap();
    assert_eq!(plane.sample(1, 1, 0, 1).unwrap(), 4);
    cases += 1;
    let rgba = Rgba16ImageBuffer {
        width: 1,
        height: 1,
        rgba: vec![1, 2, 3, 0],
    };
    let view = rgba.view_ex(ColorMetadataRef::default()).unwrap();
    let ImageViewEx::Packed(p) = view else {
        panic!()
    };
    assert_eq!(p.plane.layout.valid_bits, 16);
    cases += 1;
    cases
}
