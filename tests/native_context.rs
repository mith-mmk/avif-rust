use avif_rust::av1::{ColorConfig, ColorRange, FrameBuffers, PlaneBuffer, PlaneLayout};
use avif_rust::{
    DecodedFrame, DecoderError, ImageBuffer, NativeAvifInformation, NativeDecodeLimits,
    NativeDecodedFrame, Rgba16ImageBuffer, RichAvifInfo, decode_frame_bytes_strict,
    decode_frame_bytes_strict_with_limits, image_from_bytes,
};

fn decode(data: &[u8]) -> NativeDecodedFrame {
    let limits = NativeDecodeLimits::new(
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
    );
    decode_frame_bytes_strict_with_limits(data, &limits).expect("checked-in native fixture")
}

#[test]
fn complete_context_and_native_sample_owners_survive_the_handoff() {
    for data in [
        include_bytes!("WML2Viewer.avif").as_slice(),
        include_bytes!("plum-blossom-small.profile1.8bpc.yuv444.alpha-full.avif").as_slice(),
    ] {
        let native = decode(data);
        let expected_frame = native.frame().clone();
        let expected_information = native.information().clone();
        assert!(!native.information().ordered_primary_properties().is_empty());
        let properties_owner = native.information().ordered_primary_properties().as_ptr();
        let sample_owners: Vec<_> = native
            .frame()
            .buffers
            .planes
            .iter()
            .map(|plane| plane.samples.as_ptr())
            .collect();

        let (frame, information) = native.into_parts();
        assert_eq!(frame, expected_frame);
        assert_eq!(information, expected_information);
        assert_eq!(information.input_bytes(), data.len());
        assert_eq!(
            information.ordered_primary_properties().as_ptr(),
            properties_owner
        );
        for (plane, owner) in frame.buffers.planes.iter().zip(sample_owners) {
            assert_eq!(plane.samples.as_ptr(), owner);
        }
    }
}

#[test]
fn existing_consuming_projections_keep_their_previous_results() {
    let native = decode(include_bytes!(
        "plum-blossom-small.profile1.8bpc.yuv444.alpha-full.avif"
    ));
    let old_frame = native.clone().into_frame();
    let (old_frame_and_rich, old_rich) = native.clone().into_frame_and_rich();
    let (frame, information) = native.into_parts();
    assert_eq!(frame, old_frame);
    assert_eq!(frame, old_frame_and_rich);
    assert_eq!(information.rich(), &old_rich);
    assert_eq!(frame.buffers.planes.len(), 4);
}

#[test]
fn legacy_public_literals_and_entry_point_signatures_remain_available() {
    let _rgba8 = ImageBuffer {
        width: 1,
        height: 1,
        rgba: vec![1, 2, 3, 0],
    };
    let _rgba16 = Rgba16ImageBuffer {
        width: 1,
        height: 1,
        rgba: vec![1, 2, 3, 0],
    };
    let frame = DecodedFrame {
        width: 1,
        height: 1,
        render_width: 1,
        render_height: 1,
        bit_depth: 12,
        color_config: ColorConfig {
            high_bitdepth: true,
            twelve_bit: true,
            bit_depth: 12,
            monochrome: true,
            color_description: None,
            color_range: ColorRange::Full,
            subsampling_x: true,
            subsampling_y: true,
            chroma_sample_position: None,
            separate_uv_delta_q: false,
        },
        color_information: None,
        alpha_premultiplied: false,
        buffers: FrameBuffers {
            width: 1,
            height: 1,
            planes: vec![PlaneBuffer {
                layout: PlaneLayout {
                    plane: 0,
                    width: 1,
                    height: 1,
                    subsampling_x: 0,
                    subsampling_y: 0,
                    sample_count: 1,
                },
                samples: vec![4095],
            }],
        },
    };
    assert_eq!(frame.buffers.planes[0].samples, [4095]);
    let _: fn(&[u8]) -> Result<ImageBuffer, DecoderError> = image_from_bytes;
    let _: fn(&[u8]) -> Result<DecodedFrame, DecoderError> = decode_frame_bytes_strict;
    let _: fn(NativeDecodedFrame) -> DecodedFrame = NativeDecodedFrame::into_frame;
    let _: fn(NativeDecodedFrame) -> (DecodedFrame, RichAvifInfo) =
        NativeDecodedFrame::into_frame_and_rich;
    let _: fn(NativeDecodedFrame) -> (DecodedFrame, NativeAvifInformation) =
        NativeDecodedFrame::into_parts;
}
