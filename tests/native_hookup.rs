use std::path::PathBuf;

use avif_rust::container::parse_avif;
use avif_rust::{NativeDecodeLimits, decode_frame_bytes_strict_with_limits};

fn sample_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("test/images/external/avif/unsupported")
        .join(name)
}

fn bundled_sample_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("samples")
        .join(name)
}

fn native_limits(input_len: usize) -> NativeDecodeLimits {
    NativeDecodeLimits::new(
        input_len.saturating_mul(2),
        8192,
        8192,
        1 << 28,
        1 << 30,
        1 << 24,
        1 << 24,
        128,
        256,
        256,
        8,
        8,
    )
}

#[test]
fn native_still_retains_a_displayed_master_frame() {
    let data = std::fs::read(bundled_sample_path("WML2Viewer.avif"))
        .expect("native hookup fixture must be present");
    let info = parse_avif(&data).expect("native hookup fixture must parse");
    assert!(info.alpha_auxiliary_items.is_empty());

    let decoded = decode_frame_bytes_strict_with_limits(&data, &native_limits(data.len()))
        .expect("displayed native still must decode");
    assert!(decoded.frame().width > 0);
    assert!(decoded.frame().height > 0);
}

#[test]
fn native_still_decodes_selected_alpha_after_header_preflight() {
    let data = std::fs::read(sample_path(
        "plum-blossom-small.profile1.8bpc.yuv444.alpha-full.avif",
    ))
    .expect("native alpha hookup fixture must be present");
    let info = parse_avif(&data).expect("native alpha hookup fixture must parse");
    assert_eq!(info.alpha_auxiliary_items.len(), 1);

    let decoded = decode_frame_bytes_strict_with_limits(&data, &native_limits(data.len()))
        .expect("native alpha still must decode after both header checks");
    let alpha = decoded
        .frame()
        .buffers
        .planes
        .iter()
        .find(|plane| plane.layout.plane == 3)
        .expect("native alpha plane must be retained");
    assert_eq!(alpha.layout.width, decoded.frame().width);
    assert_eq!(alpha.layout.height, decoded.frame().height);
    assert!(!alpha.samples.is_empty());
}

fn boxed(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut output = u32::try_from(payload.len() + 8)
        .unwrap()
        .to_be_bytes()
        .to_vec();
    output.extend_from_slice(kind);
    output.extend_from_slice(payload);
    output
}

fn native_container(properties: Vec<Vec<u8>>, payload: &[u8], items: u16) -> Vec<u8> {
    let ftyp = boxed(b"ftyp", b"avif\0\0\0\0avif");
    let mut iinf = vec![0, 0, 0, 0];
    iinf.extend_from_slice(&items.to_be_bytes());
    for id in 1..=items {
        let mut infe = vec![2, 0, 0, 0];
        infe.extend_from_slice(&id.to_be_bytes());
        infe.extend_from_slice(&[0, 0]);
        infe.extend_from_slice(b"av01");
        infe.push(0);
        iinf.extend_from_slice(&boxed(b"infe", &infe));
    }

    let property_count = properties.len();
    let mut iprp_payload = properties.into_iter().flatten().collect::<Vec<_>>();
    let ipco = boxed(b"ipco", &iprp_payload);
    iprp_payload.clear();
    iprp_payload.extend_from_slice(&ipco);
    for id in 1..=items {
        let mut ipma = vec![0, 0, 0, 0, 0, 0, 0, 1];
        ipma.extend_from_slice(&id.to_be_bytes());
        ipma.push(u8::try_from(property_count).unwrap());
        ipma.extend((1..=property_count).map(|index| u8::try_from(index).unwrap()));
        iprp_payload.extend_from_slice(&boxed(b"ipma", &ipma));
    }

    let mut meta = vec![0, 0, 0, 0];
    meta.extend_from_slice(&boxed(b"pitm", &[0, 0, 0, 0, 0, 1]));
    meta.extend_from_slice(&boxed(b"iinf", &iinf));
    meta.extend_from_slice(&boxed(b"iprp", &iprp_payload));

    let data_offset = ftyp.len() + 8 + meta.len() + (8 + 8 + 14 * usize::from(items)) + 8;
    let mut iloc = vec![0, 0, 0, 0, 0x44, 0];
    iloc.extend_from_slice(&items.to_be_bytes());
    for id in 1..=items {
        iloc.extend_from_slice(&id.to_be_bytes());
        iloc.extend_from_slice(&[0, 0, 0, 1]);
        iloc.extend_from_slice(&u32::try_from(data_offset).unwrap().to_be_bytes());
        iloc.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_be_bytes());
    }
    meta.extend_from_slice(&boxed(b"iloc", &iloc));

    let mut file = ftyp;
    file.extend_from_slice(&boxed(b"meta", &meta));
    file.extend_from_slice(&boxed(b"mdat", payload));
    file
}

fn box_start(data: &[u8], kind: &[u8; 4]) -> usize {
    data.windows(4)
        .position(|value| value == kind)
        .expect("synthetic box must be present")
        - 4
}

fn append_meta_child(data: &mut Vec<u8>, child: &[u8]) {
    let meta = box_start(data, b"meta");
    let meta_size = u32::from_be_bytes(data[meta..meta + 4].try_into().unwrap()) as usize;
    let iloc = box_start(data, b"iloc");
    let item_count = u16::from_be_bytes(data[iloc + 14..iloc + 16].try_into().unwrap()) as usize;
    for item_index in 0..item_count {
        let offset = iloc + 22 + item_index * 14;
        let value = u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap());
        data[offset..offset + 4].copy_from_slice(
            &value
                .checked_add(u32::try_from(child.len()).unwrap())
                .unwrap()
                .to_be_bytes(),
        );
    }
    data[meta..meta + 4].copy_from_slice(
        &u32::try_from(meta_size + child.len())
            .unwrap()
            .to_be_bytes(),
    );
    data.splice(meta + meta_size..meta + meta_size, child.iter().copied());
}

fn append_obu(output: &mut Vec<u8>, obu_kind: u8, payload: &[u8]) {
    output.push((obu_kind << 3) | 2);
    let mut size = payload.len();
    loop {
        let next = size >> 7;
        output.push((size & 127) as u8 | if next == 0 { 0 } else { 128 });
        size = next;
        if size == 0 {
            break;
        }
    }
    output.extend_from_slice(payload);
}

fn obu_start_for_payload(stream: &[u8], payload: &[u8]) -> usize {
    let payload_offset = payload.as_ptr() as usize - stream.as_ptr() as usize;
    for start in (payload_offset.saturating_sub(10)..payload_offset).rev() {
        let Ok(obus) = avif_rust::obu::parse_obu_stream(&stream[start..]) else {
            continue;
        };
        if obus
            .first()
            .is_some_and(|obu| obu.payload.as_ptr() == payload.as_ptr())
        {
            return start;
        }
    }
    panic!("OBU header for the selected frame payload must be present");
}

fn native_container_with_alpha(primary: &[u8], alpha: &[u8]) -> Vec<u8> {
    let sequence_payload =
        avif_rust::obu::find_obu_payload(primary, avif_rust::obu::ObuType::SequenceHeader)
            .unwrap()
            .unwrap();
    let sequence = avif_rust::av1::parse_sequence_header(sequence_payload).unwrap();
    let frame_payload = avif_rust::obu::find_obu_payload(primary, avif_rust::obu::ObuType::Frame)
        .unwrap()
        .or(
            avif_rust::obu::find_obu_payload(primary, avif_rust::obu::ObuType::FrameHeader)
                .unwrap(),
        )
        .unwrap();
    let frame = avif_rust::av1::parse_frame_header(frame_payload, &sequence).unwrap();
    let mut ispe = vec![0, 0, 0, 0];
    ispe.extend_from_slice(&frame.upscaled_width.to_be_bytes());
    ispe.extend_from_slice(&frame.frame_height.to_be_bytes());
    let channels = if sequence.color_config.monochrome {
        1
    } else {
        3
    };
    let mut pixi = vec![0, 0, 0, 0, channels];
    pixi.resize(5 + channels as usize, sequence.color_config.bit_depth);
    let flags = ((sequence.color_config.high_bitdepth as u8) << 6)
        | ((sequence.color_config.twelve_bit as u8) << 5)
        | ((sequence.color_config.monochrome as u8) << 4)
        | ((sequence.color_config.subsampling_x as u8) << 3)
        | ((sequence.color_config.subsampling_y as u8) << 2)
        | sequence
            .color_config
            .chroma_sample_position
            .map_or(0, |value| value as u8);
    let config = [
        0x81,
        sequence.seq_profile << 5 | sequence.seq_level_idx_0,
        flags,
        0,
    ];
    let mut data = native_container(
        vec![
            boxed(b"ispe", &ispe),
            boxed(b"pixi", &pixi),
            boxed(b"av1C", &config),
        ],
        primary,
        2,
    );
    let mut iref = vec![0, 0, 0, 0];
    iref.extend_from_slice(&boxed(b"auxl", &[0, 1, 0, 1, 0, 2]));
    append_meta_child(&mut data, &boxed(b"iref", &iref));

    let iloc = box_start(&data, b"iloc");
    let offset = iloc + 22 + 14;
    let alpha_offset = data.len() + 8;
    data[offset..offset + 4].copy_from_slice(&u32::try_from(alpha_offset).unwrap().to_be_bytes());
    data[offset + 4..offset + 8]
        .copy_from_slice(&u32::try_from(alpha.len()).unwrap().to_be_bytes());
    data.extend_from_slice(&boxed(b"mdat", alpha));
    data
}

#[test]
fn native_rejects_an_extra_alpha_frame() {
    let data = std::fs::read(sample_path(
        "plum-blossom-small.profile1.8bpc.yuv444.alpha-full.avif",
    ))
    .expect("native alpha hookup fixture must be present");
    let info = parse_avif(&data).expect("native alpha hookup fixture must parse");
    let primary = &info.primary_item_payload;
    let alpha = &info.alpha_auxiliary_items[0].payload;
    let valid = native_container_with_alpha(primary, alpha);
    let valid_decoded = decode_frame_bytes_strict_with_limits(&valid, &native_limits(valid.len()))
        .expect("the unmodified synthetic alpha container must decode");
    assert!(
        valid_decoded
            .frame()
            .buffers
            .planes
            .iter()
            .any(|plane| plane.layout.plane == 3)
    );
    let (obu_kind, frame) = if let Some(frame) =
        avif_rust::obu::find_obu_payload(alpha, avif_rust::obu::ObuType::Frame).unwrap()
    {
        (6, frame)
    } else {
        (
            3,
            avif_rust::obu::find_obu_payload(alpha, avif_rust::obu::ObuType::FrameHeader)
                .unwrap()
                .expect("alpha frame must be present"),
        )
    };
    let frame_start = obu_start_for_payload(alpha, frame);
    let mut duplicate = Vec::new();
    append_obu(&mut duplicate, obu_kind, frame);
    let mut invalid_alpha = alpha.to_vec();
    invalid_alpha.splice(frame_start..frame_start, duplicate);
    let invalid = native_container_with_alpha(primary, &invalid_alpha);
    let parsed = avif_rust::parse_native_info(&invalid, &native_limits(invalid.len()))
        .expect("synthetic multiple-alpha container must parse");
    assert_eq!(
        parsed.info().alpha_auxiliary_items[0].payload,
        invalid_alpha,
        "synthetic alpha extent must retain the appended frame header"
    );

    let error = decode_frame_bytes_strict_with_limits(&invalid, &native_limits(invalid.len()))
        .expect_err("a second alpha frame must be rejected");
    // The independent native-hookup boundary harness observes zero master
    // tile copies for this same fixture; this tracked test fixes the parser
    // regression and the valid-container positive control without installing
    // a second global allocator in this integration crate.
    assert!(
        matches!(error, avif_rust::DecoderError::Unsupported(ref message) if message.contains("alpha")),
        "unexpected error for multiple alpha frames: {error:?}"
    );
}
