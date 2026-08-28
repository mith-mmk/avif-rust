use avif_rust::{NativeDecodeLimits, NativePropertyRecord, parse_native_info};

fn boxed(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut output = (u32::try_from(payload.len() + 8).unwrap())
        .to_be_bytes()
        .to_vec();
    output.extend_from_slice(kind);
    output.extend_from_slice(payload);
    output
}

fn limits(metadata: usize, properties: usize, frames: usize) -> NativeDecodeLimits {
    NativeDecodeLimits::new(
        1 << 20,
        64,
        64,
        4096,
        8192,
        metadata,
        1 << 20,
        32,
        properties,
        32,
        8,
        frames,
    )
}

fn properties() -> Vec<Vec<u8>> {
    vec![
        boxed(b"ispe", &[0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 16]),
        boxed(b"pixi", &[0, 0, 0, 0, 3, 8, 8, 8]),
        boxed(b"av1C", &[0x81, 0x20, 0, 0]),
    ]
}

fn avif_container(properties: Vec<Vec<u8>>, payload: &[u8], items: u16) -> Vec<u8> {
    let mut meta = vec![0, 0, 0, 0];
    meta.extend_from_slice(&boxed(b"pitm", &[0, 0, 0, 0, 0, 1]));
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
    meta.extend_from_slice(&boxed(b"iinf", &iinf));

    let mut ipco = Vec::new();
    for property in properties {
        ipco.extend_from_slice(&property);
    }
    let property_count = ipco_property_count(&ipco);
    let mut iprp = boxed(b"ipco", &ipco);
    for id in 1..=items {
        let mut ipma = vec![0, 0, 0, 0, 0, 0, 0, 1];
        ipma.extend_from_slice(&id.to_be_bytes());
        ipma.push(property_count as u8);
        ipma.extend((1..=property_count).map(|index| index as u8));
        iprp.extend_from_slice(&boxed(b"ipma", &ipma));
    }
    meta.extend_from_slice(&boxed(b"iprp", &iprp));

    let mut iloc = vec![0, 0, 0, 0, 0x44, 0];
    iloc.extend_from_slice(&items.to_be_bytes());
    let iloc_box_size = 8usize
        .checked_add(8usize.checked_add(14usize * usize::from(items)).unwrap())
        .unwrap();
    let ftyp_size = boxed(b"ftyp", b"avif\0\0\0\0avif").len();
    let data_offset = u32::try_from(ftyp_size + 8 + meta.len() + iloc_box_size + 8).unwrap();
    for id in 1..=items {
        iloc.extend_from_slice(&id.to_be_bytes());
        iloc.extend_from_slice(&[0, 0, 0, 1]);
        iloc.extend_from_slice(&data_offset.to_be_bytes());
        iloc.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    }
    meta.extend_from_slice(&boxed(b"iloc", &iloc));

    let mut data = boxed(b"ftyp", b"avif\0\0\0\0avif");
    data.extend_from_slice(&boxed(b"meta", &meta));
    data.extend_from_slice(&boxed(b"mdat", payload));
    data
}

fn ipco_property_count(payload: &[u8]) -> usize {
    let mut offset = 0;
    let mut count = 0;
    while offset < payload.len() {
        let size = u32::from_be_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
        count += 1;
        offset += size;
    }
    count
}

fn box_start(data: &[u8], kind: &[u8; 4]) -> usize {
    data.windows(4).position(|value| value == kind).unwrap() - 4
}

fn append_meta_child(data: &mut Vec<u8>, child: &[u8]) {
    let meta = box_start(data, b"meta");
    let meta_size = u32::from_be_bytes(data[meta..meta + 4].try_into().unwrap()) as usize;
    data[meta..meta + 4].copy_from_slice(
        &u32::try_from(meta_size + child.len())
            .unwrap()
            .to_be_bytes(),
    );
    data.splice(meta + meta_size..meta + meta_size, child.iter().copied());
}

fn vec_bytes<T>(value: &Vec<T>) -> usize {
    value.capacity() * std::mem::size_of::<T>()
}

fn pixi_bytes(value: &avif_rust::PixelInformation) -> usize {
    vec_bytes(&value.bits_per_channel) + value.extended_channels.as_ref().map_or(0, vec_bytes)
}

#[test]
fn aggregate_owned_metadata_copies_are_bounded() {
    let mut properties = properties();
    let mut icc = b"prof".to_vec();
    icc.resize(4100, 1);
    properties.push(boxed(b"colr", &icc));
    let data = avif_container(properties, &[0x12, 0], 1);
    assert!(parse_native_info(&data, &limits(data.len(), 8, 1)).is_err());
    let info = parse_native_info(&data, &limits(data.len() * 4, 8, 1)).unwrap();
    let retained = info
        .info()
        .color_information
        .as_ref()
        .unwrap()
        .payload
        .len()
        + info.color_information().icc_profile.as_ref().unwrap().len()
        + info
            .ordered_primary_properties()
            .iter()
            .map(|property| match property {
                NativePropertyRecord::ColorInformation(color) => color.payload.len(),
                _ => 0,
            })
            .sum::<usize>();
    assert!(retained <= data.len() * 4);
    assert!(info.metadata_bytes() >= retained);
}

#[test]
fn all_ipma_associations_count_toward_the_property_limit() {
    let data = avif_container(properties(), &[0x12, 0], 4);
    assert!(parse_native_info(&data, &limits(1 << 20, 3, 1)).is_err());
}

#[test]
fn still_metadata_accepts_zero_movie_samples_at_zero_frame_limit() {
    let data = avif_container(properties(), &[0x12, 0], 1);
    assert!(parse_native_info(&data, &limits(1 << 20, 8, 0)).is_ok());
}

#[test]
fn reduced_still_disable_cdf_is_not_show_existing() {
    let sequence = [0x38, 0x0c, 0xff, 0xd8, 0x40, 0x43, 0x40, 0x08];
    let frame = [0xc4, 0x00, 0x00, 0xc2];
    let parsed = avif_rust::av1::parse_sequence_header(&sequence).unwrap();
    let header = avif_rust::av1::parse_frame_header(&frame, &parsed).unwrap();
    assert!(parsed.reduced_still_picture_header && header.disable_cdf_update);
    assert!(!header.show_existing_frame);
    let mut payload = vec![0x0a, sequence.len() as u8];
    payload.extend_from_slice(&sequence);
    payload.extend_from_slice(&[0x32, frame.len() as u8]);
    payload.extend_from_slice(&frame);
    let data = avif_container(properties(), &payload, 1);
    let result = avif_rust::decode_frame_bytes_strict_with_limits(&data, &limits(1 << 20, 8, 1));
    assert!(matches!(
        result,
        Err(avif_rust::DecoderError::NotEnoughData(message))
            if message.contains("entropy tile payload is too short")
    ));
}

#[test]
fn native_prefix_rejects_plane_limit_before_decode_materialization() {
    let sequence = [0x38, 0x0c, 0xff, 0xd8, 0x40, 0x43, 0x40, 0x08];
    let frame = [0xc4, 0x00, 0x00, 0xc2];
    let parsed = avif_rust::av1::parse_sequence_header(&sequence).unwrap();
    let header = avif_rust::av1::parse_frame_header(&frame, &parsed).unwrap();
    let mut payload = vec![0x0a, sequence.len() as u8];
    payload.extend_from_slice(&sequence);
    payload.extend_from_slice(&[0x32, 0x10]);
    payload.extend(std::iter::repeat_n(0, 16));
    let data = avif_container(properties(), &payload, 1);
    let result = avif_rust::decode_frame_bytes_strict_with_limits(
        &data,
        &NativeDecodeLimits::new(
            1 << 20,
            header.frame_width as usize,
            header.frame_height as usize,
            1 << 24,
            1,
            1 << 20,
            1 << 20,
            32,
            32,
            32,
            8,
            1,
        ),
    );
    assert!(matches!(
        result,
        Err(avif_rust::DecoderError::InvalidParam(message))
            if message.contains("plane") && message.contains("limit")
    ));
}

#[test]
fn native_context_rejects_movie_before_sequence_materialization() {
    let mut data = avif_container(properties(), &[0x12, 0], 1);
    data.extend_from_slice(&boxed(b"moov", &[]));
    let error = avif_rust::parse_native_info(&data, &limits(1 << 20, 8, 1)).unwrap_err();
    assert!(matches!(
        error,
        avif_rust::DecoderError::Unsupported(message)
            if message.contains("movie containers")
    ));
}

#[test]
fn native_derived_primary_rejects_before_payload_materialization() {
    let mut data = avif_container(properties(), &[1; 4096], 1);
    let kind = data.windows(4).position(|value| value == b"av01").unwrap();
    data[kind..kind + 4].copy_from_slice(b"grid");
    let error = parse_native_info(&data, &limits(1 << 20, 16, 10)).unwrap_err();
    assert!(
        matches!(error, avif_rust::DecoderError::Unsupported(message) if message.contains("primary av01") || message.contains("derived"))
    );
}

#[test]
fn native_derived_alpha_rejects_before_master_payload() {
    let mut data = avif_container(properties(), &[1; 4096], 2);
    let kind = data
        .windows(4)
        .enumerate()
        .filter(|(_, value)| *value == b"av01")
        .nth(1)
        .unwrap()
        .0;
    data[kind..kind + 4].copy_from_slice(b"grid");
    let mut iref = vec![0, 0, 0, 0];
    iref.extend_from_slice(&boxed(b"auxl", &[0, 1, 0, 1, 0, 2]));
    append_meta_child(&mut data, &boxed(b"iref", &iref));
    let error = parse_native_info(&data, &limits(1 << 20, 16, 10)).unwrap_err();
    assert!(
        matches!(error, avif_rust::DecoderError::Unsupported(message) if message.contains("alpha item"))
    );
}

#[test]
fn native_invalid_extent_is_rejected_before_payload_reserve() {
    let mut data = avif_container(properties(), &[1, 2], 1);
    let iloc = box_start(&data, b"iloc");
    data[iloc + 24..iloc + 28].copy_from_slice(&u32::MAX.to_be_bytes());
    data[iloc + 28..iloc + 32].copy_from_slice(&65_536u32.to_be_bytes());
    assert!(parse_native_info(&data, &limits(1 << 20, 16, 10)).is_err());
}

#[test]
fn metadata_report_covers_owned_projection_storage() {
    let mut property_boxes = properties();
    let mut icc = b"prof".to_vec();
    icc.resize(4100, 1);
    property_boxes.push(boxed(b"colr", &icc));
    let data = avif_container(property_boxes, &[0x12, 0], 1);
    let info = parse_native_info(&data, &limits(1 << 20, 8, 1)).unwrap();
    let legacy = info.info();
    let rich = info.color_information();
    let mut owned = vec_bytes(&legacy.compatible_brands)
        + legacy.pixel_information.as_ref().map_or(0, pixi_bytes)
        + legacy
            .color_information
            .as_ref()
            .map_or(0, |color| vec_bytes(&color.payload))
        + legacy.av1_config.as_ref().map_or(0, vec_bytes)
        + vec_bytes(&legacy.alpha_auxiliary_items)
        + vec_bytes(&legacy.sequence_sample_payloads)
        + rich.icc_profile.as_ref().map_or(0, vec_bytes)
        + vec_bytes(&rich.unknown_colr)
        + rich
            .unknown_colr
            .iter()
            .map(|color| vec_bytes(&color.payload))
            .sum::<usize>()
        + std::mem::size_of_val(info.ordered_primary_properties());
    for property in info.ordered_primary_properties() {
        owned += match property {
            NativePropertyRecord::AuxiliaryType(value) => value.capacity(),
            NativePropertyRecord::PixelInformation(value) => pixi_bytes(value),
            NativePropertyRecord::Av1Config(value) => vec_bytes(value),
            NativePropertyRecord::ColorInformation(value) => vec_bytes(&value.payload),
            _ => 0,
        };
    }
    assert!(info.metadata_bytes() >= owned);
}

#[test]
fn pasp_and_unknown_properties_keep_source_order() {
    let mut property_boxes = properties();
    property_boxes.push(boxed(b"pasp", &[0, 0, 0, 4, 0, 0, 0, 3]));
    property_boxes.push(boxed(b"zzzz", &[4, 3, 2, 1]));
    let data = avif_container(property_boxes, &[0x12, 0], 1);
    let info = parse_native_info(&data, &limits(1 << 20, 8, 1)).unwrap();
    assert!(matches!(
        info.ordered_primary_properties()[3],
        NativePropertyRecord::PixelAspectRatio(value) if value.h_spacing == 4 && value.v_spacing == 3
    ));
    assert_eq!(
        info.ordered_primary_properties()[4],
        NativePropertyRecord::Other(*b"zzzz")
    );
}
