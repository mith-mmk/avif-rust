use avif_rust::{DecoderError, NativeDecodeLimits, parse_native_info};

fn boxed(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut output = (payload.len() as u32 + 8).to_be_bytes().to_vec();
    output.extend_from_slice(kind);
    output.extend_from_slice(payload);
    output
}

fn limits(input: usize, metadata: usize) -> NativeDecodeLimits {
    NativeDecodeLimits::new(
        input,
        64,
        64,
        4096,
        8192,
        metadata,
        1 << 20,
        32,
        16,
        32,
        8,
        1,
    )
}

fn properties() -> Vec<Vec<u8>> {
    vec![
        boxed(b"ispe", &[0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 16]),
        boxed(b"pixi", &[0, 0, 0, 0, 3, 8, 8, 8]),
        boxed(b"av1C", &[0x81, 0x20, 0, 0]),
    ]
}

fn container(properties: Vec<Vec<u8>>, payload: &[u8], items: u16) -> Vec<u8> {
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
    let mut property_count = 0;
    let mut property_offset = 0;
    while property_offset < ipco.len() {
        let size = u32::from_be_bytes(
            ipco[property_offset..property_offset + 4]
                .try_into()
                .unwrap(),
        ) as usize;
        property_count += 1;
        property_offset += size;
    }
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
    let data_offset = (boxed(b"ftyp", b"avif\0\0\0\0avif").len()
        + 8
        + meta.len()
        + 8
        + 8
        + 14 * usize::from(items)
        + 8) as u32;
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

fn with_alpha(payload: &[u8]) -> Vec<u8> {
    let mut data = container(properties(), payload, 2);
    let mut iref = vec![0, 0, 0, 0];
    iref.extend_from_slice(&boxed(b"auxl", &[0, 1, 0, 1, 0, 2]));
    append_meta_child(&mut data, &boxed(b"iref", &iref));
    data
}

fn append_meta_child(data: &mut Vec<u8>, child: &[u8]) {
    let meta = data.windows(4).position(|value| value == b"meta").unwrap() - 4;
    let size = u32::from_be_bytes(data[meta..meta + 4].try_into().unwrap()) as usize;
    data[meta..meta + 4].copy_from_slice(&(size as u32 + child.len() as u32).to_be_bytes());
    data.splice(meta + size..meta + size, child.iter().copied());
}

#[test]
fn selected_primary_and_alpha_payloads_share_the_native_limit() {
    let data = with_alpha(&[1; 4096]);
    let result = parse_native_info(&data, &limits(data.len(), 1 << 20));
    assert!(matches!(
        result,
        Err(DecoderError::InvalidParam(message))
            if message.contains("payload") && message.contains("limit")
    ));
}

#[test]
fn retained_alpha_auxiliary_type_counts_as_metadata() {
    let data = with_alpha(&[1, 2]);
    let info = parse_native_info(&data, &limits(1 << 20, 1 << 20)).unwrap();
    let auxiliary = &info.info().alpha_auxiliary_items[0];
    let owned = auxiliary.aux_type.capacity()
        + info.info().alpha_auxiliary_items.capacity() * std::mem::size_of_val(auxiliary)
        + std::mem::size_of_val(info.ordered_primary_properties());
    assert!(info.metadata_bytes() >= owned);
}

#[test]
fn repeated_iinf_replacement_releases_parser_owned_names() {
    let mut data = container(properties(), &[1, 2], 1);
    for _ in 0..20 {
        let mut infe = vec![2, 0, 0, 0, 0, 1, 0, 0];
        infe.extend_from_slice(b"av01");
        infe.extend(std::iter::repeat_n(b'x', 512));
        infe.push(0);
        let mut iinf = vec![0, 0, 0, 0, 0, 1];
        iinf.extend_from_slice(&boxed(b"infe", &infe));
        let child = boxed(b"iinf", &iinf);
        let meta = data.windows(4).position(|value| value == b"meta").unwrap() - 4;
        let size = u32::from_be_bytes(data[meta..meta + 4].try_into().unwrap()) as usize;
        data[meta..meta + 4].copy_from_slice(&(size as u32 + child.len() as u32).to_be_bytes());
        data.splice(meta + size..meta + size, child);
    }
    assert!(parse_native_info(&data, &limits(1 << 20, 8192)).is_ok());
}

#[test]
fn repeated_alpha_references_count_unique_selected_payload_owners() {
    let mut data = container(properties(), &[1; 2048], 2);
    let mut iref = vec![0, 0, 0, 0];
    iref.extend_from_slice(&boxed(b"auxl", &[0, 1, 0, 2, 0, 2, 0, 2]));
    append_meta_child(&mut data, &boxed(b"iref", &iref));
    let info = parse_native_info(&data, &limits(1 << 20, 1 << 20)).unwrap();
    assert_eq!(info.info().alpha_auxiliary_items.len(), 1);
    let bounds =
        NativeDecodeLimits::new(4096, 64, 64, 4096, 8192, 1 << 20, 1 << 20, 32, 16, 32, 8, 1);
    let result = parse_native_info(&data, &bounds);
    assert!(
        result.is_ok(),
        "unique selected alpha owner was rejected: {result:?}"
    );
}

#[test]
fn empty_auxl_reference_keeps_auxiliary_property_fallback_bounded() {
    let mut data = container(properties(), &[1; 4096], 2);
    let mut auxc = vec![0, 0, 0, 0];
    auxc.extend_from_slice(b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha\0");
    let mut iprp = boxed(b"ipco", &boxed(b"auxC", &auxc));
    iprp.extend_from_slice(&boxed(b"ipma", &[0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 4]));
    append_meta_child(&mut data, &boxed(b"iprp", &iprp));
    let mut iref = vec![0, 0, 0, 0];
    iref.extend_from_slice(&boxed(b"auxl", &[0, 1, 0, 0]));
    append_meta_child(&mut data, &boxed(b"iref", &iref));
    let info = parse_native_info(&data, &limits(1 << 20, 1 << 20)).unwrap();
    assert_eq!(info.info().alpha_auxiliary_items.len(), 1);
    assert!(matches!(
        parse_native_info(&data, &limits(data.len(), 1 << 20)),
        Err(DecoderError::InvalidParam(message)) if message.contains("payload")
    ));
}

#[test]
fn repeated_iref_and_grpl_replacements_release_nested_storage() {
    for kind in [*b"iref", *b"grpl"] {
        let mut data = container(properties(), &[1, 2], 1);
        for _ in 0..24 {
            let child = if kind == *b"iref" {
                let mut refs = vec![0, 0, 0, 0];
                let mut entry = vec![0, 1, 0, 32];
                for _ in 0..32 {
                    entry.extend_from_slice(&1u16.to_be_bytes());
                }
                refs.extend_from_slice(&boxed(b"zzzz", &entry));
                boxed(b"iref", &refs)
            } else {
                let mut altr = vec![0, 0, 0, 0, 0, 0, 0, 1];
                altr.extend_from_slice(&32u32.to_be_bytes());
                for _ in 0..32 {
                    altr.extend_from_slice(&1u32.to_be_bytes());
                }
                boxed(b"grpl", &boxed(b"altr", &altr))
            };
            append_meta_child(&mut data, &child);
        }
        assert!(parse_native_info(&data, &limits(4096, 4096)).is_ok());
    }
}

#[test]
fn repeated_iloc_and_consumed_ipma_replacements_release_nested_storage() {
    for (kind, budget) in [(*b"iloc", 4096), (*b"ipma", 1024)] {
        let mut data = container(properties(), &[1, 2], 1);
        for _ in 0..24 {
            let child = if kind == *b"iloc" {
                let mut location = vec![0, 0, 0, 0, 0x44, 0, 0, 1, 0, 1, 0, 0, 0, 8];
                location.extend([0; 8 * 8]);
                boxed(b"iloc", &location)
            } else {
                boxed(b"iprp", &boxed(b"ipma", &[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]))
            };
            append_meta_child(&mut data, &child);
        }
        assert!(
            parse_native_info(&data, &limits(budget, 1 << 20)).is_ok(),
            "repeated {kind:?} owners must be released before the next replacement"
        );
    }
}

#[test]
fn consumed_alpha_selection_outer_is_released_before_final_projection() {
    let mut properties = properties();
    let mut icc = b"prof".to_vec();
    icc.extend(std::iter::repeat_n(7, 4096));
    properties.push(boxed(b"colr", &icc));
    let mut data = container(properties, &[1, 2], 2);
    let mut iref = vec![0, 0, 0, 0];
    iref.extend_from_slice(&boxed(b"auxl", &[0, 1, 0, 1, 0, 2]));
    append_meta_child(&mut data, &boxed(b"iref", &iref));
    let high = parse_native_info(&data, &limits(1 << 20, 1 << 20)).unwrap();
    assert_eq!(high.info().alpha_auxiliary_items.len(), 1);
    assert_eq!(
        high.color_information().icc_profile.as_ref().unwrap().len(),
        4096
    );
    // The allocator-observer gate checks the exact peak. This standalone
    // regression keeps the same ICC/alpha ownership fixture under a bounded
    // but non-fragile metadata allowance.
    let result = parse_native_info(&data, &limits(data.len(), high.metadata_bytes() + 8192));
    assert!(
        result.is_ok(),
        "temporary alpha selection storage must be released before retained ICC projection: {result:?}"
    );
}

#[test]
fn legacy_primary_payload_error_precedes_alpha_materialization() {
    let mut data = with_alpha(&[1, 2]);
    let iloc = data.windows(4).position(|value| value == b"iloc").unwrap() - 4;
    data[iloc + 26..iloc + 30].copy_from_slice(&65536_u32.to_be_bytes());
    data[iloc + 36..iloc + 40].copy_from_slice(&u32::MAX.to_be_bytes());
    let result = avif_rust::container::parse_avif(&data);
    assert!(matches!(
        result,
        Err(DecoderError::Bitstream(message))
            if message.contains("item extent payload length exceeds file size")
    ));
}
