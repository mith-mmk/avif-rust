use avif_rust::{NativeDecodeLimits, parse_native_info};

fn boxed(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let size = u32::try_from(payload.len() + 8).expect("test box fits");
    let mut output = Vec::with_capacity(payload.len() + 8);
    output.extend_from_slice(&size.to_be_bytes());
    output.extend_from_slice(kind);
    output.extend_from_slice(payload);
    output
}

fn ipma_with_associations(count: u8) -> Vec<u8> {
    let mut payload = vec![0, 0, 0, 0];
    payload.extend_from_slice(&1u32.to_be_bytes());
    payload.extend_from_slice(&1u16.to_be_bytes());
    payload.push(count);
    payload.extend(std::iter::repeat_n(1u8, usize::from(count)));
    boxed(b"ipma", &payload)
}

fn limits(max_input_bytes: usize, max_items: usize, max_width: usize) -> NativeDecodeLimits {
    NativeDecodeLimits::new(
        max_input_bytes,
        max_width,
        max_width,
        max_width.saturating_mul(max_width),
        usize::MAX,
        usize::MAX,
        usize::MAX,
        max_items,
        usize::MAX,
        usize::MAX,
        usize::MAX,
        usize::MAX,
    )
}

fn ftyp() -> Vec<u8> {
    boxed(b"ftyp", b"avif\0\0\0\0avif")
}

fn association_limits() -> NativeDecodeLimits {
    NativeDecodeLimits::new(
        4096,
        16,
        16,
        256,
        usize::MAX,
        usize::MAX,
        usize::MAX,
        8,
        3,
        usize::MAX,
        usize::MAX,
        usize::MAX,
    )
}

#[test]
fn bounded_parser_rejects_input_before_box_parsing() {
    let data = ftyp();
    let error = parse_native_info(&data, &limits(4, 1, 16)).unwrap_err();
    assert!(
        matches!(error, avif_rust::DecoderError::InvalidParam(message) if message.contains("input size"))
    );
}

#[test]
fn bounded_parser_rejects_iinf_count_before_reserve() {
    let mut iinf_payload = vec![1, 0, 0, 0];
    iinf_payload.extend_from_slice(&u32::MAX.to_be_bytes());
    let iinf = boxed(b"iinf", &iinf_payload);
    let mut meta_payload = vec![0, 0, 0, 0];
    meta_payload.extend_from_slice(&iinf);
    let mut data = ftyp();
    data.extend_from_slice(&boxed(b"meta", &meta_payload));
    let error = parse_native_info(&data, &limits(4096, 8, 16)).unwrap_err();
    assert!(
        matches!(error, avif_rust::DecoderError::InvalidParam(message) if message.contains("item count"))
    );
}

#[test]
fn bounded_parser_rejects_oversized_ispe_before_structured_parse() {
    let mut ispe = vec![0, 0, 0, 0];
    ispe.extend_from_slice(&u32::MAX.to_be_bytes());
    ispe.extend_from_slice(&1u32.to_be_bytes());
    let ipco = boxed(b"ipco", &boxed(b"ispe", &ispe));
    let iprp = boxed(b"iprp", &ipco);
    let mut meta_payload = vec![0, 0, 0, 0];
    meta_payload.extend_from_slice(&iprp);
    let mut data = ftyp();
    data.extend_from_slice(&boxed(b"meta", &meta_payload));
    let error = parse_native_info(&data, &limits(4096, 8, 16)).unwrap_err();
    assert!(
        matches!(error, avif_rust::DecoderError::InvalidParam(message) if message.contains("dimensions"))
    );
}

#[test]
fn bounded_parser_rejects_deep_movie_nesting_without_recursion() {
    let mut nested = Vec::new();
    for _ in 0..70 {
        nested = boxed(b"moov", &nested);
    }
    let mut data = ftyp();
    data.extend_from_slice(&nested);
    let error = parse_native_info(&data, &limits(4096, 8, 16)).unwrap_err();
    assert!(matches!(
        error,
        avif_rust::DecoderError::InvalidParam(message)
            if message.contains("nesting") || message.contains("worklist")
    ));
}

#[test]
fn bounded_parser_accumulates_associations_across_ipma_boxes() {
    let mut iprp_payload = Vec::new();
    for _ in 0..4 {
        iprp_payload.extend_from_slice(&ipma_with_associations(3));
    }
    let iprp = boxed(b"iprp", &iprp_payload);
    let mut meta_payload = vec![0, 0, 0, 0];
    meta_payload.extend_from_slice(&iprp);
    let mut data = ftyp();
    data.extend_from_slice(&boxed(b"meta", &meta_payload));
    let error = parse_native_info(&data, &association_limits()).unwrap_err();
    assert!(matches!(
        error,
        avif_rust::DecoderError::InvalidParam(message)
            if message.contains("property association")
    ));
}
