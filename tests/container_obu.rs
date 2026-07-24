mod support;

use std::{
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

use avif_rust::DecoderError;
use avif_rust::compat::{
    CallbackResponse, DataMap, DecodeOptions, DrawCallback, DrawOptions, InitOptions,
    TerminateOptions, VerboseOptions,
};
use avif_rust::container::{
    AvifSequenceSampleKind, classify_av1_sequence_sample, is_avif_file, parse_avif,
};
use avif_rust::obu::{ObuType, count_obus, find_obu_payload, find_obu_payloads, parse_obu_stream};
use bin_rs::reader::BytesReader;
use support::sample_path;

static NEXT_AVIS_SAMPLE: AtomicUsize = AtomicUsize::new(0);

fn sample_avif() -> Vec<u8> {
    std::fs::read(sample_path("WML2Viewer.avif")).expect("sample AVIF should exist")
}

fn external_star_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist")
        .join("test/images/external/avif/unsupported/star-8bpc.avifs")
}

#[derive(Default)]
struct RecordingDrawer {
    init: Option<(usize, usize, InitOptions)>,
    draw_buffers: Vec<Vec<u8>>,
    terminated: bool,
}

impl DrawCallback for RecordingDrawer {
    fn init(
        &mut self,
        width: usize,
        height: usize,
        option: Option<InitOptions>,
    ) -> Result<Option<CallbackResponse>, Box<dyn std::error::Error>> {
        self.init = Some((
            width,
            height,
            option.expect("AVIS callback should provide init options"),
        ));
        Ok(Some(CallbackResponse::cont()))
    }

    fn draw(
        &mut self,
        _start_x: usize,
        _start_y: usize,
        _width: usize,
        _height: usize,
        data: &[u8],
        _option: Option<DrawOptions>,
    ) -> Result<Option<CallbackResponse>, Box<dyn std::error::Error>> {
        self.draw_buffers.push(data.to_vec());
        Ok(Some(CallbackResponse::cont()))
    }

    fn terminate(
        &mut self,
        _term: Option<TerminateOptions>,
    ) -> Result<Option<CallbackResponse>, Box<dyn std::error::Error>> {
        self.terminated = true;
        Ok(Some(CallbackResponse::cont()))
    }

    fn verbose(
        &mut self,
        _verbose: &str,
        _option: Option<VerboseOptions>,
    ) -> Result<Option<CallbackResponse>, Box<dyn std::error::Error>> {
        Ok(Some(CallbackResponse::cont()))
    }

    fn set_metadata(
        &mut self,
        _key: &str,
        _value: DataMap,
    ) -> Result<Option<CallbackResponse>, Box<dyn std::error::Error>> {
        Ok(Some(CallbackResponse::cont()))
    }
}

fn generated_all_key_avis_sample() -> Option<Vec<u8>> {
    let root = std::env::temp_dir().join(format!(
        ".test-avif-avis-sequence-api-{}-{}",
        std::process::id(),
        NEXT_AVIS_SAMPLE.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&root).expect("temporary AVIS API directory should be creatable");
    let output = root.join("all-key.avifs");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "color=c=red:size=64x64:rate=1"])
        .args(["-frames:v", "4", "-c:v", "libaom-av1"])
        .args([
            "-still-picture",
            "0",
            "-g",
            "1",
            "-lag-in-frames",
            "0",
            "-auto-alt-ref",
            "0",
            "-f",
            "avif",
        ])
        .arg(&output)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping sequence API sample");
        let _ = std::fs::remove_dir_all(&root);
        return None;
    };
    if !status.success() {
        eprintln!("libaom AVIS encoder is unavailable; skipping sequence API sample");
        let _ = std::fs::remove_dir_all(&root);
        return None;
    }
    let data = std::fs::read(&output).expect("all-key AVIS sample should be readable");
    let _ = std::fs::remove_dir_all(&root);
    Some(data)
}

fn generated_film_grain_avis_sample() -> Option<Vec<u8>> {
    let root = std::env::temp_dir().join(format!(
        ".test-avif-film-grain-{}-{}",
        std::process::id(),
        NEXT_AVIS_SAMPLE.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&root)
        .expect("temporary film-grain AVIS directory should be creatable");
    let output = root.join("film-grain.avifs");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "color=c=gray:size=64x64:rate=2"])
        .args(["-frames:v", "3", "-c:v", "libaom-av1"])
        .args([
            "-still-picture",
            "0",
            "-g",
            "3",
            "-lag-in-frames",
            "0",
            "-auto-alt-ref",
            "0",
            "-aom-params",
            "film-grain-test=1",
            "-f",
            "avif",
        ])
        .arg(&output)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated film-grain sample");
        let _ = std::fs::remove_dir_all(&root);
        return None;
    };
    if !status.success() {
        eprintln!("libaom film-grain encoder option is unavailable; skipping sample");
        let _ = std::fs::remove_dir_all(&root);
        return None;
    }
    let data = std::fs::read(&output).expect("generated film-grain AVIS should be readable");
    let _ = std::fs::remove_dir_all(&root);
    Some(data)
}

#[test]
fn sample_container_metadata_is_exposed_through_public_api() {
    let data = sample_avif();
    let info = parse_avif(&data).unwrap();

    assert!(is_avif_file(&data));
    assert!(info.is_avif_brand());
    assert_eq!(info.width, Some(900));
    assert_eq!(info.height, Some(900));
    assert_eq!(
        info.pixel_information
            .as_ref()
            .map(|pixi| pixi.bits_per_channel.as_slice()),
        Some(&[8, 8, 8][..])
    );
    assert!(!info.primary_item_payload.is_empty());
}

#[test]
fn official_extended_pixi_sample_exposes_channel_descriptors() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist")
        .join("test/images/external/avif/unsupported/extended_pixi.avif");
    if !path.is_file() {
        eprintln!("extended pixi sample is unavailable; skipping metadata check");
        return;
    }
    let data = std::fs::read(path).expect("extended pixi sample should be readable");
    let info = parse_avif(&data).expect("extended pixi sample should parse");
    let pixi = info
        .pixel_information
        .expect("extended pixi sample should expose pixi");
    let channels = pixi
        .extended_channels
        .expect("extended pixi sample should expose channel descriptors");
    assert_eq!(channels.len(), 3);
    assert_eq!(
        channels[0].subsampling,
        Some(avif_rust::container::PixelSubsampling {
            subsampling_type: 0,
            subsampling_location: 0,
        })
    );
    assert!(channels[1..].iter().all(|channel| channel.subsampling
        == Some(avif_rust::container::PixelSubsampling {
            subsampling_type: 2,
            subsampling_location: 0,
        })));
    let image = avif_rust::image_from_bytes(&data).expect("extended pixi image should decode");
    assert_eq!((image.width, image.height), (4, 4));
}

#[test]
fn avio_major_brand_is_accepted_for_intra_only_image_items() {
    let mut data = sample_avif();
    assert_eq!(&data[4..8], b"ftyp");
    data[8..12].copy_from_slice(b"avio");

    assert!(is_avif_file(&data));
    let info = parse_avif(&data).expect("AVIO-branded image should parse");
    assert_eq!(&info.major_brand, b"avio");
    assert_eq!(info.sequence_sample_payloads.len(), 0);
    let frame = avif_rust::decode_frame_bytes(&data).expect("AVIO image should decode");
    assert_eq!((frame.width, frame.height), (900, 900));
}

#[test]
fn public_obu_helpers_find_sample_frame_payloads() {
    let data = sample_avif();
    let info = parse_avif(&data).unwrap();
    let obus = parse_obu_stream(&info.primary_item_payload).unwrap();

    assert!(
        obus.iter()
            .any(|obu| matches!(obu.obu_type, ObuType::SequenceHeader))
    );
    assert!(
        obus.iter()
            .any(|obu| matches!(obu.obu_type, ObuType::Frame | ObuType::FrameHeader))
    );
    assert!(
        obus.iter().all(
            |obu| !obu.payload.is_empty() || matches!(obu.obu_type, ObuType::TemporalDelimiter)
        )
    );

    let payload = find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader)
        .unwrap()
        .expect("sequence header OBU should exist");
    assert!(!payload.is_empty());

    let [sequence, frame] = find_obu_payloads(
        &info.primary_item_payload,
        [ObuType::SequenceHeader, ObuType::Frame],
    )
    .unwrap();
    assert!(sequence.is_some());
    assert!(frame.is_some());
}

#[test]
fn public_obu_helpers_count_repeated_types() {
    let data = [
        0x22, 0x01, 0xaa, // tile group, one-byte payload
        0x0a, 0x01, 0xbb, // sequence header, one-byte payload
        0x22, 0x02, 0xcc, 0xdd, // tile group, two-byte payload
    ];

    assert_eq!(count_obus(&data, ObuType::TileGroup).unwrap(), 2);
    assert_eq!(count_obus(&data, ObuType::SequenceHeader).unwrap(), 1);
    assert_eq!(count_obus(&data, ObuType::Frame).unwrap(), 0);
}

#[test]
fn avis_container_exposes_track_samples_without_concatenating_obus() {
    let root = std::env::temp_dir().join(format!(".test-avif-avis-{}", std::process::id()));
    std::fs::create_dir_all(&root).expect("temporary AVIS directory should be creatable");
    let output = root.join("sequence.avif");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "testsrc2=size=64x64:rate=4"])
        .args([
            "-frames:v",
            "8",
            "-c:v",
            "libaom-av1",
            "-still-picture",
            "0",
            "-g",
            "8",
            "-lag-in-frames",
            "8",
            "-auto-alt-ref",
            "1",
            "-f",
            "avif",
        ])
        .arg(&output)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated AVIS sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom AVIS encoder is unavailable; skipping generated sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output).expect("generated AVIS sample should be readable");
    let info = parse_avif(&data).expect("generated AVIS metadata should parse");
    assert_eq!(&info.major_brand, b"avis");
    assert_eq!(info.sequence_sample_payloads.len(), 8);
    assert_eq!(info.sequence_sample_payloads[0], info.primary_item_payload);
    let kinds: Vec<_> = info
        .sequence_sample_payloads
        .iter()
        .map(|payload| classify_av1_sequence_sample(payload).unwrap())
        .collect();
    assert!(matches!(
        kinds.first(),
        Some(Some(AvifSequenceSampleKind::Key))
    ));
    assert!(
        kinds
            .iter()
            .skip(1)
            .any(|kind| matches!(kind, Some(AvifSequenceSampleKind::Inter)))
    );
    let frame = avif_rust::decode_frame_bytes(&data).expect("primary AVIS frame should decode");
    assert_eq!((frame.width, frame.height), (64, 64));
    for payload in &info.sequence_sample_payloads {
        assert!(
            !parse_obu_stream(payload)
                .expect("AVIS sample should be an independently framed OBU stream")
                .is_empty()
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_avis_exposes_inter_and_show_existing_reference_samples() {
    let root =
        std::env::temp_dir().join(format!(".test-avif-avis-reference-{}", std::process::id()));
    std::fs::create_dir_all(&root).expect("temporary AVIS reference directory should be creatable");
    let output = root.join("reference-sequence.avif");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "testsrc2=size=64x64:rate=30"])
        .args(["-frames:v", "60", "-c:v", "libaom-av1"])
        .args([
            "-still-picture",
            "0",
            "-g",
            "60",
            "-lag-in-frames",
            "25",
            "-auto-alt-ref",
            "1",
            "-f",
            "avif",
        ])
        .arg(&output)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated AVIS reference sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom AVIS encoder is unavailable; skipping reference sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output).expect("generated AVIS reference sample should be readable");
    let info = parse_avif(&data).expect("generated AVIS reference metadata should parse");
    assert_eq!(&info.major_brand, b"avis");
    assert_eq!(info.sequence_sample_payloads.len(), 60);
    let kinds: Vec<_> = info
        .sequence_sample_payloads
        .iter()
        .map(|payload| classify_av1_sequence_sample(payload).unwrap())
        .collect();
    assert!(matches!(
        kinds.first(),
        Some(Some(AvifSequenceSampleKind::Key))
    ));
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, Some(AvifSequenceSampleKind::Inter)))
    );
    assert!(
        kinds
            .iter()
            .any(|kind| { matches!(kind, Some(AvifSequenceSampleKind::ShowExisting { .. })) })
    );
    let show_existing_index = kinds
        .iter()
        .position(|kind| matches!(kind, Some(AvifSequenceSampleKind::ShowExisting { .. })))
        .expect("generated AVIS reference sample should expose a show-existing index");
    let frames = avif_rust::decode_sequence_frames_bytes(&data)
        .expect("generated AVIS reference sample should decode all frames");
    assert_eq!(frames.len(), info.sequence_sample_payloads.len());
    let shown = avif_rust::decode_sequence_frame_bytes(&data, show_existing_index)
        .expect("generated AVIS show-existing sample should decode");
    assert_eq!((shown.width, shown.height), (64, 64));
    let frame = avif_rust::decode_frame_bytes(&data)
        .expect("primary AVIS reference sample should still decode");
    assert_eq!((frame.width, frame.height), (64, 64));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_all_key_avis_samples_decode_by_index() {
    let Some(data) = generated_all_key_avis_sample() else {
        return;
    };
    let info = parse_avif(&data).expect("all-key AVIS metadata should parse");
    assert_eq!(info.sequence_sample_payloads.len(), 4);
    for payload in &info.sequence_sample_payloads {
        assert!(matches!(
            classify_av1_sequence_sample(payload).unwrap(),
            Some(AvifSequenceSampleKind::Key | AvifSequenceSampleKind::IntraOnly)
        ));
    }

    let first = avif_rust::decode_sequence_frame_bytes(&data, 0)
        .expect("first all-key AVIS sample should decode");
    assert_eq!((first.width, first.height), (64, 64));
    for index in 1..4 {
        let frame = avif_rust::decode_sequence_frame_bytes(&data, index)
            .expect("all-key AVIS sample should decode by index");
        assert_eq!((frame.width, frame.height), (64, 64));
        assert_eq!(frame.buffers, first.buffers);
    }
    let frames = avif_rust::decode_sequence_frames_bytes(&data)
        .expect("all-key AVIS samples should decode as an animation batch");
    assert_eq!(frames.len(), 4);
    assert!(frames.iter().all(|frame| frame.buffers == first.buffers));
    let error = avif_rust::decode_sequence_frame_bytes(&data, 4).unwrap_err();
    assert!(matches!(error, DecoderError::InvalidParam(message) if message.contains("outside")));
}

#[test]
fn generated_all_key_avis_samples_emit_animation_callback_frames() {
    let Some(data) = generated_all_key_avis_sample() else {
        return;
    };
    let mut drawer = RecordingDrawer::default();
    let mut options = DecodeOptions::new(&mut drawer);
    avif_rust::decode(&mut BytesReader::new(&data), &mut options)
        .expect("supported AVIS sequence should emit callback frames");

    let (width, height, init) = drawer.init.expect("callback should be initialized");
    assert_eq!((width, height), (64, 64));
    assert_eq!(init.loop_count, 1);
    assert!(init.animation);
    assert_eq!(drawer.draw_buffers.len(), 4);
    assert!(
        drawer
            .draw_buffers
            .windows(2)
            .all(|pair| pair[0] == pair[1])
    );
    assert!(drawer.terminated);
}

#[test]
fn generated_film_grain_avis_samples_decode_with_reference_parameters() {
    let Some(data) = generated_film_grain_avis_sample() else {
        return;
    };
    let info = parse_avif(&data).expect("generated film-grain AVIS metadata should parse");
    assert!(info.sequence_sample_payloads.len() >= 2);
    assert!(matches!(
        classify_av1_sequence_sample(&info.sequence_sample_payloads[0]).unwrap(),
        Some(AvifSequenceSampleKind::Key)
    ));
    assert!(info.sequence_sample_payloads.iter().skip(1).any(|payload| {
        matches!(
            classify_av1_sequence_sample(payload).unwrap(),
            Some(AvifSequenceSampleKind::Inter)
        )
    }));
    for index in 0..info.sequence_sample_payloads.len() {
        let frame = avif_rust::decode_sequence_frame_bytes(&data, index)
            .unwrap_or_else(|error| panic!("film-grain sample {index} should decode: {error}"));
        assert_eq!((frame.width, frame.height), (64, 64));
    }
}

#[test]
fn external_avis_sequence_exposes_all_track_samples_when_present() {
    let path = external_star_path();
    if !path.is_file() {
        eprintln!("external AVIS sequence sample is unavailable; skipping track audit");
        return;
    }
    let data = std::fs::read(&path).expect("external AVIS sequence should be readable");
    let info = parse_avif(&data).expect("external AVIS metadata should parse");
    assert_eq!(&info.major_brand, b"avis");
    assert_eq!(info.sequence_sample_payloads.len(), 5);
    assert_eq!(info.sequence_sample_payloads[0], info.primary_item_payload);
    let kinds: Vec<_> = info
        .sequence_sample_payloads
        .iter()
        .map(|payload| classify_av1_sequence_sample(payload).unwrap())
        .collect();
    assert!(matches!(
        kinds.first(),
        Some(Some(AvifSequenceSampleKind::Key))
    ));
    assert!(
        kinds
            .iter()
            .skip(1)
            .any(|kind| matches!(kind, Some(AvifSequenceSampleKind::Inter)))
    );
    for payload in &info.sequence_sample_payloads {
        assert!(
            !parse_obu_stream(payload)
                .expect("external AVIS sample should be independently framed")
                .is_empty()
        );
    }
}

#[test]
fn sequence_api_decodes_inter_sample_without_partial_output() {
    let path = external_star_path();
    if !path.is_file() {
        eprintln!("external AVIS sequence sample is unavailable; skipping API boundary test");
        return;
    }
    let data = std::fs::read(&path).expect("external AVIS sequence should be readable");
    let frame = avif_rust::decode_sequence_frame_bytes(&data, 1)
        .expect("inter sample should decode without partial output");
    assert_eq!((frame.width, frame.height), (159, 159));
    assert_eq!(frame.buffers.planes[0].samples.len(), 159 * 159);
    for index in 0..5 {
        avif_rust::decode_sequence_frame_bytes(&data, index)
            .unwrap_or_else(|error| panic!("sample {index} should decode: {error}"));
    }
}

#[test]
fn callback_decodes_inter_sequence_after_initialization() {
    let path = external_star_path();
    if !path.is_file() {
        eprintln!("external AVIS sequence sample is unavailable; skipping callback boundary test");
        return;
    }
    let data = std::fs::read(&path).expect("external AVIS sequence should be readable");
    let mut drawer = RecordingDrawer::default();
    let mut options = DecodeOptions::new(&mut drawer);
    avif_rust::decode(&mut BytesReader::new(&data), &mut options)
        .expect("callback should decode inter sequence");
    assert!(drawer.init.is_some());
    assert_eq!(drawer.draw_buffers.len(), 5);
    assert!(!drawer.draw_buffers.is_empty());
    assert!(drawer.terminated);
}

#[test]
fn external_animated_libavif_sample_metadata_is_parsed_when_present() {
    let Some(root) = std::env::var_os("AVIF_ANIMATED_SAMPLE_DIR") else {
        eprintln!("external animated AVIF sample directory is unavailable; skipping");
        return;
    };
    let path = std::path::PathBuf::from(root).join("colors-animated-8bpc.avif");
    if !path.is_file() {
        eprintln!("external animated AVIF sample is unavailable; skipping");
        return;
    }
    let data = std::fs::read(&path).expect("animated AVIF sample should be readable");
    let info = parse_avif(&data).expect("animated AVIF metadata should parse");
    eprintln!(
        "animated samples={} kinds={:?}",
        info.sequence_sample_payloads.len(),
        info.sequence_sample_payloads
            .iter()
            .map(|payload| classify_av1_sequence_sample(payload))
            .collect::<Result<Vec<_>, _>>()
    );
    assert_eq!(info.sequence_sample_payloads.len(), 5);
    assert_eq!(info.sequence_sample_payloads[0], info.primary_item_payload);
    let frames = avif_rust::decode_sequence_frames_bytes(&data)
        .expect("animated AVIF sample should decode every frame");
    assert_eq!(frames.len(), 5);
    assert!(
        frames
            .iter()
            .all(|frame| (frame.width, frame.height) == (150, 150))
    );
}

#[test]
fn external_animated_libavif_12bpc_sample_decodes_when_present() {
    let Some(root) = std::env::var_os("AVIF_ANIMATED_SAMPLE_DIR") else {
        eprintln!("external animated AVIF sample directory is unavailable; skipping");
        return;
    };
    let path = std::path::PathBuf::from(root).join("colors-animated-12bpc-keyframes-0-2-3.avif");
    if !path.is_file() {
        eprintln!("external animated 12bpc AVIF sample is unavailable; skipping");
        return;
    }
    let data = std::fs::read(&path).expect("animated 12bpc AVIF sample should be readable");
    let frames = avif_rust::decode_sequence_frames_bytes(&data)
        .expect("animated 12bpc AVIF sample should decode every frame");
    assert!(!frames.is_empty());
    assert!(frames.iter().all(|frame| frame.bit_depth == 12));
    assert!(frames.iter().all(|frame| {
        frame
            .buffers
            .planes
            .iter()
            .flat_map(|plane| plane.samples.iter())
            .all(|&sample| sample <= 4095)
    }));
}

#[test]
fn public_parsers_reject_truncated_and_malformed_headers() {
    let err = parse_avif(&[0, 0, 0]).unwrap_err();
    assert!(matches!(err, DecoderError::NotEnoughData(message) if message.contains("box header")));

    let err = parse_obu_stream(&[0x0e]).unwrap_err();
    assert!(
        matches!(err, DecoderError::NotEnoughData(message) if message.contains("extension header"))
    );

    let err = parse_obu_stream(&[0x0a, 0x80]).unwrap_err();
    assert!(matches!(err, DecoderError::NotEnoughData(message) if message.contains("leb128 size")));

    let err = parse_obu_stream(&[0x0a, 0x02, 0xff]).unwrap_err();
    assert!(
        matches!(err, DecoderError::NotEnoughData(message) if message.contains("payload extends"))
    );

    let err = parse_obu_stream(&[0x80]).unwrap_err();
    assert!(err.to_string().contains("forbidden bit"));
}
