use avif_rust::{
    CallbackResponse, DataMap, DecodeOptions, DrawCallback, DrawOptions, InitOptions,
    TerminateOptions, VerboseOptions,
};
use bin_rs::reader::{BinaryReader, BytesReader, StreamReader};
use std::io::{BufReader, Cursor, SeekFrom};

const IMAGE: &[u8] = include_bytes!("lossless-sub8x8-palette.avif");
type ResultCallback = Result<Option<CallbackResponse>, Box<dyn std::error::Error>>;

#[derive(Default)]
struct Capture {
    width: usize,
    height: usize,
    rgba: Vec<u8>,
}
impl DrawCallback for Capture {
    fn init(&mut self, width: usize, height: usize, _: Option<InitOptions>) -> ResultCallback {
        self.width = width;
        self.height = height;
        self.rgba = vec![0; width * height * 4];
        Ok(None)
    }
    fn draw(
        &mut self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        data: &[u8],
        _: Option<DrawOptions>,
    ) -> ResultCallback {
        for row in 0..height {
            let start = ((y + row) * self.width + x) * 4;
            self.rgba[start..start + width * 4]
                .copy_from_slice(&data[row * width * 4..(row + 1) * width * 4]);
        }
        Ok(None)
    }
    fn terminate(&mut self, _: Option<TerminateOptions>) -> ResultCallback {
        Ok(None)
    }
    fn verbose(&mut self, _: &str, _: Option<VerboseOptions>) -> ResultCallback {
        Ok(None)
    }
    fn set_metadata(&mut self, _: &str, _: DataMap) -> ResultCallback {
        Ok(None)
    }
}

fn check_reader(reader: &mut impl BinaryReader, start: u64) {
    let expected_info = avif_rust::container::parse_avif(IMAGE).unwrap();
    reader.seek(SeekFrom::Start(start)).unwrap();
    let info = avif_rust::parse_info(reader).unwrap();
    assert_eq!(info.primary_item_id, expected_info.primary_item_id);
    assert_eq!(info.major_brand, expected_info.major_brand);
    assert_eq!(reader.offset().unwrap(), start + IMAGE.len() as u64);
    assert!(avif_rust::parse_info(reader).is_err());
    reader.seek(SeekFrom::Start(start)).unwrap();
    let expected = avif_rust::image_from_bytes(IMAGE).unwrap();
    let mut capture = Capture::default();
    avif_rust::decode(reader, &mut DecodeOptions::new(&mut capture)).unwrap();
    assert_eq!(
        (capture.width, capture.height),
        (expected.width, expected.height)
    );
    assert_eq!(capture.rgba, expected.rgba);
    assert_eq!(reader.offset().unwrap(), start + IMAGE.len() as u64);
    assert!(avif_rust::decode(reader, &mut DecodeOptions::new(&mut Capture::default())).is_err());
}

#[test]
fn published_reader_types_support_offsets_tiny_buffers_and_replay() {
    for prefix in [0usize, 7] {
        let mut input = vec![0xff; prefix];
        input.extend_from_slice(IMAGE);
        check_reader(&mut BytesReader::new(&input), prefix as u64);
        for capacity in [1, 2, 7] {
            let mut reader =
                StreamReader::new(BufReader::with_capacity(capacity, Cursor::new(&input)));
            check_reader(&mut reader, prefix as u64);
        }
    }
}

#[test]
fn public_reader_apis_reject_empty_and_truncated_input() {
    for input in [&[][..], &IMAGE[..7], &IMAGE[..IMAGE.len() / 2]] {
        let mut bytes = BytesReader::new(input);
        let mut stream = StreamReader::new(BufReader::with_capacity(1, Cursor::new(input)));
        assert!(avif_rust::parse_info(&mut bytes).is_err());
        assert!(avif_rust::parse_info(&mut stream).is_err());
        bytes.seek(SeekFrom::Start(0)).unwrap();
        stream.seek(SeekFrom::Start(0)).unwrap();
        assert!(
            avif_rust::decode(&mut bytes, &mut DecodeOptions::new(&mut Capture::default()))
                .is_err()
        );
        assert!(
            avif_rust::decode(
                &mut stream,
                &mut DecodeOptions::new(&mut Capture::default())
            )
            .is_err()
        );
    }
}
