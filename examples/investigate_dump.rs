use std::{env, fs, io::Write};

fn main() {
    let a: Vec<String> = env::args().collect();
    let data = fs::read(&a[1]).unwrap();
    let info = avif_rust::container::parse_avif(&data).unwrap();
    fs::write(format!("{}.obu", a[2]), &info.primary_item_payload).unwrap();
    for (i, alpha) in info.alpha_auxiliary_items.iter().enumerate() {
        fs::write(format!("{}.alpha{i}.obu", a[2]), &alpha.payload).unwrap();
    }
    let frame = avif_rust::decode_frame_bytes(&data).unwrap();
    let mut out = fs::File::create(format!("{}.native", a[2])).unwrap();
    println!(
        "{} {} {} {:?}",
        frame.width, frame.height, frame.bit_depth, frame.color_config
    );
    for p in &frame.buffers.planes {
        println!("plane {:?}", p.layout);
        for y in 0..p.layout.height {
            for x in 0..p.layout.width {
                let s = p.samples[y * p.layout.stride() + x];
                if frame.bit_depth > 8 {
                    out.write_all(&s.to_le_bytes()).unwrap();
                } else {
                    out.write_all(&[s as u8]).unwrap();
                }
            }
        }
    }
}
