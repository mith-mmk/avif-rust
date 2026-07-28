use std::path::PathBuf;

pub(crate) fn wml2viewer_avif() -> Option<Vec<u8>> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut paths = vec![manifest.join("test_data/images/WML2Viewer.avif")];
    if let Some(parent) = manifest.parent() {
        paths.push(parent.join("samples/WML2Viewer.avif"));
    }
    paths.into_iter().find_map(|path| std::fs::read(path).ok())
}
