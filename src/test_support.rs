use std::path::PathBuf;

pub(crate) fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(name)
}

pub(crate) fn wml2viewer_avif() -> Option<Vec<u8>> {
    std::fs::read(fixture_path("WML2Viewer.avif")).ok()
}
