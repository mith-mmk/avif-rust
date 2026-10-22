"""Generate synthetic AVIF/oracles with FFmpeg/libaom; verify dav1d agrees.

Usage: python scripts/generate_native_pixel_fixtures.py OUTPUT_DIRECTORY
No user input images are used. See tests/FIXTURES.md.
"""

import json
import math
from pathlib import Path
import struct
import subprocess
import sys


def generate(output):
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    width, height = 85, 73
    for depth in (8, 10, 12):
        for fmt, sx, sy in [("420", 1, 1), ("422", 1, 0), ("444", 0, 0), ("gray", 0, 0)]:
            for lossless in (False, True):
                pixel_format = "gray" if fmt == "gray" else "yuv" + fmt + "p"
                if depth > 8:
                    pixel_format += str(depth) + "le"
                name = pixel_format + ("-lossless" if lossless else "-lossy")
                source = bytearray()
                for plane in range(1 if fmt == "gray" else 3):
                    pw = (width + (1 << sx) - 1) >> sx if plane else width
                    ph = (height + (1 << sy) - 1) >> sy if plane else height
                    for y in range(ph):
                        for x in range(pw):
                            value = round(
                                (100 + 28 * math.sin(x / 13) + 19 * math.cos(y / 7)
                                 + ((x // 11 + y // 9 + plane) % 3) * 8)
                                * (1 << (depth - 8))
                            )
                            source.extend(struct.pack("<H", value) if depth > 8 else bytes([value]))
                source_path = output / (name + ".source")
                source_path.write_bytes(source)
                avif = output / (name + ".avif")
                subprocess.run([
                    "ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
                    "-pixel_format", pixel_format, "-video_size", f"{width}x{height}",
                    "-i", str(source_path), "-frames:v", "1", "-c:v", "libaom-av1",
                    "-cpu-used", "4", "-threads", "1", "-still-picture", "1",
                    "-aom-params", "enable-cdef=0:enable-restoration=0",
                    "-crf", "0" if lossless else "26", "-b:v", "0", str(avif),
                ], check=True)
                references = []
                for decoder in ("libaom-av1", "libdav1d"):
                    references.append(subprocess.check_output([
                        "ffmpeg", "-v", "error", "-threads", "1", "-c:v", decoder,
                        "-i", str(avif), "-frames:v", "1", "-c:v", "rawvideo",
                        "-threads", "1", "-f", "rawvideo", "-",
                    ]))
                if not references[0] or references[0] != references[1]:
                    raise RuntimeError(f"Independent references disagree: {name}")
                (output / (name + ".yuv")).write_bytes(references[0])
                manifest.append(dict(name=name, width=width, height=height, depth=depth,
                                     format=fmt, lossless=lossless, bytes=len(references[0])))
                print(name, flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    generate(Path(sys.argv[1]))
