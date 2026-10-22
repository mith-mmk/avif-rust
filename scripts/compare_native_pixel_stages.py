"""Compare same-sized native plane dumps, including per-sample main regressions.

Dumps must already use the same stage, visible dimensions, plane order, range,
subsampling and grain setting. 10/12-bit samples use little-endian u16.
This script does not decode files or convert colour. See docs/native-pixel-validation.md.
"""
import argparse
import json
from itertools import islice
from pathlib import Path
import struct


def compare(reference, baseline, candidate):
    if not reference or len(reference) != len(baseline) or len(reference) != len(candidate):
        raise ValueError("Native sample lengths differ or are empty")

    def stats(values):
        errors = [abs(a - b) for a, b in zip(values, reference)]
        return dict(count=sum(e != 0 for e in errors), max=max(errors), sum_abs=sum(errors))

    return dict(
        samples=len(reference), main=stats(baseline), candidate=stats(candidate),
        newly_incorrect=sum(m == r and c != r for m, c, r in zip(baseline, candidate, reference)),
        error_increased=sum(abs(c-r) > abs(m-r) for m, c, r in zip(baseline, candidate, reference)),
        changed_remaining_errors=sum(c != r and c != m for m, c, r in zip(baseline, candidate, reference)),
        first_errors=list(islice((dict(index=i, main=m, candidate=c, reference=r)
                      for i, (m, c, r) in enumerate(zip(baseline, candidate, reference)) if c != r), 8)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bit-depth", type=int, choices=(8, 10, 12), required=True)
    parser.add_argument("reference", type=Path)
    parser.add_argument("main", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()

    def read(path):
        data = path.read_bytes()
        if args.bit_depth == 8:
            return data
        if len(data) % 2:
            raise ValueError("Truncated u16 sample")
        return [v[0] for v in struct.iter_unpack("<H", data)]

    print(json.dumps(compare(read(args.reference), read(args.main), read(args.candidate)), indent=2))
