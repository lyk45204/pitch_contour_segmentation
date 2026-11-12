#!/usr/bin/env python3
"""Join the additional labels to locally obtained SMC2016 segment boundaries.

Python 3.9+; standard library only. No data are downloaded by this script.
"""

import argparse
import csv
import hashlib
import io
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_table(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def restore_recording(info, source_dir):
    source = source_dir / info["source_segment_file"]
    source_bytes = source.read_bytes()
    if sha256(source_bytes) != info["source_segment_sha256"]:
        raise ValueError(f"Source checksum mismatch: {source.name}. Use the pinned SMC2016 revision in README.md.")
    source_rows = list(csv.reader(io.StringIO(source_bytes.decode("utf-8"))))
    if len(source_rows) != int(info["source_segment_rows"]):
        raise ValueError(f"Unexpected source row count: {source.name}")

    assignments = read_table(HERE / info["label_file"])
    if len(assignments) != int(info["label_count"]):
        raise ValueError(f"Unexpected label count: {info['label_file']}")
    selected, previous_source_row = [], 0
    for item in assignments:
        source_row = int(item["source_row"])
        label = int(item["label"])
        if label not in (-1, 0, 1):
            raise ValueError(f"Invalid label in {info['label_file']}")
        if not previous_source_row < source_row <= len(source_rows):
            raise ValueError(f"Invalid, duplicate or unordered source row in {info['label_file']}")
        previous_source_row = source_row
        row = source_rows[source_row - 1]
        start_text, duration_text = row[0].strip(), row[2].strip()
        start, duration = float(start_text), float(duration_text)
        if not math.isfinite(start) or not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid interval in {source.name}, row {source_row}")
        selected.append([start_text, str(label), duration_text])

    # Repeat the four overlap truncations already present in the released label
    # version. This does not introduce a new annotation decision.
    truncations = 0
    for left, right in zip(selected, selected[1:]):
        start, end = float(left[0]), float(left[0]) + float(left[2])
        next_start = float(right[0])
        if next_start < end - 1e-9:
            duration = next_start - start
            if duration <= 0:
                raise ValueError(f"Nonpositive cleaned duration: {source.name}")
            left[2] = f"{duration:.9f}"
            truncations += 1

    for left, right in zip(selected, selected[1:]):
        if float(left[0]) + float(left[2]) > float(right[0]) + 1e-9:
            raise ValueError(f"Overlapping restored intervals: {source.name}")
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\r\n").writerows(selected)
    data = output.getvalue().encode("utf-8")
    if sha256(data) != info["restored_sha256"]:
        raise ValueError(f"Restored annotation checksum mismatch: {info['recording_id']}")
    split = {"train": "Train", "development_test": "Test"}[info["split"]]
    filename = info["recording_id"] + "_GongSegLabel_cleaned.csv"
    relative = Path(split) / "Seg_Gong_MajorityLabel_cleaned" / filename
    return relative, data, truncations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path,
                        help="The original SMC2016/dataset/groundtruth directory")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="A new directory for the restored, headerless label files")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    records = read_table(HERE / "recordings.csv")
    # Validate all inputs and restored checksums before writing any output.
    restored = [restore_recording(info, args.source_dir) for info in records]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for relative, data, _ in restored:
        destination = args.output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    count = sum(int(info["label_count"]) for info in records)
    truncations = sum(item[2] for item in restored)
    print(f"Restored {count} labels across {len(records)} recordings; all checksums match.")
    print(f"Reproduced {truncations} existing overlap truncations. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
