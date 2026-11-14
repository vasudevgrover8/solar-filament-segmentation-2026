"""Remove shared pixels in the rank order of the submitted filament IDs."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

SIDE = 2048
PIXELS = SIDE * SIDE
COLUMNS = ["filament_id", "segmentation_rle"]

def read_counts(encoded):
    runs = []
    cursor = 0
    while cursor < len(encoded):
        count = shift = 0
        while True:
            if cursor == len(encoded):
                raise ValueError("Truncated COCO RLE run")
            code = ord(encoded[cursor]) - 48
            if not 0 <= code <= 63:
                raise ValueError("Invalid COCO RLE character")
            cursor += 1
            count |= (code & 31) << shift
            shift += 5
            if not code & 32:
                if code & 16:
                    count -= 1 << shift
                break
        if len(runs) > 2:
            count += runs[-2]
        if count < 0:
            raise ValueError("Negative COCO RLE run")
        runs.append(count)
    if sum(runs) != PIXELS:
        raise ValueError("Mask does not cover the native pixel grid")
    return runs

def decode(encoded):
    return np.repeat(np.arange(len(runs := read_counts(encoded)), dtype=np.int64) % 2, runs).astype(bool, copy=False)

def encode(mask):
    edges = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    runs = np.diff(np.r_[0, edges, PIXELS]).tolist()
    if mask[0]:
        runs.insert(0, 0)
    tokens = []
    for i, run in enumerate(runs):
        count = run - runs[i - 2] if i > 2 else run
        while True:
            code = count & 31
            count >>= 5
            more = count != (-1 if code & 16 else 0)
            tokens.append(chr(48 + code + (32 if more else 0)))
            if not more:
                break
    return "".join(tokens)

def read_csv(path):
    groups = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"Invalid columns in {path}: {reader.fieldnames}")
        ids = set()
        for row in reader:
            ident = row["filament_id"]
            if not ident or ident in ids or not row["segmentation_rle"] or None in row:
                raise ValueError(f"Invalid or duplicate prediction at line {reader.line_num}")
            ids.add(ident)
            image, rank = ident.rsplit("_", 1)
            if not image or int(rank) < 1:
                raise ValueError(f"Invalid filament ID: {ident}")
            groups[image].append((int(rank), row["segmentation_rle"]))
    return groups



def repair_csv(source, out):
    """Write disjoint COCO masks and verify the saved file image by image."""
    if source.resolve() == out.resolve():
        raise ValueError("Choose a separate output file.")
    groups = read_csv(source)
    emitted = []
    union_area = {}
    modified = dropped = affected_images = removed_pixels = original_pixels = 0
    for image, rows in groups.items():
        occupied = np.zeros(PIXELS, dtype=bool)
        affected = False
        retained = 0
        for _, encoded in sorted(rows):
            mask = decode(encoded)
            area = int(mask.sum())
            assert area
            original_pixels += area
            remaining = mask & ~occupied
            occupied |= mask
            left = int(remaining.sum())
            if left < area:
                modified += 1
                affected = True
                removed_pixels += area - left
            if left == 0:
                dropped += 1
                continue
            retained += 1
            repaired = encoded if left == area else encode(remaining)
            if left < area:
                assert np.array_equal(decode(repaired), remaining)
            emitted.append((f"{image}_{retained}", repaired))
        affected_images += affected
        union_area[image] = int(occupied.sum())

    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        writer.writerows(emitted)

    # Read the deliverable back and verify zero overlap and identical foreground union area.
    reloaded = read_csv(out)
    assert reloaded.keys() == groups.keys()
    assert sum(map(len, reloaded.values())) == len(emitted)
    for image, rows in reloaded.items():
        occupied = np.zeros(PIXELS, dtype=bool)
        for expected, (rank, encoded) in enumerate(rows, 1):
            assert rank == expected
            mask = decode(encoded)
            assert mask.any() and not np.any(occupied & mask), image
            occupied |= mask
        assert int(occupied.sum()) == union_area[image], image


    return {
        "input_masks": sum(map(len, groups.values())),
        "output_masks": len(emitted),
        "images_with_predictions": len(reloaded),
        "images_with_overlaps": affected_images,
        "trimmed_masks": modified - dropped,
        "dropped_masks": dropped,
        "overlap_pixels_removed": removed_pixels,
        "input_foreground_pixels": original_pixels,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(repair_csv(args.source, args.output))
