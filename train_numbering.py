"""Stable G-series numbers shared by independent experiment cases.

Copy index 0 retains the original number; positive indices use a persistent,
globally collision-free pool. Each case starts its own occurrence counters.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
from collections import Counter
from pathlib import Path

NUMBERING_VERSION = "original-and-copy-pool-v1"
REGISTRY_PATH = Path(__file__).resolve().parent / "train_number_registry.csv"
FIELDS = ("source_id", "copy_index", "train_id")


def source_key(value):
    if not re.fullmatch(r"G[1-9][0-9]*", str(value)):
        raise ValueError(f"Expected an original G-series train number: {value!r}")
    return int(value[1:])


class TrainNumberRegistry:
    def __init__(self, rows=()):
        self.numbers = {}
        used = set()
        for row in rows:
            source, number = str(row["source_id"]), str(row["train_id"])
            source_key(source)
            source_key(number)
            copy = int(row["copy_index"])
            if copy < 0 or str(copy) != str(row["copy_index"]):
                raise ValueError("copy_index must be a nonnegative integer")
            key = (source, copy)
            if key in self.numbers or number in used:
                raise ValueError("Duplicate source/copy slot or train number in registry")
            if copy == 0 and number != source:
                raise ValueError("Original train numbers must be preserved")
            self.numbers[key] = number
            used.add(number)
        for source, copy in self.numbers:
            if any((source, index) not in self.numbers for index in range(copy)):
                raise ValueError(f"Missing earlier copy slot for {source}")

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            return cls()
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(FIELDS):
                raise ValueError(f"Unexpected numbering registry columns: {path}")
            return cls(list(reader))

    def ensure(self, source_ids, max_copies=3):
        sources = sorted(set(source_ids), key=source_key)
        if not sources or max_copies < 0:
            raise ValueError("Registry requires source IDs and a nonnegative copy capacity")
        used = set(self.numbers.values())
        # Reserve all original names before allocating a single copy number.
        for source in sources:
            if (source, 0) not in self.numbers:
                if source in used:
                    raise ValueError(f"New original {source} collides with a saved copy number")
                self.numbers[source, 0] = source
                used.add(source)
        candidate = max(9001, max(map(source_key, used)) + 1)
        for source in sources:
            for copy in range(1, max_copies + 1):
                if (source, copy) not in self.numbers:
                    while f"G{candidate}" in used:
                        candidate += 1
                    number = f"G{candidate}"
                    self.numbers[source, copy] = number
                    used.add(number)
                    candidate += 1
        return self

    def assign(self, sources):
        counts, numbers, indices = Counter(), [], []
        for source in sources:
            copy = counts[source]
            key = (source, copy)
            if key not in self.numbers:
                raise ValueError(f"Registry has no copy slot {key}; extend it before generation")
            numbers.append(self.numbers[key])
            indices.append(copy)
            counts[source] += 1
        if len(set(numbers)) != len(numbers):
            raise ValueError("Repeated train number inside one case")
        return numbers, indices

    def to_bytes(self):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for (source, copy), number in sorted(self.numbers.items(), key=lambda item: (source_key(item[0][0]), item[0][1])):
            writer.writerow(dict(source_id=source, copy_index=copy, train_id=number))
        return stream.getvalue().encode("utf-8-sig")

    def save(self, path):
        path = Path(path)
        content = self.to_bytes()
        if path.exists() and path.read_bytes() == content:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)

    @property
    def fingerprint(self):
        return hashlib.sha256(self.to_bytes()).hexdigest()
