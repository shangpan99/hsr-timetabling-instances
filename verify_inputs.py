"""Check the release manifest and all 50 target inputs without solving or writing.

Uses only Python's standard library. This checks data integrity and structure;
operational feasibility is checked by trainset_tools/check_feasibility.py.
"""
from pathlib import Path
import ast
import csv
import hashlib
import math
from collections import defaultdict

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    manifest = read_csv(ROOT / 'MANIFEST.csv')
    paths = [row['path'] for row in manifest]
    require(len(paths) == len(set(paths)), 'Duplicate manifest paths')
    scripts = 0
    for row in manifest:
        path = (ROOT / row['path']).resolve()
        require(ROOT in path.parents, f'Unsafe manifest path: {row["path"]}')
        raw = path.read_bytes()
        require(len(raw) == int(row['bytes']), f'File size mismatch: {row["path"]}')
        require(hashlib.sha256(raw).hexdigest() == row['sha256'], f'SHA-256 mismatch: {row["path"]}')
        if path.suffix == '.py':
            tree = ast.parse(raw.decode('utf-8-sig'), filename=row['path'])
            compile(tree, row['path'], 'exec')
            scripts += 1
            # The TS/ALNS source file stores additional Python modules as strings.
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript):
                    target = node.targets[0]
                    if isinstance(target.value, ast.Name) and target.value.id == '_MODULE_SOURCES':
                        compile(ast.literal_eval(node.value), row['path'] + '::embedded', 'exec')
    expected = {f'K{k}_G{g:02d}': k for k in (200, 300, 400, 500, 600) for g in range(1, 11)}
    actual = {p.name for p in (ROOT / 'instances').iterdir() if p.is_dir()}
    require(actual == set(expected), 'Expected exactly 50 instance directories')
    plan = read_csv(ROOT / 'case_plan.csv')
    require(len(plan) == 50, 'Expected 50 case-plan rows')
    require({f'K{r["K"]}_G{int(r["group"]):02d}' for r in plan} == set(expected), 'Case plan mismatch')
    for row in plan:
        k = int(row['K'])
        require(int(row['local_size']) == k // 10, 'Incorrect local subset size')
        require(int(row['stratified_size']) == int(row['coverage_size']) == k // 5, 'Incorrect subset size')
        require([int(row['sampling_' + m + '_count']) for m in ('local', 'stratified', 'coverage')]
                == [60, 120, 120], 'Incorrect 300-subset allocation')
    total_trains = total_rows = 0
    for name, k in sorted(expected.items()):
        folder = ROOT / 'instances' / name
        stations = read_csv(folder / 'stations.csv')
        segments = read_csv(folder / 'segments.csv')
        require(len(stations) == 23 and len(segments) == 22, f'{name}: infrastructure size mismatch')
        station_names = {r['station'] for r in stations}
        segment_pairs = {(r['from'], r['to']) for r in segments}
        require(len(station_names) == 23 and len(segment_pairs) == 22, f'{name}: duplicate infrastructure')
        for filename in ('stations.csv', 'segments.csv'):
            require((folder / filename).read_bytes() == (ROOT / 'trainset_tools/data' / filename).read_bytes(),
                    f'{name}: infrastructure differs from reference')
        trains = defaultdict(list)
        for row in read_csv(folder / 'trains.csv'):
            require(bool(row['车次']), f'{name}: empty train ID')
            trains[row['车次']].append(row)
            for field in ('到达时间', '出发时间'):
                if row[field]:
                    value = float(row[field])
                    require(math.isfinite(value) and 370 <= value <= 1429, f'{name}: invalid reference event time')
            require(row['是否停站'].upper() in ('TRUE', 'FALSE'), f'{name}: invalid stopping flag')
            total_rows += 1
        require(len(trains) == k, f'{name}: wrong train count')
        for train, rows in trains.items():
            rows.sort(key=lambda r: int(r['站序']))
            sequence = [r['车站'] for r in rows]
            require(len(sequence) >= 2 and len(sequence) == len(set(sequence)), f'{name}/{train}: invalid route')
            require(len({r['站序'] for r in rows}) == len(rows), f'{name}/{train}: duplicate station sequence')
            require(set(sequence) <= station_names, f'{name}/{train}: unknown station')
            require(all((a, b) in segment_pairs for a, b in zip(sequence, sequence[1:])), f'{name}/{train}: broken route')
            types = {r['train_type'] for r in rows}
            require(len(types) == 1 and types <= {'300', '350'}, f'{name}/{train}: invalid speed class')
            require(bool(rows[0]['出发时间']) and bool(rows[-1]['到达时间']), f'{name}/{train}: missing endpoint time')
        total_trains += k
    references = read_csv(ROOT / 'trainset_tools/data/trains_344_original.csv')
    require(len({r['车次'] for r in references}) == 344, 'Expected 344 reference trains')
    print(f'PASS: {len(manifest)} manifest entries; {scripts} Python files and embedded modules compile.')
    print(f'PASS: 50 cases, {total_trains} train records across cases, {total_rows} station-event rows.')
    print('No solver called. No files written. Operational feasibility was not re-solved.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
