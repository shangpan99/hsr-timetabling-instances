"""Generate cases and replace their three input CSVs in data/instances/.

The input destination is shared with every experiment entry point through
settings/generation_config.csv. Generation reports are saved under results/.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src'
if str(SOURCE) in sys.path:
    sys.path.remove(str(SOURCE))
sys.path.insert(0, str(SOURCE))
for name in list(sys.modules):
    if name in ('config_loader', 'case_manager', 'train_numbering') or name == 'trainset_tools' or name.startswith('trainset_tools.'):
        sys.modules.pop(name, None)

from case_manager import load_common, instance_root, results_root, input_fingerprint
from trainset_tools.hsrgen.data import Config, Source
from train_numbering import TrainNumberRegistry

INPUT_FILES = ('trains.csv', 'stations.csv', 'segments.csv')


def replace_file(path, content):
    """Replace one complete file; never expose a partly written CSV."""
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', choices=[200, 300, 400, 500, 600])
    parser.add_argument('--groups', type=int, choices=range(1, 11), help='Generate groups G01 through this number')
    parser.add_argument('--check', action='store_true', help='Check inputs and paths without generating or replacing files')
    args = parser.parse_args(argv)
    common = load_common()
    config_path = ROOT / 'settings/generation_config.csv'
    config = Config(config_path)
    model = {
        'model.fast_type': str(common['fast_train_type']),
        'model.fast_departure_window': str(common['dep_window_fast']),
        'model.normal_departure_window': str(common['dep_window_normal']),
        'model.allow_early_departure': str(common['allow_early_departure']),
        'model.dwell_upper_enabled': 'FALSE', 'model.run_upper_enabled': 'FALSE',
    }
    config.values.update(model)
    source = Source(config)
    registry_path = ROOT / 'data/train_number_registry.csv'
    registry = TrainNumberRegistry.load(registry_path)
    if source.n != 344:
        raise ValueError(f'Expected 344 reference trains, found {source.n}')
    if any((train.source, 0) not in registry.numbers for train in source.trains):
        raise ValueError('Numbering registry is incomplete')
    sizes = args.sizes or [int(k) for k in config.list('generation.target_sizes')]
    groups = args.groups or config.int('generation.groups_per_size')
    if not sizes or len(sizes) != len(set(sizes)) or any(k not in (200, 300, 400, 500, 600) for k in sizes):
        parser.error('Use distinct sizes from 200, 300, 400, 500, 600')
    if not 1 <= groups <= 10:
        parser.error('groups_per_size must be between 1 and 10')
    destination = instance_root(common)
    report_root = results_root(common) / 'generation'
    expected = [(f'K{k}_G{g:02d}', k) for k in sizes for g in range(1, groups + 1)]
    for case_id, k in expected:
        target = destination / case_id
        if target.resolve().parent != destination.resolve():
            raise ValueError(f'Case folder leaves the configured input directory: {target}')
        if target.exists() and not target.is_dir():
            raise ValueError(f'Case folder is occupied by a file: {target}')
    print(f'Input destination: {destination} (replace {len(expected)} selected cases)')
    print(f'Generation reports: {report_root}/<run-id>')
    if args.check:
        print(f'Source inputs OK: {source.n} reference trains; no solver called or files written.')
        return 0

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    reports = report_root / run_id
    reports.mkdir(parents=True, exist_ok=False)
    registry_copy = reports / 'train_number_registry.csv'
    shutil.copy2(registry_path, registry_copy)
    from trainset_tools.hsrgen import main as generator
    generator.REGISTRY_PATH = registry_copy
    _, statuses = generator.run(config_path, sizes=sizes, groups=groups,
                                destination=reports, model_values=model)
    if len(statuses) != len(sizes) or any(r['status'] != 'COMPLETE' for r in statuses):
        raise RuntimeError(f'Generation incomplete; existing case inputs were not replaced. Reports: {reports}')

    # Validate every requested case before replacing any existing input.
    publication = []
    for case_id, k in expected:
        folder = reports / case_id
        input_fingerprint(folder, k)
        publication.append((destination / case_id,
                            {name: (folder / name).read_bytes() for name in INPUT_FILES}))
    TrainNumberRegistry.load(registry_copy)
    destination.mkdir(parents=True, exist_ok=True)
    for target, content in publication:
        target.mkdir(parents=True, exist_ok=True)
        for name, raw in content.items():
            replace_file(target / name, raw)
    replace_file(registry_path, registry_copy.read_bytes())
    print(f'Replaced {len(publication)} cases in {destination}')
    print(f'Generation reports: {reports}')
    print('Experiment entries automatically check regenerated cases before solving; check_feasibility.py is also available separately.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
