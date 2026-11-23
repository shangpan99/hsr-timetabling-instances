"""Regenerate target instances in a separate empty directory.

Uses the original hsrgen implementation, original reference data and numbering
registry. The published instances and registry are never overwritten.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config_loader import load_config
from trainset_tools.hsrgen.data import Config, Source
from train_numbering import TrainNumberRegistry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'regenerated_instances')
    parser.add_argument('--sizes', type=int, nargs='+', choices=[200, 300, 400, 500, 600])
    parser.add_argument('--groups', type=int, choices=range(1, 11))
    parser.add_argument('--check', action='store_true', help='Read source data and config only; no solve or writes')
    args = parser.parse_args(argv)
    config_path = Path(__file__).with_name('generation_config.csv')
    common = load_config(ROOT / 'config.csv')
    model = {
        'model.fast_type': str(common['fast_train_type']),
        'model.fast_departure_window': str(common['dep_window_fast']),
        'model.normal_departure_window': str(common['dep_window_normal']),
        'model.allow_early_departure': str(common['allow_early_departure']),
        'model.dwell_upper_enabled': 'FALSE', 'model.run_upper_enabled': 'FALSE',
    }
    config = Config(config_path)
    config.values.update(model)
    source = Source(config)
    registry = TrainNumberRegistry.load(ROOT / 'train_number_registry.csv')
    if source.n != 344:
        raise ValueError(f'Expected 344 reference trains, found {source.n}')
    if any((train.source, 0) not in registry.numbers for train in source.trains):
        raise ValueError('Numbering registry is incomplete')
    if args.sizes and len(args.sizes) != len(set(args.sizes)):
        parser.error('--sizes must not repeat a size')
    destination = args.output_dir.resolve()
    protected = (ROOT / 'instances', ROOT / 'trainset_tools')
    if destination == ROOT or destination in ROOT.parents or any(
            destination == p or p in destination.parents for p in protected):
        parser.error('Use a separate output directory outside instances/ and trainset_tools/')
    if args.check:
        print(f'Source inputs OK: {source.n} reference trains; no solver called or files written.')
        return 0
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        parser.error('Output directory must be empty; choose a new path to preserve earlier data')
    destination.mkdir(parents=True, exist_ok=True)
    registry_copy = destination / 'train_number_registry.csv'
    shutil.copy2(ROOT / 'train_number_registry.csv', registry_copy)
    from trainset_tools.hsrgen import main as generator
    # The original generator maintains a registry; direct it to the new run.
    generator.REGISTRY_PATH = registry_copy
    _, statuses = generator.run(config_path, sizes=args.sizes, groups=args.groups,
                                destination=destination, model_values=model)
    expected_sizes = args.sizes or [int(k) for k in config.list('generation.target_sizes')]
    if len(statuses) != len(expected_sizes) or any(r['status'] != 'COMPLETE' for r in statuses):
        raise RuntimeError('Generation did not complete; partial files are retained in the output directory')
    print(f'Generated inputs: {destination}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
