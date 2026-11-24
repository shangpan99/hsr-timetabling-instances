# -*- coding: utf-8 -*-
"""Check generated groups under the shared main model; no terminal-time upper bound.

Default: all generated groups. --selected-only: enabled rows in case_plan.csv.
Only status results are saved; no timetable, TT, or finish-time statistics.
"""
from pathlib import Path
import sys
_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / 'src'
if str(_SOURCE) in sys.path:
    sys.path.remove(str(_SOURCE))
sys.path.insert(0, str(_SOURCE))
for _name in list(sys.modules):
    if _name in ('case_manager', 'config_loader', 'train_numbering', 'feasibility_core') or _name == 'trainset_tools' or _name.startswith('trainset_tools.'):
        sys.modules.pop(_name, None)

import argparse
import re
from case_manager import (load_common, instance_root, read_cases, input_fingerprint,
                          feasibility_results_path, feasibility_settings, check_feasibility_groups)
from trainset_tools.feasibility_core import self_test


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected-only', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Inspect input and output paths without solving or writing')
    args = parser.parse_args(argv)
    common = load_common()
    feasibility_settings(common)
    root = instance_root()
    if args.selected_only:
        groups = [(root / c.case_id, c.k, c.group) for c in read_cases()]
    else:
        groups = []
        for p in root.glob('K*_G*'):
            m = re.fullmatch(r'K(\d+)_G(\d+)', p.name)
            if p.is_dir() and m:
                groups.append((p, int(m[1]), int(m[2])))
        groups.sort(key=lambda x: (x[1], x[2]))
    if not groups and not args.self_test:
        if args.selected_only:
            print('No selected cases; set enabled=TRUE in case_plan.csv.')
            return 0
        raise FileNotFoundError('No generated groups. Run scripts/generate_instances.py first.')
    if args.dry_run:
        for folder, k, group in groups:
            _, count = input_fingerprint(folder, k)
            print(f'{folder.name}: {count} trains; input={folder}')
        print(f'Feasibility output: {feasibility_results_path(common)}')
        print('Dry run complete. No solver was called. No files written.')
        return 0
    if args.self_test:
        import gurobipy as gp
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            self_test(gp, env)
        return 0
    print(f'Checking {len(groups)} groups; terminal-time upper bound: NONE', flush=True)
    try:
        tested = check_feasibility_groups(common, groups)
    except KeyboardInterrupt:
        return 130
    counts = {s: sum(r['result'] == s for r in tested) for s in ('FEASIBLE', 'INFEASIBLE', 'UNKNOWN', 'ERROR')}
    print(f'Results: {feasibility_results_path(common)}\n{counts}')
    return 0 if all(r['result'] == 'FEASIBLE' for r in tested) else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)
