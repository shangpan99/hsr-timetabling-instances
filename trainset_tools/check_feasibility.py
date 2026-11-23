# -*- coding: utf-8 -*-
"""Check generated groups under the shared main model; no terminal-time upper bound.

Default: all generated groups. --selected-only: enabled rows in case_plan.csv.
Only status results are saved; no timetable, TT, or finish-time statistics.
"""
from pathlib import Path
import sys
_TOOLS_DIR = Path(__file__).resolve().parent
_ROOT = _TOOLS_DIR.parent
if str(_ROOT) in sys.path:
    sys.path.remove(str(_ROOT))
sys.path.insert(0, str(_ROOT))
for _name in list(sys.modules):
    if _name in ('case_manager', 'config_loader', 'feasibility_core') or _name == 'hsrgen' or _name.startswith('hsrgen.') or _name == 'trainset_tools' or _name.startswith('trainset_tools.'):
        sys.modules.pop(_name, None)

import argparse
import gc
import math
import re
from datetime import datetime
from types import SimpleNamespace
from case_manager import (load_common, instance_root, read_cases, input_fingerprint,
                          model_fingerprint, feasibility_records, write_csv_atomic, FEAS_FIELDS,
                          positive_int)
from trainset_tools.feasibility_core import load_data, check, self_test


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected-only', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args(argv)
    common = load_common()
    import gurobipy as gp
    settings = SimpleNamespace(lp_seconds=float(common.get('feasibility_lp_seconds', 30)),
                               mip_seconds=float(common.get('feasibility_mip_seconds', 300)),
                               threads=positive_int(common.get('feasibility_threads', 1), 'feasibility_threads'),
                               candidates=positive_int(common.get('feasibility_candidates', 3), 'feasibility_candidates', 0))
    if any(not math.isfinite(v) or v <= 0 for v in (settings.lp_seconds, settings.mip_seconds)) or settings.candidates > 3:
        raise ValueError('Feasibility limits must be positive; candidates must be 0..3')
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
        raise FileNotFoundError('No generated groups. Run trainset_tools/generate_trainsets.py first.')
    records = feasibility_records()
    tested = []
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        if args.self_test:
            self_test(gp, env)
            return 0
        print(f'Checking {len(groups)} groups; terminal-time upper bound: NONE', flush=True)
        for index, (folder, k, group) in enumerate(groups, 1):
            print(f'[{index}/{len(groups)}] {folder.name}', flush=True)
            sha, status, stage, solver_status, message = '', 'ERROR', 'load_or_solve', '', ''
            interrupted = False
            try:
                sha, _ = input_fingerprint(folder, k)
                data = load_data(folder, common)
                status, stage, solver_status, message = check(data, gp, env, settings)
                del data
            except KeyboardInterrupt:
                status, stage, message = 'UNKNOWN', 'interrupted', 'Interrupted by user'
                interrupted = True
            except Exception as exc:
                message = f'{type(exc).__name__}: {exc}'
            row = dict(zip(FEAS_FIELDS, [folder.name, k, group, status, stage, solver_status, message,
                                       sha, model_fingerprint(common), datetime.now().isoformat(timespec='seconds')]))
            records[folder.name] = row
            tested.append(row)
            write_csv_atomic(root / 'feasibility_results.csv',
                             sorted(records.values(), key=lambda r: (int(r['K']), int(r['group']))), FEAS_FIELDS)
            print(f'  {status}: {message}', flush=True)
            if interrupted:
                return 130
            gc.collect()
    counts = {s: sum(r['result'] == s for r in tested) for s in ('FEASIBLE', 'INFEASIBLE', 'UNKNOWN', 'ERROR')}
    print(f'Results: {root / "feasibility_results.csv"}\n{counts}')
    return 0 if all(r['result'] == 'FEASIBLE' for r in tested) else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)
