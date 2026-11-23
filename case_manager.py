# -*- coding: utf-8 -*-
"""Three-table configuration and isolated execution of selected train sets."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import shutil
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config_loader import load_config, as_bool, project_root
from trainset_tools.hsrgen.data import Config

ROOT = project_root()
PLAN_FIELDS = ['enabled', 'K', 'group', 'local_size', 'stratified_size', 'coverage_size',
               'sampling_local_candidate_pool_size', 'sampling_local_count',
               'sampling_stratified_count', 'sampling_coverage_count']
CASE_KEYS = {'train_limit', 'sample_size', 'data_dir', 'instance_tag', 'create_case_subfolder',
             'sampling_local_size', 'sampling_stratified_size', 'sampling_coverage_size',
             'sampling_local_candidate_pool_size', 'sampling_local_count',
             'sampling_stratified_count', 'sampling_coverage_count', 'sample_count'}
MODEL_KEYS = ('fast_train_type', 'dep_window_fast', 'dep_window_normal',
              'allow_early_departure', 'big_m', 'validator_tolerance')
FEAS_FIELDS = ['case_id', 'K', 'group', 'result', 'stage', 'solver_status', 'message',
               'input_sha256', 'model_sha256', 'checked_at']
SCHEMA = 'hsr-integrated-1'


def positive_int(value, label, minimum=1):
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{label} must be an integer: {value!r}') from None
    if not math.isfinite(x) or not x.is_integer() or x < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}: {value!r}')
    return int(x)


def strict_bool(value):
    if str(value).strip().lower() not in ('true', 'false', '1', '0', 'yes', 'no', '是', '否'):
        raise ValueError(f'enabled must be TRUE or FALSE: {value!r}')
    return str(value).strip().lower() in ('true', '1', 'yes', '是')


def resolve_path(value):
    p = Path(str(value)).expanduser()
    return (p if p.is_absolute() else ROOT / p).resolve()


def load_common():
    cfg = load_config(ROOT / 'config.csv')
    conflict = CASE_KEYS.intersection(cfg)
    if conflict:
        raise ValueError('Per-case settings belong to case_plan.csv, not config.csv: ' + ', '.join(sorted(conflict)))
    for key in MODEL_KEYS:
        if key not in cfg:
            raise ValueError(f'config.csv is missing {key}')
    if str(cfg.get('sampling_allocation_mode', '')).lower() != 'manual':
        raise ValueError('Use sampling_allocation_mode=manual with the three explicit sampling counts')
    return cfg


def instance_root():
    g = Config(ROOT / 'trainset_tools' / 'generation_config.csv')
    p = g.path_value('generation.output_root')
    if p is None or p == ROOT or p in ROOT.parents:
        raise ValueError('generation.output_root must be a dedicated directory, e.g. instances')
    for source in ('trains', 'stations', 'segments'):
        src = g.path_value('input.' + source)
        if src and (src == p or p in src.parents):
            raise ValueError('Generation output cannot contain its own input source files')
    return p


@dataclass(frozen=True)
class Case:
    enabled: bool
    k: int
    group: int
    local: int
    stratified: int
    coverage: int
    local_pool_size: int
    local_count: int
    stratified_count: int
    coverage_count: int

    @property
    def case_id(self):
        return f'K{self.k}_G{self.group:02d}'

    @property
    def sample_count(self):
        return self.local_count + self.stratified_count + self.coverage_count

    def row(self):
        return dict(zip(PLAN_FIELDS, ['TRUE' if self.enabled else 'FALSE', self.k, self.group,
                                     self.local, self.stratified, self.coverage,
                                     self.local_pool_size, self.local_count,
                                     self.stratified_count, self.coverage_count]))


def read_cases(selected=True):
    path = ROOT / 'case_plan.csv'
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != PLAN_FIELDS:
            raise ValueError(f'case_plan.csv columns must be: {PLAN_FIELDS}')
        cases, seen = [], set()
        for line, row in enumerate(reader, 2):
            if not any(str(v or '').strip() for v in row.values()):
                continue
            if None in row:
                raise ValueError(f'case_plan.csv line {line}: too many columns')
            c = Case(strict_bool(row['enabled']), positive_int(row['K'], 'K', 3),
                     positive_int(row['group'], 'group'),
                     *(positive_int(row[n], n, 2) for n in PLAN_FIELDS[3:6]),
                     positive_int(row['sampling_local_candidate_pool_size'],
                                  'sampling_local_candidate_pool_size'),
                     *(positive_int(row[n], n, 0) for n in PLAN_FIELDS[7:]))
            if c.case_id in seen:
                raise ValueError(f'Duplicate case: {c.case_id}')
            if max(c.local, c.stratified, c.coverage) >= c.k:
                raise ValueError(f'{c.case_id}: each restricted size must be < K')
            if not c.local <= c.local_pool_size <= c.k:
                raise ValueError(f'{c.case_id}: sampling_local_candidate_pool_size must be between local_size and K')
            if c.sample_count <= 0:
                raise ValueError(f'{c.case_id}: the three sampling counts must sum to a positive number')
            seen.add(c.case_id)
            if c.enabled or not selected:
                cases.append(c)
    return cases


def write_csv_atomic(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def sync_case_plan(sizes, groups):
    # Existing settings are never overwritten. Newly added cases start disabled.
    path = ROOT / 'case_plan.csv'
    existing = read_cases(False) if path.exists() else []
    keys = {c.case_id for c in existing}
    added = False
    for k in sizes:
        for g in range(1, groups + 1):
            local_size = max(2, k // 10)
            c = Case(False, k, g, local_size, max(2, k // 5), max(2, k // 5),
                     min(k, 2 * local_size), 60, 120, 120)
            if c.case_id not in keys:
                existing.append(c)
                keys.add(c.case_id)
                added = True
    if added:
        write_csv_atomic(path, [c.row() for c in existing], PLAN_FIELDS)


def effective_config(common, case):
    cfg = dict(common)
    cfg.update(train_limit=case.k, sample_size=max(case.local, case.stratified, case.coverage),
               sampling_local_size=case.local, sampling_stratified_size=case.stratified,
               sampling_coverage_size=case.coverage,
               sampling_local_candidate_pool_size=case.local_pool_size,
               sampling_local_count=case.local_count,
               sampling_stratified_count=case.stratified_count,
               sampling_coverage_count=case.coverage_count,
               sample_count=case.sample_count, instance_tag=case.case_id,
               create_case_subfolder=True, data_dir=str(instance_root() / case.case_id),
               output_root=str(resolve_path(common.get('output_root', 'outputs'))))
    # Manuscript budgets: 1800 s for K<=400, 3600 s for K=500/600.
    cfg['direct_time_limit'] = common.get(
        f'direct_time_limit_K{case.k}', cfg.get('direct_time_limit', 1800))
    inp, out = Path(cfg['data_dir']), Path(cfg['output_root'])
    ir = instance_root()
    if out == ROOT or out in ROOT.parents or out == ir or out in ir.parents or ir in out.parents:
        raise ValueError('output_root must be separate from source code and generated input directories')
    return cfg


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def input_fingerprint(folder, expected_k=None):
    folder = Path(folder)
    hashes = {}
    for name in ('trains.csv', 'stations.csv', 'segments.csv'):
        p = folder / name
        if not p.is_file():
            raise FileNotFoundError(f'{p} is missing; run trainset_tools/generate_trainsets.py first')
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    with (folder / 'trains.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    ids = {r.get('车次', '').strip() for r in rows}
    if not ids or '' in ids:
        raise ValueError(f'{folder}: empty train ID or empty input')
    if expected_k is not None and len(ids) != expected_k:
        raise ValueError(f'{folder.name}: expected exactly {expected_k} trains, found {len(ids)}; no prefix selection is allowed')
    if any(r.get('train_type', '').strip() not in ('300', '350', '300.0', '350.0') for r in rows):
        raise ValueError(f'{folder.name}: generated train_type must be explicit on every row')
    return digest(hashes), len(ids)


def model_fingerprint(cfg):
    return digest({'schema': SCHEMA, 'terminal_upper_bound': None,
                   **{k: cfg[k] for k in MODEL_KEYS}})


def feasibility_records():
    path = instance_root() / 'feasibility_results.csv'
    if not path.exists():
        return {}
    with path.open(encoding='utf-8-sig', newline='') as f:
        return {r['case_id']: r for r in csv.DictReader(f)}


def feasibility_state(case_id, cfg, input_sha, records):
    row = records.get(case_id)
    if not row:
        return 'NOT_CHECKED'
    if row.get('input_sha256') != input_sha or row.get('model_sha256') != model_fingerprint(cfg):
        return 'STALE'
    return row.get('result', 'UNKNOWN')


def run_signature(cfg, input_sha):
    # Execution-selection switches do not change experiment definitions.
    ignored = {'fresh_full_run', 'sample_resume', 'sample_clear_output', 'continue_on_error',
               'after_direct_require_direct_result', 'require_feasibility_pass',
               'data_dir', 'output_root'}
    params = {k: v for k, v in cfg.items() if not k.startswith(('_', 'feasibility_')) and k not in ignored}
    return {'schema': SCHEMA, 'input_sha256': input_sha, 'parameters': params,
            'signature': digest({'schema': SCHEMA, 'input': input_sha, 'parameters': params})}


def check_saved_signature(outdir, stamp, allow_reset=False):
    path = outdir / 'case_signature.json'
    if not outdir.exists() or not any(outdir.iterdir()) or allow_reset:
        return
    if not path.exists():
        raise ValueError(f'{outdir}: existing results lack case identity; use a new output_root or a fresh full run')
    old = json.loads(path.read_text(encoding='utf-8'))
    if old.get('signature') != stamp['signature']:
        raise ValueError(f'{outdir.name}: input or experiment parameters changed; do not reuse old results. '
                         'Use a new output_root, or set fresh_full_run=TRUE and run RUN_1_TO_6.py')


def run_entry(action, argv=None):
    parser = argparse.ArgumentParser(description=f'Run {action} for enabled rows of case_plan.csv')
    parser.add_argument('--dry-run', action='store_true', help='Inspect selected cases without solving or writing results')
    args = parser.parse_args(argv)
    common = load_common()
    if action == 'collect':
        import hsr_pipeline as hp
        print(hp.collect_all_case_rows(common))
        return 0
    cases = read_cases()
    if not cases:
        print('No enabled cases. Set enabled=TRUE in case_plan.csv.')
        return 0
    records = feasibility_records()
    prepared = []
    for case in cases:
        cfg = effective_config(common, case)
        sha, count = input_fingerprint(cfg['data_dir'], case.k)
        state = feasibility_state(case.case_id, cfg, sha, records)
        outdir = Path(cfg['output_root']) / case.case_id
        stamp = run_signature(cfg, sha)
        reset = action == 'full' and as_bool(cfg.get('fresh_full_run'), False)
        print(f'{case.case_id}: K={count}, Local={case.local}, Stratified={case.stratified}, '
              f'Coverage={case.coverage}, feasibility={state}\n  input={cfg["data_dir"]}\n  output={outdir}', flush=True)
        if not args.dry_run and action not in ('inspect', 'aggregate'):
            if as_bool(cfg.get('require_feasibility_pass'), True) and state != 'FEASIBLE':
                raise ValueError(f'{case.case_id}: feasibility={state}; run trainset_tools/check_feasibility.py with the current config first')
        if not args.dry_run and action != 'inspect':
            check_saved_signature(outdir, stamp, reset)
        prepared.append((case, cfg, outdir, stamp, reset))
    if args.dry_run:
        if action in ('full', 'method'):
            import hsr_pipeline as hp
            from resume_manager import ResumeManager
            for case, cfg, outdir, stamp, reset in prepared:
                check_saved_signature(outdir, stamp, reset)
                if reset:
                    print(f'{case.case_id}: fresh_full_run=TRUE; all stages will run')
                    continue
                manager = ResumeManager(cfg, hp.standalone_context(cfg), stamp['signature'], action)
                ready, reasons = manager.inspect()
                for stage in manager.stages:
                    print(f'  {case.case_id} {stage}: {"SKIP" if ready[stage] else "RUN"} ({reasons[stage]})')
        print('Dry run complete. No solver was called.')
        return 0
    import hsr_pipeline as hp
    actions = {'direct': hp.run_direct, 'restricted': hp.run_restricted_instances,
               'vote': hp.run_vote, 'packing': hp.run_packing,
               'inverse': hp.run_inverse_all_alphas, 'fullfix': hp.run_fullfix_all_alphas}
    results = []
    fields = ['case_id', 'K', 'group', 'action', 'status', 'message', 'finished_at']
    for index, (case, cfg, outdir, stamp, reset) in enumerate(prepared, 1):
        print(f'\n[{index}/{len(prepared)}] {case.case_id}: {action}', flush=True)
        status, message = 'COMPLETED', ''
        try:
            ctx = hp.standalone_context(cfg)
            if action == 'method' and as_bool(cfg.get('after_direct_require_direct_result'), True):
                if not (ctx.direct_dir / 'direct_summary.csv').is_file():
                    raise FileNotFoundError('RUN_2_TO_6 requires a saved Direct result with the current settings')
            if action in ('full', 'method') and not reset:
                from resume_manager import ResumeManager
                manager = ResumeManager(cfg, ctx, stamp['signature'], action)
                ready, _ = manager.inspect()
                if all(ready.values()):
                    status, message = 'SKIPPED_COMPLETED', 'All required stages already complete; existing results retained'
                    print(f'[SKIP CASE] {case.case_id}: {message}', flush=True)
                    continue
            data = hp.HSRData(cfg)
            validation = hp.validate_config(cfg, ctx, data)
            if action == 'inspect':
                print(f'Configuration/data PASS; sampling={validation["method_targets"]}; '
                      f'alpha={validation["alpha_list"]}; no terminal-time upper bound.')
                continue
            if reset and outdir.exists():
                shutil.rmtree(outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / 'case_signature.json').write_text(json.dumps(stamp, ensure_ascii=False, indent=2), encoding='utf-8')
            # Reset was handled above, before writing identity and effective parameters.
            cfg = dict(cfg, fresh_full_run=False)
            hp.write_effective_parameters(cfg, ctx, data, validation)
            if action in ('full', 'method'):
                from resume_manager import ResumeManager
                ResumeManager(cfg, ctx, stamp['signature'], action).run(hp, data)
            elif action == 'aggregate':
                hp.aggregate_experiment(cfg, ctx)
            else:
                actions[action](cfg, ctx, data)
            del data
        except KeyboardInterrupt:
            status, message = 'INTERRUPTED', 'Interrupted by user'
            raise
        except Exception as exc:
            status, message = 'ERROR', f'{type(exc).__name__}: {exc}'
            traceback.print_exc()
            if not as_bool(common.get('continue_on_error'), True):
                raise
        finally:
            results.append(dict(zip(fields, [case.case_id, case.k, case.group, action, status,
                                             message, datetime.now().isoformat(timespec='seconds')])))
            if action != 'inspect':
                write_csv_atomic(resolve_path(common.get('output_root', 'outputs')) / 'run_status.csv', results, fields)
            gc.collect()
    # COMPLETED means that the stage returned; solver feasibility/status lives in its own summaries.
    return 1 if any(r['status'] not in ('COMPLETED', 'SKIPPED_COMPLETED') for r in results) else 0
