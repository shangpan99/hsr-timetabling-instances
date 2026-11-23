# -*- coding: utf-8 -*-
"""顶部选择规模和种子；固定输出TS_ALNS，主实验config.csv同级，直接F5。"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import traceback
import types

VERSION = 'TS80_ALNS_FIXED_FOLDER_1'
SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parent
# ================== 通常只修改下面这两项 ==================
# 选中的每个规模都会测试全部10个案例 G01–G10。
# 后续只测400–600：K_VALUES = (400, 500, 600)
# 测全部规模：K_VALUES = (200, 300, 400, 500, 600)
K_VALUES = (200, 300, 400, 500, 600)
SEEDS = (6, 7, 8, 9, 10)  # 更换种子时，已有结果目录会触发提醒并停止。
# 输出固定为同级 TS_ALNS；同种子续跑，异种子停止并提醒手动处理。
# ================== 以下保持算法和停止规则不变 ==================
MAX_WORKERS = 20
SECONDS_200_400 = 1800
SECONDS_500_600 = 3600
METHODS = ('TS', 'ALNS')
MAX_ACTIVE_GROUPS = 4
SCHEDULER_POLICY = 'case_method_five_seeds_whole_group_refill_v1'
FORMAL_STOPS = {'FEASIBLE_NO_IMPROVEMENT', 'TIME_LIMIT', 'NO_FEASIBLE_TIME_LIMIT'}
INPUT_FILES = ('trains.csv', 'stations.csv', 'segments.csv')
HISTORY_FIELDS = [
    'iteration', 'elapsed_sec', 'evaluations', 'extra_decodes', 'phase',
    'current_TT', 'current_origin_violations', 'current_excess_sum', 'best_search_TT',
    'best_origin_violations', 'best_excess_sum', 'best_feasible_TT', 'accepted', 'improved_best',
    'no_improve', 'feasible_no_improve', 'improved_best_feasible_TT', 'first_feasible_iteration',
    'move', 'moved_train', 'source_index', 'destination_index', 'candidate_count', 'destroy_operator',
    'swapped_train_1', 'swapped_train_2',
    'repair_operator', 'removed_count', 'candidate_TT', 'candidate_origin_violations', 'delta_TT',
    'accept_reason', 'temperature', 'destroy_weight', 'repair_weight',
]
_ENGINES = None


def engines():
    """Load original modules in memory; no core folder or external scripts needed."""
    global _ENGINES
    if _ENGINES is not None:
        return _ENGINES
    prefix = '_hsr50_' + hashlib.sha256(str(SCRIPT).encode()).hexdigest()[:12]
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + '.'):
            del sys.modules[name]
    for name in (prefix, prefix + '.baseline_lib'):
        module = types.ModuleType(name)
        module.__path__ = []
        module.__package__ = name
        sys.modules[name] = module
    for short, original in _MODULE_SOURCES.items():
        name = prefix + '.' + short
        module = types.ModuleType(name)
        module.__file__ = str(SCRIPT)
        module.__package__ = name.rpartition('.')[0]
        sys.modules[name] = module
        source = original
        for imported in ('baseline_lib', 'insert_search', 'fast_decode', 'swap_search_phased'):
            source = source.replace('from ' + imported, 'from ' + prefix + '.' + imported)
        if short == 'csv_settings':
            source = source.replace('DEFAULT_K = (200, 300, 400, 500)',
                                    'DEFAULT_K = (200, 300, 400, 500, 600)')
        exec(compile(source, str(SCRIPT) + '::' + short, 'exec'), module.__dict__)
        setattr(sys.modules[module.__package__], short.rsplit('.', 1)[-1], module)
    _ENGINES = {n: sys.modules[prefix + '.' + n] for n in _MODULE_SOURCES}
    return _ENGINES


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def csv_value(value):
    if value is None:
        return ''
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return value


def replace_csv(temp, path, attempts=7):
    """Bounded retries for transient Windows sharing/antivirus locks."""
    for attempt in range(attempts):
        try:
            os.replace(temp, path)
            return
        except OSError as exc:
            if not isinstance(exc, PermissionError) and getattr(exc, 'winerror', None) not in (5, 32, 33):
                raise
            if attempt == attempts - 1:
                raise
            time.sleep(min(0.01 * 2 ** attempt, 0.2))


def save_csv(path, rows, fields=None, replace_attempts=7):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='raise')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(v) for k, v in row.items()})
        f.flush()
        os.fsync(f.fileno())
    replace_csv(temp, path, attempts=replace_attempts)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def read_one(path):
    rows = read_csv(path)
    if len(rows) != 1:
        raise ValueError(f'Expected one record: {path}')
    return rows[0]


@contextmanager
def progress_access(path):
    """Nonblocking interprocess guard: our reader never overlaps replacement.

    The lock file is stable for the whole batch, including seed retries. The OS
    releases its byte lock if a worker is killed. A busy snapshot can be skipped.
    """
    with Path(path).with_suffix('.lock').open('a+b') as file:
        if file.tell() == 0:
            file.write(b'0')
            file.flush()
        file.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            file.seek(0)
            if os.name == 'nt':
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def read_progress(path):
    with progress_access(path) as acquired:
        if not acquired:
            raise BlockingIOError('Progress snapshot is being updated')
        return read_one(path)


def yes(value):
    return str(value).strip().lower() in {'1', 'true'}


def validate_selection():
    if not isinstance(K_VALUES, (list, tuple)) or not K_VALUES:
        raise ValueError('K_VALUES 必须至少包含一个规模，例如 (200, 300)；单个规模写 (300,)')
    if any(type(k) is not int or k not in (200, 300, 400, 500, 600) for k in K_VALUES):
        raise ValueError('规模必须是 200/300/400/500/600')
    if len(K_VALUES) != len(set(K_VALUES)):
        raise ValueError('K_VALUES 不能包含重复规模')
    result = {k: list(range(1, 11)) for k in sorted(K_VALUES)}
    if not isinstance(SEEDS, (list, tuple, range)) or not 1 <= len(SEEDS) <= MAX_WORKERS:
        raise ValueError(f'SEEDS 应包含 1–{MAX_WORKERS} 个不重复种子，例如 (6, 7, 8, 9, 10)')
    if any(type(x) is not int or not 0 <= x <= 4294967295 for x in SEEDS) or len(SEEDS) != len(set(SEEDS)):
        raise ValueError('随机种子应为 0–4294967295 之间不重复的整数')
    return dict(sorted(result.items())), sorted(SEEDS)


def implementation_hash():
    # Only these two user-controlled selection assignments are excluded. The
    # algorithm, all embedded sources, budgets, scheduler and identity code
    # remain protected. Formatting/comments do not change Python semantics.
    tree = ast.parse(SCRIPT.read_text(encoding='utf-8-sig'))
    allowed, found, kept = {'K_VALUES', 'SEEDS'}, set(), []
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in allowed):
            name = node.targets[0].id
            if name in found:
                raise ValueError(f'顶部选择项 {name} 被重复赋值，不能安全续跑')
            found.add(name)
        else:
            kept.append(node)
    if found != allowed:
        raise ValueError('请保留顶部 K_VALUES 和 SEEDS 的独立赋值')
    tree.body = kept
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode('utf-8')).hexdigest()


def plan_key(row):
    return row['case_id'], row['method'], int(row['seed'])


LEGACY_VERSION = 'TS80_ALNS_50CASES_GROUP_PARALLEL_1'
LEGACY_SCRIPT_HASH = 'd5a4de5b445e2a02fd3c2acaf9aafd2aa4fb26e6ab1e78ee7b28d05f7c603e0d'
SELECTABLE_VERSION = 'TS80_ALNS_SELECTABLE_GROUP_PARALLEL_1'
SELECTABLE_CODE_HASH = '54bef8b52296e9152846d80aa220c9f5f01580b2e2c4e56cfba5bf98f8a60fb9'


def output_conflict(output, reason):
    raise ValueError(f'\n[停止：已有结果不能混用]\n{reason}\n结果目录：{output}\n'
                     '请先结束使用此目录的其他脚本。若要重新测试，请自行备份后删除整个已有结果目录，'
                     '或手动将整个目录改名保留，再按F5。\n本脚本不会自动删除、改名或清空已有结果。')


def guard_output(settings, output):
    """Read-only gate; run before loading instances, locking or writing metadata."""
    if not output.exists():
        return
    if not output.is_dir():
        output_conflict(output, '输出路径已被一个文件占用。')
    contents = [p for p in output.iterdir() if p.name != '.runner.lock']
    if not contents:
        return
    manifest = output / 'run_plan.csv'
    if not manifest.is_file():
        output_conflict(output, '目录非空，但缺少 run_plan.csv，无法确认已有种子与实验设置。')
    previous = read_csv(manifest)
    if not previous:
        output_conflict(output, '保存的运行计划为空，无法安全续跑。')
    try:
        saved_seeds = sorted({int(r['seed']) for r in previous})
    except (KeyError, ValueError):
        output_conflict(output, '保存的运行计划中随机种子无效。')
    configured = settings['GENERAL']['seeds']
    if saved_seeds != configured:
        output_conflict(output, f'已有结果的随机种子：{saved_seeds}\n当前代码的随机种子：{configured}\n'
                        '随机种子已改变，已停止；已有结果保持原样。也可以恢复原种子组继续续跑。')
    grouped = {}
    for row in previous:
        pair = row['case_id'], row['method']
        grouped.setdefault(pair, []).append(int(row['seed']))
    if any(sorted(seeds) != configured for seeds in grouped.values()):
        output_conflict(output, '已有计划存在重复种子或不完整的种子组。')
    parameter_file = output / 'effective_parameters.csv'
    if parameter_file.exists():
        rows = read_csv(parameter_file)
        saved = {(r['section'], r['parameter']): r['value'] for r in rows}
        if len(rows) != len(saved):
            output_conflict(output, '保存的参数表含重复项。')
        for section in ('GENERAL', *METHODS, 'MODEL'):
            for name, value in settings[section].items():
                if isinstance(value, Path) or name in ('mode', 'K_values', 'case_groups'):
                    continue
                if saved.get((section, name)) != str(csv_value(value)):
                    output_conflict(output, f'参数改变或缺失：{section}.{name}。')


def compatible_identity(settings, row, files, current_hash):
    """Recognize only the two verified predecessors, preserving committed identities."""
    version = row['version']
    if version == VERSION:
        code_hash = current_hash
    elif version == SELECTABLE_VERSION:
        code_hash = SELECTABLE_CODE_HASH
    elif version == LEGACY_VERSION:
        if settings['GENERAL']['seeds'] != [1, 2, 3, 4, 5]:
            raise ValueError('旧固定1–5种子版本的计划与当前种子组不一致。')
        code_hash = current_hash
    else:
        raise ValueError(f'未识别的旧脚本版本：{version}')
    expected = run_identity(settings, row['case_id'], row['K'], row['group'], row['method'],
                            row['seed'], row['time_limit_sec'], row['run_class'], files, code_hash)
    expected['version'] = version
    if version == LEGACY_VERSION:
        expected.pop('code_sha256')
        expected.pop('experiment_signature')
        expected['script_sha256'] = LEGACY_SCRIPT_HASH
    expected.pop('run_signature')
    expected['run_signature'] = digest(dict(expected, params=settings[row['method']], model=settings['MODEL']))
    return expected


def merge_plan(settings, output, current, inputs):
    guard_output(settings, output)
    manifest = output / 'run_plan.csv'
    if not manifest.exists():
        return current, inputs
    previous = read_csv(manifest)
    previous_inputs = read_csv(output / 'input_fingerprints.csv')
    fingerprints = {(r['case_id'], r['file']): r for r in previous_inputs}
    if len(fingerprints) != len(previous_inputs):
        output_conflict(output, '保存的输入指纹有重复项。')
    for row in inputs:
        key = row['case_id'], row['file']
        if key in fingerprints and fingerprints[key]['sha256'] != row['sha256']:
            output_conflict(output, f'案例输入已改变：{key}。')
        fingerprints[key] = row
    merged = {}
    current_hash = implementation_hash()
    run_class = current[0]['run_class']
    for raw in previous:
        # Mixed-version plans have blank columns for the other identity format.
        row = {k: v for k, v in raw.items() if v != ''}
        for name in ('K', 'group', 'seed', 'max_active_groups', 'seeds_per_group'):
            row[name] = int(row[name])
        row['time_limit_sec'] = float(row['time_limit_sec'])
        key = plan_key(row)
        if key in merged or row['method'] not in METHODS:
            output_conflict(output, f'保存的计划含重复运行或未知方法：{key}。')
        if row['case_id'] != f'K{row["K"]}_G{row["group"]:02d}':
            output_conflict(output, f'案例编号与规模不一致：{key}。')
        seconds = (current[0]['time_limit_sec'] if run_class == 'diagnostic' else
                   settings['GENERAL']['seconds_200_400' if row['K'] <= 400 else 'seconds_500_600'])
        if row['run_class'] != run_class or row['time_limit_sec'] != seconds:
            output_conflict(output, f'时间预算或正式/诊断模式不同：{key}。')
        files = {name: fingerprints[row['case_id'], name]['sha256'] for name in INPUT_FILES}
        expected = compatible_identity(settings, row, files, current_hash)
        if row != expected:
            output_conflict(output, f'算法、物理参数、输入指纹或运行身份不一致：{key}。')
        merged[key] = row
    for row in current:
        key = plan_key(row)
        old = merged.get(key)
        # Completed predecessor results stay byte-for-byte intact. An unfinished
        # selected run uses the new identity when restarted from its initial state.
        if old is None or old == row or completed_record(output, old) is None:
            merged[key] = row
    ordered = sorted(merged.values(), key=lambda r: (r['K'], r['group'], METHODS.index(r['method']), r['seed']))
    return ordered, [fingerprints[k] for k in sorted(fingerprints)]


def read_settings(main_dir=None):
    """Use the script's directory; read only physical settings from the existing main config.

    Search seeds, budgets, algorithms, and case selection are fixed here, and do
    not inherit the main experiment's seed/direct_time_limit/case_plan switches.
    The main config is read-only; no new settings file is required or written.
    """
    selection, seeds = validate_selection()
    base = (Path(main_dir) if main_dir is not None else ROOT).resolve()
    path = base / 'config.csv'
    if not path.is_file():
        raise FileNotFoundError(f'请把 RUN_TS80_ALNS_FIXED.py 放在主实验 config.csv 同级：{path}')
    raw = path.read_bytes()
    rows = None
    for encoding in ('utf-8-sig', 'utf-16', 'gb18030'):
        try:
            text = raw.decode(encoding)
            reader = csv.DictReader(io.StringIO(text, newline=''))
            if not {'key', 'value'}.issubset(reader.fieldnames or []):
                continue
            rows = list(reader)
            break
        except (UnicodeError, csv.Error):
            continue
    if rows is None:
        raise ValueError('同级config.csv应是主实验原有的key,value表；请将本脚本放进主实验目录，不使用前一版附加配置。')
    modules = engines()
    original = {}
    for row in rows:
        key = str(row.get('key') or '').strip()
        if key and not key.startswith('#'):
            if key in original:
                raise ValueError(f'主实验config.csv有重复键：{key}')
            original[key] = modules['baseline_lib.config_loader'].cast_value(row.get('value'))
    physical_keys = ('train_selection_rule', 'fast_train_type', 'dep_window_fast',
                     'dep_window_normal', 'allow_early_departure')
    missing = [k for k in physical_keys if original.get(k) is None]
    if missing:
        raise ValueError(f'主实验config.csv缺少物理参数：{missing}')
    model = {k: original[k] for k in physical_keys}
    model.update(
        validator_tolerance=1e-5,  # Original TS/ALNS feasibility tolerance.
        validator_objective_tolerance=original.get('validator_objective_tolerance', 1e-3),
        validator_max_detail_rows=original.get('validator_max_detail_rows', 20000),
        write_validation_details=original.get('write_validation_details', True),
        paper_output_mode=True,
    )
    for key in ('dep_window_fast', 'dep_window_normal', 'validator_objective_tolerance'):
        if isinstance(model[key], bool) or not math.isfinite(float(model[key])) or float(model[key]) <= 0:
            raise ValueError(f'主实验参数{key}必须是正数')
    if not isinstance(model['allow_early_departure'], bool):
        raise ValueError('主实验allow_early_departure应为TRUE或FALSE')
    sections = {
        'GENERAL': dict(mode='ALL', instances_dir=base / 'instances', output_dir=base / 'TS_ALNS',
                        K_values=list(selection), case_groups=selection, seeds=seeds,
                        seconds_200_400=SECONDS_200_400, seconds_500_600=SECONDS_500_600,
                        max_active_groups=min(MAX_ACTIVE_GROUPS, MAX_WORKERS // len(seeds)), seeds_per_group=len(seeds),
                        scheduler_policy=SCHEDULER_POLICY),
        'MODEL': model,
        '_main_config_bytes': raw,
    }
    sections['TS'] = dict(candidate_list_size=80, tabu_tenure=10,
                          max_feasible_no_improve=100)
    # Exactly the original ALNS parameters, including its own stopping patience.
    sections['ALNS'] = dict(min_remove=3, max_remove=12, max_insertion_positions=20,
                            segment_length=25, reaction_factor=0.25,
                            cooling_factor=0.995, rewards=[10.0, 5.0, 1.0],
                            initial_temperature_fraction=0.03,
                            initial_temperature_floor=1.0, max_feasible_no_improve=80)
    return sections


def run_identity(settings, case_id, k, group, method, seed, seconds, run_class,
                 files=None, script_hash=None):
    g = settings['GENERAL']
    if files is None:
        files = {n: sha(g['instances_dir'] / case_id / n) for n in INPUT_FILES}
    code_hash = script_hash or implementation_hash()
    experiment = dict(code_sha256=code_hash, model=settings['MODEL'],
                      methods={m: settings[m] for m in METHODS}, seeds=g['seeds'],
                      budgets=[g['seconds_200_400'], g['seconds_500_600']],
                      diagnostic_seconds=float(seconds) if run_class == 'diagnostic' else None,
                      run_class=run_class, max_active_groups=g['max_active_groups'],
                      seeds_per_group=g['seeds_per_group'], scheduler_policy=g['scheduler_policy'])
    identity = dict(case_id=case_id, K=k, group=group, method=method, seed=seed,
                    time_limit_sec=float(seconds), run_class=run_class,
                    decoder_backend='python', version=VERSION,
                    input_sha256=digest(files), code_sha256=code_hash,
                    experiment_signature=digest(experiment),
                    max_active_groups=g['max_active_groups'],
                    seeds_per_group=g['seeds_per_group'], scheduler_policy=g['scheduler_policy'])
    identity['run_signature'] = digest(dict(identity, params=settings[method], model=settings['MODEL']))
    return identity


def make_plan(settings, diagnostic_seconds=None):
    g = settings['GENERAL']
    script_hash = implementation_hash()
    plan, inputs = [], []
    for k in sorted(g['K_values']):
        for group in sorted(g['case_groups'][k]):
            case_id = f'K{k}_G{group:02d}'
            case_dir = g['instances_dir'] / case_id
            files = {}
            for name in INPUT_FILES:
                path = case_dir / name
                if not path.is_file():
                    raise FileNotFoundError(f'Missing case input: {path}')
                files[name] = sha(path)
                inputs.append(dict(case_id=case_id, file=name, sha256=files[name]))
            seconds = float(g['seconds_200_400'] if k <= 400 else g['seconds_500_600'])
            if diagnostic_seconds is not None:
                seconds = float(diagnostic_seconds)
            for method in METHODS:
                for seed in g['seeds']:
                    identity = run_identity(settings, case_id, k, group, method, seed, seconds,
                                            'diagnostic' if diagnostic_seconds is not None else 'formal',
                                            files, script_hash)
                    plan.append(identity)
    return plan, inputs


def paths_for(output, identity):
    directory = output / identity['case_id']
    prefix = f'{identity["method"]}_seed{identity["seed"]}_'
    return directory, prefix


def completed_record(output, identity, verify=True):
    directory, prefix = paths_for(output, identity)
    path = directory / (prefix + 'result.csv')
    if not path.exists():
        return None
    # A power loss between writing result.csv and its commit manifest is an
    # unfinished attempt, and is safe to rerun from this seed's initial state.
    if not (directory / (prefix + 'artifacts.csv')).exists():
        return None
    record = read_one(path)
    if record.get('run_signature') != identity['run_signature']:
        raise ValueError(f'输入、算法参数或脚本版本不同，不能混用结果：{path}。请使用新的输出目录。')
    for key in ('case_id', 'method', 'seed', 'K', 'time_limit_sec', 'run_class', 'decoder_backend'):
        expected, actual = str(identity[key]), record.get(key, '')
        if actual != expected:
            raise ValueError(f'Run identity mismatch ({key}): {path}')
    if not yes(record.get('completed')) or record.get('stop_reason') not in FORMAL_STOPS:
        raise ValueError(f'Invalid completed record: {path}')
    feasible = yes(record.get('feasible'))
    if feasible != (record['stop_reason'] != 'NO_FEASIBLE_TIME_LIMIT'):
        raise ValueError(f'Invalid feasible status: {path}')
    if feasible and (not yes(record.get('validator_pass')) or not math.isfinite(float(record['best_TT']))):
        raise ValueError(f'Unvalidated feasible result: {path}')
    if not feasible and record.get('best_TT'):
        raise ValueError(f'Infeasible result contains a formal TT: {path}')
    if verify:
        artifacts = read_csv(directory / (prefix + 'artifacts.csv'))
        listed = {r['file'] for r in artifacts}
        required = {prefix + 'result.csv', prefix + 'history.csv'}
        if feasible:
            required |= {prefix + 'timetable.csv', prefix + 'order.csv', prefix + 'validation.csv'}
        if not required.issubset(listed) or len(listed) != len(artifacts):
            raise ValueError(f'Incomplete artifact manifest: {path}')
        for r in artifacts:
            name = r['file']
            if Path(name).name != name or not name.startswith(prefix):
                raise ValueError(f'Invalid artifact path: {name}')
            file = directory / name
            if not file.is_file() or sha(file) != r['sha256']:
                raise ValueError(f'结果文件缺失或内容已改变：{file}。请保留原结果并使用新的 output_dir 重跑。')
    return record


@contextmanager
def batch_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.runner.lock').open('a+b') as f:
        if f.tell() == 0:
            f.write(b'0')
            f.flush()
        f.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('这个结果目录已有脚本正在运行，请勿同时启动第二份。') from exc
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class History:
    def __init__(self, directory, prefix, identity):
        self.path = directory / (prefix + 'history.csv')
        self.progress_path = directory / (prefix + 'progress.csv')
        self.identity = identity
        self.file = self.path.open('w', encoding='utf-8-sig', newline='')
        self.writer = None
        self.fields = None
        self.rows = 0
        self.last = {}
        self.last_flush = self.last_report = time.perf_counter()
        self.progress_write_failures = 0
        self.last_progress_warning = None

    def __call__(self, row):
        if self.writer is None:
            self.fields = HISTORY_FIELDS
            self.writer = csv.DictWriter(self.file, fieldnames=self.fields)
            self.writer.writeheader()
        self.writer.writerow(row)
        self.rows += 1
        self.last = dict(row)
        current = time.perf_counter()
        if self.rows == 1 or current - self.last_flush >= 2:
            self.file.flush()
            self.last_flush = current
        self.progress(row)

    def progress(self, row, force=False, status='RUNNING'):
        current = time.perf_counter()
        if not force and current - self.last_report < 15:
            return
        self.last_report = current
        try:
            with progress_access(self.progress_path) as acquired:
                if acquired:
                    save_csv(self.progress_path,
                             [dict(self.identity, **row, status=status, updated_utc=now())],
                             replace_attempts=1)
        except OSError as exc:
            # Progress is observational. A locked progress file must never
            # discard a valid search or a committed result. History/result
            # writes still propagate errors and are verified before resuming.
            self.progress_write_failures += 1
            if self.last_progress_warning is None or current - self.last_progress_warning >= 60:
                print(f'[PROGRESS WARNING] 进度快照暂未写入，搜索继续：{exc}', flush=True)
                self.last_progress_warning = current
        print(f'  {self.identity["case_id"]} {self.identity["method"]} seed{self.identity["seed"]}: '
              f't={float(row.get("elapsed_sec", 0)):.1f}s; '
              f'iterations={row.get("iteration", 0)}; bestTT={row.get("best_feasible_TT")}', flush=True)

    def close(self):
        if not self.file.closed:
            if self.writer is None:
                self.file.write('iteration,elapsed_sec,best_feasible_TT\n')
            self.file.flush()
            os.fsync(self.file.fileno())
            self.file.close()


def run_one(instance, identity, settings, output, evaluator=None):
    directory, prefix = paths_for(output, identity)
    directory.mkdir(parents=True, exist_ok=True)
    # Only incomplete files belonging to this exact seed are replaced.
    for file in directory.glob(prefix + '*'):
        if file.is_file() and file.name not in {prefix + 'worker.log', prefix + 'progress.lock',
                                              prefix + 'progress.csv', prefix + 'progress.csv.tmp'}:
            file.unlink()
    started = time.perf_counter()
    record = dict(identity, script_sha256=sha(SCRIPT), started_utc=now(), worker_pid=os.getpid(), completed=False, feasible=False,
                  best_TT=None, diagnostic_TT=None, status='RUNNING', validator_pass=None,
                  time_limit_scope='entire_search', termination_is_optimality_certificate=False)
    save_csv(directory / (prefix + 'attempt.csv'), [record])
    stream = History(directory, prefix, identity)
    method = identity['method']
    modules = engines()
    params = dict(settings[method])
    patience = params.pop('max_feasible_no_improve')
    print(f'[START] {identity["case_id"]} {method} seed{identity["seed"]}, '
          f'limit={identity["time_limit_sec"]:g}s, patience={patience}', flush=True)
    try:
        kwargs = dict(seed=identity['seed'], infeasible_seconds=identity['time_limit_sec'],
                      max_feasible_no_improve=patience, history_sink=stream, history_keep=False)
        if method == 'TS':
            state = modules['swap_search_phased'].run_swap_phased(instance, params, evaluator, **kwargs)
        elif method == 'ALNS':
            state = modules['alns_search_phased'].run_alns_phased(instance, params, progress=stream.progress, **kwargs)
        else:
            raise ValueError(f'Unknown method: {method}')
        stream.close()
        record.update(stop_reason=state.stop_reason, search_runtime_sec=state.search_runtime_sec,
                      process_cpu_sec=state.process_cpu_sec, evaluations=state.evaluations,
                      extra_decodes=getattr(state, 'extra_decodes', 0), iterations=state.iteration,
                      completed_iterations=state.completed_iterations,
                      first_feasible_sec=state.first_feasible_sec, best_feasible_sec=state.best_feasible_sec,
                      best_iteration=state.best_iteration, feasible_no_improve=state.final_feasible_stall,
                      total_deadline_overrun_sec=state.total_deadline_overrun_sec,
                      initial_TT=state.initial.tt if state.initial else None,
                      initial_violations=state.initial.violation_count if state.initial else None,
                      history_rows=stream.rows, progress_write_failures=stream.progress_write_failures)
        if state.stop_reason not in FORMAL_STOPS:
            record['status'] = state.stop_reason
        else:
            expected_rows = state.completed_iterations + 1 if state.initial is not None else 0
            if expected_rows != stream.rows:
                raise AssertionError('Incomplete search history')
            if (state.best_feasible is not None) != (state.stop_reason != 'NO_FEASIBLE_TIME_LIMIT'):
                raise AssertionError('Stop reason disagrees with feasible incumbent')
            if state.stop_reason == 'FEASIBLE_NO_IMPROVEMENT' and state.final_feasible_stall != patience:
                raise AssertionError('Incorrect stagnation counter')
            if state.stop_reason in {'TIME_LIMIT', 'NO_FEASIBLE_TIME_LIMIT'} and state.search_runtime_sec < identity['time_limit_sec']:
                raise AssertionError('Search stopped before its deadline')
            validation_started = time.perf_counter()
            chosen = state.best_feasible or state.best
            if chosen is not None:
                final = modules['baseline_lib.common'].decode(instance, chosen.order, materialize=True)
                if final.key != chosen.key:
                    raise AssertionError('Final decoder did not reproduce search result')
                name = 'timetable.csv' if final.feasible else 'diagnostic_timetable.csv'
                timetable = directory / (prefix + name)
                save_csv(timetable, modules['baseline_lib.common'].timetable_rows(instance, final))
                save_csv(directory / (prefix + 'order.csv'), [
                    dict(rank=i + 1, train_index=k, train_id=instance.ids[k])
                    for i, k in enumerate(final.order)])
                # Original validator writes in this same directory. Rename its
                # method-and-seed-specific files to the flat prefix immediately.
                # Both methods validate the same seed concurrently: tags must differ.
                tag = prefix.rstrip('_')
                validation = modules['baseline_lib.validation'].validate_saved(
                    instance, final.order, timetable, final.tt, directory, tag)
                for file in directory.glob(f'validation_{tag}_*.csv'):
                    suffix = file.name[len(f'validation_{tag}_'):]
                    file.replace(directory / (prefix + 'validation_' + suffix))
                save_csv(directory / (prefix + 'validation.csv'),
                         [dict(parameter=k, value=v) for k, v in validation.items()])
                if final.feasible and validation['combined_pass'] != 1:
                    raise AssertionError('Independent validator rejected the feasible timetable')
                record.update(feasible=final.feasible, best_TT=final.tt if final.feasible else None,
                              diagnostic_TT=None if final.feasible else final.tt,
                              validator_pass=validation['combined_pass'],
                              violation_count=final.violation_count, violation_sum=final.violation_sum)
            if getattr(state, 'operators', None):
                save_csv(directory / (prefix + 'operators.csv'), state.operators)
            record.update(completed=True, status='FEASIBLE' if record['feasible'] else 'NO_FEASIBLE',
                          validation_runtime_sec=time.perf_counter() - validation_started)
    except KeyboardInterrupt:
        record.update(completed=False, status='INTERRUPTED', stop_reason='INTERRUPTED')
    except Exception as exc:
        record.update(completed=False, feasible=False, best_TT=None, status='ERROR', stop_reason='ERROR',
                      error=f'{type(exc).__name__}: {exc}')
        (directory / (prefix + 'error.txt')).write_text(traceback.format_exc(), encoding='utf-8')
        print(traceback.format_exc(), flush=True)
    finally:
        stream.close()
    record.update(finished_utc=now(), total_runtime_sec=time.perf_counter() - started)
    if record['completed']:
        result_path = directory / (prefix + 'result.csv')
        try:
            save_csv(result_path, [record])
            artifacts = [p for p in sorted(directory.glob(prefix + '*')) if p.is_file()
                         and p.suffix == '.csv'
                         and p.name not in {prefix + 'artifacts.csv', prefix + 'progress.csv', prefix + 'attempt.csv'}]
            save_csv(directory / (prefix + 'artifacts.csv'),
                     [dict(file=p.name, sha256=sha(p)) for p in artifacts])
            (directory / (prefix + 'attempt.csv')).unlink(missing_ok=True)
        except BaseException:
            result_path.unlink(missing_ok=True)
            raise
    else:
        save_csv(directory / (prefix + 'attempt.csv'), [record])
    stream.progress(stream.last, force=True, status=record['status'])
    print(f'[END] {identity["case_id"]} {method} seed{identity["seed"]}: '
          f'{record["status"]}, TT={record["best_TT"]}, stop={record.get("stop_reason")}', flush=True)
    return record


def summarize(output, plan, records):
    rows = []
    by_case = {}
    for identity in plan:
        key = (identity['case_id'], identity['method'], identity['seed'])
        record = records.get(key)
        if record is None:
            record = dict(identity, status='PENDING', completed=False, feasible=False, best_TT=None)
        rows.append(record)
        by_case.setdefault(identity['case_id'], {}).setdefault(identity['method'], []).append(record)
    case_rows, comparison_rows = [], []
    for case_id, methods in by_case.items():
        row = dict(case_id=case_id, K=int(case_id.split('_')[0][1:]))
        for method in METHODS:
            runs = methods[method]
            completed = [r for r in runs if yes(r.get('completed'))]
            valid = [r for r in completed if yes(r.get('feasible')) and yes(r.get('validator_pass'))]
            values = [float(r['best_TT']) for r in valid]
            all_complete = len(completed) == len(runs)
            all_feasible = len(valid) == len(runs)
            times = [float(r['search_runtime_sec']) for r in completed]
            stats = dict(planned_runs=len(runs), completed_runs=len(completed), feasible_runs=len(valid),
                         feasible_fraction=f'{len(valid)}/{len(runs)}',
                         completed=int(all_complete),
                         mean_TT=statistics.mean(values) if all_feasible else None,
                         SD=statistics.stdev(values) if all_feasible and len(values) > 1 else None,
                         feasible_mean_TT=statistics.mean(values) if values else None,
                         feasible_SD=statistics.stdev(values) if len(values) > 1 else None,
                         best_TT=min(values) if values else None,
                         mean_search_sec=statistics.mean(times) if all_complete else None,
                         error_runs=sum(r.get('status') == 'ERROR' for r in runs),
                         pending_seeds=';'.join(str(r['seed']) for r in runs if not yes(r.get('completed'))))
            row.update({method + '_' + key: value for key, value in stats.items()})
            comparison_rows.append(dict(
                case_id=case_id, K=row['K'], method=method,
                **{key: stats[key] for key in ('planned_runs', 'completed_runs', 'feasible_runs',
                    'mean_TT', 'SD', 'best_TT', 'feasible_mean_TT', 'feasible_SD',
                    'mean_search_sec', 'error_runs', 'pending_seeds')}))
        case_rows.append(row)
    save_csv(output / 'all_runs.csv', rows)
    save_csv(output / 'case_summary.csv', case_rows)
    save_csv(output / 'comparison.csv', comparison_rows)


def safe_summary(output, plan, records):
    try:
        summarize(output, plan, records)
    except PermissionError as exc:
        print(f'[SUMMARY] 汇总文件被占用，请关闭 Excel 后以 SUMMARY 模式更新。单次结果已保留：{exc}', flush=True)


def load_case(settings, case_id, k):
    modules = engines()
    cfg = dict(settings['MODEL'], data_dir=str(settings['GENERAL']['instances_dir'] / case_id))
    data = modules['baseline_lib.reference_model'].HSRData(cfg)
    count = data.trains_df_all['车次'].nunique()
    if count != k:
        raise ValueError(f'{case_id}: 需要完整的 {k} 列车案例，实际为 {count}；禁止截取前K列车。')
    instance = modules['baseline_lib.common'].Instance(data, cfg, k)
    if instance.K != k:
        raise ValueError(f'{case_id}: incorrect selected train count')
    return instance


def worker_main():
    # A separate interpreter also works when the parent is launched with Spyder
    # F5. It does not pickle Spyder's __main__ or the embedded dynamic modules.
    line = sys.stdin.readline()
    if not line:
        raise RuntimeError('Worker lost its parent before receiving its job')
    payload = json.loads(line)
    parent_pipe = sys.stdin.fileno()
    if os.name == 'nt':
        import ctypes
        import msvcrt
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        peek_pipe = kernel32.PeekNamedPipe
        peek_pipe.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong,
                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        peek_pipe.restype = ctypes.c_int
        pipe_handle = msvcrt.get_osfhandle(parent_pipe)

    def watch_parent():
        # Only this parent owns the pipe's writing end. A hard parent/kernel
        # exit closes it, so workers cannot silently remain running afterward.
        # Never hold a blocking stdin read during native-library startup.
        # PeekNamedPipe is nonblocking and reports a broken pipe on parent exit.
        try:
            if os.name == 'nt':
                while peek_pipe(pipe_handle, None, 0, None, None, None):
                    time.sleep(0.25)
            else:
                while os.read(parent_pipe, 4096):
                    pass
        except OSError:
            pass
        os._exit(130)

    threading.Thread(target=watch_parent, daemon=True, name='parent-lifetime').start()
    settings = read_settings(payload['main_dir'])
    identity = payload['identity']
    actual = run_identity(settings, identity['case_id'], identity['K'], identity['group'],
                          identity['method'], identity['seed'], identity['time_limit_sec'],
                          identity['run_class'])
    if actual != identity:
        raise ValueError('输入、参数或脚本在调度后发生改变，子进程拒绝运行。')
    instance = load_case(settings, identity['case_id'], identity['K'])
    evaluator = (engines()['fast_decode'].FastDecoder(instance, use_cpp=False)
                 if identity['method'] == 'TS' else None)
    try:
        record = run_one(instance, identity, settings, Path(payload['output']), evaluator)
    finally:
        if evaluator is not None:
            evaluator.close()
    return 0 if yes(record.get('completed')) else (130 if record['status'] == 'INTERRUPTED' else 1)


def launch_worker(identity, settings, output):
    directory, prefix = paths_for(output, identity)
    directory.mkdir(parents=True, exist_ok=True)
    log = (directory / (prefix + 'worker.log')).open('wb')
    executable = Path(sys.executable)
    if executable.name.lower() == 'pythonw.exe':
        executable = executable.with_name('python.exe')
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', BLIS_NUM_THREADS='1')
    process = None
    try:
        process = subprocess.Popen(
            [str(executable), str(SCRIPT), '--_worker'], cwd=str(ROOT), env=env,
            stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', close_fds=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        payload = dict(main_dir=str(settings['GENERAL']['instances_dir'].parent),
                       output=str(output), identity=identity)
        process.stdin.write(json.dumps(payload, ensure_ascii=False) + '\n')
        process.stdin.flush()
    except BaseException:
        if process is not None:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)
            process.stdin.close()
        log.close()
        raise
    return dict(identity=identity, process=process, log=log, started_utc=now())


def close_worker(worker):
    worker['process'].stdin.close()
    worker['log'].close()


def collect_worker(worker, output, interrupted=False):
    identity = worker['identity']
    record = completed_record(output, identity)
    if record is not None:
        return record
    directory, prefix = paths_for(output, identity)
    attempt = directory / (prefix + 'attempt.csv')
    if attempt.exists():
        record = read_one(attempt)
        if record.get('run_signature') != identity['run_signature']:
            raise ValueError(f'Worker identity mismatch: {attempt}')
        if record.get('status') in {'ERROR', 'INTERRUPTED'}:
            return record
    status = 'INTERRUPTED' if interrupted else 'ERROR'
    record = dict(identity, completed=False, feasible=False, best_TT=None,
                  status=status, stop_reason=status, worker_pid=worker['process'].pid,
                  worker_exit_code=worker['process'].returncode,
                  started_utc=worker['started_utc'], finished_utc=now(),
                  error='运行被中断；此种子下次从头运行。' if interrupted else
                        '子进程未提交完整结果，请查看同目录对应 TS_seedN_worker.log 或 ALNS_seedN_worker.log。')
    save_csv(attempt, [record])
    return record


class SchedulerEvents:
    def __init__(self, output):
        path = output / 'scheduler_events.csv'
        has_header = path.exists() and path.stat().st_size > 0
        self.file = path.open('a', encoding='utf-8-sig', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=[
            'session', 'utc', 'elapsed_sec', 'event', 'case_id', 'method', 'seed',
            'pid', 'active_groups', 'active_workers', 'status'])
        self.session, self.start = now() + '_' + str(os.getpid()), time.perf_counter()
        if not has_header:
            self.writer.writeheader()

    def write(self, event, identity=None, active=(), pid='', status=''):
        identity = identity or {}
        self.writer.writerow(dict(session=self.session, utc=now(),
            elapsed_sec=round(time.perf_counter() - self.start, 6), event=event,
            case_id=identity.get('case_id', ''), method=identity.get('method', ''),
            seed=identity.get('seed', ''), pid=pid, active_groups=len(active),
            active_workers=sum(len(g['workers']) for g in active), status=status))
        self.file.flush()

    def close(self):
        self.file.close()


def run_group_queue(selected, settings, output, plan, records):
    # Each slot belongs to a WHOLE case/method group, including while some of
    # its seeds have finished. Only an empty group releases a slot.
    grouped = {}
    for identity in selected:
        grouped.setdefault((identity['case_id'], identity['method']), []).append(identity)
    queue, skipped = [], 0
    for (case_id, method), jobs in grouped.items():
        pending = []
        for identity in jobs:
            key = (case_id, method, identity['seed'])
            if yes(records.get(key, {}).get('completed')):
                skipped += 1
                print(f'[SKIP] {case_id} {method} seed{identity["seed"]}: 已完成且文件校验通过。', flush=True)
            else:
                pending.append(identity)
        if pending:
            queue.append(dict(case_id=case_id, method=method, jobs=pending, workers=[], errors=0))
    limit = settings['GENERAL']['max_active_groups']
    active, next_group, ran, errors = [], 0, 0, 0
    events = SchedulerEvents(output)
    last_report = time.perf_counter()
    try:
        events.write('BATCH_START', active=active)
        while active or next_group < len(queue):
            while len(active) < limit and next_group < len(queue):
                group = queue[next_group]
                next_group += 1
                active.append(group)
                events.write('GROUP_START', group, active)
                print(f'[GROUP START] {group["case_id"]} {group["method"]}: '
                      f'启动{len(group["jobs"])}个未完成种子；活动组{len(active)}/{limit}。', flush=True)
                for identity in group['jobs']:
                    worker = launch_worker(identity, settings, output)
                    group['workers'].append(worker)
                    key = (identity['case_id'], identity['method'], identity['seed'])
                    records[key] = dict(identity, completed=False, feasible=False, best_TT=None,
                                        status='RUNNING', worker_pid=worker['process'].pid)
                    events.write('SEED_START', identity, active, pid=worker['process'].pid)
                safe_summary(output, plan, records)
            changed = False
            for group in list(active):
                for worker in list(group['workers']):
                    if worker['process'].poll() is None:
                        continue
                    record = collect_worker(worker, output)
                    identity = worker['identity']
                    records[(identity['case_id'], identity['method'], identity['seed'])] = record
                    close_worker(worker)
                    group['workers'].remove(worker)
                    ran += 1
                    failed = not yes(record.get('completed'))
                    errors += int(failed)
                    group['errors'] += int(failed)
                    changed = True
                    events.write('SEED_END', identity, active, pid=worker['process'].pid, status=record['status'])
                    print(f'[END] {identity["case_id"]} {identity["method"]} seed{identity["seed"]}: '
                          f'{record["status"]}, TT={record.get("best_TT") or "—"}；'
                          f'本组剩余{len(group["workers"])}个种子。', flush=True)
                    if record['status'] == 'INTERRUPTED':
                        raise KeyboardInterrupt
                if not group['workers']:
                    active.remove(group)
                    events.write('GROUP_END', group, active, status='ERROR' if group['errors'] else 'COMPLETE')
                    print(f'[GROUP END] {group["case_id"]} {group["method"]}: '
                          '整组结束，释放一个组名额。', flush=True)
            if changed:
                safe_summary(output, plan, records)
            if time.perf_counter() - last_report >= 15:
                last_report = time.perf_counter()
                print(f'[PROGRESS] 活动组{len(active)}/{limit}，'
                      f'未回收子进程{sum(len(g["workers"]) for g in active)}，'
                      f'等待组{len(queue) - next_group}。', flush=True)
                for group in active:
                    snippets = []
                    for worker in group['workers']:
                        identity = worker['identity']
                        directory, prefix = paths_for(output, identity)
                        try:
                            progress = read_progress(directory / (prefix + 'progress.csv'))
                            if progress.get('run_signature') != identity['run_signature']:
                                raise ValueError('old progress')
                            snippets.append(f's{identity["seed"]}: {float(progress["elapsed_sec"]):.0f}s '
                                            f'TT={progress.get("best_feasible_TT") or "None"}')
                        except (OSError, ValueError, KeyError, csv.Error):
                            snippets.append(f's{identity["seed"]}: 启动/加载中')
                    print(f'  {group["case_id"]} {group["method"]} | ' + ' | '.join(snippets), flush=True)
            if active:
                time.sleep(0.2)
        events.write('BATCH_END', active=active, status='ERROR' if errors else 'COMPLETE')
        return skipped, ran, errors
    finally:
        # Covers Ctrl+C, Spyder's normal interrupt, startup errors, and failures
        # while reading results. A killed parent is covered by the stdin pipe.
        workers = [w for g in active for w in g['workers']]
        for worker in workers:
            process = worker['process']
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
        for worker in workers:
            process = worker['process']
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            finally:
                close_worker(worker)
        try:
            for group in active:
                for worker in group['workers']:
                    identity = worker['identity']
                    records[(identity['case_id'], identity['method'], identity['seed'])] = collect_worker(
                        worker, output, interrupted=True)
                events.write('GROUP_INTERRUPTED', group, active, status='INTERRUPTED')
            if active:
                safe_summary(output, plan, records)
        finally:
            events.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description='顶部修改规模 K_VALUES / 种子 SEEDS，TS80 与原 ALNS；主实验 config.csv 同级，直接 F5。')
    parser.add_argument('--main-dir', type=Path, default=ROOT)
    parser.add_argument('--out-dir', type=Path, help='可选输出目录；默认固定为同级 TS_ALNS')
    parser.add_argument('--check', action='store_true', help='只检查当前选中案例，不开始搜索')
    parser.add_argument('--summary', action='store_true', help='只更新固定结果目录的累计汇总')
    parser.add_argument('--seeds', nargs='+', type=int, help='可选：只执行顶部种子组中的部分种子')
    parser.add_argument('--methods', nargs='+', choices=METHODS)
    parser.add_argument('--diagnostic-seconds', type=float, help='短时诊断，自动使用独立 _diagnostic 目录')
    args = parser.parse_args(argv)
    settings = read_settings(args.main_dir)
    g = settings['GENERAL']
    if args.out_dir is not None:
        g['output_dir'] = args.out_dir.resolve()
    if g['output_dir'] == g['instances_dir'] or g['output_dir'].is_relative_to(g['instances_dir']):
        raise ValueError('结果目录不能覆盖案例输入目录')
    if args.check and args.summary:
        parser.error('--check 与 --summary 不能同时使用')
    if args.diagnostic_seconds is not None:
        if not math.isfinite(args.diagnostic_seconds) or args.diagnostic_seconds <= 0:
            parser.error('--diagnostic-seconds 必须是正数')
        g['output_dir'] = g['output_dir'].with_name(g['output_dir'].name + '_diagnostic')
    guard_output(settings, g['output_dir'])
    current_plan, inputs = make_plan(settings, args.diagnostic_seconds)
    all_cases = list(dict.fromkeys(r['case_id'] for r in current_plan))
    cases, seeds, methods = all_cases, args.seeds or g['seeds'], args.methods or list(METHODS)
    if len(seeds) != len(set(seeds)) or set(seeds) - set(g['seeds']):
        parser.error('--seeds 必须属于顶部 SEEDS，且不能重复；更换整组种子请修改顶部 SEEDS')
    selected = [r for r in current_plan if r['case_id'] in cases and r['seed'] in seeds and r['method'] in methods]
    print(f'输入目录: {g["instances_dir"]}\n结果目录: {g["output_dir"]}', flush=True)
    print(f'顶部配置: {len(all_cases)}案例 × {len(METHODS)}方法 × {len(g["seeds"])}种子 = {len(current_plan)}次；本次选择{len(selected)}次。', flush=True)
    print(f'随机种子: {g["seeds"]}；方法: TS（均匀交换80候选）、ALNS（原版）。', flush=True)
    print(f'全程限时: K200–400 每种子 {g["seconds_200_400"]}s；K500–600 每种子 {g["seconds_500_600"]}s。'
          '已有可行解后，最佳可行TT连续 TS 100轮 / ALNS 80轮无下降即停止。', flush=True)
    print(f'每个案例×方法为一组（{g["seeds_per_group"]}种子），最多{g["max_active_groups"]}组、'
          f'{g["max_active_groups"] * g["seeds_per_group"]}个独立搜索进程；整组结束才补入下一组。', flush=True)
    if args.diagnostic_seconds is not None:
        print(f'[DIAGNOSTIC] 每运行仅 {args.diagnostic_seconds:g}s，结果单独保存。', flush=True)
    if args.check:
        for case_id in cases:
            instance = load_case(settings, case_id, int(case_id.split('_')[0][1:]))
            print(f'[CHECK] {case_id}: {instance.K} trains, {instance.event_count} events', flush=True)
        print(f'[CHECK OK] {len(cases)}个选中案例可读，未启动搜索或写入结果。', flush=True)
        return 0
    output = g['output_dir']
    if args.summary and not output.exists():
        raise FileNotFoundError('结果目录不存在，SUMMARY不会启动搜索。')
    with batch_lock(output):
        plan, all_inputs = merge_plan(settings, output, current_plan, inputs)
        identities = {plan_key(row): row for row in plan}
        selected = [identities[plan_key(row)] for row in selected]
        records = {}
        for identity in plan:
            record = completed_record(output, identity)
            if record is None:
                directory, prefix = paths_for(output, identity)
                attempt = directory / (prefix + 'attempt.csv')
                if attempt.exists():
                    prior = read_one(attempt)
                    if prior.get('run_signature') == identity['run_signature']:
                        record = prior
            if record is not None:
                records[(identity['case_id'], identity['method'], identity['seed'])] = record
        # Write provenance first: extra unreferenced input rows after a crash
        # are harmless, whereas a plan must always have all its fingerprints.
        save_csv(output / 'input_fingerprints.csv', all_inputs)
        save_csv(output / 'run_plan.csv', plan)
        if not (output / 'effective_parameters.csv').exists():
            save_csv(output / 'effective_parameters.csv', [
                dict(section=section, parameter=k, value=v)
                for section in ('GENERAL', *METHODS, 'MODEL')
                for k, v in settings[section].items()
                if not isinstance(v, Path) and k not in ('K_values', 'case_groups')])
            (output / 'main_config_snapshot.csv').write_bytes(settings['_main_config_bytes'])
        history_path = output / 'selection_history.csv'
        history = read_csv(history_path) if history_path.exists() else []
        history.append(dict(utc=now(), cases=cases, seeds=seeds, methods=methods,
                            configured_seeds=g['seeds'], selected_runs=len(selected),
                            registered_runs=len(plan), code_sha256=implementation_hash(),
                            script_sha256=sha(SCRIPT), main_config_sha256=hashlib.sha256(settings['_main_config_bytes']).hexdigest(),
                            mode='SUMMARY' if args.summary else 'RUN'))
        save_csv(history_path, history)
        safe_summary(output, plan, records)
        if args.summary:
            print(f'[SUMMARY] {output / "comparison.csv"}', flush=True)
            return 0
        skipped, ran, errors = run_group_queue(selected, settings, output, plan, records)
        print(f'[DONE] 本次运行{ran}次，跳过{skipped}次，错误{errors}次。累计汇总: {output / "comparison.csv"}', flush=True)
        print('本次选择已结束。再次F5续跑相同选择；新增规模只需修改顶部K_VALUES。', flush=True)
        return 1 if errors else 0


def launch():
    if sys.argv[1:] == ['--_worker']:
        try:
            status = worker_main()
        except KeyboardInterrupt:
            status = 130
        except Exception:
            traceback.print_exc()
            status = 1
        raise SystemExit(status)
    # Spyder F5 keeps the console alive and ignores IPython's own kernel flags.
    in_spyder = any('spyder' in name.lower() for name in sys.modules)
    try:
        status = main([] if in_spyder else None)
    except KeyboardInterrupt:
        print('\n已中断。已完成的种子保留；再次运行自动续跑。', flush=True)
        status = 130
    except Exception:
        traceback.print_exc()
        print('运行未完成，请按上面的错误信息处理后再次运行。', flush=True)
        status = 1
    if not in_spyder:
        raise SystemExit(status)


_MODULE_SOURCES = {}
_MODULE_SOURCES['baseline_lib.config_loader'] = '# -*- coding: utf-8 -*-\n"""统一实验配置读取工具。\n\n支持 UTF-8 / UTF-8-BOM / UTF-16 / GB18030 / GBK 等常见编码。\n即使配置文件的注释因 Windows 编辑器被保存成混合编码，只要参数键和值仍可识别，\n也会尽量继续读取并给出明确警告，而不会把最后一次解码失败误报成唯一原因。\n"""\nfrom __future__ import annotations\n\nimport csv\nimport io\nimport warnings\nfrom pathlib import Path\nfrom typing import Any, Dict, List, Mapping, Tuple\n\n_STRICT_ENCODINGS = (\n    "utf-8-sig",\n    "utf-8",\n    "utf-16",\n    "utf-16-le",\n    "utf-16-be",\n    "gb18030",\n    "gbk",\n    "cp1252",\n)\n_FALLBACK_ENCODINGS = ("utf-8-sig", "gb18030", "gbk", "cp1252", "latin-1")\n_SENTINELS = {"", "none", "null", "nil", "nan"}\n_TRUE = {"true", "yes", "y", "on", "是"}\n_FALSE = {"false", "no", "n", "off", "否"}\n\n# 这些键均为 ASCII。它们用于避免把 UTF-16 数据误当成含大量 NUL 的 UTF-8 文本，\n# 也用于确认一次“宽松解码”确实找到了配置内容，而非偶然解出了乱码。\n_SIGNATURE_KEYS = {"train_limit", "sample_size", "sample_count"}\n\n\ndef cast_value(value: Any) -> Any:\n    """将配置值转换为 None / bool / int / float / str。\n\n    数字 0/1 保留为整数，使用 ``as_bool`` 时仍可按布尔值读取；\n    这样可以避免 alpha=1、min_obs=1 被误读为 True。\n    """\n    if value is None:\n        return None\n    text = str(value).strip()\n    low = text.lower()\n    if low in _SENTINELS:\n        return None\n    if low in _TRUE:\n        return True\n    if low in _FALSE:\n        return False\n    try:\n        if low.startswith("0x"):\n            return int(low, 16)\n        if any(ch in text for ch in (".", "e", "E")):\n            return float(text)\n        return int(text)\n    except ValueError:\n        return text\n\n\ndef _extract_config_keys(text: str) -> set[str]:\n    """在不转换值的情况下提取有效配置键，用于验证解码结果。"""\n    if not text:\n        return set()\n    # 正常解码后的配置不应包含 NUL；含 NUL 通常说明 UTF-16 被误按单字节编码读取。\n    if "\\x00" in text:\n        return set()\n\n    keys: set[str] = set()\n    try:\n        reader = csv.reader(io.StringIO(text, newline=""))\n        for row in reader:\n            if not row:\n                continue\n            key = str(row[0]).lstrip("\\ufeff").strip()\n            if not key or key.startswith("#") or key.lower() == "key":\n                continue\n            keys.add(key)\n    except (csv.Error, UnicodeError):\n        return set()\n    return keys\n\n\ndef _looks_like_expected_config(text: str) -> bool:\n    keys = _extract_config_keys(text)\n    return _SIGNATURE_KEYS.issubset(keys)\n\n\ndef _encoding_candidates(raw: bytes) -> Tuple[str, ...]:\n    """根据 BOM/NUL 特征调整尝试顺序，同时保留完整候选集合。"""\n    preferred: list[str] = []\n    if raw.startswith(b"\\xef\\xbb\\xbf"):\n        preferred.append("utf-8-sig")\n    elif raw.startswith((b"\\xff\\xfe", b"\\xfe\\xff")):\n        preferred.append("utf-16")\n    elif raw[:200].count(b"\\x00") >= 5:\n        # 无 BOM 的 UTF-16 配置文件在 Windows/Excel 中并不少见。\n        preferred.extend(("utf-16-le", "utf-16-be"))\n\n    ordered: list[str] = []\n    for enc in (*preferred, *_STRICT_ENCODINGS):\n        if enc not in ordered:\n            ordered.append(enc)\n    return tuple(ordered)\n\n\ndef _decode_config_bytes(path: Path) -> Tuple[str, str, bool, Dict[str, str]]:\n    """读取并解码配置文件。\n\n    Returns\n    -------\n    text, encoding_label, used_replacement, diagnostics\n    """\n    raw = path.read_bytes()\n    if not raw:\n        raise ValueError(f"配置文件为空：{path}")\n\n    diagnostics: Dict[str, str] = {}\n\n    # 第一轮：严格解码。只接受确实包含关键配置项的结果。\n    for encoding in _encoding_candidates(raw):\n        try:\n            text = raw.decode(encoding, errors="strict")\n        except (UnicodeDecodeError, LookupError) as exc:\n            diagnostics[encoding] = f"{type(exc).__name__}: {exc}"\n            continue\n        if _looks_like_expected_config(text):\n            return text, encoding, False, diagnostics\n        diagnostics[encoding] = "decoded, but required configuration keys were not found"\n\n    # 第二轮：容错解码。主要处理“ASCII 参数行正常、中文注释有个别坏字节”的混合文件。\n    # 参数键和值在本项目默认配置中均为 ASCII，因此注释里的替换字符不会改变实验参数。\n    for encoding in _FALLBACK_ENCODINGS:\n        try:\n            text = raw.decode(encoding, errors="replace")\n        except LookupError as exc:\n            diagnostics[f"{encoding}+replace"] = f"{type(exc).__name__}: {exc}"\n            continue\n        if _looks_like_expected_config(text):\n            return text, f"{encoding}+replace", True, diagnostics\n        diagnostics[f"{encoding}+replace"] = (\n            "decoded with replacement, but required configuration keys were not found"\n        )\n\n    details = "\\n".join(f"  - {enc}: {msg}" for enc, msg in diagnostics.items())\n    raise UnicodeError(\n        "无法识别 config.csv 的编码，或文件内容已损坏。\\n"\n        f"文件：{path}\\n"\n        "请用 Windows 记事本打开该文件，选择“另存为”，编码选 UTF-8；"\n        "也可用随包提供的原始 config.csv 覆盖。\\n"\n        f"尝试记录：\\n{details}"\n    )\n\n\ndef load_config(\n    path: str | Path = "config.csv",\n    defaults: Mapping[str, Any] | None = None,\n) -> Dict[str, Any]:\n    path = Path(path)\n    if not path.exists():\n        if defaults is None:\n            raise FileNotFoundError(f"未找到配置文件：{path}")\n        return dict(defaults)\n\n    text, encoding, used_replacement, diagnostics = _decode_config_bytes(path)\n\n    cfg: Dict[str, Any] = {}\n    reader = csv.reader(io.StringIO(text, newline=""))\n    for line_no, row in enumerate(reader, start=1):\n        if not row:\n            continue\n        key = str(row[0]).lstrip("\\ufeff").strip()\n        if not key or key.startswith("#") or key.lower() == "key":\n            continue\n        if len(row) < 2:\n            raise ValueError(\n                f"config.csv 第 {line_no} 行缺少 value 列：{row!r}；"\n                "每个参数应写成 key,value"\n            )\n        cfg[key] = cast_value(row[1])\n\n    missing = sorted(_SIGNATURE_KEYS - set(cfg))\n    if missing:\n        raise ValueError(f"config.csv 缺少必要参数：{missing}")\n\n    if defaults:\n        for key, value in defaults.items():\n            cfg.setdefault(key, value)\n\n    cfg["_config_encoding"] = encoding\n    cfg["_config_decode_replaced"] = int(used_replacement)\n    cfg["_config_path"] = str(path.resolve())\n\n    if used_replacement:\n        warnings.warn(\n            "config.csv 含有混合编码或无效字节；程序已在不影响可识别参数行的前提下"\n            "进行容错读取。建议稍后用 Windows 记事本将 config.csv 另存为 UTF-8。"\n            f" 当前读取方式：{encoding}",\n            RuntimeWarning,\n            stacklevel=2,\n        )\n\n    return cfg\n\n\ndef as_bool(value: Any, default: bool = False) -> bool:\n    if value is None:\n        return default\n    if isinstance(value, bool):\n        return value\n    if isinstance(value, (int, float)):\n        return bool(value)\n    low = str(value).strip().lower()\n    if low in _TRUE or low == "1":\n        return True\n    if low in _FALSE or low == "0":\n        return False\n    return default\n\n\ndef as_int(value: Any, default: int | None = None) -> int | None:\n    if value is None:\n        return default\n    return int(float(value))\n\n\ndef as_float(value: Any, default: float | None = None) -> float | None:\n    if value is None:\n        return default\n    return float(value)\n\n\ndef parse_list(value: Any, *, sep: str = ";") -> List[str]:\n    if value is None:\n        return []\n    if isinstance(value, (list, tuple)):\n        return [str(x).strip() for x in value if str(x).strip()]\n    text = str(value).strip().replace("，", sep).replace(",", sep)\n    return [x.strip() for x in text.split(sep) if x.strip()]\n\n\ndef parse_int_list(value: Any) -> List[int]:\n    return [int(float(x)) for x in parse_list(value)]\n\n\ndef parse_float_list(value: Any) -> List[float]:\n    return [float(x) for x in parse_list(value)]\n\n\ndef parse_named_designs(value: Any) -> Dict[str, List[int]]:\n    """解析 ``D2:80+100;D4:60+80+100+120``。"""\n    result: Dict[str, List[int]] = {}\n    for item in parse_list(value):\n        if ":" not in item:\n            raise ValueError(f"multi-R design 格式错误：{item!r}；应为 D2:80+100")\n        name, sizes_text = item.split(":", 1)\n        sizes = [int(float(x.strip())) for x in sizes_text.replace(",", "+").split("+") if x.strip()]\n        if not name.strip() or not sizes:\n            raise ValueError(f"multi-R design 格式错误：{item!r}")\n        result[name.strip()] = sizes\n    return result\n\n\ndef get_k_list(cfg: Mapping[str, Any], prefix: str, k: int) -> List[int]:\n    return parse_int_list(cfg.get(f"{prefix}_K{k}"))\n\n\ndef project_root() -> Path:\n    return Path(__file__).resolve().parent\n\n\ndef resolve_project_path(value: Any, default: str | None = None) -> Path:\n    raw = value if value not in {None, ""} else default\n    if raw is None:\n        raise ValueError("路径配置为空")\n    path = Path(str(raw)).expanduser()\n    if not path.is_absolute():\n        path = project_root() / path\n    return path.resolve()\n'
_MODULE_SOURCES['baseline_lib.reference_model'] = '# -*- coding: utf-8 -*-\n"""Solver-free, source-exact extract of the supplied 2026-09-15 main pipeline.\n\nHSRData and validate_timetable_csv below are copied without semantic changes.\nOnly imports and PairKey are provided here to avoid importing Gurobi.\nSee SOURCE_PROVENANCE.json for source ranges and hashes.\n"""\nfrom __future__ import annotations\nfrom collections import Counter\nimport math\nimport time\nfrom pathlib import Path\nfrom typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple\nimport pandas as pd\nfrom .config_loader import as_bool, as_float, as_int, resolve_project_path\nPairKey = Tuple[str, str, str, str]\n\ndef write_csv(df: pd.DataFrame, path: Path) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    tmp = path.with_suffix(path.suffix + ".tmp")\n    df.to_csv(tmp, index=False, encoding="utf-8-sig")\n    tmp.replace(path)\n\n\ndef ensure_dir(path: Path) -> Path:\n    path.mkdir(parents=True, exist_ok=True)\n    return path\n\n\ndef parse_stop(value: Any) -> bool:\n    if isinstance(value, bool):\n        return value\n    if value is None or (isinstance(value, float) and math.isnan(value)):\n        return False\n    return str(value).strip().lower() in {"true", "1", "yes", "y", "是"}\n\n\ndef infer_train_type(train_id: str) -> int:\n    try:\n        digits = "".join(ch for ch in str(train_id) if ch.isdigit())\n        number = int(digits)\n        return 350 if 1 <= number <= 50 else 300\n    except Exception:\n        return 300\n\n\ndef train_id_sort_key(train_id: str) -> Tuple[str, int, str]:\n    text = str(train_id)\n    digits = "".join(ch for ch in text if ch.isdigit())\n    return (text[:1], int(digits) if digits else 0, text)\n\n\ndef normalize_train_selection_rule(value: Any) -> str:\n    raw = str(value or "entry_time").strip().lower().replace("-", "_").replace(" ", "_")\n    aliases = {\n        "entry_time": "entry_time",\n        "corridor_entry_time": "entry_time",\n        "arrival_else_departure": "entry_time",\n        "file_order": "file_order",\n        "first_departure": "first_departure",\n        "train_id": "train_id",\n    }\n    if raw not in aliases:\n        raise ValueError(\n            "train_selection_rule 仅支持 entry_time / file_order / first_departure / train_id，"\n            f"当前值为 {value!r}"\n        )\n    return aliases[raw]\n\n\nclass HSRData:\n    def __init__(self, cfg: Mapping[str, Any]):\n        self.cfg = cfg\n        self.data_dir = resolve_project_path(cfg.get("data_dir"), ".")\n        trains_path = self.data_dir / "trains.csv"\n        stations_path = self.data_dir / "stations.csv"\n        segments_path = self.data_dir / "segments.csv"\n        for path in (trains_path, stations_path, segments_path):\n            if not path.exists():\n                raise FileNotFoundError(f"缺少输入文件：{path}")\n\n        self.trains_df_all = pd.read_csv(trains_path, encoding="utf-8-sig")\n        self.stations_df = pd.read_csv(stations_path, encoding="utf-8-sig")\n        self.segments_df = pd.read_csv(segments_path, encoding="utf-8-sig")\n\n        required_train_cols = {"车次", "车站", "到达时间", "出发时间", "是否停站", "站序"}\n        missing = required_train_cols - set(self.trains_df_all.columns)\n        if missing:\n            raise ValueError(f"trains.csv 缺少列：{sorted(missing)}")\n\n        self.trains_df_all["车次"] = self.trains_df_all["车次"].astype(str)\n        self.trains_df_all["车站"] = self.trains_df_all["车站"].astype(str)\n        self.stations_df["station"] = self.stations_df["station"].astype(str)\n        self.segments_df["from"] = self.segments_df["from"].astype(str)\n        self.segments_df["to"] = self.segments_df["to"].astype(str)\n\n        self.stations: List[str] = self.stations_df["station"].tolist()\n        self.station_index: Dict[str, int] = {s: idx for idx, s in enumerate(self.stations)}\n        self.segments: List[Tuple[str, str]] = list(zip(self.segments_df["from"], self.segments_df["to"]))\n\n        self.all_trains: List[str] = list(pd.unique(self.trains_df_all["车次"]))\n        self.global_train_index: Dict[str, int] = {k: idx for idx, k in enumerate(self.all_trains)}\n        self.train_selection_rule = normalize_train_selection_rule(cfg.get("train_selection_rule", "entry_time"))\n\n        self.station_params: Dict[str, Dict[str, float]] = {}\n        for _, row in self.stations_df.iterrows():\n            self.station_params[str(row["station"])] = {\n                "h_arr": float(row.get("h_arr", 0) or 0),\n                "h_dep": float(row.get("h_dep", 0) or 0),\n                "dwell_min": float(row.get("dwell_min", 0) or 0),\n            }\n\n        self.segment_params: Dict[Tuple[str, str], Dict[str, float]] = {}\n        for _, row in self.segments_df.iterrows():\n            self.segment_params[(str(row["from"]), str(row["to"]))] = {\n                "tau_350": float(row.get("tau_350", 0) or 0),\n                "tau_300": float(row.get("tau_300", 0) or 0),\n                "eta_start_350": float(row.get("eta_start_350", 0) or 0),\n                "eta_end_350": float(row.get("eta_end_350", 0) or 0),\n                "eta_start_300": float(row.get("eta_start_300", 0) or 0),\n                "eta_end_300": float(row.get("eta_end_300", 0) or 0),\n            }\n\n        self.rows_by_train: Dict[str, pd.DataFrame] = {}\n        self.path_by_train: Dict[str, List[str]] = {}\n        self.stop_at: Dict[Tuple[str, str], bool] = {}\n        self.first_arrival: Dict[str, Optional[float]] = {}\n        self.first_dep: Dict[str, float] = {}\n        self.entry_time: Dict[str, float] = {}\n        self.entry_time_source: Dict[str, str] = {}\n        self.intermediate_stop_count: Dict[str, int] = {}\n        self.train_type: Dict[str, int] = {}\n        self.direction: Dict[str, str] = {}\n\n        has_type_col = "train_type" in self.trains_df_all.columns\n        has_direction_col = "方向" in self.trains_df_all.columns\n        for train in self.all_trains:\n            sub = self.trains_df_all[self.trains_df_all["车次"] == train].sort_values("站序").copy()\n            if sub.empty:\n                raise ValueError(f"列车 {train} 没有站点记录")\n            self.rows_by_train[train] = sub\n            self.path_by_train[train] = sub["车站"].astype(str).tolist()\n            for _, row in sub.iterrows():\n                self.stop_at[(train, str(row["车站"]))] = parse_stop(row["是否停站"])\n\n            first = sub.iloc[0]\n            if pd.isna(first["出发时间"]):\n                raise ValueError(f"列车 {train} 的首站没有出发时间")\n            self.first_dep[train] = float(first["出发时间"])\n            if pd.notna(first["到达时间"]):\n                self.first_arrival[train] = float(first["到达时间"])\n                self.entry_time[train] = float(first["到达时间"])\n                self.entry_time_source[train] = "first_arrival"\n            else:\n                self.first_arrival[train] = None\n                self.entry_time[train] = self.first_dep[train]\n                self.entry_time_source[train] = "first_departure"\n\n            self.intermediate_stop_count[train] = sum(\n                int(self.stop_at.get((train, station), False))\n                for station in self.path_by_train[train][1:-1]\n            )\n            if has_type_col and pd.notna(first.get("train_type")):\n                self.train_type[train] = int(float(first["train_type"]))\n            else:\n                self.train_type[train] = infer_train_type(train)\n            self.direction[train] = str(first["方向"]) if has_direction_col else ""\n\n    def ordered_trains(self, rule: Optional[str] = None) -> List[str]:\n        selected_rule = normalize_train_selection_rule(rule or self.train_selection_rule)\n        if selected_rule == "entry_time":\n            return sorted(self.all_trains, key=lambda k: (self.entry_time[k], self.global_train_index[k]))\n        if selected_rule == "file_order":\n            return list(self.all_trains)\n        if selected_rule == "first_departure":\n            return sorted(self.all_trains, key=lambda k: (self.first_dep[k], self.global_train_index[k]))\n        if selected_rule == "train_id":\n            return sorted(self.all_trains, key=lambda k: (train_id_sort_key(k), self.global_train_index[k]))\n        raise AssertionError(f"未处理的 train selection rule: {selected_rule}")\n\n    def select_trains(self, target_k: int) -> List[str]:\n        ordered = self.ordered_trains()\n        if target_k <= 0 or target_k > len(ordered):\n            raise ValueError(f"train_limit={target_k} 必须位于 1..{len(ordered)}")\n        return ordered[:target_k]\n\n    def canonical_pair(self, i: str, j: str) -> Tuple[str, str]:\n        if i == j:\n            raise ValueError("同一列车不能组成 precedence pair")\n        return (i, j) if self.global_train_index[i] < self.global_train_index[j] else (j, i)\n\n    def has_event(self, train: str, station: str, kind: str) -> bool:\n        seq = self.path_by_train.get(train, [])\n        if not seq or station not in seq:\n            return False\n        if kind == "arr":\n            return station != seq[0]\n        if kind == "dep":\n            return station != seq[-1]\n        raise ValueError(f"未知事件类型：{kind}")\n\n    def event_universe(self, trains: Sequence[str]) -> List[PairKey]:\n        universe: List[PairKey] = []\n        for station in self.stations:\n            for kind in ("dep", "arr"):\n                present = [k for k in trains if self.has_event(k, station, kind)]\n                for p in range(len(present)):\n                    for q in range(p + 1, len(present)):\n                        i, j = self.canonical_pair(present[p], present[q])\n                        universe.append((kind, station, i, j))\n        return universe\n\n    def required_run_time(self, train: str, upstream: str, downstream: str) -> float:\n        params = self.segment_params.get((upstream, downstream))\n        if params is None:\n            raise KeyError(f"segments.csv 缺少区间：{upstream}->{downstream}")\n        suffix = "350" if self.train_type.get(train, 300) == 350 else "300"\n        return (\n            params[f"tau_{suffix}"]\n            + (params[f"eta_start_{suffix}"] if self.stop_at.get((train, upstream), False) else 0.0)\n            + (params[f"eta_end_{suffix}"] if self.stop_at.get((train, downstream), False) else 0.0)\n        )\n\n\ndef validate_timetable_csv(\n    data: HSRData,\n    cfg: Mapping[str, Any],\n    trains: Sequence[str],\n    timetable_path: Path,\n    *,\n    expected_objective: Optional[float],\n    fixed_precedence: Optional[Mapping[PairKey, int]],\n    output_dir: Path,\n    validation_tag: str,\n) -> Dict[str, Any]:\n    """Independently verify a saved timetable against the source data.\n\n    The validator deliberately does not inspect Gurobi constraints.  It reads\n    the exported CSV and reconstructs the operational checks from ``trains``,\n    ``stations`` and ``segments``.  This catches both numerical violations and\n    implementation omissions in the optimization model.\n    """\n    wall_start = time.perf_counter()\n    cpu_start = time.process_time()\n    tolerance = float(as_float(cfg.get("validator_tolerance"), 1e-5) or 1e-5)\n    objective_tolerance = float(\n        as_float(cfg.get("validator_objective_tolerance"), 1e-3) or 1e-3\n    )\n    detail_limit = int(as_int(cfg.get("validator_max_detail_rows"), 20000) or 20000)\n    write_details = as_bool(cfg.get("write_validation_details"), True)\n    write_details_on_failure_only = _cfg_flag(cfg, "write_validation_details_on_failure_only", paper_default=True, normal_default=False)\n    output_dir = ensure_dir(output_dir)\n\n    counts: Counter[str] = Counter()\n    details: List[Dict[str, Any]] = []\n    max_violation = 0.0\n    total_violation_count = 0\n\n    def add_violation(category: str, magnitude: float = 1.0, **fields: Any) -> None:\n        nonlocal max_violation, total_violation_count\n        mag = max(0.0, float(magnitude))\n        counts[category] += 1\n        total_violation_count += 1\n        max_violation = max(max_violation, mag)\n        if len(details) < detail_limit:\n            details.append({\n                "category": category,\n                "violation_magnitude": mag,\n                **fields,\n            })\n\n    required_columns = {"车次", "车站", "到达时间", "出发时间"}\n    if not timetable_path.exists():\n        add_violation("missing_timetable_file", 1.0, path=str(timetable_path))\n        df = pd.DataFrame(columns=sorted(required_columns))\n    else:\n        df = pd.read_csv(timetable_path, encoding="utf-8-sig")\n        missing_columns = required_columns - set(df.columns)\n        for column in sorted(missing_columns):\n            add_violation("missing_csv_column", 1.0, column=column)\n        for column in required_columns - missing_columns:\n            if column in {"车次", "车站"}:\n                df[column] = df[column].astype(str)\n        if missing_columns:\n            for column in missing_columns:\n                df[column] = None\n\n    selected = list(trains)\n    selected_set = set(selected)\n    expected_station_rows = sum(len(data.path_by_train[k]) for k in selected)\n\n    # Duplicate and unexpected rows are checked before creating the lookup map.\n    if not df.empty:\n        duplicate_mask = df.duplicated(subset=["车次", "车站"], keep=False)\n        if duplicate_mask.any():\n            duplicate_groups = (\n                df.loc[duplicate_mask]\n                .groupby(["车次", "车站"], dropna=False)\n                .size()\n                .reset_index(name="row_count")\n            )\n            for rec in duplicate_groups.to_dict("records"):\n                extra = max(1, int(rec["row_count"]) - 1)\n                add_violation(\n                    "duplicate_train_station_row",\n                    float(extra),\n                    train=rec["车次"],\n                    station=rec["车站"],\n                    row_count=rec["row_count"],\n                )\n        for rec in df[["车次", "车站"]].drop_duplicates().to_dict("records"):\n            train = str(rec["车次"])\n            station = str(rec["车站"])\n            if train not in selected_set:\n                add_violation("unexpected_train_row", 1.0, train=train, station=station)\n            elif station not in data.path_by_train[train]:\n                add_violation("unexpected_station_row", 1.0, train=train, station=station)\n\n    row_map: Dict[Tuple[str, str], Mapping[str, Any]] = {}\n    for rec in df.to_dict("records"):\n        key = (str(rec.get("车次")), str(rec.get("车站")))\n        row_map.setdefault(key, rec)\n\n    arr: Dict[Tuple[str, str], float] = {}\n    dep: Dict[Tuple[str, str], float] = {}\n\n    def read_numeric(rec: Mapping[str, Any], column: str) -> Optional[float]:\n        value = rec.get(column)\n        if value is None or pd.isna(value) or str(value).strip() == "":\n            return None\n        try:\n            result = float(value)\n            return result if math.isfinite(result) else None\n        except Exception:\n            return None\n\n    for train in selected:\n        seq = data.path_by_train[train]\n        for idx, station in enumerate(seq):\n            rec = row_map.get((train, station))\n            if rec is None:\n                add_violation(\n                    "missing_train_station_row", 1.0, train=train, station=station\n                )\n                continue\n            arr_value = read_numeric(rec, "到达时间")\n            dep_value = read_numeric(rec, "出发时间")\n            if idx > 0:\n                if arr_value is None:\n                    add_violation(\n                        "missing_arrival_event", 1.0, train=train, station=station\n                    )\n                else:\n                    arr[(train, station)] = arr_value\n            elif arr_value is not None:\n                add_violation(\n                    "unexpected_origin_arrival_event",\n                    abs(arr_value),\n                    train=train,\n                    station=station,\n                    actual=arr_value,\n                )\n            if idx < len(seq) - 1:\n                if dep_value is None:\n                    add_violation(\n                        "missing_departure_event", 1.0, train=train, station=station\n                    )\n                else:\n                    dep[(train, station)] = dep_value\n            elif dep_value is not None:\n                add_violation(\n                    "unexpected_destination_departure_event",\n                    abs(dep_value),\n                    train=train,\n                    station=station,\n                    actual=dep_value,\n                )\n\n    # Origin windows.\n    allow_early = as_bool(cfg.get("allow_early_departure"), True)\n    fast_window = float(as_float(cfg.get("dep_window_fast"), 45.0) or 45.0)\n    normal_window = float(as_float(cfg.get("dep_window_normal"), 60.0) or 60.0)\n    fast_type = int(as_int(cfg.get("fast_train_type"), 350) or 350)\n    for train in selected:\n        origin = data.path_by_train[train][0]\n        if (train, origin) not in dep:\n            continue\n        actual = dep[(train, origin)]\n        expected = data.first_dep[train]\n        window = fast_window if data.train_type.get(train, 300) == fast_type else normal_window\n        lower = expected - window if allow_early else expected\n        upper = expected + window\n        if actual < lower - tolerance:\n            add_violation(\n                "origin_window_violation",\n                lower - actual,\n                train=train,\n                station=origin,\n                side="lower",\n                actual=actual,\n                required_bound=lower,\n            )\n        if actual > upper + tolerance:\n            add_violation(\n                "origin_window_violation",\n                actual - upper,\n                train=train,\n                station=origin,\n                side="upper",\n                actual=actual,\n                required_bound=upper,\n            )\n\n    # Running time, dwell time and pass-through checks.\n    for train in selected:\n        seq = data.path_by_train[train]\n        for upstream, downstream in zip(seq, seq[1:]):\n            if (train, upstream) not in dep or (train, downstream) not in arr:\n                continue\n            actual = arr[(train, downstream)] - dep[(train, upstream)]\n            required = data.required_run_time(train, upstream, downstream)\n            if actual < required - tolerance:\n                add_violation(\n                    "running_time_violation",\n                    required - actual,\n                    train=train,\n                    upstream=upstream,\n                    downstream=downstream,\n                    actual=actual,\n                    required=required,\n                )\n        for station in seq[1:-1]:\n            if (train, station) not in arr or (train, station) not in dep:\n                continue\n            actual = dep[(train, station)] - arr[(train, station)]\n            if data.stop_at.get((train, station), False):\n                required = data.station_params.get(station, {}).get("dwell_min", 0.0)\n                if actual < required - tolerance:\n                    add_violation(\n                        "dwell_time_violation",\n                        required - actual,\n                        train=train,\n                        station=station,\n                        actual=actual,\n                        required=required,\n                    )\n            elif abs(actual) > tolerance:\n                add_violation(\n                    "pass_through_violation",\n                    abs(actual),\n                    train=train,\n                    station=station,\n                    actual=actual,\n                    required=0.0,\n                )\n\n    # Station-event headways and, for Full-Fix, agreement with the fixed order.\n    universe = data.event_universe(selected)\n    for key in universe:\n        kind, station, i, j = key\n        time_map = dep if kind == "dep" else arr\n        if (i, station) not in time_map or (j, station) not in time_map:\n            continue\n        ti = time_map[(i, station)]\n        tj = time_map[(j, station)]\n        headway = data.station_params.get(station, {}).get(\n            "h_dep" if kind == "dep" else "h_arr", 0.0\n        )\n        separation = abs(tj - ti)\n        if separation < headway - tolerance:\n            add_violation(\n                f"{kind}_headway_violation",\n                headway - separation,\n                train_i=i,\n                train_j=j,\n                station=station,\n                actual=separation,\n                required=headway,\n            )\n        if fixed_precedence is not None and key in fixed_precedence:\n            value = int(fixed_precedence[key])\n            directional_separation = (tj - ti) if value == 1 else (ti - tj)\n            if directional_separation < headway - tolerance:\n                add_violation(\n                    "fixed_precedence_violation",\n                    headway - directional_separation,\n                    kind=kind,\n                    train_i=i,\n                    train_j=j,\n                    station=station,\n                    fixed_value=value,\n                    actual=directional_separation,\n                    required=headway,\n                )\n\n    # Segment order consistency reconstructed from event times.\n    for upstream, downstream in data.segments:\n        present = [\n            k\n            for k in selected\n            if upstream in data.path_by_train[k]\n            and downstream in data.path_by_train[k]\n            and data.path_by_train[k].index(downstream)\n            == data.path_by_train[k].index(upstream) + 1\n            and (k, upstream) in dep\n            and (k, downstream) in arr\n        ]\n        for p in range(len(present)):\n            for q in range(p + 1, len(present)):\n                i, j = data.canonical_pair(present[p], present[q])\n                dep_i, dep_j = dep[(i, upstream)], dep[(j, upstream)]\n                arr_i, arr_j = arr[(i, downstream)], arr[(j, downstream)]\n                dep_order = 1 if dep_i < dep_j - tolerance else 0 if dep_j < dep_i - tolerance else None\n                arr_order = 1 if arr_i < arr_j - tolerance else 0 if arr_j < arr_i - tolerance else None\n                if dep_order is not None and arr_order is not None and dep_order != arr_order:\n                    add_violation(\n                        "segment_order_violation",\n                        min(abs(dep_i - dep_j), abs(arr_i - arr_j)),\n                        train_i=i,\n                        train_j=j,\n                        upstream=upstream,\n                        downstream=downstream,\n                        departure_order=dep_order,\n                        arrival_order=arr_order,\n                    )\n\n    # Stop-dependent overtaking reconstructed from arrival/departure order changes.\n    for station in data.stations:\n        present = [\n            k\n            for k in selected\n            if (k, station) in arr and (k, station) in dep\n        ]\n        for p in range(len(present)):\n            for q in range(p + 1, len(present)):\n                i, j = data.canonical_pair(present[p], present[q])\n                ai, aj = arr[(i, station)], arr[(j, station)]\n                di, dj = dep[(i, station)], dep[(j, station)]\n                if ai < aj - tolerance and di > dj + tolerance:\n                    if not data.stop_at.get((i, station), False):\n                        add_violation(\n                            "stop_overtaking_violation",\n                            min(aj - ai, di - dj),\n                            overtaken_train=i,\n                            passing_train=j,\n                            station=station,\n                        )\n                elif aj < ai - tolerance and dj > di + tolerance:\n                    if not data.stop_at.get((j, station), False):\n                        add_violation(\n                            "stop_overtaking_violation",\n                            min(ai - aj, dj - di),\n                            overtaken_train=j,\n                            passing_train=i,\n                            station=station,\n                        )\n\n    recomputed_tt: Optional[float] = None\n    objective_abs_diff: Optional[float] = None\n    if all(\n        (train, data.path_by_train[train][0]) in dep\n        and (train, data.path_by_train[train][-1]) in arr\n        for train in selected\n    ):\n        recomputed_tt = sum(\n            arr[(train, data.path_by_train[train][-1])]\n            - dep[(train, data.path_by_train[train][0])]\n            for train in selected\n        )\n        if expected_objective is not None and math.isfinite(float(expected_objective)):\n            objective_abs_diff = abs(recomputed_tt - float(expected_objective))\n            if objective_abs_diff > objective_tolerance:\n                add_violation(\n                    "objective_recalculation_violation",\n                    objective_abs_diff,\n                    recomputed_TT=recomputed_tt,\n                    reported_TT=float(expected_objective),\n                    tolerance=objective_tolerance,\n                )\n\n    wall_sec = time.perf_counter() - wall_start\n    cpu_sec = time.process_time() - cpu_start\n    passed = total_violation_count == 0\n    summary: Dict[str, Any] = {\n        "validator_ran": 1,\n        "validator_pass": int(passed),\n        "validator_feasible_text": "Yes" if passed else "No",\n        "validator_tolerance": tolerance,\n        "validator_objective_tolerance": objective_tolerance,\n        "validator_timetable_rows": len(df),\n        "validator_expected_rows": expected_station_rows,\n        "validator_total_violations": total_violation_count,\n        "validator_max_violation": max_violation,\n        "validator_recomputed_TT": recomputed_tt,\n        "validator_reported_TT": expected_objective,\n        "validator_objective_abs_diff": objective_abs_diff,\n        "validator_detail_rows_written": (len(details) if write_details and ((not write_details_on_failure_only) or (not passed)) else 0),\n        "validator_details_truncated": int(total_violation_count > len(details) and write_details and ((not write_details_on_failure_only) or (not passed))),\n        "validator_wall_sec": wall_sec,\n        "validator_process_cpu_sec": cpu_sec,\n        "validator_missing_row_count": counts["missing_train_station_row"],\n        "validator_missing_event_count": (\n            counts["missing_arrival_event"] + counts["missing_departure_event"]\n        ),\n        "validator_duplicate_row_count": counts["duplicate_train_station_row"],\n        "validator_origin_window_violation_count": counts["origin_window_violation"],\n        "validator_running_time_violation_count": counts["running_time_violation"],\n        "validator_dwell_time_violation_count": counts["dwell_time_violation"],\n        "validator_pass_through_violation_count": counts["pass_through_violation"],\n        "validator_departure_headway_violation_count": counts["dep_headway_violation"],\n        "validator_arrival_headway_violation_count": counts["arr_headway_violation"],\n        "validator_fixed_precedence_violation_count": counts["fixed_precedence_violation"],\n        "validator_segment_order_violation_count": counts["segment_order_violation"],\n        "validator_stop_overtaking_violation_count": counts["stop_overtaking_violation"],\n        "validator_objective_violation_count": counts["objective_recalculation_violation"],\n    }\n\n    summary_path = output_dir / f"validation_{validation_tag}_summary.csv"\n    details_path = output_dir / f"validation_{validation_tag}_violations.csv"\n    write_csv(pd.DataFrame([summary]), summary_path)\n    should_write_details = write_details and ((not write_details_on_failure_only) or (not passed))\n    if should_write_details:\n        detail_columns = [\n            "category", "violation_magnitude", "train", "train_i", "train_j",\n            "overtaken_train", "passing_train", "station", "upstream", "downstream",\n            "kind", "side", "actual", "required", "required_bound", "fixed_value",\n            "departure_order", "arrival_order", "recomputed_TT", "reported_TT",\n            "tolerance", "path", "column", "row_count",\n        ]\n        detail_df = pd.DataFrame(details)\n        for column in detail_columns:\n            if column not in detail_df.columns:\n                detail_df[column] = None\n        write_csv(detail_df[detail_columns], details_path)\n    elif details_path.exists():\n        details_path.unlink()\n    return summary\n\n\ndef _paper_output_mode(cfg: Mapping[str, Any]) -> bool:\n    return as_bool(cfg.get("paper_output_mode"), False)\n\n\ndef _cfg_flag(cfg: Mapping[str, Any], key: str, *, paper_default: bool, normal_default: bool) -> bool:\n    if key in cfg and cfg.get(key) not in {None, ""}:\n        return as_bool(cfg.get(key), paper_default if _paper_output_mode(cfg) else normal_default)\n    return paper_default if _paper_output_mode(cfg) else normal_default\n\n'
_MODULE_SOURCES['baseline_lib.common'] = '# -*- coding: utf-8 -*-\n"""Input compilation and the shared Appendix-C earliest-event decoder.\n\nNo LP/MILP solver, learned precedence or packing information is used here.\nAll timetable/event times are in minutes; elapsed runtimes are in seconds.\n"""\nfrom __future__ import annotations\nimport csv\nimport hashlib\nimport json\nimport math\nimport time\nfrom dataclasses import dataclass\nfrom pathlib import Path\nfrom typing import Optional, Sequence\nimport pandas as pd\nfrom .config_loader import load_config, as_bool\nfrom .reference_model import HSRData\n\n\nclass DeadlineReached(Exception):\n    """A candidate that has not finished decoding must not be used."""\n\n\nclass Budget:\n    def __init__(self, seconds: float):\n        if not math.isfinite(seconds) or seconds <= 0:\n            raise ValueError(\'time_limit_sec must be finite and positive\')\n        self.start = time.perf_counter()\n        self.deadline = self.start + seconds\n        self.cpu_start = time.process_time()\n        self.seconds = seconds\n\n    def check(self) -> None:\n        if time.perf_counter() >= self.deadline:\n            raise DeadlineReached()\n\n    @property\n    def elapsed(self) -> float:\n        return time.perf_counter() - self.start\n\n\ndef file_sha(path: Path) -> str:\n    return hashlib.sha256(path.read_bytes()).hexdigest()\n\n\ndef digest(value) -> str:\n    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,\n                                     separators=(\',\', \':\'), allow_nan=False).encode()).hexdigest()\n\n\ndef atomic_json(path: Path, value) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    tmp = path.with_suffix(path.suffix + \'.tmp\')\n    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding=\'utf-8\')\n    tmp.replace(path)\n\n\ndef write_rows(path: Path, rows: list[dict], fields: Optional[list[str]] = None) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    if fields is None:\n        fields = list(dict.fromkeys(k for row in rows for k in row))\n    tmp = path.with_suffix(path.suffix + \'.tmp\')\n    with tmp.open(\'w\', encoding=\'utf-8-sig\', newline=\'\') as stream:\n        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction=\'ignore\')\n        writer.writeheader()\n        writer.writerows(rows)\n    tmp.replace(path)\n\n\ndef load_inputs(data_dir: Path, config_path: Optional[Path] = None):\n    data_dir = data_dir.resolve()\n    config_path = (config_path or data_dir / \'config.csv\').resolve()\n    cfg = load_config(str(config_path))\n    # An explicit data directory always wins over the source config\'s relative \'.\'.\n    cfg[\'data_dir\'] = str(data_dir)\n    cfg.setdefault(\'train_selection_rule\', \'entry_time\')\n    for key in (\'dep_window_fast\', \'dep_window_normal\'):\n        # Positive windows avoid the legacy validator\'s `value or default` zero case.\n        if not math.isfinite(float(cfg.get(key, 0))) or float(cfg[key]) <= 0:\n            raise ValueError(f\'{key} must be positive; zero-width windows require a validator update\')\n    data = HSRData(cfg)\n    paths = [data_dir / n for n in (\'trains.csv\', \'stations.csv\', \'segments.csv\')]\n    source = {\'files\': {p.name: file_sha(p) for p in paths},\n              \'config_sha256\': file_sha(config_path), \'data_dir\': str(data_dir),\n              \'config_path\': str(config_path)}\n    return data, cfg, source\n\n\n@dataclass(frozen=True)\nclass Train:\n    name: str\n    stations: tuple[int, ...]\n    run: tuple[float, ...]\n    dwell: tuple[Optional[float], ...]  # None: a nonstopping intermediate event\n    orig_arr: tuple[Optional[float], ...]\n    orig_dep: tuple[Optional[float], ...]\n    lower: float\n    upper: float\n    terminal_reference: Optional[float]\n\n\nclass Instance:\n    def __init__(self, data: HSRData, cfg: dict, K: int):\n        self.data = data\n        self.cfg = cfg\n        self.ids = tuple(data.select_trains(K))\n        self.K = len(self.ids)\n        self.station_names = tuple(data.stations)\n        self.harr = tuple(data.station_params[s][\'h_arr\'] for s in data.stations)\n        self.hdep = tuple(data.station_params[s][\'h_dep\'] for s in data.stations)\n        self.tolerance = float(cfg.get(\'validator_tolerance\', 1e-5))\n        self.trains: list[Train] = []\n        allow_early = as_bool(cfg.get(\'allow_early_departure\'), True)\n        fast_type = int(cfg.get(\'fast_train_type\', 350))\n        for name in self.ids:\n            path = data.path_by_train[name]\n            if len(path) < 2 or len(set(path)) != len(path):\n                raise ValueError(f\'{name}: route must contain at least two distinct, nonrepeated stations\')\n            for u, v in zip(path, path[1:]):\n                if (u, v) not in data.segment_params:\n                    raise ValueError(f\'{name}: missing segment {u}->{v}\')\n            rows = data.rows_by_train[name].to_dict(\'records\')\n            def optional(x):\n                if pd.isna(x):\n                    return None\n                value = float(x)\n                if not math.isfinite(value):\n                    raise ValueError(f\'{name}: nonfinite original event time\')\n                return value\n            oa = [optional(r[\'到达时间\']) for r in rows]\n            od = [optional(r[\'出发时间\']) for r in rows]\n            terminal_reference = oa[-1] if oa[-1] is not None else od[-1]\n            oa[0] = None  # no origin arrival event\n            od[-1] = None  # no destination departure event\n            width = float(cfg[\'dep_window_fast\'] if data.train_type[name] == fast_type\n                          else cfg[\'dep_window_normal\'])\n            ref = data.first_dep[name]\n            train = Train(name, tuple(data.station_index[s] for s in path),\n                          tuple(data.required_run_time(name, u, v) for u, v in zip(path, path[1:])),\n                          tuple(data.station_params[s][\'dwell_min\']\n                                if data.stop_at[(name, s)] else None for s in path),\n                          tuple(oa), tuple(od), ref - width if allow_early else ref,\n                          ref + width, terminal_reference)\n            self.trains.append(train)\n        self.trains = tuple(self.trains)\n        numeric = list(self.harr) + list(self.hdep)\n        numeric += [x for tr in self.trains for x in tr.run]\n        numeric += [x for tr in self.trains for x in tr.dwell if x is not None]\n        if any(not math.isfinite(x) or x < 0 for x in numeric):\n            raise ValueError(\'Headways, running times and dwell minima must be finite and nonnegative\')\n        self.fcfs = tuple(sorted(range(self.K), key=lambda i: (\n            data.first_dep[self.ids[i]], data.global_train_index[self.ids[i]])))\n        self.base_rank = {i: p for p, i in enumerate(self.fcfs)}\n        self.event_count = sum(2 * len(tr.stations) - 2 for tr in self.trains)\n\n    def selected_rows(self):\n        return [dict(selection_rank=p + 1, train_id=k,\n                     selection_rule=self.data.train_selection_rule,\n                     entry_time=self.data.entry_time[k],\n                     entry_time_source=self.data.entry_time_source[k],\n                     first_departure=self.data.first_dep[k],\n                     train_type=self.data.train_type[k],\n                     origin=self.data.path_by_train[k][0],\n                     destination=self.data.path_by_train[k][-1])\n                for p, k in enumerate(self.ids)]\n\n\n@dataclass\nclass Decoded:\n    order: tuple[int, ...]\n    F: tuple[float, float, float, float]\n    violation_count: int\n    violation_sum: float\n    violation_max: float\n    terminal_delay: tuple[float, ...]\n    event_delay: tuple[float, ...]\n    origin_excess: tuple[float, ...]\n    # The fast search does not need all event times. Materialize only for export.\n    times: Optional[tuple] = None\n\n    @property\n    def feasible(self) -> bool:\n        return self.violation_count == 0\n\n    @property\n    def tt(self) -> float:\n        return self.F[0]\n\n    @property\n    def key(self) -> tuple:\n        # Feasibility dominance before the unmodified Appendix-C tuple F.\n        # Violation components are all exactly zero for feasible candidates.\n        return (self.violation_count, self.violation_sum, self.violation_max) + self.F\n\n\ndef decode(instance: Instance, order: Sequence[int], budget: Optional[Budget] = None,\n           *, materialize: bool = False) -> Decoded:\n    """Forward earliest construction for one *complete* global train order.\n\n    Earlier trains provide resource lower bounds to later trains. At a pass\n    station the arrival and departure variables are contracted into one event.\n    The origin upper bound is checked, never clipped or relaxed for reporting.\n    Original-time lower bounds are the conservative rule stated in Appendix C.\n    Segment entry/exit headways are covered by the same physical station events;\n    eta_start/eta_end are ONLY running-time supplements, never headways.\n    """\n    order = tuple(order)\n    if len(order) != instance.K or set(order) != set(range(instance.K)):\n        raise ValueError(\'Decoder requires a full permutation; partial orders are not timetables\')\n    if budget:\n        budget.check()\n    nstations = len(instance.station_names)\n    last_arr = [-math.inf] * nstations\n    last_dep = [-math.inf] * nstations\n    terminal = [0.0] * instance.K\n    event = [0.0] * instance.K\n    excesses = [0.0] * instance.K\n    all_times = [None] * instance.K if materialize else None\n    tt = 0.0\n    violations = 0\n    excess_sum = excess_max = 0.0\n    for i in order:\n        if budget:\n            budget.check()\n        tr = instance.trains[i]\n        first = tr.stations[0]\n        original = tr.orig_dep[0]\n        assert original is not None\n        departure = max(0.0, tr.lower, original, last_dep[first] + instance.hdep[first])\n        first_dep = departure\n        last_dep[first] = departure\n        event_i = max(0.0, departure - original)\n        bad = max(0.0, tr.lower - departure, departure - tr.upper)\n        if bad > instance.tolerance:\n            violations += 1\n            excess_sum += bad\n            excess_max = max(excess_max, bad)\n            excesses[i] = bad\n        times = [(None, departure)] if materialize else None\n        final_index = len(tr.stations) - 1\n        for p in range(1, len(tr.stations)):\n            s = tr.stations[p]\n            oa = tr.orig_arr[p]\n            arrival = max(departure + tr.run[p - 1],\n                          last_arr[s] + instance.harr[s], 0.0)\n            if oa is not None:\n                arrival = max(arrival, oa)\n            if p == final_index:\n                last_arr[s] = arrival\n                if oa is not None:\n                    event_i += max(0.0, arrival - oa)\n                if materialize:\n                    times.append((arrival, None))\n                break\n            od = tr.orig_dep[p]\n            dep_lb = max(last_dep[s] + instance.hdep[s],\n                         od if od is not None else 0.0)\n            if tr.dwell[p] is None:\n                # Passing: any departure-resource wait moves the arrival too.\n                arrival = departure = max(arrival, dep_lb)\n            else:\n                departure = max(arrival + tr.dwell[p], dep_lb)\n            last_arr[s], last_dep[s] = arrival, departure\n            if oa is not None:\n                event_i += max(0.0, arrival - oa)\n            if od is not None:\n                event_i += max(0.0, departure - od)\n            if materialize:\n                times.append((arrival, departure))\n        tt += arrival - first_dep\n        terminal[i] = max(0.0, arrival - tr.terminal_reference) if tr.terminal_reference is not None else 0.0\n        event[i] = event_i\n        if materialize:\n            all_times[i] = tuple(times)\n    if budget:\n        budget.check()\n    return Decoded(order, (tt, sum(terminal), max(terminal, default=0.0), sum(event)),\n                   violations, excess_sum, excess_max, tuple(terminal), tuple(event),\n                   tuple(excesses), tuple(all_times) if materialize else None)\n\n\ndef timetable_rows(instance: Instance, result: Decoded) -> list[dict]:\n    if result.times is None:\n        raise ValueError(\'Materialized times required for export\')\n    rows = []\n    for i, tr in enumerate(instance.trains):\n        for p, (a, d) in enumerate(result.times[i]):\n            rows.append({\'车次\': tr.name, \'车站\': instance.station_names[tr.stations[p]],\n                         \'到达时间\': a, \'出发时间\': d, \'站序\': p,\n                         \'是否停站\': instance.data.stop_at[(tr.name, instance.station_names[tr.stations[p]])],\n                         \'train_type\': instance.data.train_type[tr.name]})\n    return rows\n'
_MODULE_SOURCES['baseline_lib.search'] = '# -*- coding: utf-8 -*-\n"""Appendix-C TS-P and ALNS-R; all candidate evaluations share one deadline."""\nfrom __future__ import annotations\nimport math\nimport random\nimport time\nfrom dataclasses import dataclass, field\nfrom typing import Callable, Optional\nfrom .common import Budget, Decoded, DeadlineReached, Instance, decode\n\nDESTROY = (\'random_removal\', \'worst_event_delay_removal\', \'worst_terminal_delay_removal\',\n           \'time_window_removal\', \'current_block_removal\')\nREPAIR = (\'random_insertion\', \'original_order_insertion\', \'greedy_best_insertion\', \'regret2_insertion\')\n\n\n@dataclass\nclass SearchState:\n    best: Optional[Decoded] = None           # best feasibility-dominance key\n    best_feasible: Optional[Decoded] = None # ONLY full, operationally feasible candidates\n    first_feasible_sec: Optional[float] = None\n    best_feasible_sec: Optional[float] = None\n    best_iteration: int = 0\n    evaluations: int = 0\n    iteration: int = 0\n    current: Optional[Decoded] = None\n    initial: Optional[Decoded] = None\n    history: list[dict] = field(default_factory=list)\n    operators: list[dict] = field(default_factory=list)\n    stop_reason: str = \'\'\n    search_runtime_sec: float = 0.0\n    process_cpu_sec: float = 0.0\n\n\nclass Search:\n    def __init__(self, instance: Instance, seconds: float, seed: int,\n                 progress: Optional[Callable[[dict], None]] = None):\n        self.instance = instance\n        self.budget = Budget(seconds)\n        self.rng = random.Random(seed)\n        self.state = SearchState()\n        self.progress = progress\n        self.last_progress = 0.0\n\n    def evaluate(self, order) -> Decoded:\n        result = decode(self.instance, order, self.budget)\n        self.state.evaluations += 1\n        if self.progress and self.budget.elapsed - self.last_progress >= 15.0:\n            self.emit_progress(\'SEARCHING\')\n        return result\n\n    def consider_incumbent(self, result: Decoded) -> None:\n        """Commit an initialized/accepted complete outer-iteration candidate.\n\n        Inner insertion trials are scoring evaluations, not accepted repairs.\n        An interrupted ALNS repair therefore leaves the prior incumbent intact.\n        """\n        if self.state.best is None or result.key < self.state.best.key:\n            self.state.best = result\n        if result.feasible and (self.state.best_feasible is None or result.F < self.state.best_feasible.F):\n            elapsed = self.budget.elapsed\n            if self.state.first_feasible_sec is None:\n                self.state.first_feasible_sec = elapsed\n            self.state.best_feasible = result\n            self.state.best_feasible_sec = elapsed\n            self.state.best_iteration = self.state.iteration\n\n    def emit_progress(self, stage: str):\n        best = self.state.best\n        self.last_progress = self.budget.elapsed\n        if self.progress:\n            self.progress(dict(stage=stage, iteration=self.state.iteration,\n                               elapsed_sec=self.budget.elapsed,\n                               evaluations=self.state.evaluations,\n                               best_feasible_TT=self.state.best_feasible.tt if self.state.best_feasible else None,\n                               best_violation_count=best.violation_count if best else None,\n                               best_violation_sum=best.violation_sum if best else None))\n\n    def initialize(self):\n        initial = self.evaluate(self.instance.fcfs)\n        self.state.initial = self.state.current = initial\n        self.consider_incumbent(initial)\n        self.emit_progress(\'INITIALIZED\')\n        self.record(iteration=0, move=\'FCFS\', accepted=1, no_improve=0)\n        return initial\n\n    def record(self, **fields):\n        current, best, feasible = self.state.current, self.state.best, self.state.best_feasible\n        row = dict(elapsed_sec=self.budget.elapsed, evaluations=self.state.evaluations,\n                   current_TT=current.tt if current else None,\n                   current_origin_violations=current.violation_count if current else None,\n                   current_excess_sum=current.violation_sum if current else None,\n                   best_search_TT=best.tt if best else None,\n                   best_origin_violations=best.violation_count if best else None,\n                   best_excess_sum=best.violation_sum if best else None,\n                   best_feasible_TT=feasible.tt if feasible else None,\n                   **fields)\n        self.state.history.append(row)\n\n    def finish(self, reason):\n        self.state.stop_reason = reason\n        self.state.search_runtime_sec = self.budget.elapsed\n        self.state.process_cpu_sec = time.process_time() - self.budget.cpu_start\n        self.emit_progress(reason)\n        return self.state\n\n\ndef candidate_swaps(instance: Instance, current: Decoded, rng: random.Random,\n                    params: dict) -> list[tuple[int, int]]:\n    """Appendix-C local/adjacent/random swaps with bounded source quotas.\n\n    While no feasible current order exists, origin excess precedes terminal\n    delay in the intensification ranking; the swap operators do not change.\n    """\n    order = current.order\n    n = len(order)\n    size = int(params[\'candidate_list_size\'])\n    indices = {k: p for p, k in enumerate(order)}\n    delayed = sorted(order, key=lambda k: (\n        -current.origin_excess[k] if not current.feasible else 0.0,\n        -current.terminal_delay[k], instance.base_rank[k]))[:params[\'delayed_top_k\']]\n    moves = []\n    for k in delayed:\n        i = indices[k]\n        if i > 0:\n            moves.append((i - 1, i))\n        if i + 1 < n:\n            moves.append((i, i + 1))\n        for j in range(max(0, i - params[\'local_window\']), min(n, i + params[\'local_window\'] + 1)):\n            if j != i:\n                moves.append((i, j))\n    # Appendix C requires all three move sources. Do NOT let a large local\n    # pool suppress random long-range moves (a defect in the supplied code).\n    # With size 80: 40 local + 20 adjacent + 20 long-range, with deduplication.\n    target = min(size, n * (n - 1) // 2)\n    local = list(dict.fromkeys((min(i, j), max(i, j)) for i, j in moves))\n    rng.shuffle(local)\n    selected = local[:min(len(local), (target + 1) // 2)]\n    seen = set(selected)\n    adjacent = [(i, i + 1) for i in range(n - 1) if (i, i + 1) not in seen]\n    rng.shuffle(adjacent)\n    for move in adjacent[:target // 4]:\n        selected.append(move)\n        seen.add(move)\n    # Prefer genuinely nonlocal pairs where the instance size permits them.\n    minimum_distance = params[\'local_window\'] + 1 if n > params[\'local_window\'] + 1 else 2\n    for _ in range(max(20, 20 * target)):\n        if len(selected) >= target or n < 2:\n            break\n        i, j = sorted(rng.sample(range(n), 2))\n        if j - i >= minimum_distance and (i, j) not in seen:\n            selected.append((i, j))\n            seen.add((i, j))\n    # Small-instance / exhausted-pool fallback; bounded, no retry-forever loop.\n    if len(selected) < target:\n        remainder = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in seen]\n        rng.shuffle(remainder)\n        selected.extend(remainder[:target - len(selected)])\n    rng.shuffle(selected)\n    return selected\n\n\ndef run_tsp(instance: Instance, params: dict, seconds: float, seed: int,\n            max_iterations: Optional[int] = None, progress=None) -> SearchState:\n    search = Search(instance, seconds, seed, progress)\n    tabu: dict[tuple[int, int], int] = {}\n    stagnant = 0\n    reason = \'NO_IMPROVEMENT\'\n    try:\n        current = search.initialize()\n        while True:\n            search.budget.check()\n            if stagnant >= params[\'max_no_improve\']:\n                break\n            if max_iterations is not None and search.state.iteration >= max_iterations:\n                reason = \'ITERATION_LIMIT\'\n                break\n            search.state.iteration += 1\n            it = search.state.iteration\n            best_before = search.state.best.key\n            chosen = None\n            chosen_move = None\n            timeout = False\n            for i, j in candidate_swaps(instance, current, search.rng, params):\n                trial = list(current.order)\n                attr = tuple(sorted((trial[i], trial[j])))\n                trial[i], trial[j] = trial[j], trial[i]\n                # Aspiration compares with the incumbent BEFORE evaluating this move.\n                aspiration_reference = search.state.best.key\n                try:\n                    candidate = search.evaluate(trial)\n                except DeadlineReached:\n                    timeout = True\n                    break\n                is_tabu = tabu.get(attr, -1) >= it\n                admissible = (not is_tabu) or candidate.key < aspiration_reference\n                if admissible and (chosen is None or candidate.key < chosen.key):\n                    chosen, chosen_move = candidate, attr\n            if chosen is not None:\n                current = chosen\n                search.state.current = current\n                search.consider_incumbent(current)\n                tabu[chosen_move] = it + params[\'tabu_tenure\']\n                tabu = {k: expiry for k, expiry in tabu.items() if expiry >= it}\n            improved = search.state.best.key < best_before\n            stagnant = 0 if improved else stagnant + 1\n            search.record(iteration=it, move=\'pairwise_swap\',\n                          swapped_train_1=instance.ids[chosen_move[0]] if chosen_move else \'\',\n                          swapped_train_2=instance.ids[chosen_move[1]] if chosen_move else \'\',\n                          accepted=int(chosen is not None), improved_best=int(improved),\n                          no_improve=stagnant)\n            if timeout:\n                reason = \'TIME_LIMIT\'\n                break\n            # A temporarily all-tabu candidate list counts as a stagnant iteration;\n            # tenure expires naturally instead of inventing a new neighborhood.\n    except DeadlineReached:\n        reason = \'TIME_LIMIT\'\n    except KeyboardInterrupt:\n        reason = \'INTERRUPTED\'\n    return search.finish(reason)\n\n\ndef destroy(instance: Instance, current: Decoded, operator: str, q: int,\n            rng: random.Random) -> tuple[list[int], list[int]]:\n    order = list(current.order)\n    q = max(1, min(q, len(order) - 1))\n    rank = instance.base_rank\n    if operator == \'random_removal\':\n        removed = rng.sample(order, q)\n    elif operator == \'worst_event_delay_removal\':\n        ranked = sorted(order, key=lambda k: (-current.event_delay[k], rank[k]))\n        elite = ranked[:max(q, min(len(order), q * 3))]\n        removed = elite[:max(1, int(round(q * 0.75)))]\n        pool = [k for k in elite if k not in removed] + [k for k in order if k not in removed]\n        while len(removed) < q:\n            k = rng.choice(pool)\n            removed.append(k)\n            pool = [x for x in pool if x != k]\n    elif operator == \'worst_terminal_delay_removal\':\n        removed = sorted(order, key=lambda k: (-current.terminal_delay[k], rank[k]))[:q]\n    elif operator == \'time_window_removal\':\n        k0 = rng.choice(order)\n        t0 = instance.data.first_dep[instance.ids[k0]]\n        removed = sorted(order, key=lambda k: (\n            abs(instance.data.first_dep[instance.ids[k]] - t0), rank[k]))[:q]\n    elif operator == \'current_block_removal\':\n        start = rng.randrange(len(order) - q + 1)\n        removed = order[start:start + q]\n    else:\n        raise ValueError(f\'Unknown destroy operator: {operator}\')\n    selected = set(removed)\n    return [k for k in order if k not in selected], removed\n\n\ndef insertion_positions(order: list[int], k: int, rank: dict, limit: int,\n                        rng: random.Random) -> list[int]:\n    n = len(order)\n    if n + 1 <= limit:\n        return list(range(n + 1))\n    target = sum(rank[x] < rank[k] for x in order)\n    positions = {0, n, max(0, min(n, target))}\n    for delta in (-8, -5, -3, -2, -1, 1, 2, 3, 5, 8):\n        if 0 <= target + delta <= n:\n            positions.add(target + delta)\n    while len(positions) < limit:\n        positions.add(rng.randrange(n + 1))\n    result = sorted(positions)\n    rng.shuffle(result)\n    return result[:limit]\n\n\ndef insert(order, pos, k):\n    return order[:pos] + [k] + order[pos:]\n\n\ndef regret_priority(results: list[tuple[Decoded, list[int]]]):\n    """Return (priority, best insertion) without confusing violations with TT.\n\n    Only TT is differenced for feasible positions. With none feasible, restore\n    feasibility using the candidate key, not a fictitious feasible objective.\n    """\n    results.sort(key=lambda item: item[0].key)\n    feasible = [x for x in results if x[0].feasible]\n    if len(feasible) == 1:\n        return (2, 0.0), feasible[0]\n    if len(feasible) >= 2:\n        return (1, feasible[1][0].tt - feasible[0][0].tt), feasible[0]\n    return (0, 0.0), results[0]\n\n\ndef repair(search: Search, partial: list[int], removed: list[int], operator: str,\n           max_positions: int) -> Decoded:\n    instance, rng = search.instance, search.rng\n    rank = instance.base_rank\n    order, remaining = list(partial), list(removed)\n    if operator == \'random_insertion\':\n        rng.shuffle(remaining)\n        for k in remaining:\n            search.budget.check()\n            order.insert(rng.randrange(len(order) + 1), k)\n        return search.evaluate(order)\n    if operator == \'original_order_insertion\':\n        for k in sorted(remaining, key=rank.__getitem__):\n            search.budget.check()\n            pos = sum(rank[x] < rank[k] for x in order)\n            order.insert(pos, k)\n        return search.evaluate(order)\n    if operator == \'greedy_best_insertion\':\n        rng.shuffle(remaining)\n        final = None\n        while remaining:\n            search.budget.check()\n            k = remaining.pop(0)\n            chosen = None\n            for p in insertion_positions(order, k, rank, max_positions, rng):\n                search.budget.check()\n                trial_partial = insert(order, p, k)\n                # Preserve the supplied implementation\'s full-order completion.\n                # Remaining removed trains are appended in their FCFS order.\n                trial_full = trial_partial + sorted(remaining, key=rank.__getitem__)\n                candidate = search.evaluate(trial_full)\n                if chosen is None or candidate.key < chosen[0].key:\n                    chosen = (candidate, trial_partial)\n            final, order = chosen\n        assert final is not None and tuple(order) == final.order\n        return final\n    if operator == \'regret2_insertion\':\n        final = None\n        while remaining:\n            search.budget.check()\n            chosen = None\n            chosen_priority = None\n            chosen_k = None\n            for k in remaining:\n                other = [x for x in remaining if x != k]\n                results = []\n                for p in insertion_positions(order, k, rank, max_positions, rng):\n                    search.budget.check()\n                    trial_partial = insert(order, p, k)\n                    trial_full = trial_partial + sorted(other, key=rank.__getitem__)\n                    results.append((search.evaluate(trial_full), trial_partial))\n                priority, best = regret_priority(results)\n                if (chosen is None or priority > chosen_priority or\n                        (priority == chosen_priority and best[0].key < chosen[0].key)):\n                    chosen, chosen_priority, chosen_k = best, priority, k\n            final, order = chosen\n            remaining.remove(chosen_k)\n        assert final is not None and tuple(order) == final.order\n        return final\n    raise ValueError(f\'Unknown repair operator: {operator}\')\n\n\ndef sa_accept(current: Decoded, candidate: Decoded, temperature: float,\n              rng: random.Random) -> tuple[bool, str]:\n    if candidate.key < current.key:\n        return True, \'improve_current\'\n    if current.feasible and not candidate.feasible:\n        return False, \'reject_infeasible\'\n    if not candidate.feasible:\n        # Appendix C does not prescribe an infeasibility annealing schedule.\n        # Use monotone feasibility recovery, with neutral plateau moves allowed.\n        return candidate.key == current.key, \'recovery_neutral\' if candidate.key == current.key else \'recovery_reject\'\n    delta_tt = candidate.tt - current.tt\n    probability = math.exp(-max(0.0, delta_tt) / temperature) if temperature > 1e-12 else 0.0\n    accepted = rng.random() < probability\n    return accepted, \'sa_accept\' if accepted else \'reject\'\n\n\ndef roulette(weights: dict[str, float], rng: random.Random) -> str:\n    value = rng.random() * sum(weights.values())\n    cumulative = 0.0\n    for operator, weight in weights.items():\n        cumulative += weight\n        if value <= cumulative:\n            return operator\n    return next(reversed(weights))\n\n\ndef smooth(weights, scores, uses, rho, minimum_weight=0.05):\n    for op in weights:\n        if uses[op]:\n            weights[op] = max(minimum_weight, (1 - rho) * weights[op] + rho * scores[op] / uses[op])\n        else:\n            weights[op] = max(minimum_weight, weights[op])\n        scores[op], uses[op] = 0.0, 0\n\n\ndef run_alns(instance: Instance, params: dict, seconds: float, seed: int,\n             max_iterations: Optional[int] = None, progress=None) -> SearchState:\n    search = Search(instance, seconds, seed, progress)\n    dweights = {op: 1.0 for op in DESTROY}\n    rweights = {op: 1.0 for op in REPAIR}\n    scores = {op: 0.0 for op in DESTROY + REPAIR}\n    uses = {op: 0 for op in DESTROY + REPAIR}\n    totals = {op: 0 for op in DESTROY + REPAIR}\n    total_scores = {op: 0.0 for op in DESTROY + REPAIR}\n    stagnant = 0\n    reason = \'NO_IMPROVEMENT\'\n    try:\n        current = search.initialize()\n        temperature = max(float(params[\'initial_temperature_floor\']),\n                          float(params[\'initial_temperature_fraction\']) * max(1.0, current.tt))\n        while True:\n            search.budget.check()\n            if stagnant >= params[\'max_no_improve\']:\n                break\n            if max_iterations is not None and search.state.iteration >= max_iterations:\n                reason = \'ITERATION_LIMIT\'\n                break\n            search.state.iteration += 1\n            it = search.state.iteration\n            best_before = search.state.best.key\n            current_before = current\n            d = roulette(dweights, search.rng)\n            r = roulette(rweights, search.rng)\n            q = search.rng.randint(params[\'min_remove\'], params[\'max_remove\'])\n            partial, removed = destroy(instance, current, d, q, search.rng)\n            candidate = repair(search, partial, removed, r, params[\'max_insertion_positions\'])\n            accepted, accept_reason = sa_accept(current, candidate, temperature, search.rng)\n            if accepted:\n                current = candidate\n                search.state.current = current\n                search.consider_incumbent(current)\n            # Appendix C updates the incumbent/reward after the complete repair\n            # has been accepted, not on temporary insertion-scoring completions.\n            improved_best = search.state.best.key < best_before\n            improved_current = candidate.key < current_before.key\n            stagnant = 0 if improved_best else stagnant + 1\n            reward = (params[\'rewards\'][0] if improved_best else params[\'rewards\'][1]\n                      if accepted and improved_current else params[\'rewards\'][2] if accepted else 0.0)\n            for op in (d, r):\n                scores[op] += reward\n                uses[op] += 1\n                totals[op] += 1\n                total_scores[op] += reward\n            search.record(iteration=it, destroy_operator=d, repair_operator=r,\n                          removed_count=len(removed), candidate_TT=candidate.tt,\n                          candidate_origin_violations=candidate.violation_count,\n                          delta_TT=candidate.tt - current_before.tt,\n                          accepted=int(accepted), accept_reason=accept_reason,\n                          improved_best=int(improved_best), no_improve=stagnant,\n                          temperature=temperature, destroy_weight=dweights[d], repair_weight=rweights[r])\n            if it % params[\'segment_length\'] == 0:\n                smooth(dweights, scores, uses, params[\'reaction_factor\'])\n                smooth(rweights, scores, uses, params[\'reaction_factor\'])\n            temperature *= params[\'cooling_factor\']\n    except DeadlineReached:\n        reason = \'TIME_LIMIT\'\n    except KeyboardInterrupt:\n        reason = \'INTERRUPTED\'\n    smooth(dweights, scores, uses, params[\'reaction_factor\'])\n    smooth(rweights, scores, uses, params[\'reaction_factor\'])\n    search.state.operators = [dict(operator_type=\'destroy\' if op in DESTROY else \'repair\',\n                                   operator=op, uses=totals[op], total_reward=total_scores[op],\n                                   final_weight=(dweights if op in DESTROY else rweights)[op])\n                              for op in DESTROY + REPAIR]\n    return search.finish(reason)\n'
_MODULE_SOURCES['insert_search'] = '"""Transparent, evaluation-budgeted TS insertion/swap sensitivity experiments.\n\nNo IO targets or expected performance enter either optimizer. Objective keys,\nFCFS initialization, aspiration and accepted-incumbent archiving follow the\nsupplied baseline. A search evaluation is one scored complete permutation;\ndecoding an already-scored accepted order only to recover train-level metadata\nis reported separately, not charged a second search evaluation.\n"""\nfrom __future__ import annotations\n\nfrom dataclasses import dataclass, field\nimport math\nimport random\nimport time\nfrom typing import Any\n\n\n@dataclass\nclass StudyState:\n    best: Any = None\n    best_feasible: Any = None\n    current: Any = None\n    initial: Any = None\n    first_feasible_sec: float | None = None\n    best_feasible_sec: float | None = None\n    best_iteration: int = 0\n    evaluations: int = 0\n    extra_decodes: int = 0\n    iteration: int = 0\n    history: list[dict] = field(default_factory=list)\n    stop_reason: str = ""\n    search_runtime_sec: float = 0.0\n    process_cpu_sec: float = 0.0\n    params: dict = field(default_factory=dict)\n\n\ndef _integer(value, name, minimum=1):\n    if isinstance(value, bool) or int(value) != value or int(value) < minimum:\n        raise ValueError(f"{name} must be an integer >= {minimum}")\n    return int(value)\n\n\ndef relocate(order, i, j):\n    """Remove index i, reinsert so the moved train\'s final index is j."""\n    trial = list(order)\n    train = trial.pop(i)\n    trial.insert(j, train)\n    return trial\n\n\ndef _identity(i, j):\n    # Adjacent relocations in opposite directions have identical outcomes.\n    return (min(i, j), max(i, j)) if abs(i - j) == 1 else (i, j)\n\n\ndef candidate_insertions(instance, current, rng, params):\n    """Unique resulting orders; half guided / half uniform if requested.\n\n    Uniform means uniform selection of a source index, then a valid destination\n    conditional on that source. Boundary sources can have fewer destinations.\n    After repeated draws are exhausted, a shuffled enumeration fills the list.\n    """\n    n = len(current.order)\n    if n < 2:\n        return []\n    radius = params["radius"] or n - 1\n    radius = min(radius, n - 1)\n    distinct = 2 * radius * n - radius * (radius + 1) - (n - 1)\n    target = min(params["candidate_list_size"], distinct)\n    moves, seen = [], set()\n\n    def draw(indices, count):\n        for _ in range(max(50, 30 * count)):\n            if len(moves) >= count:\n                return\n            i = rng.choice(indices)\n            lo, hi = max(0, i - radius), min(n - 1, i + radius)\n            j = rng.randrange(lo, hi)  # hi-lo destinations excluding source.\n            if j >= i:\n                j += 1\n            identity = _identity(i, j)\n            if identity not in seen:\n                seen.add(identity)\n                moves.append((i, j))\n\n    if params["selection"] == "guided":\n        positions = {k: i for i, k in enumerate(current.order)}\n        delayed = sorted(current.order, key=lambda k: (\n            -current.origin_excess[k] if not current.feasible else 0.0,\n            -current.terminal_delay[k], instance.base_rank[k]))[:15]\n        draw([positions[k] for k in delayed], (target + 1) // 2)\n    draw(range(n), target)\n    if len(moves) < target:\n        rest = [(i, j) for i in range(n)\n                for j in range(max(0, i - radius), min(n, i + radius + 1))\n                if i != j and _identity(i, j) not in seen]\n        rng.shuffle(rest)\n        for i, j in rest:\n            identity = _identity(i, j)\n            if identity not in seen:\n                seen.add(identity)\n                moves.append((i, j))\n                if len(moves) == target:\n                    break\n    rng.shuffle(moves)\n    return moves\n\n\ndef _run(instance, params, evaluator, *, neighborhood, max_evaluations,\n         seed, seconds, max_no_improve):\n    params = dict(params)\n    for name in ("candidate_list_size", "tabu_tenure"):\n        params[name] = _integer(params.get(name, 80 if name == "candidate_list_size" else 10), name)\n    max_evaluations = _integer(max_evaluations, "max_evaluations")\n    max_no_improve = _integer(max_no_improve, "max_no_improve")\n    if not math.isfinite(seconds) or seconds <= 0:\n        raise ValueError("seconds must be finite and positive")\n    if neighborhood == "insertion":\n        params["radius"] = _integer(params.get("radius", 0), "radius", 0)\n        params.setdefault("selection", "uniform")\n        if params["selection"] not in ("guided", "uniform"):\n            raise ValueError("selection must be \'guided\' or \'uniform\'")\n        generate = candidate_insertions\n    else:\n        from baseline_lib.search import candidate_swaps\n        params["local_window"] = _integer(params.get("local_window", 8), "local_window")\n        params["delayed_top_k"] = _integer(params.get("delayed_top_k", 15), "delayed_top_k")\n        generate = candidate_swaps\n    state = StudyState(params=params)\n    start, cpu_start = time.perf_counter(), time.process_time()\n    deadline = start + seconds\n    rng, tabu = random.Random(seed), {}\n    stagnant = 0\n\n    def consider(candidate):\n        elapsed = time.perf_counter() - start\n        if state.best is None or candidate.key < state.best.key:\n            state.best = candidate\n        if candidate.feasible and (state.best_feasible is None or candidate.F < state.best_feasible.F):\n            if state.first_feasible_sec is None:\n                state.first_feasible_sec = elapsed\n            state.best_feasible = candidate\n            state.best_feasible_sec = elapsed\n            state.best_iteration = state.iteration\n\n    def record(accepted, improved=False, moved=None, source=None, destination=None,\n               candidate_count=0):\n        current, best, feasible = state.current, state.best, state.best_feasible\n        state.history.append(dict(\n            iteration=state.iteration, elapsed_sec=time.perf_counter() - start,\n            evaluations=state.evaluations, extra_decodes=state.extra_decodes,\n            current_TT=current.tt, current_origin_violations=current.violation_count,\n            current_excess_sum=current.violation_sum, best_search_TT=best.tt,\n            best_origin_violations=best.violation_count,\n            best_excess_sum=best.violation_sum,\n            best_feasible_TT=feasible.tt if feasible else None,\n            accepted=int(accepted), improved_best=int(improved), no_improve=stagnant,\n            move="FCFS" if not state.iteration else neighborhood,\n            moved_train=moved, source_index=source, destination_index=destination,\n            candidate_count=candidate_count))\n\n    state.initial = state.current = evaluator.decode(instance.fcfs)\n    state.evaluations = 1\n    consider(state.current)\n    record(True)\n    reason = "EVALUATION_LIMIT"\n    try:\n        while state.evaluations < max_evaluations:\n            if time.perf_counter() >= deadline:\n                reason = "TIME_LIMIT"\n                break\n            if stagnant >= max_no_improve:\n                reason = "NO_IMPROVEMENT"\n                break\n            current = state.current\n            moves = generate(instance, current, rng, params)\n            moves = moves[:max_evaluations - state.evaluations]\n            if not moves:\n                reason = "EMPTY_NEIGHBORHOOD"\n                break\n            state.iteration += 1\n            iteration = state.iteration\n            best_before = state.best.key\n            trials, attrs = [], []\n            for i, j in moves:\n                if neighborhood == "insertion":\n                    trials.append(relocate(current.order, i, j))\n                    attrs.append((current.order[i], j))\n                else:\n                    trial = list(current.order)\n                    attrs.append(tuple(sorted((trial[i], trial[j]))))\n                    trial[i], trial[j] = trial[j], trial[i]\n                    trials.append(trial)\n            # Complete batches are committed even if the deadline passes during\n            # a native scoring call. Thus timeout overshoot is at most one batch\n            # plus the accepted-order metadata decode, and is visible in runtime.\n            keys = evaluator.scores(trials)\n            if len(keys) != len(trials):\n                raise ValueError("evaluator returned the wrong number of scores")\n            state.evaluations += len(trials)\n            chosen = None\n            chosen_key = None\n            for index, raw_key in enumerate(keys):\n                key = tuple(float(x) for x in raw_key)\n                if len(key) != 7 or not all(math.isfinite(x) for x in key):\n                    raise ValueError("each evaluator score must be a finite 7-component key")\n                admissible = tabu.get(attrs[index], -1) < iteration or key < best_before\n                if admissible and (chosen_key is None or key < chosen_key):\n                    chosen, chosen_key = index, key\n            moved = source = destination = None\n            if chosen is not None:\n                candidate = evaluator.decode(trials[chosen])\n                state.extra_decodes += 1\n                if candidate.key != chosen_key:\n                    raise ValueError("batch score and decoded accepted order disagree")\n                source, destination = moves[chosen]\n                moved = current.order[source]\n                expiry = iteration + params["tabu_tenure"]\n                if neighborhood == "insertion":\n                    tabu[(moved, source)] = expiry\n                    if abs(source - destination) == 1:\n                        # The same adjacent relocation has two representations.\n                        # Forbid the inverse under either moved-train identity.\n                        tabu[(current.order[destination], destination)] = expiry\n                else:\n                    tabu[attrs[chosen]] = expiry\n                state.current = candidate\n                consider(candidate)\n            tabu = {attr: expiry for attr, expiry in tabu.items() if expiry >= iteration}\n            improved = state.best.key < best_before\n            stagnant = 0 if improved else stagnant + 1\n            record(chosen is not None, improved, moved, source, destination, len(trials))\n            if time.perf_counter() >= deadline:\n                reason = "TIME_LIMIT"\n                break\n        else:\n            reason = "EVALUATION_LIMIT"\n    except KeyboardInterrupt:\n        reason = "INTERRUPTED"\n    state.stop_reason = reason\n    state.search_runtime_sec = time.perf_counter() - start\n    state.process_cpu_sec = time.process_time() - cpu_start\n    return state\n\n\ndef run_insert(instance, params, evaluator, *, max_evaluations=800001,\n               seed=1, seconds=3600, max_no_improve=100):\n    """Return StudyState with .best, .best_feasible and complete outer history."""\n    return _run(instance, params, evaluator, neighborhood="insertion",\n                max_evaluations=max_evaluations, seed=seed, seconds=seconds,\n                max_no_improve=max_no_improve)\n\n\ndef run_swap(instance, params, evaluator, *, max_evaluations=800001,\n             seed=1, seconds=3600, max_no_improve=100):\n    """Original candidate_swaps/tabu semantics, same accelerated budget engine."""\n    return _run(instance, params, evaluator, neighborhood="swap",\n                max_evaluations=max_evaluations, seed=seed, seconds=seconds,\n                max_no_improve=max_no_improve)\n'
_MODULE_SOURCES['fast_decode'] = '"""Optional exact C++ accelerator for the unchanged earliest-event decoder.\n\nPublic API:\n    FastDecoder(instance).decode(order, budget=None, materialize=False)\n    FastDecoder(instance).scores(orders, budget=None)  # shape (n, 7), full key\n\nOnly execution speed changes. All validation/export uses the original Python\ndecoder when materialize=True. If compilation/loading is unavailable (e.g. a\nWindows machine without a compiler), the original decoder is used transparently.\n"""\nfrom __future__ import annotations\n\nimport ctypes\nimport hashlib\nimport os\nfrom pathlib import Path\nimport shutil\nimport subprocess\nimport sys\nimport warnings\n\nimport numpy as np\n\nfrom baseline_lib.common import Decoded, decode as python_decode\n\n\n_LIBRARY = None\n_LOAD_ATTEMPTED = False\n\n\ndef _load_library():\n    global _LIBRARY, _LOAD_ATTEMPTED\n    if _LOAD_ATTEMPTED:\n        return _LIBRARY\n    _LOAD_ATTEMPTED = True\n    source = Path(__file__).with_name("fast_decode.cpp")\n    compiler = shutil.which("g++") or shutil.which("clang++")\n    if os.name == "nt" or compiler is None:\n        warnings.warn("C++ decoder unavailable; using original Python decoder.")\n        return None\n    signature = hashlib.sha256(source.read_bytes()).hexdigest()[:16]\n    build = source.parent / ".build"\n    build.mkdir(exist_ok=True)\n    target = build / f"fast_decode_{signature}.so"\n    try:\n        if not target.exists():\n            temporary = target.with_suffix(f".{os.getpid()}.tmp.so")\n            subprocess.run(\n                [compiler, "-std=c++17", "-O3", "-ffp-contract=off", "-fPIC",\n                 "-shared", str(source), "-o", str(temporary)],\n                check=True, capture_output=True, text=True, timeout=120,\n            )\n            temporary.replace(target)\n        lib = ctypes.CDLL(str(target))\n        ip = ctypes.POINTER(ctypes.c_int32)\n        dp = ctypes.POINTER(ctypes.c_double)\n        lib.decoder_create.argtypes = [ctypes.c_int, ctypes.c_int, ip, ip] + [dp] * 9 + [ctypes.c_double, ctypes.c_int]\n        lib.decoder_create.restype = ctypes.c_void_p\n        lib.decoder_destroy.argtypes = [ctypes.c_void_p]\n        lib.decoder_destroy.restype = None\n        lib.decoder_many.argtypes = [ctypes.c_void_p, ctypes.c_int, ip, dp, dp]\n        lib.decoder_many.restype = None\n        _LIBRARY = lib\n    except (OSError, subprocess.SubprocessError) as exc:\n        warnings.warn(f"C++ decoder unavailable ({exc}); using original Python decoder.")\n    return _LIBRARY\n\n\ndef _double_pointer(array):\n    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))\n\n\ndef _integer_pointer(array):\n    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))\n\n\nclass FastDecoder:\n    def __init__(self, instance, *, use_cpp=True):\n        self.instance = instance\n        self._handle = None\n        self._lib = _load_library() if use_cpp else None\n        self.backend = "python"\n        if self._lib is None:\n            return\n        offsets = [0]\n        stations, run, dwell, oa, od = [], [], [], [], []\n        nullable = lambda x: float("nan") if x is None else float(x)\n        for train in instance.trains:\n            stations.extend(train.stations)\n            run.extend((0.0,) + train.run)\n            dwell.extend(map(nullable, train.dwell))\n            oa.extend(map(nullable, train.orig_arr))\n            od.extend(map(nullable, train.orig_dep))\n            offsets.append(len(stations))\n        i_arrays = [np.asarray(x, dtype=np.int32) for x in (offsets, stations)]\n        d_arrays = [np.asarray(x, dtype=np.float64) for x in (\n            run, dwell, oa, od, [tr.lower for tr in instance.trains],\n            [tr.upper for tr in instance.trains],\n            [nullable(tr.terminal_reference) for tr in instance.trains],\n            instance.harr, instance.hdep,\n        )]\n        self._handle = self._lib.decoder_create(\n            instance.K, len(instance.station_names),\n            *map(_integer_pointer, i_arrays), *map(_double_pointer, d_arrays),\n            instance.tolerance, int(sys.version_info >= (3, 12)),\n        )\n        self.backend = "cpp"\n\n    def close(self):\n        if self._handle is not None:\n            self._lib.decoder_destroy(self._handle)\n            self._handle = None\n\n    def __del__(self):\n        self.close()\n\n    def _orders_array(self, orders):\n        values = np.asarray(orders)\n        if values.ndim != 2 or values.shape[1] != self.instance.K:\n            raise ValueError("Decoder requires complete train permutations")\n        if not np.issubdtype(values.dtype, np.integer):\n            raise ValueError("Train indices must be integers")\n        if not np.all(np.sort(values, axis=1) == np.arange(self.instance.K)):\n            raise ValueError("Decoder requires complete train permutations")\n        return np.ascontiguousarray(values, dtype=np.int32)\n\n    def decode(self, order, budget=None, *, materialize=False):\n        order = tuple(order)\n        if materialize or self._lib is None:\n            return python_decode(self.instance, order, budget, materialize=materialize)\n        if self._handle is None:\n            raise RuntimeError("FastDecoder has been closed")\n        if budget is not None:\n            budget.check()\n        values = self._orders_array([order])\n        key = np.empty((1, 7), dtype=np.float64)\n        details = np.empty((3, self.instance.K), dtype=np.float64)\n        self._lib.decoder_many(self._handle, 1, _integer_pointer(values),\n                               _double_pointer(key), _double_pointer(details))\n        if budget is not None:\n            budget.check()\n        k = key[0]\n        return Decoded(order, tuple(map(float, k[3:])), int(k[0]), float(k[1]),\n                       float(k[2]), tuple(map(float, details[0])),\n                       tuple(map(float, details[1])), tuple(map(float, details[2])))\n\n    def scores(self, orders, budget=None):\n        """Return full keys; budget checked before/after every <=64 candidates.\n\n        No partial batch is returned if its deadline is exceeded. Timed search\n        should request one iteration\'s candidate set at a time.\n        """\n        if not isinstance(orders, np.ndarray):\n            orders = list(orders)\n        if len(orders) == 0:\n            return np.empty((0, 7), dtype=np.float64)\n        if self._lib is None:\n            return np.asarray([python_decode(self.instance, order, budget).key\n                               for order in orders], dtype=np.float64)\n        if self._handle is None:\n            raise RuntimeError("FastDecoder has been closed")\n        if budget is not None:\n            budget.check()\n        values = self._orders_array(orders)\n        keys = np.empty((len(values), 7), dtype=np.float64)\n        for begin in range(0, len(values), 64):\n            if budget is not None:\n                budget.check()\n            end = min(begin + 64, len(values))\n            self._lib.decoder_many(self._handle, end - begin,\n                                   _integer_pointer(values[begin:end]),\n                                   _double_pointer(keys[begin:end]), None)\n            if budget is not None:\n                budget.check()\n        return keys\n\n    batch_scores = scores\n'
_MODULE_SOURCES['baseline_lib.validation'] = '# -*- coding: utf-8 -*-\n"""Read-back validation of the additional Appendix-C decoder contract."""\nfrom __future__ import annotations\nimport csv\nimport math\nfrom pathlib import Path\nfrom .common import Instance, write_rows\nfrom .reference_model import validate_timetable_csv\n\n\ndef validate_saved(instance: Instance, order, path: Path, expected_tt: float,\n                   output_dir: Path, tag: str) -> dict:\n    # The main-pipeline independent validator is source-exact, not reimplemented.\n    main = validate_timetable_csv(instance.data, instance.cfg, list(instance.ids), path,\n                                  expected_objective=expected_tt, fixed_precedence=None,\n                                  output_dir=output_dir, validation_tag=tag)\n    with path.open(\'r\', encoding=\'utf-8-sig\', newline=\'\') as stream:\n        rows = {(r[\'车次\'], r[\'车站\']): r for r in csv.DictReader(stream)}\n    details = []\n    tol = instance.tolerance\n    last = {}\n    for i in order:\n        tr = instance.trains[i]\n        for p, s in enumerate(tr.stations):\n            station = instance.station_names[s]\n            row = rows.get((tr.name, station), {})\n            for kind, column, original in [(\'arr\', \'到达时间\', tr.orig_arr[p]),\n                                            (\'dep\', \'出发时间\', tr.orig_dep[p])]:\n                if (kind == \'arr\' and p == 0) or (kind == \'dep\' and p == len(tr.stations) - 1):\n                    continue\n                try:\n                    value = float(row[column])\n                    if not math.isfinite(value):\n                        raise ValueError(\'nonfinite\')\n                except (KeyError, ValueError, TypeError):\n                    details.append(dict(category=\'contract_missing_event\', train=tr.name, station=station, kind=kind))\n                    continue\n                if value < -tol:\n                    details.append(dict(category=\'negative_event_time\', train=tr.name, station=station, kind=kind, actual=value))\n                if original is not None and value < original - tol:\n                    details.append(dict(category=\'original_time_lower_bound\', train=tr.name, station=station,\n                                        kind=kind, actual=value, required=original))\n                key = (s, kind)\n                headway = instance.harr[s] if kind == \'arr\' else instance.hdep[s]\n                if key in last and value < last[key] + headway - tol:\n                    details.append(dict(category=\'global_order_headway\', train=tr.name, station=station,\n                                        kind=kind, actual=value, required=last[key] + headway))\n                last[key] = value\n    write_rows(output_dir / f\'validation_{tag}_decoder_contract.csv\', details,\n               [\'category\', \'train\', \'station\', \'kind\', \'actual\', \'required\'])\n    return dict(main, decoder_contract_pass=int(not details),\n                decoder_contract_violations=len(details),\n                combined_pass=int(main[\'validator_pass\'] == 1 and not details))\n'
_MODULE_SOURCES['alns_search_phased'] = '"""Original ALNS operators with a global search deadline and TT stagnation.\n\nThe wall-clock deadline applies from initialization through the entire search,\nincluding after a feasible incumbent is accepted. The original decoder and\ninsertion repair check this deadline internally; an interrupted repair never\ncommits its temporary full-order scoring trials. Once a feasible incumbent\nexists, strict best-feasible-TT stagnation can also stop the search early.\nOther ALNS choices and RNG draws retain the semantics of\n``baseline_lib.search.run_alns``.\n"""\nfrom __future__ import annotations\n\nfrom dataclasses import dataclass, field\nimport math\nimport random\nimport time\nfrom typing import Callable\n\nfrom baseline_lib.common import Budget, DeadlineReached, decode\nfrom baseline_lib.search import (\n    DESTROY, REPAIR, Search, SearchState, destroy, repair, roulette, sa_accept,\n    smooth,\n)\n\n\n@dataclass\nclass PhasedALNSState(SearchState):\n    params: dict = field(default_factory=dict)\n    extra_decodes: int = 0\n    first_feasible_iteration: int | None = None\n    final_feasible_stall: int = 0\n    termination_phase: str = "SEARCH_FEASIBLE"\n    infeasible_seconds: float = 3600.0\n    max_feasible_no_improve: int = 80\n    first_feasible_after_deadline: bool = False\n    pre_feasible_deadline_overrun_sec: float = 0.0\n    total_deadline_overrun_sec: float = 0.0\n    completed_iterations: int = 0\n\n\nclass _PhasedBudget(Budget):\n    def __init__(self, seconds, clock):\n        self.clock = clock\n        self.start = clock()\n        self.deadline = self.start + seconds\n        self.cpu_start = time.process_time()\n        self.seconds = seconds\n\n    def check(self):\n        if self.clock() >= self.deadline:\n            raise DeadlineReached()\n\n    @property\n    def elapsed(self):\n        return self.clock() - self.start\n\n\nclass _PhasedSearch(Search):\n    def __init__(self, instance, params, seconds, seed, patience, history_sink,\n                 history_keep, clock, progress):\n        self.instance = instance\n        self.budget = _PhasedBudget(seconds, clock)\n        self.rng = random.Random(seed)\n        self.state = PhasedALNSState(\n            params=dict(params), infeasible_seconds=float(seconds),\n            max_feasible_no_improve=patience)\n        self.progress = progress\n        self.last_progress = 0.0\n        self.history_sink = history_sink\n        self.history_keep = history_keep\n\n    def evaluate(self, order):\n        # Keep the original Python decoder, including its internal deadline\n        # checks. Evaluations count completed decodes, never interrupted ones.\n        self.budget.check()\n        result = decode(self.instance, order, self.budget)\n        # A decoder may finish after its final internal check. Do not expose\n        # an overtime result to repair or the accepted-incumbent archive.\n        self.budget.check()\n        self.state.evaluations += 1\n        if self.progress and self.budget.elapsed - self.last_progress >= 15.0:\n            self.emit_progress("SEARCHING")\n        return result\n\n    def initialize(self):\n        initial = self.evaluate(self.instance.fcfs)\n        # Commit before assigning current, so an initialization timeout leaves\n        # an empty, internally consistent state.\n        self.consider_incumbent(initial)\n        self.state.initial = self.state.current = initial\n        self.emit_progress("INITIALIZED")\n        self.record(iteration=0, move="FCFS", accepted=1, no_improve=0)\n        return initial\n\n    def consider_incumbent(self, result):\n        elapsed = self.budget.elapsed\n        if elapsed >= self.state.infeasible_seconds:\n            raise DeadlineReached()\n        if self.state.best is None or result.key < self.state.best.key:\n            self.state.best = result\n        if result.feasible and (\n                self.state.best_feasible is None\n                or result.F < self.state.best_feasible.F):\n            if self.state.first_feasible_sec is None:\n                self.state.first_feasible_sec = elapsed\n                self.state.first_feasible_iteration = self.state.iteration\n                self.state.first_feasible_after_deadline = (\n                    elapsed > self.state.infeasible_seconds)\n                self.state.pre_feasible_deadline_overrun_sec = max(\n                    0.0, elapsed - self.state.infeasible_seconds)\n            # Preserve F-lexicographic archiving and best_iteration semantics.\n            self.state.best_feasible = result\n            self.state.best_feasible_sec = elapsed\n            self.state.best_iteration = self.state.iteration\n\n    def record(self, **fields):\n        current, best = self.state.current, self.state.best\n        feasible = self.state.best_feasible\n        fields.setdefault("improved_best_feasible_TT", int(\n            self.state.iteration == 0 and feasible is not None))\n        row = dict(\n            elapsed_sec=self.budget.elapsed,\n            evaluations=self.state.evaluations,\n            extra_decodes=0,\n            current_TT=current.tt if current else None,\n            current_origin_violations=current.violation_count if current else None,\n            current_excess_sum=current.violation_sum if current else None,\n            best_search_TT=best.tt if best else None,\n            best_origin_violations=best.violation_count if best else None,\n            best_excess_sum=best.violation_sum if best else None,\n            best_feasible_TT=feasible.tt if feasible else None,\n            phase="IMPROVE_FEASIBLE" if feasible else "SEARCH_FEASIBLE",\n            feasible_no_improve=self.state.final_feasible_stall,\n            first_feasible_iteration=self.state.first_feasible_iteration,\n            **fields)\n        if self.history_keep:\n            self.state.history.append(row)\n        if self.history_sink is not None:\n            self.history_sink(row)\n\n    def emit_progress(self, stage):\n        best = self.state.best\n        self.last_progress = self.budget.elapsed\n        if self.progress:\n            self.progress(dict(\n                stage=stage, iteration=self.state.iteration,\n                elapsed_sec=self.budget.elapsed,\n                evaluations=self.state.evaluations,\n                best_feasible_TT=(self.state.best_feasible.tt\n                                  if self.state.best_feasible else None),\n                best_violation_count=best.violation_count if best else None,\n                best_violation_sum=best.violation_sum if best else None,\n                phase=("IMPROVE_FEASIBLE" if self.state.best_feasible\n                       else "SEARCH_FEASIBLE"),\n                feasible_no_improve=self.state.final_feasible_stall))\n\n\ndef _positive_integer(value, name):\n    if isinstance(value, bool) or not isinstance(value, int) or value < 1:\n        raise ValueError(f"{name} must be a positive integer")\n    return value\n\n\ndef run_alns_phased(instance, params, *, seed=1, infeasible_seconds=3600,\n                    max_feasible_no_improve=80,\n                    history_sink: Callable[[dict], None] | None = None,\n                    history_keep=False, clock=None, progress=None):\n    """Run ALNS with no evaluation/iteration cap and return PhasedALNSState.\n\n    ``infeasible_seconds`` retains its old name for configuration compatibility;\n    it now limits the entire search, including initialization and feasible\n    improvement. The deadline is never reset or disabled after finding a solution.\n    A TT tie with better secondary F components updates the feasible archive\n    and original ALNS rewards, but does not reset the stopping counter.\n    ``params[\'max_no_improve\']`` from old configurations is not a termination\n    condition; use ``max_feasible_no_improve`` for the feasible phase.\n    History is streamed and/or retained once per complete outer iteration.\n    """\n    params = dict(params)\n    for name in ("min_remove", "max_remove", "max_insertion_positions",\n                 "segment_length"):\n        params[name] = _positive_integer(params[name], name)\n    if params["max_remove"] < params["min_remove"]:\n        raise ValueError("max_remove must be at least min_remove")\n    for name in ("reaction_factor", "cooling_factor"):\n        value = params[name]\n        if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:\n            raise ValueError(f"{name} must be finite and in (0, 1]")\n    for name in ("initial_temperature_floor", "initial_temperature_fraction"):\n        value = params[name]\n        if isinstance(value, bool) or not math.isfinite(value) or value < 0:\n            raise ValueError(f"{name} must be finite and nonnegative")\n    rewards = params["rewards"]\n    if len(rewards) != 3 or any(\n            isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in rewards):\n        raise ValueError("rewards must contain three finite nonnegative values")\n    max_feasible_no_improve = _positive_integer(\n        max_feasible_no_improve, "max_feasible_no_improve")\n    if (isinstance(infeasible_seconds, bool)\n            or not math.isfinite(infeasible_seconds) or infeasible_seconds <= 0):\n        raise ValueError("infeasible_seconds must be finite and positive")\n    if history_sink is not None and not callable(history_sink):\n        raise TypeError("history_sink must be callable or None")\n    clock = time.perf_counter if clock is None else clock\n    if not callable(clock):\n        raise TypeError("clock must be callable")\n    search = _PhasedSearch(\n        instance, params, infeasible_seconds, seed, max_feasible_no_improve,\n        history_sink, history_keep, clock, progress)\n    dweights = {op: 1.0 for op in DESTROY}\n    rweights = {op: 1.0 for op in REPAIR}\n    scores = {op: 0.0 for op in DESTROY + REPAIR}\n    uses = {op: 0 for op in DESTROY + REPAIR}\n    totals = {op: 0 for op in DESTROY + REPAIR}\n    total_scores = {op: 0.0 for op in DESTROY + REPAIR}\n    stagnant = 0  # Original full-key diagnostic, never a stopping condition.\n    reason = ""\n    try:\n        current = search.initialize()\n        temperature = max(float(params["initial_temperature_floor"]),\n                          float(params["initial_temperature_fraction"])\n                          * max(1.0, current.tt))\n        while True:\n            search.budget.check()\n            if (search.state.best_feasible is not None\n                    and search.state.final_feasible_stall >= max_feasible_no_improve):\n                reason = "FEASIBLE_NO_IMPROVEMENT"\n                break\n            search.state.iteration += 1\n            it = search.state.iteration\n            best_before = search.state.best.key\n            feasible_before = search.state.best_feasible\n            feasible_tt_before = feasible_before.tt if feasible_before else None\n            current_before = current\n            d = roulette(dweights, search.rng)\n            r = roulette(rweights, search.rng)\n            q = search.rng.randint(params["min_remove"], params["max_remove"])\n            partial, removed = destroy(instance, current, d, q, search.rng)\n            candidate = repair(search, partial, removed, r, params["max_insertion_positions"])\n            search.budget.check()\n            accepted, accept_reason = sa_accept(current, candidate, temperature, search.rng)\n            search.budget.check()\n            if accepted:\n                search.consider_incumbent(candidate)\n                current = candidate\n                search.state.current = current\n            improved_best = search.state.best.key < best_before\n            improved_current = candidate.key < current_before.key\n            stagnant = 0 if improved_best else stagnant + 1\n            improved_feasible_tt = search.state.best_feasible is not None and (\n                feasible_tt_before is None\n                or search.state.best_feasible.tt < feasible_tt_before)\n            if search.state.best_feasible is not None:\n                search.state.final_feasible_stall = (\n                    0 if improved_feasible_tt else search.state.final_feasible_stall + 1)\n            reward = (params["rewards"][0] if improved_best else params["rewards"][1]\n                      if accepted and improved_current else params["rewards"][2] if accepted else 0.0)\n            for op in (d, r):\n                scores[op] += reward\n                uses[op] += 1\n                totals[op] += 1\n                total_scores[op] += reward\n            search.state.completed_iterations = it\n            search.record(\n                iteration=it, destroy_operator=d, repair_operator=r,\n                removed_count=len(removed), candidate_TT=candidate.tt,\n                candidate_origin_violations=candidate.violation_count,\n                delta_TT=candidate.tt - current_before.tt,\n                accepted=int(accepted), accept_reason=accept_reason,\n                improved_best=int(improved_best), no_improve=stagnant,\n                improved_best_feasible_TT=int(improved_feasible_tt),\n                temperature=temperature,\n                destroy_weight=dweights[d], repair_weight=rweights[r])\n            if it % params["segment_length"] == 0:\n                smooth(dweights, scores, uses, params["reaction_factor"])\n                smooth(rweights, scores, uses, params["reaction_factor"])\n            temperature *= params["cooling_factor"]\n    except DeadlineReached:\n        reason = ("TIME_LIMIT" if search.state.best_feasible is not None\n                  else "NO_FEASIBLE_TIME_LIMIT")\n    except KeyboardInterrupt:\n        reason = "INTERRUPTED"\n    smooth(dweights, scores, uses, params["reaction_factor"])\n    smooth(rweights, scores, uses, params["reaction_factor"])\n    search.state.operators = [dict(\n        operator_type="destroy" if op in DESTROY else "repair",\n        operator=op, uses=totals[op], total_reward=total_scores[op],\n        final_weight=(dweights if op in DESTROY else rweights)[op])\n        for op in DESTROY + REPAIR]\n    search.state.termination_phase = (\n        "IMPROVE_FEASIBLE" if search.state.best_feasible else "SEARCH_FEASIBLE")\n    state = search.finish(reason)\n    state.total_deadline_overrun_sec = max(\n        0.0, state.search_runtime_sec - infeasible_seconds)\n    if state.best_feasible is None:\n        state.pre_feasible_deadline_overrun_sec = max(\n            0.0, state.search_runtime_sec - infeasible_seconds)\n    return state\n'
_MODULE_SOURCES['swap_search_phased'] = '"""Uniform random swap TS with the current total deadline and TT stagnation.\n\nThe deadline runs continuously from initialization through feasible improvement.\nThe legacy ``infeasible_seconds`` argument now limits the entire search. Only\nafter an accepted feasible incumbent exists can strict best-feasible-TT\nstagnation stop the search early. Each candidate list uniformly samples distinct\nunordered position pairs without replacement. Sampling does not inspect delay,\ntime-window violations, feasibility, or any comparison-method result. Tabu\nattributes are train pairs, as in the original run_tsp. Initialization, scoring\nand decoder are shared with the existing experiment. This deliberately does\nnot restore the original infeasible-phase/full-key stagnation stop.\n\nThe same deadline is checked inside decoding and between candidate batches.\nAn interrupted batch is not committed. Final validation and CSV export occur\nafter search timing; unavoidable deadline-check latency is recorded explicitly.\n"""\nfrom __future__ import annotations\n\nfrom bisect import bisect_right\nfrom dataclasses import dataclass\nimport math\nimport random\nimport time\nfrom typing import Callable\n\nfrom baseline_lib.common import DeadlineReached\nfrom insert_search import StudyState, _integer\n\n\ndef candidate_uniform_swaps(instance, current, rng, params):\n    """Sample position pairs uniformly; only the order length is inspected.\n\n    Lexicographic pair ranks map bijectively onto 0 <= i < j < n. Sampling\n    ranks avoids guidance and duplicate rejection, without constructing the\n    full quadratic neighborhood. All pairs are used when fewer than requested\n    exist; their evaluation order is still randomly permuted.\n    """\n    n = len(current.order)\n    if n < 2:\n        return []\n    total = n * (n - 1) // 2\n    starts = [i * (2 * n - i - 1) // 2 for i in range(n - 1)]\n    pairs = []\n    for rank in rng.sample(range(total), min(params[\'candidate_list_size\'], total)):\n        i = bisect_right(starts, rank) - 1\n        pairs.append((i, i + 1 + rank - starts[i]))\n    return pairs\n\n\n@dataclass\nclass PhasedStudyState(StudyState):\n    first_feasible_iteration: int | None = None\n    final_feasible_stall: int = 0\n    termination_phase: str = "SEARCH_FEASIBLE"\n    infeasible_seconds: float = 1800.0\n    max_feasible_no_improve: int = 100\n    first_feasible_after_deadline: bool = False\n    pre_feasible_deadline_overrun_sec: float = 0.0\n    total_deadline_overrun_sec: float = 0.0\n    completed_iterations: int = 0\n\n\nclass _SearchBudget:\n    def __init__(self, clock, deadline):\n        self.clock, self.deadline = clock, deadline\n\n    def check(self):\n        if self.clock() >= self.deadline:\n            raise DeadlineReached()\n\n\ndef run_swap_phased(instance, params, evaluator, *, seed=1,\n                      infeasible_seconds=1800, max_feasible_no_improve=100,\n                      history_sink: Callable[[dict], None] | None = None,\n                      history_keep=False, clock=None):\n    """Run TS and return ``PhasedStudyState``; do not retain history by default.\n\n    ``history_sink``, when supplied, is called once for the initial solution and\n    after every completed iteration. ``clock`` is a monotonic seconds callable;\n    it defaults to ``time.perf_counter`` and is injectable for deterministic\n    stopping-rule tests. A secondary feasible F improvement updates the archive\n    exactly as before, but does not reset the TT-only stagnation counter.\n    """\n    params = dict(params)\n    for name in ("candidate_list_size", "tabu_tenure"):\n        params[name] = _integer(\n            params.get(name, 80 if name == "candidate_list_size" else 10), name)\n    max_feasible_no_improve = _integer(\n        max_feasible_no_improve, "max_feasible_no_improve")\n    if (isinstance(infeasible_seconds, bool)\n            or not math.isfinite(infeasible_seconds) or infeasible_seconds <= 0):\n        raise ValueError("infeasible_seconds must be finite and positive")\n    if history_sink is not None and not callable(history_sink):\n        raise TypeError("history_sink must be callable or None")\n    clock = time.perf_counter if clock is None else clock\n    if not callable(clock):\n        raise TypeError("clock must be callable")\n    state = PhasedStudyState(\n        params=params, infeasible_seconds=float(infeasible_seconds),\n        max_feasible_no_improve=max_feasible_no_improve)\n    start, cpu_start = clock(), time.process_time()\n    deadline = start + infeasible_seconds\n    budget = _SearchBudget(clock, deadline)\n    rng, tabu = random.Random(seed), {}\n    stagnant = 0  # Original full-key statistic, never a stopping condition.\n    feasible_stagnant = 0\n\n    def consider(candidate):\n        elapsed = clock() - start\n        if elapsed >= infeasible_seconds:\n            raise DeadlineReached()\n        if state.best is None or candidate.key < state.best.key:\n            state.best = candidate\n        if candidate.feasible and (\n                state.best_feasible is None or candidate.F < state.best_feasible.F):\n            if state.first_feasible_sec is None:\n                state.first_feasible_sec = elapsed\n                state.first_feasible_iteration = state.iteration\n                state.first_feasible_after_deadline = elapsed > infeasible_seconds\n                state.pre_feasible_deadline_overrun_sec = max(\n                    0.0, elapsed - infeasible_seconds)\n            state.best_feasible = candidate\n            state.best_feasible_sec = elapsed\n            state.best_iteration = state.iteration\n\n    def record(accepted, improved=False, moved=None, source=None, destination=None,\n               swapped=None,\n               candidate_count=0, improved_feasible_tt=False):\n        current, best, feasible = state.current, state.best, state.best_feasible\n        row = dict(\n            iteration=state.iteration, elapsed_sec=clock() - start,\n            evaluations=state.evaluations, extra_decodes=state.extra_decodes,\n            current_TT=current.tt, current_origin_violations=current.violation_count,\n            current_excess_sum=current.violation_sum, best_search_TT=best.tt,\n            best_origin_violations=best.violation_count,\n            best_excess_sum=best.violation_sum,\n            best_feasible_TT=feasible.tt if feasible is not None else None,\n            accepted=int(accepted), improved_best=int(improved), no_improve=stagnant,\n            move="FCFS" if not state.iteration else "pairwise_swap",\n            moved_train=moved, source_index=source, destination_index=destination,\n            swapped_train_1=instance.ids[moved] if moved is not None else None,\n            swapped_train_2=instance.ids[swapped] if swapped is not None else None,\n            candidate_count=candidate_count,\n            phase="IMPROVE_FEASIBLE" if feasible is not None else "SEARCH_FEASIBLE",\n            feasible_no_improve=feasible_stagnant,\n            improved_best_feasible_TT=int(improved_feasible_tt),\n            first_feasible_iteration=state.first_feasible_iteration)\n        if history_keep:\n            state.history.append(row)\n        if history_sink is not None:\n            history_sink(row)\n\n    reason = ""\n    try:\n        initial = evaluator.decode(instance.fcfs, budget=budget)\n        budget.check()\n        state.evaluations = 1\n        consider(initial)\n        state.initial = state.current = initial\n        record(True, improved_feasible_tt=state.best_feasible is not None)\n        while True:\n            budget.check()\n            if (state.best_feasible is not None\n                    and feasible_stagnant >= max_feasible_no_improve):\n                reason = "FEASIBLE_NO_IMPROVEMENT"\n                break\n            current = state.current\n            moves = candidate_uniform_swaps(instance, current, rng, params)\n            if not moves:\n                reason = "EMPTY_NEIGHBORHOOD"\n                break\n            state.iteration += 1\n            iteration = state.iteration\n            best_before = state.best.key\n            feasible_before = state.best_feasible\n            feasible_tt_before = feasible_before.tt if feasible_before is not None else None\n            trials, attrs = [], []\n            for i, j in moves:\n                trial = list(current.order)\n                attrs.append(tuple(sorted((trial[i], trial[j]))))\n                trial[i], trial[j] = trial[j], trial[i]\n                trials.append(tuple(trial))\n            keys = evaluator.scores(trials, budget=budget)\n            budget.check()\n            if len(keys) != len(trials):\n                raise ValueError("evaluator returned the wrong number of scores")\n            state.evaluations += len(trials)\n            chosen = chosen_key = None\n            for index, raw_key in enumerate(keys):\n                key = tuple(float(x) for x in raw_key)\n                if len(key) != 7 or not all(math.isfinite(x) for x in key):\n                    raise ValueError("each evaluator score must be a finite 7-component key")\n                admissible = tabu.get(attrs[index], -1) < iteration or key < best_before\n                if admissible and (chosen_key is None or key < chosen_key):\n                    chosen, chosen_key = index, key\n            budget.check()\n            moved = swapped = source = destination = None\n            if chosen is not None:\n                candidate = evaluator.decode(trials[chosen], budget=budget)\n                budget.check()\n                state.extra_decodes += 1\n                if candidate.key != chosen_key:\n                    raise ValueError("batch score and decoded accepted order disagree")\n                consider(candidate)\n                source, destination = moves[chosen]\n                moved = current.order[source]\n                swapped = current.order[destination]\n                expiry = iteration + params["tabu_tenure"]\n                tabu[attrs[chosen]] = expiry\n                state.current = candidate\n            tabu = {attr: expiry for attr, expiry in tabu.items() if expiry >= iteration}\n            improved = state.best.key < best_before\n            stagnant = 0 if improved else stagnant + 1\n            improved_feasible_tt = state.best_feasible is not None and (\n                feasible_tt_before is None or state.best_feasible.tt < feasible_tt_before)\n            if state.best_feasible is not None:\n                feasible_stagnant = 0 if improved_feasible_tt else feasible_stagnant + 1\n            state.completed_iterations = iteration\n            record(chosen is not None, improved, moved, source, destination,\n                   swapped, len(trials), improved_feasible_tt)\n    except DeadlineReached:\n        reason = "TIME_LIMIT" if state.best_feasible is not None else "NO_FEASIBLE_TIME_LIMIT"\n    except KeyboardInterrupt:\n        reason = "INTERRUPTED"\n    state.stop_reason = reason\n    state.search_runtime_sec = clock() - start\n    state.process_cpu_sec = time.process_time() - cpu_start\n    state.total_deadline_overrun_sec = max(0.0, state.search_runtime_sec - infeasible_seconds)\n    state.final_feasible_stall = feasible_stagnant\n    state.termination_phase = (\n        "IMPROVE_FEASIBLE" if state.best_feasible is not None else "SEARCH_FEASIBLE")\n    if state.best_feasible is None:\n        state.pre_feasible_deadline_overrun_sec = max(\n            0.0, state.search_runtime_sec - infeasible_seconds)\n    return state\n'

if __name__ == "__main__":
    launch()
