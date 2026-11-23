"""Stage-level restart for the integrated entry points; no solver-state restart."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from config_loader import as_bool, parse_float_list, project_root


STAGES = ('direct', 'samples', 'vote', 'packing', 'inverse', 'fullfix', 'aggregate')
DEPENDENCIES = {
    'direct': (), 'samples': (), 'vote': ('samples',), 'packing': ('samples',),
    'inverse': ('vote', 'packing'), 'fullfix': ('vote', 'inverse'),
    'aggregate': ('direct', 'samples', 'vote', 'packing', 'inverse', 'fullfix'),
}
FUNCTIONS = dict(zip(STAGES, (
    'run_direct', 'run_restricted_instances', 'run_vote', 'run_packing',
    'run_inverse_all_alphas', 'run_fullfix_all_alphas', 'aggregate_experiment',
)))
# A normal solver limit is a finished experiment, even without an incumbent.
# INTERRUPTED, ERROR and SKIPPED_NO_PRIORS must never count as completed.
FINISHED = {'OPTIMAL', 'INFEASIBLE', 'UNBOUNDED', 'TIME_LIMIT', 'NODE_LIMIT',
            'ITERATION_LIMIT', 'SOLUTION_LIMIT', 'USER_OBJ_LIMIT', 'WORK_LIMIT'}


def _rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def _one(path):
    rows = _rows(path)
    if len(rows) != 1:
        raise ValueError(f'Expected one summary row: {path}')
    return rows[0]


def _integer(value):
    return int(float(value))


def _tag(alpha):
    return 'alpha_' + f'{float(alpha):g}'.replace('-', 'm').replace('.', 'p')


class ResumeManager:
    def __init__(self, cfg, ctx, signature, action='full'):
        self.cfg, self.ctx, self.signature = cfg, ctx, signature
        self.stages = STAGES if action == 'full' else STAGES[1:]
        self.path = ctx.output_dir / 'resume_state.json'
        self.state = {'schema': 1, 'signature': signature, 'stages': {}}
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding='utf-8'))
            if saved.get('schema') != 1 or saved.get('signature') != signature:
                raise ValueError(f'{ctx.experiment_tag}: resume checkpoint does not match this experiment')
            self.state = saved

    def _dependencies(self, stage):
        return tuple(s for s in DEPENDENCIES[stage] if s in self.stages)

    def _solution_files(self, row, directory, tag, timetable):
        if not as_bool(row.get('has_solution'), False):
            return []
        report = directory / f'validation_{tag}_summary.csv'
        if not as_bool(row.get('validator_pass'), False) or not as_bool(_one(report).get('validator_pass'), False):
            raise ValueError('Saved timetable did not pass validation')
        return [report, directory / timetable]

    def _files(self, stage):
        """Return completion summary first, followed by required artifacts."""
        c, cfg = self.ctx, self.cfg
        if stage == 'direct':
            summary = c.direct_dir / 'direct_summary.csv'
            row = _one(summary)
            if row.get('status_name') not in FINISHED:
                raise ValueError('Direct did not finish normally')
            target = c.direct_dir / 'target_train_set.csv'
            # Old entry points rewrite this before starting Direct. It detects
            # an unfinished rerun even if an earlier complete summary remains.
            if target.stat().st_mtime_ns > summary.stat().st_mtime_ns:
                raise ValueError('Direct was restarted after the saved summary')
            return [summary, target] + self._solution_files(row, c.direct_dir, 'direct', 'timetable_direct.csv')
        if stage == 'samples':
            summary = c.sample_dir / 'sample_solve_summary.csv'
            completion = c.sample_dir / 'sampling_completion_record.json'
            row = _one(summary)
            saved = json.loads(completion.read_text(encoding='utf-8'))
            manifest = c.sample_dir / 'sample_manifest.csv'
            samples = _rows(manifest)
            count = int(c.sample_counts[0])
            if len(samples) != count or _integer(row['final_usable_samples']) != count:
                raise ValueError('Sample pool is incomplete')
            if not row.get('sampling_config_signature') or saved.get('summary', {}).get('sampling_config_signature') != row.get('sampling_config_signature'):
                raise ValueError('Sample completion record is inconsistent')
            counts = Counter(r['sampling_method'] for r in samples)
            if any(counts[m] != int(cfg[f'sampling_{m}_count']) for m in ('local', 'stratified', 'coverage')):
                raise ValueError('Sample composition does not match the case plan')
            files = [completion, summary, manifest]
            for rec in samples:
                pairwise = rec.get('pairwise_csv', '')
                if not pairwise:
                    raise ValueError('Sample precedence evidence is missing')
                files.append(project_root() / pairwise)
            return files
        if stage == 'vote':
            summary = c.vote_dir / 'vote_summary.csv'
            _one(summary)
            assignment = c.vote_dir / 'vote_priors_fullfix_cache.pkl'
            if not assignment.exists():
                assignment = c.vote_dir / 'vote_priors_fullfix.csv'
            return [summary, c.vote_dir / 'evidence_diagnostics.csv', assignment]
        if stage == 'packing':
            summary = c.packing_dir / 'packing_summary.csv'
            row = _one(summary)
            if row.get('primal_status') != 'OPTIMAL' or row.get('dual_status') != 'OPTIMAL':
                raise ValueError('Packing did not finish optimally')
            return [summary, c.packing_dir / 'packing_train_prices.csv', c.packing_dir / 'packing_runs_used.csv']
        if stage == 'inverse':
            summary = c.inverse_dir / 'inverse_all_alpha_summary.csv'
            rows = _rows(summary)
            alphas = parse_float_list(cfg.get('alpha_list'))
            if len(rows) != len(alphas) or {float(r['alpha']) for r in rows} != set(alphas):
                raise ValueError('Some alpha results are missing')
            files = [summary]
            for row in rows:
                directory = c.inverse_dir / _tag(row['alpha'])
                detail = directory / 'inverse_summary.csv'
                record = _one(detail)
                if any(r.get('status_name') not in FINISHED or not as_bool(r.get('accepted_for_fullfix'), False)
                       for r in (row, record)):
                    raise ValueError('Inverse result is incomplete or rejected')
                cache = directory / 'learned_core_components_cache.pkl'
                if not cache.exists():
                    cache = directory / 'learned_core_precedence.csv'
                files.extend([detail, cache])
            return files
        if stage == 'fullfix':
            summary = c.fullfix_dir / 'fullfix_all_alpha_summary.csv'
            rows = _rows(summary)
            expected = {'IO_' + _tag(a) for a in parse_float_list(cfg.get('alpha_list'))}
            if as_bool(cfg.get('run_vote_fullfix'), True):
                expected.add('VOTE')
            if len(rows) != len(expected) or {r['variant'] for r in rows} != expected:
                raise ValueError('Some Full-Fix results are missing')
            files = [summary]
            for row in rows:
                directory = c.fullfix_dir / row['variant']
                detail = directory / 'fullfix_summary.csv'
                record = _one(detail)
                if any(r.get('status_name') not in FINISHED for r in (row, record)):
                    raise ValueError('Full-Fix did not finish normally')
                files.extend([detail] + self._solution_files(record, directory, row['variant'], 'timetable_fullfix.csv'))
            return files
        directory = c.aggregate_dir
        _one(directory / 'main_experiment_summary.csv')
        manifest = json.loads((directory / 'experiment_manifest.json').read_text(encoding='utf-8'))
        tables = manifest.get('table_files', [])
        if not tables:
            raise ValueError('Aggregate tables are incomplete')
        return [directory / 'experiment_manifest.json', directory / 'main_experiment_summary.csv',
                directory / 'experiment_results_long.csv', directory / 'experiment_results_wide.csv',
                *(directory / 'chapter6_table_rows' / name for name in tables)]

    def _snapshot(self, stage):
        files = self._files(stage)
        snapshot = {str(p.resolve()): [p.stat().st_size, p.stat().st_mtime_ns] for p in files}
        if any(size <= 0 for size, _ in snapshot.values()):
            raise ValueError('Required artifact is empty')
        # Required artifacts are written before their completion summary.
        # Aggregate also writes a timing copy later; that file is not a prerequisite.
        finished = files[0].stat().st_mtime_ns
        if any(p.stat().st_mtime_ns > finished for p in files[1:]):
            raise ValueError('A stage artifact is newer than its completion summary')
        return snapshot, finished

    def inspect(self):
        ready, stamps, reasons = {}, {}, {}
        for stage in self.stages:
            ready[stage] = False
            try:
                if stage == 'samples' and not as_bool(self.cfg.get('sample_resume'), True):
                    raise ValueError('sample_resume=FALSE requests new samples')
                if any(not ready[d] for d in self._dependencies(stage)):
                    raise ValueError('An upstream stage needs to run')
                snapshot, finished = self._snapshot(stage)
                if any(stamps[d] > finished for d in self._dependencies(stage)):
                    raise ValueError('An upstream result changed')
                direct = self.ctx.direct_dir / 'direct_summary.csv'
                if stage == 'aggregate' and direct.exists() and direct.stat().st_mtime_ns > finished:
                    raise ValueError('Direct results changed since aggregation')
                record = self.state['stages'].get(stage)
                if record is not None and (record.get('status') != 'complete' or record.get('files') != snapshot):
                    raise ValueError('Checkpoint is unfinished or its artifacts changed')
                ready[stage], stamps[stage] = True, finished
                reasons[stage] = 'saved result complete'
            except (OSError, ValueError, KeyError, TypeError) as exc:
                reasons[stage] = str(exc)
        return ready, reasons

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(self.path)

    def run(self, pipeline, data):
        # The case manager verifies the input/config signature before entering.
        for stage in self.stages:
            ready, _ = self.inspect()
            if ready[stage]:
                print(f'[SKIP] {self.ctx.experiment_tag}: {stage} already complete', flush=True)
                continue
            # Persist invalidation before starting. Old result files must not
            # masquerade as completed work after an interrupted rerun.
            dirty = {stage}
            for other in self.stages:
                if other in dirty or dirty.intersection(self._dependencies(other)):
                    dirty.add(other)
                    self.state['stages'][other] = {'status': 'pending'}
            self.state['stages'][stage] = {'status': 'running'}
            self._save()
            print(f'[RUN] {self.ctx.experiment_tag}: {stage}', flush=True)
            try:
                function = getattr(pipeline, FUNCTIONS[stage])
                if stage == 'aggregate':
                    function(self.cfg, self.ctx)
                else:
                    function(self.cfg, self.ctx, data)
                snapshot, _ = self._snapshot(stage)
                self.state['stages'][stage] = {'status': 'complete', 'files': snapshot}
                self._save()
            except BaseException:
                self.state['stages'][stage] = {'status': 'interrupted'}
                self._save()
                raise
