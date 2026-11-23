import hashlib
import json
import math
import random
from collections import Counter, defaultdict

from .constraints import copy_bounds, composition_requirements, time_requirements
from .solver import LinearModel


def solver_args(cfg, seed):
    return dict(backend=cfg.get("solver.backend", "auto"), seconds=cfg.float("solver.time_limit_seconds", 60),
                seed=seed, threads=cfg.int("solver.threads", 1), log=cfg.bool("solver.console_log"),
                gap=cfg.float("solver.relative_gap", 0.05))


def shift_limits(source, t):
    cfg = source.cfg
    lo, hi = source.horizon_start - t.start, source.horizon_end - t.end
    maximum = cfg.get("time.max_shift_minutes", "unlimited")
    if maximum.lower() not in ("", "none", "unlimited"):
        maximum = float(maximum)
        if maximum < 0: raise ValueError("max_shift_minutes cannot be negative")
        lo, hi = max(lo, -maximum), min(hi, maximum)
    if cfg.bool("time.lock_source_entry_bin"):
        b = math.floor(t.entry / source.bin_minutes)
        lo = max(lo, b * source.bin_minutes - t.entry)
        hi = min(hi, (b + 1) * source.bin_minutes - t.entry - 1e-7)
    return math.ceil(lo / source.grid - 1e-8), math.floor(hi / source.grid + 1e-8)


def compose(source, k, seed, exclusions=(), deadline=None):
    if source.cfg.get("solver.composition_mode") == "unique_score":
        from .unique import compose_unique
        return compose_unique(source, k, seed, exclusions, deadline)
    cfg = source.cfg; rng = random.Random(seed)
    lower, upper = copy_bounds(source, k)
    model = LinearModel()
    mv = [None] * source.n
    ordered = list(source.trains); rng.shuffle(ordered)
    feasible_only = cfg.get("solver.composition_mode", "feasibility") == "feasibility"
    for t in ordered:
        a, b = shift_limits(source, t)
        ub = upper if a <= b else 0
        if lower > ub:
            return None, {"status": "INFEASIBLE", "message": "Retained template has no admissible shift: " + t.source,
                          "backend": "precheck", "seconds": 0}
        mv[t.index] = model.var(lower, ub, True, 0 if feasible_only else rng.uniform(-1, 1))
    model.add({v: 1 for v in mv}, k, k, "exact_K")
    requirements = composition_requirements(source, k)
    for q in requirements: model.add({mv[i]: v for i, v in q.coefficients.items()}, q.lower, q.upper, q.family)
    balance = cfg.float("copy.balance_weight", 0.5)
    if k > source.n and balance > 0 and not feasible_only:
        for v in mv: model.absolute({v: 1}, k / source.n, balance)
    for previous in exclusions: model.exclude_integer_vector(mv, previous)
    result = model.solve(**solver_args(cfg, seed))
    x = result.pop("x", None)
    if x is None: return None, result
    m = [int(x[v]) for v in mv]
    if any(not q.satisfied(m) for q in requirements): raise RuntimeError("Composition check failed")
    return m, result


def assign_times(source, m, seed, deadline=None):
    """Assign eligible entry-time cells, then sample a grid shift inside each cell.

    Cell boundaries include both hour and broad-period boundaries. Consequently
    every draw has exactly the time-category memberships checked by the MIP.
    """
    if source.cfg.get("solver.composition_mode") == "unique_score":
        from .unique import assign_times_unique
        return assign_times_unique(source, m, seed, deadline)
    cfg = source.cfg; rng = random.Random(seed); k = sum(m)
    model = LinearModel(); options = []; by_template = defaultdict(list)
    members = defaultdict(lambda: defaultdict(list)); fixed = []
    movable = list(m)
    scope = cfg.get("time.movable", "all")
    if scope not in ("all", "copies_only"): raise ValueError("time.movable must be all or copies_only")
    if scope == "copies_only":
        for i, count in enumerate(m):
            if count:
                fixed.append((i, 0.0)); movable[i] -= 1
                t = source.trains[i]
                if t.start < source.horizon_start - 1e-7 or t.end > source.horizon_end + 1e-7:
                    return None, {"status": "INFEASIBLE", "backend": "precheck", "seconds": 0,
                                  "message": "Fixed original lies outside operating horizon: " + t.source}
    fixed_counts = source.time_counts(fixed)
    active_families = set(cfg.list("features.time_families"))
    for t, count in zip(source.trains, movable):
        if not count: continue
        left, right = shift_limits(source, t)
        if left > right:
            return None, {"status": "INFEASIBLE", "backend": "precheck", "seconds": 0, "message": "Empty shift interval"}
        # At most a few cells per train for ordinary minute/hour settings;
        # enumeration is over time-grid shifts, not subsets of trains.
        cells = {}
        for step in range(left, right + 1):
            entry = t.entry + step * source.grid
            keys = source.time_keys(t, entry)
            if any(keys[f] not in source.reference_time[f] for f in active_families): continue
            signature = tuple(keys.items())
            if signature not in cells: cells[signature] = [step, step, keys]
            else: cells[signature][1] = step
        for a, b, keys in cells.values():
            v = model.var(0, count, True, rng.uniform(-1, 1))
            options.append((v, t.index, a, b)); by_template[t.index].append(v)
            for f, key in keys.items(): members[f][key].append(v)
        model.add({v: 1 for v in by_template[t.index]}, count, count, "time_assignment")
    for f, key, lo, hi, reference_n in time_requirements(source, k):
        already = fixed_counts[f][key]
        model.add({v: 1 for v in members[f][key]}, lo - already, hi - already, f)
    if not model.lb:
        counts = source.time_counts(fixed)
        ok = all(lo <= counts[f][key] <= hi for f, key, lo, hi, _ in time_requirements(source, k))
        return (fixed if ok else None), {"status": "OPTIMAL" if ok else "INFEASIBLE", "backend": "precheck", "seconds": 0}
    result = model.solve(**solver_args(cfg, seed))
    x = result.pop("x", None)
    if x is None: return None, result
    instances = list(fixed)
    for v, i, a, b in options:
        count = int(x[v])
        # Prefer different departure minutes for copies of the same template
        # inside a cell when enough grid points exist, without adding a
        # scheduling/headway requirement.
        steps = rng.sample(range(a, b + 1), count) if b - a + 1 >= count else [rng.randint(a, b) for _ in range(count)]
        instances.extend((i, step * source.grid) for step in steps)
    instances.sort(key=lambda p: (source.trains[p[0]].entry + p[1], source.trains[p[0]].source, p[1]))
    return instances, result


def signature(source, instances, include_time=True):
    if include_time:
        # Identical copies are unlabeled. Remove a common translation from all
        # absolute entry times, so train renaming / common shifting is not new.
        anchor = min(source.trains[i].entry + d for i, d in instances)
        payload = sorted((source.trains[i].source, round(source.trains[i].entry + d - anchor, 8)) for i, d in instances)
    else:
        payload = sorted(Counter(source.trains[i].source for i, _ in instances).items())
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def audit_instances(source, k, instances):
    cfg = source.cfg; errors = []
    m = [0] * source.n
    for i, shift in instances:
        if i < 0 or i >= source.n: errors.append("unknown template"); continue
        m[i] += 1; t = source.trains[i]
        if t.start + shift < source.horizon_start - 1e-6 or t.end + shift > source.horizon_end + 1e-6:
            errors.append("time horizon: " + t.source)
        a, b = shift_limits(source, t)
        step = shift / source.grid
        if abs(step - round(step)) > 1e-6 or not a - 1e-6 <= step <= b + 1e-6:
            errors.append("shift/grid bound: " + t.source)
    if len(instances) != k: errors.append("wrong K")
    lo, hi = copy_bounds(source, k)
    if any(n < lo or n > hi for n in m): errors.append("copy / subset / retention rule")
    if cfg.get("time.movable", "all") == "copies_only":
        for i, n in enumerate(m):
            if n and not any(j == i and abs(d) <= 1e-8 for j, d in instances): errors.append("unshifted retained original missing")
    requirements = composition_requirements(source, k)
    for q in requirements:
        if not q.satisfied(m): errors.append("B: " + q.family + " " + str(q.key))
    times = source.time_counts(instances)
    for f, key, lo, hi, _ in time_requirements(source, k):
        if not lo - 1e-6 <= times[f][key] <= hi + 1e-6: errors.append("B time: " + f + " " + str(key))
    for f in cfg.list("features.time_families"):
        if any(key not in source.reference_time[f] for key in times[f]): errors.append("unobserved time category: " + f)
    return errors, m
