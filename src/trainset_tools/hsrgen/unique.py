"""Fixed integer scores, proven uniqueness, and stable whole-train shifts.

The model is built and postchecked with exact integer arithmetic. Gurobi's
optimization is floating point; an incumbent is never accepted on a time limit.
Only primary integer variables count as distinct solutions in the solution pool.
"""
import hashlib
import json
import math
import time
import copy
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction

from .constraints import composition_requirements, copy_bounds, time_requirements

GENERATOR_VERSION = "unique-score-v1"
# Preserve the score/minute namespace of the validated K200 prototype. The
# namespace is an algorithm identifier, not a restriction on supported sizes.
SCORE_VERSION = "unique-score-k200-v1"


def stable(*values):
    payload = json.dumps(values, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).digest()


def integer_optimum_closed(exact, reported, upper_bound):
    """A maximized integer objective cannot improve below the next integer.

    Call only after OPTIMAL and exact verification of the integer vector.
    Solver objective attributes are doubles: K400_G10 reported
    3688931280.999978 for the exactly recomputed score 3688931281. A fixed
    1e-5 absolute check rejects such reporting residuals at billion scale.

    Use a scale-aware consistency tolerance, capped at a quarter of the
    integer score spacing. This does not change any solver tolerance or
    feasible vector. Add the same margin to the upper bound, so numerical
    slack cannot certify a bound at or too close to the NEXT integer.
    Reject magnitudes where doubles cannot resolve this capped margin.
    """
    if (abs(exact) >= 2**50 or not math.isfinite(reported)
            or not math.isfinite(upper_bound)):
        return False
    tolerance = min(0.25, max(1e-5, abs(exact) * 1e-12))
    bound_residual = upper_bound - exact
    return (abs(reported - exact) <= tolerance
            and -tolerance <= bound_residual
            and bound_residual + tolerance < 1)


class ExactModel:
    def __init__(self):
        self.domains, self.keys, self.rows, self.exclusions = [], [], {}, []
        self.infeasible = False

    def var(self, lo, hi, key):
        if int(lo) != lo or int(hi) != hi or lo > hi:
            raise ValueError("Expected finite integer variable bounds")
        self.domains.append((int(lo), int(hi)))
        self.keys.append(key)
        return len(self.domains) - 1

    def add(self, coefficients, lo=-math.inf, hi=math.inf):
        values = {i: v if isinstance(v, Fraction) else Fraction(str(v))
                  for i, v in coefficients.items() if v}
        denominator = math.lcm(*(v.denominator for v in values.values())) if values else 1
        integers = {i: int(v * denominator) for i, v in values.items()}
        divisor = (math.gcd(*integers.values()) if integers else 1) or 1
        row = tuple(sorted((i, v // divisor) for i, v in integers.items()))
        lower = math.ceil(Fraction(str(lo)) * denominator / divisor) if math.isfinite(lo) else None
        upper = math.floor(Fraction(str(hi)) * denominator / divisor) if math.isfinite(hi) else None
        if not row:
            if (lower is not None and lower > 0) or (upper is not None and upper < 0):
                self.infeasible = True
            return
        bound = sum(abs(v) * max(abs(self.domains[i][0]), abs(self.domains[i][1])) for i, v in row)
        if bound > 10**15:
            raise ValueError("Exact integer constraint exceeds the supported numeric range")
        lower = max(-bound, lower) if lower is not None else -bound
        upper = min(bound, upper) if upper is not None else bound
        if row in self.rows:
            old_lo, old_hi = self.rows[row]
            lower, upper = max(lower, old_lo), min(upper, old_hi)
        if lower > upper:
            self.infeasible = True
        self.rows[row] = (lower, upper)

    def exclude(self, values):
        if len(values) != len(self.domains):
            raise ValueError("Wrong excluded vector length")
        self.exclusions.append(tuple(values))

    def verify(self, values):
        if self.infeasible or len(values) != len(self.domains):
            raise ValueError("Invalid integer solution")
        if any(int(v) != v or not lo <= v <= hi for v, (lo, hi) in zip(values, self.domains)):
            raise ValueError("Integer solution violates a variable bound")
        for row, (lo, hi) in self.rows.items():
            if not lo <= sum(a * values[i] for i, a in row) <= hi:
                raise ValueError("Integer solution violates an exact constraint")
        if tuple(values) in self.exclusions:
            raise ValueError("Solver returned an excluded composition")


def solve_unique(model, cfg, seed, stage, deadline=None):
    start = time.perf_counter()
    deadline = deadline if deadline is not None else start + cfg.float("solver.time_limit_seconds", 60)
    if not cfg.bool("solver.dual_search", False) or cfg.int("solver.threads", 8) < 2 or stage == "timing":
        return _solve_single(model, cfg, seed, stage, deadline)
    # Use separate models AND environments: each worker must finish its own
    # pool proof and tie-break. Built-in ConcurrentMIP does not preserve the
    # pool proof needed here. Arrival order cannot change the unique vector.
    stop = threading.Event()
    configurations = []
    total_threads = cfg.int("solver.threads", 8)
    for index, focus in enumerate((0, 3)):
        worker = copy.copy(cfg)
        worker.values = dict(cfg.values, **{
            "solver.threads": str(total_threads // 2 + (total_threads % 2 if index == 0 else 0)),
            "solver.proof_focus": str(focus)})
        configurations.append(worker)
    answer = None
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_solve_single, model, worker, seed, stage, deadline, stop)
                   for worker in configurations]
        try:
            for future in as_completed(futures):
                candidate = future.result()
                if candidate[1]["status"] in ("OPTIMAL", "INFEASIBLE"):
                    answer = candidate
                    stop.set()
                    break
                answer = candidate
        finally:
            stop.set()
        # Surface validation/programming errors even in the cancelled worker.
        for future in futures:
            other_values, other_stats = future.result()
            if (answer[1]["status"] in ("OPTIMAL", "INFEASIBLE")
                    and other_stats["status"] in ("OPTIMAL", "INFEASIBLE")
                    and (answer[1]["status"] != other_stats["status"] or answer[0] != other_values)):
                raise RuntimeError("Independent proofs disagree; no case will be accepted")
    values, stats = answer
    stats = dict(stats, search_strategy="independent_focus_0_3", seconds=time.perf_counter() - start)
    return values, stats


def _solve_single(model, cfg, seed, stage, deadline, stop=None):
    import gurobipy as gp
    start = time.perf_counter()
    deadline = deadline if deadline is not None else start + cfg.float("solver.time_limit_seconds", 60)
    scores = [1 + int.from_bytes(stable(SCORE_VERSION, seed, stage, "score", key)[:3], "big")
              for key in model.keys]
    model_hash = hashlib.sha256(json.dumps(
        [model.domains, model.keys, list(model.rows.items()), model.exclusions, scores],
        ensure_ascii=True, separators=(",", ":")).encode("ascii")).hexdigest()
    result = dict(backend="gurobi", solver_version=".".join(map(str, gp.gurobi.version())),
                  generator_version=GENERATOR_VERSION, variables=len(model.domains),
                  constraints=len(model.rows), model_sha256=model_hash,
                  objective=None, unique=False, uniqueness_method="", message="")

    def finish(values, status, message=""):
        if values is not None:
            model.verify(values)
        return values, dict(result, status=status, message=message, seconds=time.perf_counter() - start)

    if model.infeasible:
        return finish(None, "INFEASIBLE", "Contradictory exact constraints")
    if not model.domains:
        result.update(unique=True, uniqueness_method="FIXED_VECTOR", objective=0)
        return finish([], "OPTIMAL")
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", int(cfg.bool("solver.console_log")))
        env.start()
        with gp.Model(env=env) as gm:
            gm.Params.Threads = cfg.int("solver.threads", 8)
            gm.Params.ConcurrentMIP = 1
            # Selection depends on scores, never on this search seed.
            gm.Params.Seed = 0
            gm.Params.MIPGap = 0
            gm.Params.MIPGapAbs = 0
            # Closing the optimality/uniqueness proof is the expensive part.
            # Bound-focused search changes speed, not the chosen integer optimum.
            gm.Params.MIPFocus = cfg.int("solver.proof_focus", 0)
            gm.Params.FeasibilityTol = 1e-9
            gm.Params.IntFeasTol = 1e-9
            gm.Params.OptimalityTol = 1e-9
            gm.Params.PoolSearchMode = 2
            gm.Params.PoolSolutions = 2
            gm.Params.PoolGap = 0
            gm.Params.PoolGapAbs = 0
            variables = [gm.addVar(lb=lo, ub=hi, vtype=gp.GRB.BINARY if (lo, hi) == (0, 1) else gp.GRB.INTEGER,
                                   name=f"x{i}") for i, (lo, hi) in enumerate(model.domains)]
            for row, (lo, hi) in model.rows.items():
                expr = gp.LinExpr([a for _, a in row], [variables[i] for i, _ in row])
                if lo == hi:
                    gm.addConstr(expr == lo)
                else:
                    gm.addConstr(expr >= lo)
                    gm.addConstr(expr <= hi)
            for previous in model.exclusions:
                flags = []
                for v, p, (lo, hi) in zip(variables, previous, model.domains):
                    if p < hi:
                        b = gm.addVar(vtype=gp.GRB.BINARY)
                        b.PoolIgnore = 1
                        gm.addConstr(v >= p + 1 - (p + 1 - lo) * (1 - b))
                        flags.append(b)
                    if p > lo:
                        b = gm.addVar(vtype=gp.GRB.BINARY)
                        b.PoolIgnore = 1
                        gm.addConstr(v <= p - 1 + (hi - p + 1) * (1 - b))
                        flags.append(b)
                gm.addConstr(gp.quicksum(flags) >= 1)
            score_expr = gp.LinExpr(scores, variables)
            gm.setObjective(score_expr, gp.GRB.MAXIMIZE)

            def optimize():
                remaining = deadline - time.perf_counter()
                if remaining <= 0 or (stop is not None and stop.is_set()):
                    return False
                gm.Params.TimeLimit = remaining
                if stop is None:
                    gm.optimize()
                else:
                    def cancel_other_worker(current, where):
                        if stop.is_set():
                            current.terminate()
                    gm.optimize(cancel_other_worker)
                return True

            if not optimize():
                return finish(None, "UNKNOWN_LIMIT", "Group time budget exhausted")
            if gm.Status == gp.GRB.INFEASIBLE:
                return finish(None, "INFEASIBLE")
            if gm.Status != gp.GRB.OPTIMAL:
                return finish(None, "UNKNOWN_LIMIT", f"Optimum/uniqueness incomplete; solver status {gm.Status}")
            values = [round(v.X) for v in variables]
            model.verify(values)
            objective = sum(a * v for a, v in zip(scores, values))
            if not integer_optimum_closed(objective, gm.ObjVal, gm.ObjBound):
                raise RuntimeError(f"Integer score optimum has not closed: exact={objective}, "
                                   f"ObjVal={gm.ObjVal!r}, ObjBound={gm.ObjBound!r}, "
                                   f"ObjBoundC={gm.ObjBoundC!r}, "
                                   f"max_integrality_error={max(abs(v.X - round(v.X)) for v in variables)!r}")
            result["objective"] = objective
            if gm.SolCount == 1:
                result.update(unique=True, uniqueness_method="POOL_UNIQUE")
                return finish(values, "OPTIMAL")

            # Exact deterministic tie-break on primary integer variables. Mixed
            # radix handles multiplicities >1 without exponential-size weights.
            gm.addConstr(score_expr == objective)
            gm.Params.PoolSearchMode = 0
            order = sorted(range(len(model.keys)), key=lambda i: (stable(SCORE_VERSION, seed, stage, model.keys[i]), i))
            order = [i for i in order if model.domains[i][0] != model.domains[i][1]]
            while order:
                block, product = [], 1
                for i in order:
                    radix = model.domains[i][1] - model.domains[i][0] + 1
                    if block and product * radix > 2**20:
                        break
                    block.append(i)
                    product *= radix
                weights, weight = [], 1
                for i in reversed(block):
                    weights.append(weight)
                    weight *= model.domains[i][1] - model.domains[i][0] + 1
                weights.reverse()
                gm.setObjective(gp.LinExpr(weights, [variables[i] for i in block]), gp.GRB.MAXIMIZE)
                if not optimize() or gm.Status != gp.GRB.OPTIMAL:
                    return finish(None, "UNKNOWN_LIMIT", "Canonical tie-break incomplete")
                values = [round(v.X) for v in variables]
                model.verify(values)
                if sum(a * v for a, v in zip(scores, values)) != objective:
                    raise RuntimeError("Tie-break left the optimal score face")
                optimum = sum(values[i] * w for i, w in zip(block, weights))
                if not integer_optimum_closed(optimum, gm.ObjVal, gm.ObjBound):
                    raise RuntimeError("Tie-break optimum has not closed")
                for i in block:
                    variables[i].LB = variables[i].UB = values[i]
                order = order[len(block):]
            result.update(unique=True, uniqueness_method="LEXICOGRAPHIC_TIE_BREAK")
            return finish(values, "OPTIMAL")


def check_exact_source(source):
    times = [v for t in source.trains for v in t.arrivals + t.departures if v is not None]
    times += [source.horizon_start, source.horizon_end, source.grid] + source.period_edges
    if any(v != int(v) for v in times):
        raise ValueError("unique_score currently requires integral source minutes, boundaries and grid")


def compose_unique(source, k, seed, exclusions=(), deadline=None):
    from .generate import shift_limits
    check_exact_source(source)
    model = ExactModel()
    lower, upper = copy_bounds(source, k)
    for t in source.trains:
        a, b = shift_limits(source, t)
        ub = upper if a <= b else 0
        if lower > ub:
            return None, dict(status="INFEASIBLE", backend="precheck", seconds=0,
                              message="Retained template has no admissible shift: " + t.source)
        model.var(lower, ub, t.source)
    model.add({i: 1 for i in range(source.n)}, k, k)
    for q in composition_requirements(source, k, exact=True):
        model.add(q.coefficients, q.lower, q.upper)
    for previous in exclusions:
        model.exclude(previous)
    return solve_unique(model, source.cfg, seed, "composition", deadline)


def assign_times_unique(source, multiplicities, seed, deadline=None):
    from .generate import shift_limits
    check_exact_source(source)
    cfg = source.cfg
    k = sum(multiplicities)
    movable = list(multiplicities)
    fixed = []
    scope = cfg.get("time.movable", "all")
    if scope not in ("all", "copies_only"):
        raise ValueError("time.movable must be all or copies_only")
    if scope == "copies_only":
        for i, count in enumerate(movable):
            if count:
                fixed.append((i, 0))
                movable[i] -= 1
                t = source.trains[i]
                if t.start < source.horizon_start or t.end > source.horizon_end:
                    return None, dict(status="INFEASIBLE", backend="precheck", seconds=0,
                                      message="Fixed original lies outside operating horizon")
    fixed_counts = source.time_counts(fixed)
    model, options, members = ExactModel(), [], defaultdict(dict)
    active = cfg.list("features.time_families")
    for t, count in zip(source.trains, movable):
        if not count:
            continue
        left, right = shift_limits(source, t)
        cells = {}
        for step in range(left, right + 1):
            keys = source.time_keys(t, t.entry + step * source.grid)
            if any(keys[f] not in source.reference_time[f] for f in active):
                continue
            cell = tuple(keys.items())
            cells.setdefault(cell, []).append(step)
        own = {}
        for cell, steps in cells.items():
            v = model.var(0, count, (t.source, cell))
            own[v] = 1
            options.append((t.index, steps))
            for family, key in cell:
                members[family, key][v] = 1
        model.add(own, count, count)
    for family, key, lo, hi, _ in time_requirements(source, k, exact=True):
        already = fixed_counts[family][key]
        model.add(members[family, key], lo - already, hi - already)
    assignment, result = solve_unique(model, cfg, seed, "timing", deadline)
    if assignment is None:
        return None, result
    instances = list(fixed)
    for v, count in enumerate(assignment):
        if not count:
            continue
        i, steps = options[v]
        token = (SCORE_VERSION, seed, "minute", source.trains[i].source, steps[0], steps[-1])
        position = int.from_bytes(stable(*token), "big") % len(steps)
        stride = 1
        if count > 1 and len(steps) > 1:
            stride = 1 + int.from_bytes(stable(*token, "stride"), "big") % (len(steps) - 1)
            while math.gcd(stride, len(steps)) != 1:
                stride = stride % (len(steps) - 1) + 1
        instances.extend((i, steps[(position + j * stride) % len(steps)] * source.grid) for j in range(count))
    instances.sort(key=lambda p: (source.trains[p[0]].entry + p[1], source.trains[p[0]].source, p[1]))
    return instances, result
