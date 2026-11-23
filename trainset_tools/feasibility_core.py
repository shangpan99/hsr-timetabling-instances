# -*- coding: utf-8 -*-
"""可行性求解与独立数值复核；不施加运营结束边界。"""
from __future__ import annotations

import argparse
import csv
import gc
import itertools
import math
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_LP_SECONDS = 30.0
DEFAULT_MIP_SECONDS = 300.0
DEFAULT_THREADS = 1
DEFAULT_CANDIDATES = 3
ARR, DEP = "到达时间", "出发时间"
FIELDS = ["instance", "K", "result", "stage", "solver_status", "message"]


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def finite(value, label):
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{label}: non-finite value")
    return x


def boolean(value):
    s = str(value).strip().lower()
    if s in {"true", "1", "1.0", "yes", "y", "是"}:
        return True
    if s in {"false", "0", "0.0", "no", "n", "否"}:
        return False
    raise ValueError(f"Invalid boolean: {value!r}")


def read_config(path):
    cfg = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip() and not row[0].lstrip().startswith("#"):
                cfg[row[0].strip()] = row[1].strip()
    if "section" in cfg or "dep_window_fast" not in cfg or "dep_window_normal" not in cfg:
        raise ValueError("Expected the experiment config.csv inside a generated group")
    return cfg


@dataclass
class Data:
    ids: list
    paths: dict
    stops: dict
    types: dict
    first_dep: dict
    station_order: list
    station: dict
    segment: dict
    cfg: dict

    def window(self, k):
        fast = int(float(self.cfg.get("fast_train_type", 350)))
        name = "dep_window_fast" if self.types[k] == fast else "dep_window_normal"
        width = finite(self.cfg[name], name)
        if width < 0:
            raise ValueError("Departure window must be nonnegative")
        ref = self.first_dep[k]
        early = boolean(self.cfg.get("allow_early_departure", "TRUE"))
        return (ref - width if early else ref), ref + width

    def run_time(self, k, u, v):
        p = self.segment[u, v]
        suffix = "350" if self.types[k] == 350 else "300"
        return (p["tau_" + suffix]
                + (p["eta_start_" + suffix] if self.stops[k, u] else 0)
                + (p["eta_end_" + suffix] if self.stops[k, v] else 0))

    def layers(self):
        # One common nonnegative headway per station/event kind, as in the main model.
        return {(kind, s): [k for k in self.ids
                           if s in (self.paths[k][1:] if kind == "arr" else self.paths[k][:-1])]
                for s in self.station_order for kind in ("arr", "dep")}


def load_data(folder, cfg):
    folder = Path(folder)
    cfg = dict(cfg)
    stations = read_csv(folder / "stations.csv")
    segments = read_csv(folder / "segments.csv")
    rows = read_csv(folder / "trains.csv")
    station = {}
    for r in stations:
        s = r["station"].strip()
        if s in station:
            raise ValueError(f"Duplicate station: {s}")
        station[s] = {n: finite(r[n], n) for n in ("h_arr", "h_dep", "dwell_min")}
        if min(station[s].values()) < 0:
            raise ValueError("Negative station parameter")
    segment = {}
    for r in segments:
        key = (r["from"].strip(), r["to"].strip())
        if key in segment:
            raise ValueError(f"Duplicate segment: {key}")
        segment[key] = {n + "_" + t: finite(r[n + "_" + t], n + "_" + t)
                       for t in ("300", "350") for n in ("tau", "eta_start", "eta_end")}
        if min(segment[key].values()) < 0:
            raise ValueError("Negative running-time parameter")
    grouped = {}
    for r in rows:
        grouped.setdefault(r["车次"].strip(), []).append(r)
    if not grouped or "" in grouped:
        raise ValueError("No trains or empty train ID")
    paths, stops, types, first_dep = {}, {}, {}, {}
    for k, rr in grouped.items():
        rr.sort(key=lambda r: finite(r["站序"], "station sequence"))
        seq_numbers = [finite(r["站序"], "station sequence") for r in rr]
        seq = [r["车站"].strip() for r in rr]
        if len(seq) < 2 or len(set(seq)) != len(seq) or len(set(seq_numbers)) != len(seq):
            raise ValueError(f"Invalid/duplicate station sequence: {k}")
        if any(s not in station for s in seq):
            raise ValueError(f"Missing station parameters: {k}")
        if any((u, v) not in segment for u, v in zip(seq, seq[1:])):
            raise ValueError(f"Missing segment parameters: {k}")
        # Generated copies inherit an explicit type; do not infer it from a copied ID.
        tt = {finite(r["train_type"], "train_type") for r in rr}
        if len(tt) != 1 or next(iter(tt)) not in (300, 350):
            raise ValueError(f"Invalid/inconsistent train_type: {k}")
        types[k] = int(next(iter(tt)))
        first_dep[k] = finite(rr[0][DEP], f"first departure of {k}")
        paths[k] = seq
        stops.update({(k, r["车站"].strip()): boolean(r["是否停站"]) for r in rr})
    d = Data(list(grouped), paths, stops, types, first_dep, list(station), station, segment, cfg)
    for k in d.ids:
        d.window(k)
    expected = re.fullmatch(r"K(\d+)", folder.parent.name)
    if expected and int(expected.group(1)) != len(d.ids):
        raise ValueError("Actual train count differs from the K directory")
    return d


def candidate_orders(data):
    # Project the reference departure to the corridor origin using technical running times.
    # This only proposes an order; all feasibility conclusions come from constraints below.
    offset = {s: 0.0 for s in data.station_order}
    for u, v in zip(data.station_order, data.station_order[1:]):
        offset[v] = offset[u] + data.segment.get((u, v), {}).get("tau_300", 0.0)
    seen = set()
    for label, value in [
        ("reference", lambda k: data.first_dep[k]),
        ("earliest", lambda k: data.window(k)[0]),
        ("latest", lambda k: data.window(k)[1]),
    ]:
        order = tuple(sorted(data.ids, key=lambda k: (value(k) - offset[data.paths[k][0]], k)))
        if order not in seen:
            seen.add(order)
            yield label, order


def build_model(data, gp, env, seconds, threads, order=None):
    model = gp.Model("HSR_feasibility", env=env)
    try:
        model.Params.OutputFlag = 0
        model.Params.Threads = threads
        model.Params.TimeLimit = seconds
        model.Params.DualReductions = 0
        model.Params.Seed = 40
        a, d, z = {}, {}, {}
        for k in data.ids:
            path = data.paths[k]
            for s in path[1:]:
                a[k, s] = model.addVar(lb=0.0)
            for s in path[:-1]:
                d[k, s] = model.addVar(lb=0.0)
            lower, upper = data.window(k)
            model.addConstr(d[k, path[0]] >= lower)
            model.addConstr(d[k, path[0]] <= upper)
            for u, v in zip(path, path[1:]):
                model.addConstr(a[k, v] >= d[k, u] + data.run_time(k, u, v))
            for s in path[1:-1]:
                if data.stops[k, s]:
                    model.addConstr(d[k, s] >= a[k, s] + data.station[s]["dwell_min"])
                else:
                    model.addConstr(d[k, s] == a[k, s])
        layers = data.layers()
        if order is not None:
            rank = {k: i for i, k in enumerate(order)}
            for (kind, s), present in layers.items():
                times = a if kind == "arr" else d
                present = sorted(present, key=rank.__getitem__)
                h = data.station[s]["h_" + kind]
                # For a fixed order, adjacent inequalities imply every pairwise inequality.
                for i, j in zip(present, present[1:]):
                    model.addConstr(times[j, s] >= times[i, s] + h)
        else:
            big_m = finite(data.cfg.get("big_m", 10000), "big_m")
            if big_m <= 0:
                raise ValueError("big_m must be positive")
            for (kind, s), present in layers.items():
                times = a if kind == "arr" else d
                h = data.station[s]["h_" + kind]
                for i, j in itertools.combinations(present, 2):
                    v = model.addVar(vtype=gp.GRB.BINARY)
                    z[kind, s, i, j] = v
                    model.addConstr(times[j, s] >= times[i, s] + h - big_m * (1 - v))
                    model.addConstr(times[i, s] >= times[j, s] + h - big_m * v)
            for u, v in data.segment:
                present = [k for k in data.ids if (u, v) in list(zip(data.paths[k], data.paths[k][1:]))]
                for i, j in itertools.combinations(present, 2):
                    model.addConstr(z["dep", u, i, j] == z["arr", v, i, j])
            model.Params.MIPFocus = 1
            model.Params.SolutionLimit = 1
        # Replaces the main experiment's TT objective. No terminal-time upper bound is added.
        model.setObjective(0.0, gp.GRB.MINIMIZE)
        return model, a, d
    except BaseException:
        model.dispose()
        raise


def validate(data, a, d, order=None):
    """Reconstruct operational checks from data and numeric times, without solver rows."""
    tol = finite(data.cfg.get("validator_tolerance", 1e-5), "validator_tolerance")
    if tol <= 0:
        raise ValueError("validator_tolerance must be positive")
    def ge(x, y):
        return x + tol >= y
    if any(not math.isfinite(t) or t < -tol for t in list(a.values()) + list(d.values())):
        return False
    for k in data.ids:
        path = data.paths[k]
        lo, hi = data.window(k)
        if not (ge(d[k, path[0]], lo) and ge(hi, d[k, path[0]])):
            return False
        for u, v in zip(path, path[1:]):
            if not ge(a[k, v] - d[k, u], data.run_time(k, u, v)):
                return False
        for s in path[1:-1]:
            dwell = d[k, s] - a[k, s]
            if data.stops[k, s]:
                if not ge(dwell, data.station[s]["dwell_min"]):
                    return False
            elif abs(dwell) > tol:
                return False
    rank = {k: i for i, k in enumerate(order)} if order is not None else None
    for (kind, s), present in data.layers().items():
        times = a if kind == "arr" else d
        keys = sorted(present, key=(rank.__getitem__ if rank is not None else lambda k: (times[k, s], k)))
        h = data.station[s]["h_" + kind]
        if any(not ge(times[j, s] - times[i, s], h) for i, j in zip(keys, keys[1:])):
            return False
    # Check every shared-segment pair directly; no inferred order can flip between its ends.
    for u, v in data.segment:
        present = [k for k in data.ids if (u, v) in list(zip(data.paths[k], data.paths[k][1:]))]
        for i, j in itertools.combinations(present, 2):
            before, after = d[i, u] - d[j, u], a[i, v] - a[j, v]
            if (before > tol and after < -tol) or (before < -tol and after > tol):
                return False
    for s in data.station_order:
        present = [k for k in data.ids if s in data.paths[k][1:-1]]
        for i, j in itertools.combinations(present, 2):
            arr_diff, dep_diff = a[i, s] - a[j, s], d[i, s] - d[j, s]
            overtaken = i if arr_diff < -tol and dep_diff > tol else (j if arr_diff > tol and dep_diff < -tol else None)
            if overtaken is not None and not data.stops[overtaken, s]:
                return False
    return True


def solve_once(data, gp, env, seconds, threads, order=None):
    model, a, d = build_model(data, gp, env, seconds, threads, order)
    try:
        model.optimize()
        status = int(model.Status)
        usable = int(model.SolCount) > 0
        passed = False
        if usable:
            aa = {key: float(v.X) for key, v in a.items()}
            dd = {key: float(v.X) for key, v in d.items()}
            passed = validate(data, aa, dd, order)
        return status, usable, passed
    finally:
        model.dispose()


def check(data, gp, env, args):
    for n, (name, order) in enumerate(candidate_orders(data)):
        if n >= args.candidates:
            break
        print(f"  LP candidate: {name}", flush=True)
        status, usable, passed = solve_once(data, gp, env, args.lp_seconds, args.threads, order)
        if passed:
            return "FEASIBLE", "fixed_order_LP", status, "Independent validation passed"
        gc.collect()
    print("  Full MILP: searching for the first feasible solution", flush=True)
    status, usable, passed = solve_once(data, gp, env, args.mip_seconds, args.threads)
    if passed:
        return "FEASIBLE", "full_MILP", status, "Independent validation passed"
    if usable:
        return "ERROR", "full_MILP", status, "A solution was found but independent validation failed"
    if status == gp.GRB.INFEASIBLE:
        return "INFEASIBLE", "full_MILP", status, "Full model proved infeasible"
    return "UNKNOWN", "full_MILP", status, "No validated solution and no infeasibility proof within this run"


def self_test(gp, env):
    # Meaningful edge cases: fixed-order failure != global infeasibility; no end cap;
    # a real infeasibility proof; and timeout without an incumbent remains undecided.
    cfg = {"dep_window_fast": "0", "dep_window_normal": "4", "allow_early_departure": "FALSE",
           "big_m": "10000", "validator_tolerance": "0.00001"}
    sp = {"tau_300": 1., "tau_350": 1., "eta_start_300": 0., "eta_start_350": 0.,
          "eta_end_300": 0., "eta_end_350": 0.}
    data = Data(["A", "B"], {k: ["S", "T"] for k in ["A", "B"]},
                {(k, s): True for k in ["A", "B"] for s in ["S", "T"]},
                {"A": 350, "B": 300}, {"A": 10., "B": 10.}, ["S", "T"],
                {s: {"h_arr": 2., "h_dep": 2., "dwell_min": 1.} for s in ["S", "T"]},
                {("S", "T"): sp}, cfg)
    assert solve_once(data, gp, env, 5., 1, ["B", "A"])[0] == gp.GRB.INFEASIBLE
    assert solve_once(data, gp, env, 5., 1)[2]
    data.types["B"] = 350
    assert solve_once(data, gp, env, 5., 1)[0] == gp.GRB.INFEASIBLE
    data.ids = ["A"]
    data.first_dep["A"] = 1429.
    assert solve_once(data, gp, env, 5., 1, ["A"])[2]
    status, usable, passed = solve_once(data, gp, env, 0., 1)
    assert status == gp.GRB.TIME_LIMIT and not usable and not passed
    print("SELF_TEST PASS: order fallback; genuine infeasibility; after-23:49 feasibility; timeout", flush=True)

