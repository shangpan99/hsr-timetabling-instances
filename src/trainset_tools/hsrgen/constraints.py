"""All composition requirements are linear in source-template multiplicities."""
import math
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction


def quota(target, extra=0):
    if isinstance(target, Fraction):
        return max(0, math.floor(target) - extra), math.ceil(target) + extra
    return max(0, math.floor(target + 1e-9) - extra), math.ceil(target - 1e-9) + extra


def quantile(values, q):
    values = sorted(values)
    if not values: return None
    x = (len(values) - 1) * q; lo = int(x); hi = min(lo + 1, len(values) - 1)
    return values[lo] * (1 - (x - lo)) + values[hi] * (x - lo)


@dataclass
class Requirement:
    family: str
    key: object
    rule: str
    coefficients: dict
    lower: float
    upper: float
    target: float = 0
    reference_n: int = 0

    def value(self, m): return sum(v * m[i] for i, v in self.coefficients.items())
    def satisfied(self, m):
        v = self.value(m)
        return self.lower - 1e-6 <= v <= self.upper + 1e-6


def copy_bounds(source, k):
    cfg = source.cfg
    if k <= 0: raise ValueError("K must be positive")
    if k < source.n: return 0, 1
    if k == source.n: return 1, 1
    mode = cfg.get("copy.cap_mode", "adaptive")
    upper = (math.ceil(k / source.n) + cfg.int("copy.extra_allowance", 2)) if mode == "adaptive" else cfg.int("copy.max_total_occurrences", 4)
    if mode not in ("adaptive", "fixed"): raise ValueError("Unknown copy.cap_mode")
    if upper < 1: raise ValueError("Copy cap must be positive")
    return 1, upper


def composition_requirements(source, k, exact=False):
    cfg = source.cfg; out = []; scale = Fraction(k, source.n) if exact else k / source.n
    number = (lambda x: Fraction(str(x))) if exact else float
    setting = lambda key, default=0: number(cfg.get(key, default))
    ratio = (lambda a, b: Fraction(a, b)) if exact else (lambda a, b: a / b)
    epsilon = 0 if exact else 1e-9
    for family in cfg.list("features.count_families"):
        if family not in source.counts: raise ValueError("Unknown count family: " + family)
        for key, count in source.counts[family].items():
            lo, hi = quota(count * scale, cfg.int("tolerance.count_extra", 0))
            coeff = {t.index: t.categories.get(family, {}).get(key, 0) for t in source.trains}
            out.append(Requirement(family, key, "count", coeff, lo, hi, count * scale, count))
    for family in cfg.list("coverage.required_families"):
        if family not in source.counts: raise ValueError("Unknown coverage family: " + family)
        for key, count in source.counts[family].items():
            out.append(Requirement("coverage:" + family, key, "coverage",
                                   {t.index: t.categories.get(family, {}).get(key, 0) for t in source.trains},
                                   1, math.inf, 1, count))
    top = sorted(source.counts["od"].items(), key=lambda x: (-x[1], str(x[0])))[:cfg.int("coverage.top_od_count", 6)]
    for key, count in top:
        out.append(Requirement("coverage:od", key, "coverage", {t.index: int(t.od == key) for t in source.trains}, 1, math.inf, 1, count))
    for tid in cfg.list("coverage.required_source_ids"):
        if tid not in source.by_id: raise ValueError("Unknown required source train: " + tid)
        out.append(Requirement("coverage:source_id", tid, "coverage", {source.by_id[tid].index: 1}, 1, math.inf, 1, 1))
    if cfg.bool("features.station_stop_rates", True):
        delta = setting("tolerance.station_stop_rate_pp", 5) / 100
        for s in source.station_map:
            a = source.counts["station_stop"][s]; b = source.counts["station_pass"][s]
            if not a + b: continue
            p = ratio(a, a + b)
            for side, sign in (("low", -1), ("high", 1)):
                bound = max(0, min(1, p + sign * delta))
                coeff = {}
                for t in source.trains:
                    st = t.categories["station_stop"][s]; ps = t.categories["station_pass"][s]
                    coeff[t.index] = st - bound * (st + ps)
                out.append(Requirement("station_stop_rate", (s, side), "ratio", coeff,
                                       0 if side == "low" else -math.inf, math.inf if side == "low" else 0, p, a + b))
    for family in cfg.list("features.mean_families"):
        if family not in source.duration_groups: raise ValueError("Unknown mean family: " + family)
        for key, values in source.duration_groups[family].items():
            if key != "all" and len(values) < cfg.int("features.conditional_min_events", 10): continue
            if exact: values = list(map(number, values))
            mean = sum(values) / len(values)
            rel = setting("tolerance.conditional_mean_percent", 5) if key != "all" else setting("tolerance.mean_percent", 2)
            delta = max(abs(mean) * rel / 100, setting("tolerance.mean_absolute_minutes", 0))
            for side, sign in (("low", -1), ("high", 1)):
                coeff = {}
                for t in source.trains:
                    vs = t.samples.get(family, {}).get(key, [])
                    if exact: vs = list(map(number, vs))
                    coeff[t.index] = sum(vs) - (mean + sign * delta) * len(vs)
                out.append(Requirement("mean:" + family, (key, side), "mean", coeff,
                                       0 if side == "low" else -math.inf, math.inf if side == "low" else 0, mean, len(values)))
    for family in cfg.list("features.cdf_families"):
        if family not in source.duration_groups: raise ValueError("Unknown CDF family: " + family)
        for key, values in source.duration_groups[family].items():
            if key != "all" and len(values) < cfg.int("features.conditional_min_events", 10): continue
            if exact: values = list(map(number, values))
            cuts = sorted(set(quantile(values, number(q)) for q in cfg.list("features.cdf_quantiles", "0.1;0.25;0.5;0.75;0.9")))
            for cut in cuts:
                p = ratio(sum(x <= cut + epsilon for x in values), len(values))
                delta = setting("tolerance.conditional_cdf_pp", 10) / 100 if key != "all" else setting("tolerance.cdf_pp", 5) / 100
                extra = setting("tolerance.cdf_event_allowance", 1)
                for side, sign in (("low", -1), ("high", 1)):
                    bound = max(0, min(1, p + sign * delta)); coeff = {}
                    for t in source.trains:
                        vs = t.samples.get(family, {}).get(key, [])
                        if exact: vs = list(map(number, vs))
                        coeff[t.index] = sum(x <= cut + epsilon for x in vs) - bound * len(vs)
                    out.append(Requirement("cdf:" + family, (key, cut, side), "cdf", coeff,
                                           -extra if side == "low" else -math.inf, math.inf if side == "low" else extra, p, len(values)))
    return out


def time_requirements(source, k, exact=False):
    out = []
    families = source.cfg.list("features.time_families")
    for family in families:
        if family not in source.reference_time: raise ValueError("Unknown time family: " + family)
        for key, count in source.reference_time[family].items():
            target = Fraction(count * k, source.n) if exact else count * k / source.n
            lo, hi = quota(target, source.cfg.int("tolerance.time_count_extra", 0))
            out.append((family, key, lo, hi, count))
    if source.cfg.bool("coverage.entry_periods", True):
        for key, count in source.reference_time["entry_period"].items():
            out.append(("entry_period", key, 1, math.inf, count))
    return out
