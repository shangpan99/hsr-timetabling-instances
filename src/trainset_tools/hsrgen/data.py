import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


def write_csv(path, rows, fields=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        w.writeheader(); w.writerows(rows)


def read_csv(path):
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("gb18030")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Cannot decode CSV {path}: expected UTF-8 or GB18030/GBK") from exc
    with io.StringIO(text, newline="") as f:
        reader = csv.DictReader(f); rows = list(reader)
        return list(reader.fieldnames or []), rows


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(x):
    if x is None: return ""
    return str(int(round(x))) if abs(x - round(x)) < 1e-8 else format(x, ".12g")


def truth(s):
    s = str(s).strip().lower()
    if s in ("true", "1", "yes", "y", "是"): return True
    if s in ("false", "0", "no", "n", "否"): return False
    raise ValueError("Invalid boolean: " + s)


class Config:
    def __init__(self, path):
        self.path = Path(path).resolve()
        _, self.rows = read_csv(self.path)
        self.values = {}
        for r in self.rows:
            if not r.get("parameter") or r["parameter"].startswith("#"): continue
            key = r["section"].strip() + "." + r["parameter"].strip()
            if key in self.values: raise ValueError("Duplicate configuration: " + key)
            self.values[key] = (r.get("value") or "").strip()

    def get(self, key, default=""): return self.values.get(key, str(default))
    def float(self, key, default=0): return float(self.get(key, default))
    def int(self, key, default=0):
        v = self.float(key, default)
        if not v.is_integer(): raise ValueError(key + " must be an integer")
        return int(v)
    def bool(self, key, default=False): return truth(self.get(key, str(default)))
    def list(self, key, default=""): return [x.strip() for x in self.get(key, default).split(";") if x.strip()]
    def path_value(self, key):
        v = self.get(key)
        return (self.path.parent / v).resolve() if v else None


@dataclass
class Train:
    index: int
    source: str
    kind: str
    direction: str
    rows: list
    stations: tuple
    arrivals: tuple
    departures: tuple
    stops: tuple
    entry: float
    start: float
    end: float
    travel: float
    stop_count: int
    stop_group: str
    od: tuple
    pattern: tuple
    segments: tuple
    categories: dict
    samples: dict
    train_station: list
    train_segment: list


def parse_time(s, fmt):
    s = str(s).strip()
    if not s: return None
    if fmt == "hhmm":
        p = s.split(":")
        if len(p) not in (2, 3): raise ValueError("Expected HH:MM[:SS]: " + s)
        hour, minute = int(p[0]), int(p[1]); sec = float(p[2]) if len(p) == 3 else 0
        if not 0 <= minute < 60 or not 0 <= sec < 60: raise ValueError("Bad clock time: " + s)
        return hour * 60 + minute + sec / 60
    value = float(s)
    if not math.isfinite(value): raise ValueError("Nonfinite time")
    return value / 60 if fmt == "seconds" else value


def format_time(v, fmt):
    if v is None: return ""
    if fmt == "hhmm":
        seconds = round(v * 60)
        if abs(seconds / 60 - v) > 1e-7: raise ValueError("Subsecond value cannot be written in hhmm")
        return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
    return number(v * 60 if fmt == "seconds" else v)


def key_label(key):
    return json.dumps(key, ensure_ascii=False, separators=(",", ":"))


class Source:
    def __init__(self, cfg):
        self.cfg = cfg
        for key in ("tolerance.count_extra", "tolerance.time_count_extra", "tolerance.mean_percent",
                    "tolerance.conditional_mean_percent", "tolerance.mean_absolute_minutes",
                    "tolerance.cdf_pp", "tolerance.conditional_cdf_pp", "tolerance.cdf_event_allowance",
                    "tolerance.station_stop_rate_pp", "copy.extra_allowance", "copy.balance_weight"):
            if cfg.float(key, 0) < 0: raise ValueError(key + " cannot be negative")
        for q in cfg.list("features.cdf_quantiles"):
            if not 0 <= float(q) <= 1: raise ValueError("CDF quantiles must be in [0,1]")
        if cfg.int("solver.threads", 1) < 1 or cfg.float("solver.time_limit_seconds", 60) <= 0:
            raise ValueError("Solver threads and time limit must be positive")
        if cfg.get("solver.composition_mode", "feasibility") not in ("feasibility", "random_cost", "unique_score"):
            raise ValueError("Unknown solver.composition_mode")
        if cfg.get("solver.composition_mode") == "unique_score":
            if cfg.get("solver.backend") != "gurobi":
                raise ValueError("unique_score requires solver.backend=gurobi; fallback is not permitted")
            if cfg.float("solver.relative_gap", 0) != 0:
                raise ValueError("unique_score requires solver.relative_gap=0")
            if not 0 <= cfg.int("solver.proof_focus", 0) <= 3:
                raise ValueError("solver.proof_focus must be 0, 1, 2 or 3")
            cfg.bool("solver.dual_search", False)
        self.paths = {k: cfg.path_value("input." + k) for k in ("trains", "stations", "segments", "experiment_config")}
        self.fields, rows = read_csv(self.paths["trains"])
        self.station_fields, self.station_rows = read_csv(self.paths["stations"])
        self.segment_fields, self.segment_rows = read_csv(self.paths["segments"])
        self.station_map = {r["station"]: r for r in self.station_rows}
        self.segment_map = {(r["from"], r["to"]): r for r in self.segment_rows}
        if len(self.station_map) != len(self.station_rows): raise ValueError("Duplicate stations")
        if len(self.segment_map) != len(self.segment_rows): raise ValueError("Duplicate directed segments")
        self.cols = {k: cfg.get("columns." + k) for k in
                     ("id", "station", "sequence", "arrival", "departure", "stop", "type", "direction")}
        for k, col in self.cols.items():
            if col not in self.fields and k not in ("type", "direction"):
                raise ValueError("Missing train column: " + col)
        if self.cols["type"] not in self.fields: self.fields.append(self.cols["type"])
        self.fmt = cfg.get("input.time_format", "minutes")
        if self.fmt not in ("minutes", "seconds", "hhmm"): raise ValueError("Invalid time_format")
        self.extra_times = cfg.list("columns.extra_absolute_time_fields")
        for col in self.extra_times:
            if col not in self.fields: raise ValueError("Missing extra time column: " + col)
        grouped = defaultdict(list)
        for r in rows:
            tid = r[self.cols["id"]].strip()
            if not tid: raise ValueError("Empty train ID")
            grouped[tid].append(r)
        if not grouped: raise ValueError("No trains")
        self.audit = []
        self.trains = [self._train(i, tid, rr) for i, (tid, rr) in enumerate(sorted(grouped.items()))]
        self.n = len(self.trains)
        self.by_id = {t.source: t for t in self.trains}
        longest = max(len(t.stations) for t in self.trains)
        for t in self.trains:
            t.categories["route_span"] = Counter({"longest_path" if len(t.stations) == longest else "partial_path": 1})
        self.horizon_start = self._boundary("time.operating_start", min(t.start for t in self.trains))
        self.horizon_end = self._boundary("time.operating_end", max(t.end for t in self.trains))
        if self.horizon_end <= self.horizon_start: raise ValueError("Empty operating horizon")
        self.bin_minutes = cfg.int("time.bin_minutes", 60)
        self.grid = cfg.float("time.grid_minutes", 1)
        if self.bin_minutes <= 0 or self.grid <= 0: raise ValueError("Time steps must be positive")
        self.period_edges = sorted(set(float(x) for x in cfg.list("time.period_edges")))
        self.counts = defaultdict(Counter)
        self.duration_groups = defaultdict(lambda: defaultdict(list))
        for t in self.trains:
            for f, c in t.categories.items(): self.counts[f].update(c)
            for f, gs in t.samples.items():
                for g, vs in gs.items(): self.duration_groups[f][g].extend(vs)
        self.reference_time = self.time_counts([(t.index, 0.0) for t in self.trains])
        self.fingerprints = {k: sha(p) for k, p in self.paths.items() if p and p.exists()}

    def _boundary(self, key, auto):
        v = self.cfg.get(key, "auto")
        return auto if v.lower() in ("", "auto") else parse_time(v, "hhmm" if ":" in v else "minutes")

    def origin_window(self, t):
        width = self.cfg.float("model.fast_departure_window", 60) if t.kind == self.cfg.get("model.fast_type", "350") else self.cfg.float("model.normal_departure_window", 75)
        return (width if self.cfg.bool("model.allow_early_departure", True) else 0), width

    def _train(self, index, tid, rows):
        c = self.cols; cfg = self.cfg
        rows = sorted(rows, key=lambda r: float(r[c["sequence"]]))
        if len(rows) < 2: raise ValueError(tid + ": at least two route stations required")
        if len(set(float(r[c["sequence"]]) for r in rows)) != len(rows): raise ValueError(tid + ": duplicate route positions")
        kinds = {r.get(c["type"], "").strip() for r in rows if r.get(c["type"], "").strip()}
        kinds = {number(float(x)) if x.replace(".", "", 1).isdigit() else x for x in kinds}
        if len(kinds) > 1: raise ValueError(tid + ": inconsistent train types")
        if kinds: kind = next(iter(kinds))
        elif cfg.get("input.missing_type_policy", "error") == "legacy_id":
            digits = "".join(x for x in tid if x.isdigit())
            kind = "350" if digits and 1 <= int(digits) <= 50 else "300"
            self.audit.append({"train": tid, "field": c["type"], "action": "legacy_id_inference", "resolved_value": kind})
        else: raise ValueError(tid + ": missing type; configure an explicit inference policy")
        if any(not r.get(c["type"], "").strip() for r in rows) and kinds:
            self.audit.append({"train": tid, "field": c["type"], "action": "within_train_fill", "resolved_value": kind})
        directions = {r.get(c["direction"], "").strip() for r in rows}
        if len(directions) != 1: raise ValueError(tid + ": inconsistent directions")
        direction = next(iter(directions))
        stations = tuple(r[c["station"]].strip() for r in rows)
        if len(set(stations)) != len(stations): raise ValueError(tid + ": repeated stations require an occurrence-based adapter")
        if any(s not in self.station_map for s in stations): raise ValueError(tid + ": unknown station")
        arrivals, departures, stops = [], [], []
        previous, day = -math.inf, 0
        for r in rows:
            vals = []
            for col in (c["arrival"], c["departure"]):
                v = parse_time(r[col], self.fmt)
                if v is not None:
                    if self.fmt == "hhmm" and cfg.bool("input.unwrap_midnight", False):
                        while v + day < previous - 1e-8: day += 1440
                        v += day
                    if v < previous - 1e-8: raise ValueError(tid + ": nonmonotone event times")
                    previous = v
                vals.append(v)
            arrivals.append(vals[0]); departures.append(vals[1]); stops.append(truth(r[c["stop"]]))
        if departures[0] is None or arrivals[-1] is None: raise ValueError(tid + ": missing first departure or last arrival")
        for j in range(1, len(rows) - 1):
            if arrivals[j] is None or departures[j] is None: raise ValueError(tid + ": missing intermediate event")
            if not stops[j] and abs(arrivals[j] - departures[j]) > 1e-8:
                raise ValueError(tid + ": pass event has nonzero dwell")
        events = [v for p in zip(arrivals, departures) for v in p if v is not None]
        for r in rows:
            for col in self.extra_times:
                v = parse_time(r.get(col, ""), self.fmt)
                if v is not None and not math.isfinite(v): raise ValueError("Invalid additional time")
        stop_count = sum(stops[1:-1])
        stop_group = "0" if stop_count == 0 else "1-3" if stop_count <= 3 else "4-6" if stop_count <= 6 else "7+"
        pattern = tuple(stations[j] for j in range(1, len(stations) - 1) if stops[j])
        od = (stations[0], stations[-1]); segments = tuple(zip(stations, stations[1:]))
        cats = {
            "train_type": Counter({kind: 1}), "direction": Counter({direction: 1}),
            "od": Counter({od: 1}), "path": Counter({stations: 1}),
            "od_type": Counter({(od, kind): 1}), "stop_pattern": Counter({(od, pattern): 1}),
            "od_type_pattern": Counter({(od, kind, pattern): 1}),
            "stop_count": Counter({stop_count: 1}), "stop_group": Counter({stop_group: 1}),
            "segment_count": Counter({len(segments): 1}),
            "station_visit": Counter(stations), "station_type": Counter((s, kind) for s in stations),
            "station_origin": Counter({stations[0]: 1}), "station_destination": Counter({stations[-1]: 1}),
            "station_stop": Counter(stations[j] for j in range(1, len(stations) - 1) if stops[j]),
            "station_pass": Counter(stations[j] for j in range(1, len(stations) - 1) if not stops[j]),
            "segment": Counter(segments),
            "segment_type_state": Counter((seg, kind, bool(stops[j]), bool(stops[j + 1])) for j, seg in enumerate(segments)),
        }
        samples = defaultdict(lambda: defaultdict(list))
        travel = arrivals[-1] - departures[0]
        samples["travel"]["all"].append(travel)
        samples["stop_count"]["all"].append(stop_count)
        train_station, train_segment = [], []
        for j, s in enumerate(stations):
            role = "origin" if j == 0 else "destination" if j == len(stations) - 1 else "intermediate"
            dwell = departures[j] - arrivals[j] if role == "intermediate" else None
            min_field = self.station_map[s].get("dwell_min", "")
            active_min = (float(min_field) if min_field != "" else None) if stops[j] else 0
            max_field = self.station_map[s].get("dwell_max", "")
            train_station.append({"source_id": tid, "station": s, "position": j, "role": role,
                                  "type": kind, "stop": stops[j], "arrival": number(arrivals[j]), "departure": number(departures[j]),
                                  "dwell": number(dwell), "dwell_min": number(active_min),
                                  "dwell_max_field": max_field, "dwell_max_enabled": cfg.bool("model.dwell_upper_enabled"),
                                  "dwell_slack": number(dwell - active_min) if dwell is not None and stops[j] and active_min is not None else "",
                                  "event_source": "input_csv", "underlying_provenance": "not_supplied"})
            if role == "intermediate" and stops[j]:
                samples["dwell"]["all"].append(dwell)
                samples["dwell_station_type"][(s, kind)].append(dwell)
                if active_min is not None: samples["dwell_slack"]["all"].append(dwell - active_min)
        for j, seg in enumerate(segments):
            if seg not in self.segment_map: raise ValueError(tid + ": directed segment not in segments.csv: " + str(seg))
            raw = arrivals[j + 1] - departures[j]
            p = self.segment_map[seg]
            tau, e1, e2 = (p.get(prefix + kind, "") for prefix in ("tau_", "eta_start_", "eta_end_"))
            lower_known = tau != "" and (not stops[j] or e1 != "") and (not stops[j + 1] or e2 != "")
            lower = (float(tau) + int(stops[j]) * float(e1 or 0) + int(stops[j + 1]) * float(e2 or 0)) if lower_known else None
            state = (seg, kind, bool(stops[j]), bool(stops[j + 1]))
            samples["run"]["all"].append(raw)
            samples["run_segment_type_state"][state].append(raw)
            if lower is not None: samples["run_slack"]["all"].append(raw - lower)
            train_segment.append({"source_id": tid, "from": seg[0], "to": seg[1], "type": kind,
                                  "position": j, "upstream_stop": stops[j], "downstream_stop": stops[j + 1],
                                  "departure": number(departures[j]), "arrival": number(arrivals[j + 1]), "run_time": raw,
                                  "run_min": number(lower), "run_slack": number(raw - lower) if lower is not None else "",
                                  "run_max_field": p.get("run_max_" + kind, ""),
                                  "run_max_enabled": cfg.bool("model.run_upper_enabled")})
        total_dwell = sum(samples["dwell"]["all"])
        samples["dwell_total"]["all"].append(total_dwell)
        samples["run_total"]["all"].append(sum(samples["run"]["all"]))
        samples["dwell_share"]["all"].append(total_dwell / travel if travel else 0)
        return Train(index, tid, kind, direction, rows, stations, tuple(arrivals), tuple(departures), tuple(stops),
                     arrivals[0] if arrivals[0] is not None else departures[0], min(events), max(events), travel,
                     stop_count, stop_group, od, pattern, segments, cats, {f: dict(g) for f, g in samples.items()},
                     train_station, train_segment)

    def period(self, value):
        import bisect
        return bisect.bisect_right(self.period_edges, value)

    def time_keys(self, t, entry):
        b = math.floor(entry / self.bin_minutes)
        return {"entry_bin": b, "od_entry_bin": (t.od, b), "type_entry_bin": (t.kind, b), "entry_period": self.period(entry)}

    def time_counts(self, instances):
        out = defaultdict(Counter)
        for index, shift in instances:
            t = self.trains[index]
            for f, key in self.time_keys(t, t.entry + shift).items(): out[f][key] += 1
        return out

    def counts_for(self, multiplicities):
        out = defaultdict(Counter)
        for t, n in zip(self.trains, multiplicities):
            if n:
                for f, counter in t.categories.items():
                    for k, v in counter.items(): out[f][k] += n * v
        return out

    def output_rows(self, instances, generated_ids):
        result = []
        for (index, shift), gid in zip(instances, generated_ids):
            t = self.trains[index]
            for j, original in enumerate(t.rows):
                row = dict(original); row[self.cols["id"]] = gid; row[self.cols["type"]] = t.kind
                for col, v in ((self.cols["arrival"], t.arrivals[j]), (self.cols["departure"], t.departures[j])):
                    row[col] = format_time(v + shift if v is not None else None, self.fmt)
                for col in self.extra_times:
                    v = parse_time(original.get(col, ""), self.fmt)
                    row[col] = format_time(v + shift if v is not None else None, self.fmt)
                result.append(row)
        return result
