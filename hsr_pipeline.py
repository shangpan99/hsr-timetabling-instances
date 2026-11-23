# -*- coding: utf-8 -*-
"""可配置的 HSR 主实验核心：Direct、受限实例、Packing-LP、结构化学习与 Full-Fix。

每个案例由 case_plan.csv 选取；统一算法参数从 config.csv 读取。
程序自动创建独立案例目录，并输出：
- 模型规模与 Big-M 行数；
- 求解状态、目标值、界、MIP gap 与运行时间；
- 独立时刻表可行性校验；
- Packing-LP 中每个受限实例采用的有效数值来源；
- 五组 alpha 的 raw / weighted mismatch；
- 证据覆盖诊断和表6--表12可复用的单案例 CSV 行。

基础时刻表模型保持原脚本口径：
- 目标列车按 corridor-entry time 排序；
- 首站只建 departure，末站只建 arrival；
- 区间运行时间为下界，中间停站无上界，通过站 arrival=departure；
- 350/300 等级的始发窗口默认分别为 ±45/±60 分钟；
- C1=triangle branch-and-cut，C2=segment consistency，C3=stop-dependent overtaking。
"""
from __future__ import annotations

TEMPORAL_CUT_PATCH_VERSION = "2026-09-15_NO4B_STRUCTURE_ABLATION_READY"

import atexit
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import platform
import random
import shutil
import sys
import time
import traceback
from array import array
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from gurobipy import Env, GRB, LinExpr, Model, quicksum

from config_loader import (
    as_bool,
    as_float,
    as_int,
    parse_float_list,
    parse_int_list,
    parse_list,
    project_root,
    resolve_project_path,
)

PairKey = Tuple[str, str, str, str]  # kind, station, canonical_i, canonical_j


# =============================================================================
# 通用工具
# =============================================================================

STATUS_NAMES = {
    1: "LOADED",
    2: "OPTIMAL",
    3: "INFEASIBLE",
    4: "INF_OR_UNBD",
    5: "UNBOUNDED",
    6: "CUTOFF",
    7: "ITERATION_LIMIT",
    8: "NODE_LIMIT",
    9: "TIME_LIMIT",
    10: "SOLUTION_LIMIT",
    11: "INTERRUPTED",
    12: "NUMERIC",
    13: "SUBOPTIMAL",
    14: "INPROGRESS",
    15: "USER_OBJ_LIMIT",
    16: "WORK_LIMIT",
    17: "MEM_LIMIT",
}


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def status_name(code: Any) -> str:
    try:
        return STATUS_NAMES.get(int(code), f"STATUS_{int(code)}")
    except Exception:
        return str(code)


def safe_float(getter, default: Any = None) -> Any:
    try:
        return float(getter())
    except Exception:
        return default


def safe_int(getter, default: Any = None) -> Any:
    try:
        return int(getter())
    except Exception:
        return default


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def clean_dir(path: Path) -> None:
    if path.exists():
        resolved = path.resolve()
        forbidden = {Path("/").resolve(), Path.cwd().resolve(), Path.home().resolve()}
        if resolved in forbidden:
            raise RuntimeError(f"拒绝清空危险目录：{resolved}")
        shutil.rmtree(resolved)
    path.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _set_gurobi_log_file(model: Model, path: Path) -> None:
    """Open a native Gurobi log even when its Windows directory is non-ASCII."""
    log_path = Path(path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt" or str(log_path).isascii():
        model.setParam("LogFile", str(log_path))
        return

    # Python can enter Unicode directories that Gurobi's native file API cannot
    # open by absolute name. LogFile opens the file immediately and keeps its
    # handle, so only this synchronous call needs the temporary working directory.
    # Parallel restricted solves run in separate processes, each with its own cwd.
    previous_cwd = Path.cwd()
    try:
        os.chdir(log_path.parent)
        model.setParam("LogFile", log_path.name)
    finally:
        os.chdir(previous_cwd)


def parse_stop(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "是"}


def infer_train_type(train_id: str) -> int:
    try:
        digits = "".join(ch for ch in str(train_id) if ch.isdigit())
        number = int(digits)
        return 350 if 1 <= number <= 50 else 300
    except Exception:
        return 300


def train_id_sort_key(train_id: str) -> Tuple[str, int, str]:
    text = str(train_id)
    digits = "".join(ch for ch in text if ch.isdigit())
    return (text[:1], int(digits) if digits else 0, text)


def alpha_tag(alpha: float) -> str:
    text = f"{float(alpha):g}".replace("-", "m").replace(".", "p")
    return f"alpha_{text}"


def normalize_train_selection_rule(value: Any) -> str:
    raw = str(value or "entry_time").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "entry_time": "entry_time",
        "corridor_entry_time": "entry_time",
        "arrival_else_departure": "entry_time",
        "file_order": "file_order",
        "first_departure": "first_departure",
        "train_id": "train_id",
    }
    if raw not in aliases:
        raise ValueError(
            "train_selection_rule 仅支持 entry_time / file_order / first_departure / train_id，"
            f"当前值为 {value!r}"
        )
    return aliases[raw]


def normalize_pair_resolution_rule(value: Any, field_name: str) -> str:
    raw = str(value or "first_departure").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "first_departure": "first_departure",
        "file_order": "file_order",
        "train_id": "train_id",
        "entry_time": "entry_time",
    }
    if raw not in aliases:
        raise ValueError(
            f"{field_name} 仅支持 first_departure / entry_time / file_order / train_id，当前值为 {value!r}"
        )
    return aliases[raw]


SAMPLING_METHODS = ("local", "stratified", "coverage")


def configured_method_sample_sizes(cfg: Mapping[str, Any]) -> Dict[str, int]:
    """Return the restricted-instance size used by each sampling method.

    Backward compatibility: when no ``sampling_*_size`` field is supplied, all
    three methods use the legacy global ``sample_size``.
    """
    legacy = int(as_int(cfg.get("sample_size"), 0) or 0)
    explicit = any(cfg.get(f"sampling_{m}_size") is not None for m in SAMPLING_METHODS)
    if not explicit:
        if legacy <= 0:
            raise ValueError("sample_size 必须为正整数")
        return {m: legacy for m in SAMPLING_METHODS}
    result: Dict[str, int] = {}
    for method in SAMPLING_METHODS:
        raw = cfg.get(f"sampling_{method}_size")
        value = int(as_int(raw, legacy) or 0)
        if value <= 0:
            raise ValueError(f"sampling_{method}_size 必须为正整数")
        result[method] = value
    return result


def configured_sample_design(cfg: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
    """Return a backward-compatible representative size and total sample count.

    For heterogeneous sampling the representative size is the largest method
    size.  The exact per-method design is stored separately in the context and
    in every Stage-2 manifest row.
    """
    method_sizes = configured_method_sample_sizes(cfg)
    representative_size = max(method_sizes.values())
    if cfg.get("sample_count") is not None:
        total_count = int(as_int(cfg.get("sample_count"), 0) or 0)
    else:
        explicit_counts = [
            int(as_int(cfg.get(f"sampling_{m}_count"), 0) or 0)
            for m in SAMPLING_METHODS
        ]
        total_count = sum(explicit_counts)
        if total_count <= 0:
            legacy = parse_int_list(cfg.get("num_runs_list"))
            total_count = int(legacy[0]) if legacy else 0
    return [representative_size], [total_count]


def configured_sampling_design_label(cfg: Mapping[str, Any], total_count: Optional[int] = None) -> str:
    sizes = configured_method_sample_sizes(cfg)
    if total_count is None:
        _, counts = configured_sample_design(cfg)
        total_count = int(counts[0])
    targets = _method_targets(cfg, int(total_count))
    return (
        f"L{sizes['local']}x{targets['local']}_"
        f"S{sizes['stratified']}x{targets['stratified']}_"
        f"C{sizes['coverage']}x{targets['coverage']}"
    )


def configured_k_sub_design_text(cfg: Mapping[str, Any]) -> str:
    sizes = configured_method_sample_sizes(cfg)
    return f"L{sizes['local']}/S{sizes['stratified']}/C{sizes['coverage']}"


def configured_main_alpha(cfg: Mapping[str, Any]) -> float:
    return float(as_float(cfg.get("main_alpha"), 1.0) or 0.0)


def case_tag_from_config(cfg: Mapping[str, Any], k: int, sample_size: int, sample_count: int) -> str:
    raw = str(cfg.get("instance_tag") or "AUTO").strip()
    if not raw or raw.upper() == "AUTO":
        method_sizes = configured_method_sample_sizes(cfg)
        if len(set(method_sizes.values())) > 1 or any(
            cfg.get(f"sampling_{m}_size") is not None for m in SAMPLING_METHODS
        ):
            base = f"K{k}_{configured_sampling_design_label(cfg, sample_count)}"
        else:
            base = f"K{k}_Ksub{sample_size}_R{sample_count}"
        return base
    if raw in {".", ".."} or Path(raw).name != raw or "/" in raw or "\\" in raw:
        raise ValueError("instance_tag 必须是单个安全目录名，不能包含路径分隔符")
    return raw


def read_first_row(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty:
        return {}
    return {k: (None if pd.isna(v) else v) for k, v in df.iloc[0].to_dict().items()}


def relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root().resolve()).as_posix()
    except Exception:
        return path.as_posix()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return SHA-256 for a local file without loading it fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_stage_timing(ctx: "ExperimentContext", row: Mapping[str, Any]) -> None:
    path = ensure_dir(ctx.timing_dir) / "stage_timing_log.csv"
    record = {"recorded_at": now_text(), **dict(row)}
    if path.exists():
        old = pd.read_csv(path, encoding="utf-8-sig")
        columns = list(old.columns)
        for column in record:
            if column not in columns:
                columns.append(column)
        old = old.reindex(columns=columns).astype(object)
        old.loc[len(old)] = [record.get(column) for column in columns]
        out = old
    else:
        out = pd.DataFrame([record])
    write_csv(out, path)


def solver_summary(model: Model, has_solution: bool) -> Dict[str, Any]:
    num_vars = safe_int(lambda: model.NumVars, 0) or 0
    num_bin = safe_int(lambda: model.NumBinVars, 0) or 0
    return {
        "status_code": int(model.Status),
        "status_name": status_name(model.Status),
        "sol_count": safe_int(lambda: model.SolCount, 0),
        "is_optimal": int(int(model.Status) == int(GRB.OPTIMAL)),
        "has_solution": int(bool(has_solution)),
        "obj_val": safe_float(lambda: model.ObjVal) if has_solution else None,
        "obj_bound": safe_float(lambda: model.ObjBound),
        "mip_gap": safe_float(lambda: model.MIPGap) if has_solution else None,
        "mip_gap_percent": (100.0 * safe_float(lambda: model.MIPGap, 0.0)) if has_solution else None,
        "runtime_sec": safe_float(lambda: model.Runtime),
        "node_count": safe_float(lambda: model.NodeCount),
        "simplex_iterations": safe_float(lambda: model.IterCount),
        "barrier_iterations": safe_float(lambda: model.BarIterCount),
        "work_units": safe_float(lambda: model.Work),
        "num_vars": num_vars,
        "num_constrs": safe_int(lambda: model.NumConstrs),
        "num_bin_vars": num_bin,
        "num_cont_vars": num_vars - num_bin,
    }


def has_usable_solution(model: Model, keep_time_limit: bool = True) -> bool:
    status = int(model.Status)
    sol_count = safe_int(lambda: model.SolCount, 0) or 0
    allowed = {int(GRB.OPTIMAL)}
    if keep_time_limit:
        allowed.add(int(GRB.TIME_LIMIT))
    return status in allowed and sol_count > 0


def optimize_with_status_resolution(
    model: Model, cfg: Mapping[str, Any], *, stage: str, callback: Any = None
) -> Dict[str, Any]:
    """Optimize and optionally disambiguate Gurobi's INF_OR_UNBD status.

    ``callback`` is optional and is currently used by Stage 5 to separate C1
    cycle cuts lazily.  All other stages keep the previous behaviour.
    """
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    if callback is None:
        model.optimize()
    else:
        model.optimize(callback)
    runtime_total = float(safe_float(lambda: model.Runtime, 0.0) or 0.0)
    initial_status_code = int(model.Status)
    initial_status_name = status_name(model.Status)
    rerun = 0

    enabled = as_bool(cfg.get("disambiguate_inf_or_unbd"), True)
    if stage == "sample":
        enabled = as_bool(cfg.get("disambiguate_sample_inf_or_unbd"), False)
    if enabled and int(model.Status) == int(GRB.INF_OR_UNBD):
        rerun = 1
        model.setParam("DualReductions", 0)
        model.reset()
        if callback is None:
            model.optimize()
        else:
            model.optimize(callback)
        runtime_total += float(safe_float(lambda: model.Runtime, 0.0) or 0.0)

    return {
        "initial_status_code": initial_status_code,
        "initial_status_name": initial_status_name,
        "status_disambiguation_rerun": rerun,
        "solver_runtime_accumulated_sec": runtime_total,
        "optimize_wall_sec": time.perf_counter() - wall_start,
        "optimize_process_cpu_sec": time.process_time() - cpu_start,
    }


# =============================================================================
# 数据与实验上下文
# =============================================================================

@dataclass(frozen=True)
class ExperimentContext:
    experiment_group: str
    experiment_tag: str
    target_k: int
    output_root: Path
    output_dir: Path
    sample_sizes: Tuple[int, ...]
    sample_counts: Tuple[int, ...]
    use_c1: bool = True
    use_c2: bool = True
    use_c3: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def direct_dir(self) -> Path:
        return self.output_dir / "01_direct"

    @property
    def sample_dir(self) -> Path:
        return self.output_dir / "02_samples"

    @property
    def vote_dir(self) -> Path:
        return self.output_dir / "03_vote"

    @property
    def packing_dir(self) -> Path:
        return self.output_dir / "04_packing"

    @property
    def inverse_dir(self) -> Path:
        return self.output_dir / "05_inverse"

    @property
    def fullfix_dir(self) -> Path:
        return self.output_dir / "06_fullfix"

    @property
    def aggregate_dir(self) -> Path:
        return self.output_dir / "90_aggregate"

    @property
    def timing_dir(self) -> Path:
        return self.output_dir / "00_timing"

    @property
    def parameter_dir(self) -> Path:
        return self.output_dir / "00_parameters"

    @property
    def sample_manifest(self) -> Path:
        return self.sample_dir / "sample_manifest.csv"


class HSRData:
    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg = cfg
        self.data_dir = resolve_project_path(cfg.get("data_dir"), ".")
        trains_path = self.data_dir / "trains.csv"
        stations_path = self.data_dir / "stations.csv"
        segments_path = self.data_dir / "segments.csv"
        for path in (trains_path, stations_path, segments_path):
            if not path.exists():
                raise FileNotFoundError(f"缺少输入文件：{path}")

        self.trains_df_all = pd.read_csv(trains_path, encoding="utf-8-sig")
        self.stations_df = pd.read_csv(stations_path, encoding="utf-8-sig")
        self.segments_df = pd.read_csv(segments_path, encoding="utf-8-sig")

        required_train_cols = {"车次", "车站", "到达时间", "出发时间", "是否停站", "站序"}
        missing = required_train_cols - set(self.trains_df_all.columns)
        if missing:
            raise ValueError(f"trains.csv 缺少列：{sorted(missing)}")

        self.trains_df_all["车次"] = self.trains_df_all["车次"].astype(str)
        self.trains_df_all["车站"] = self.trains_df_all["车站"].astype(str)
        self.stations_df["station"] = self.stations_df["station"].astype(str)
        self.segments_df["from"] = self.segments_df["from"].astype(str)
        self.segments_df["to"] = self.segments_df["to"].astype(str)

        self.stations: List[str] = self.stations_df["station"].tolist()
        self.station_index: Dict[str, int] = {s: idx for idx, s in enumerate(self.stations)}
        self.segments: List[Tuple[str, str]] = list(zip(self.segments_df["from"], self.segments_df["to"]))

        self.all_trains: List[str] = list(pd.unique(self.trains_df_all["车次"]))
        self.global_train_index: Dict[str, int] = {k: idx for idx, k in enumerate(self.all_trains)}
        self.train_selection_rule = normalize_train_selection_rule(cfg.get("train_selection_rule", "entry_time"))

        self.station_params: Dict[str, Dict[str, float]] = {}
        for _, row in self.stations_df.iterrows():
            self.station_params[str(row["station"])] = {
                "h_arr": float(row.get("h_arr", 0) or 0),
                "h_dep": float(row.get("h_dep", 0) or 0),
                "dwell_min": float(row.get("dwell_min", 0) or 0),
            }

        self.segment_params: Dict[Tuple[str, str], Dict[str, float]] = {}
        for _, row in self.segments_df.iterrows():
            self.segment_params[(str(row["from"]), str(row["to"]))] = {
                "tau_350": float(row.get("tau_350", 0) or 0),
                "tau_300": float(row.get("tau_300", 0) or 0),
                "eta_start_350": float(row.get("eta_start_350", 0) or 0),
                "eta_end_350": float(row.get("eta_end_350", 0) or 0),
                "eta_start_300": float(row.get("eta_start_300", 0) or 0),
                "eta_end_300": float(row.get("eta_end_300", 0) or 0),
            }

        self.rows_by_train: Dict[str, pd.DataFrame] = {}
        self.path_by_train: Dict[str, List[str]] = {}
        self.stop_at: Dict[Tuple[str, str], bool] = {}
        self.first_arrival: Dict[str, Optional[float]] = {}
        self.first_dep: Dict[str, float] = {}
        self.entry_time: Dict[str, float] = {}
        self.entry_time_source: Dict[str, str] = {}
        self.intermediate_stop_count: Dict[str, int] = {}
        self.train_type: Dict[str, int] = {}
        self.direction: Dict[str, str] = {}

        has_type_col = "train_type" in self.trains_df_all.columns
        has_direction_col = "方向" in self.trains_df_all.columns
        for train in self.all_trains:
            sub = self.trains_df_all[self.trains_df_all["车次"] == train].sort_values("站序").copy()
            if sub.empty:
                raise ValueError(f"列车 {train} 没有站点记录")
            self.rows_by_train[train] = sub
            self.path_by_train[train] = sub["车站"].astype(str).tolist()
            for _, row in sub.iterrows():
                self.stop_at[(train, str(row["车站"]))] = parse_stop(row["是否停站"])

            first = sub.iloc[0]
            if pd.isna(first["出发时间"]):
                raise ValueError(f"列车 {train} 的首站没有出发时间")
            self.first_dep[train] = float(first["出发时间"])
            if pd.notna(first["到达时间"]):
                self.first_arrival[train] = float(first["到达时间"])
                self.entry_time[train] = float(first["到达时间"])
                self.entry_time_source[train] = "first_arrival"
            else:
                self.first_arrival[train] = None
                self.entry_time[train] = self.first_dep[train]
                self.entry_time_source[train] = "first_departure"

            self.intermediate_stop_count[train] = sum(
                int(self.stop_at.get((train, station), False))
                for station in self.path_by_train[train][1:-1]
            )
            if has_type_col and pd.notna(first.get("train_type")):
                self.train_type[train] = int(float(first["train_type"]))
            else:
                self.train_type[train] = infer_train_type(train)
            self.direction[train] = str(first["方向"]) if has_direction_col else ""

    def ordered_trains(self, rule: Optional[str] = None) -> List[str]:
        selected_rule = normalize_train_selection_rule(rule or self.train_selection_rule)
        if selected_rule == "entry_time":
            return sorted(self.all_trains, key=lambda k: (self.entry_time[k], self.global_train_index[k]))
        if selected_rule == "file_order":
            return list(self.all_trains)
        if selected_rule == "first_departure":
            return sorted(self.all_trains, key=lambda k: (self.first_dep[k], self.global_train_index[k]))
        if selected_rule == "train_id":
            return sorted(self.all_trains, key=lambda k: (train_id_sort_key(k), self.global_train_index[k]))
        raise AssertionError(f"未处理的 train selection rule: {selected_rule}")

    def select_trains(self, target_k: int) -> List[str]:
        ordered = self.ordered_trains()
        if target_k <= 0 or target_k > len(ordered):
            raise ValueError(f"train_limit={target_k} 必须位于 1..{len(ordered)}")
        return ordered[:target_k]

    def canonical_pair(self, i: str, j: str) -> Tuple[str, str]:
        if i == j:
            raise ValueError("同一列车不能组成 precedence pair")
        return (i, j) if self.global_train_index[i] < self.global_train_index[j] else (j, i)

    def has_event(self, train: str, station: str, kind: str) -> bool:
        seq = self.path_by_train.get(train, [])
        if not seq or station not in seq:
            return False
        if kind == "arr":
            return station != seq[0]
        if kind == "dep":
            return station != seq[-1]
        raise ValueError(f"未知事件类型：{kind}")

    def event_universe(self, trains: Sequence[str]) -> List[PairKey]:
        universe: List[PairKey] = []
        for station in self.stations:
            for kind in ("dep", "arr"):
                present = [k for k in trains if self.has_event(k, station, kind)]
                for p in range(len(present)):
                    for q in range(p + 1, len(present)):
                        i, j = self.canonical_pair(present[p], present[q])
                        universe.append((kind, station, i, j))
        return universe

    def required_run_time(self, train: str, upstream: str, downstream: str) -> float:
        params = self.segment_params.get((upstream, downstream))
        if params is None:
            raise KeyError(f"segments.csv 缺少区间：{upstream}->{downstream}")
        suffix = "350" if self.train_type.get(train, 300) == 350 else "300"
        return (
            params[f"tau_{suffix}"]
            + (params[f"eta_start_{suffix}"] if self.stop_at.get((train, upstream), False) else 0.0)
            + (params[f"eta_end_{suffix}"] if self.stop_at.get((train, downstream), False) else 0.0)
        )


# =============================================================================
# 模型构造
# =============================================================================

@dataclass
class TimetableModelArtifacts:
    model: Model
    trains: List[str]
    a: Dict[Tuple[str, str], Any]
    d: Dict[Tuple[str, str], Any]
    z: Dict[PairKey, Any]
    fixed_precedence: Dict[PairKey, int]
    universe: List[PairKey]
    model_stats: Dict[str, Any]


def configure_model(model: Model, cfg: Mapping[str, Any], stage: str) -> None:
    time_key = {
        "direct": "direct_time_limit",
        "sample": "sample_time_limit",
        "packing": "packing_time_limit",
        "inverse": "inverse_time_limit",
        "forward": "forward_time_limit",
    }.get(stage, "forward_time_limit")
    gap_key = {
        "direct": "direct_mipgap",
        "sample": "sample_mipgap",
        "inverse": "inverse_mipgap",
    }.get(stage)

    time_limit = as_float(cfg.get(time_key), as_float(cfg.get("time_limit"), None))
    # A value of 0 or a blank cell means: do not set TimeLimit, so Gurobi uses
    # its default infinity.  This is used for the restricted-instance pool.
    if time_limit is not None and time_limit > 0:
        model.setParam("TimeLimit", float(time_limit))
    if gap_key:
        gap_value = as_float(cfg.get(gap_key), as_float(cfg.get("mipgap"), 0.0))
        model.setParam("MIPGap", float(gap_value or 0.0))

    if stage == "sample":
        threads = as_int(
            cfg.get("sample_threads_per_solve"),
            as_int(cfg.get("threads"), None),
        )
    elif stage == "direct":
        # Direct can use its own thread budget.
        threads = as_int(
            cfg.get("direct_threads"),
            as_int(cfg.get("threads"), None),
        )
    elif stage == "inverse":
        threads = as_int(
            cfg.get("inverse_threads"),
            as_int(cfg.get("threads"), None),
        )
    else:
        threads = as_int(cfg.get("threads"), None)
    if threads is not None and threads > 0:
        model.setParam("Threads", int(threads))

    # NoRelHeurTime/NoRelHeurWork are deliberately NOT used in this version.
    # MIPFocus remains a normal optional inverse setting; omit the key from
    # config.csv to use Gurobi's default value.
    if stage == "inverse":
        mipfocus = as_int(cfg.get("inverse_mipfocus"), None)
        if mipfocus is not None:
            if int(mipfocus) not in (0, 1, 2, 3):
                raise ValueError("inverse_mipfocus 必须为 0/1/2/3")
            model.setParam("MIPFocus", int(mipfocus))

    if stage == "sample":
        console_log = as_bool(cfg.get("sample_console_log"), False)
        file_log = as_bool(cfg.get("sample_write_solver_log"), True)
        model.setParam("OutputFlag", 1 if (console_log or file_log) else 0)
        if console_log or file_log:
            model.setParam("LogToConsole", 1 if console_log else 0)
    elif stage == "direct":
        model.setParam("OutputFlag", 1)
        model.setParam("LogToConsole", 1 if as_bool(cfg.get("direct_console_log"), True) else 0)
    elif stage == "inverse":
        # Each alpha solve may write its own persistent Gurobi log.  OutputFlag
        # must remain enabled when either console or file logging is requested.
        console_log = as_bool(cfg.get("inverse_console_log"), True)
        file_log = as_bool(cfg.get("inverse_write_solver_log"), True)
        model.setParam("OutputFlag", 1 if (console_log or file_log) else 0)
        if console_log or file_log:
            model.setParam("LogToConsole", 1 if console_log else 0)
    else:
        model.setParam("OutputFlag", 1)


def build_timetable_model(
    data: HSRData,
    cfg: Mapping[str, Any],
    trains: Sequence[str],
    *,
    model_name: str,
    stage: str,
    fixed_precedence: Optional[Mapping[PairKey, int]] = None,
    env: Optional[Any] = None,
    log_file: Optional[Path] = None,
) -> TimetableModelArtifacts:
    """Build the presolve-independent base MILP or full-fixing LP.

    The returned ``model_stats`` are counted directly while constructing the
    formulation.  They therefore describe the mathematical model before
    Gurobi presolve and can be used directly in Table 6.
    """
    trains = list(trains)
    fixed = {k: int(v) for k, v in (fixed_precedence or {}).items()}
    universe = data.event_universe(trains)
    universe_set = set(universe)
    unknown_fixed = set(fixed) - universe_set
    if unknown_fixed:
        raise ValueError(f"fixed precedence 含事件全集外记录，例如：{list(unknown_fixed)[:5]}")

    model = Model(model_name, env=env) if env is not None else Model(model_name)
    if log_file is not None:
        log_path = Path(log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if log_path.exists():
                log_path.unlink()
        except Exception:
            pass
        # Set LogFile before the remaining solver parameters so the complete
        # optimization log (including parameter changes and MIP progress) is
        # persisted even if the solve is interrupted or runs out of memory.
        _set_gurobi_log_file(model, log_path)
    configure_model(model, cfg, stage)

    a: Dict[Tuple[str, str], Any] = {}
    d: Dict[Tuple[str, str], Any] = {}
    for train in trains:
        seq = data.path_by_train[train]
        if len(seq) < 2:
            raise ValueError(f"列车 {train} 路径少于2站")
        for idx, station in enumerate(seq):
            if idx > 0:
                a[(train, station)] = model.addVar(
                    lb=0.0, vtype=GRB.CONTINUOUS, name=f"a_{train}_{station}"
                )
            if idx < len(seq) - 1:
                d[(train, station)] = model.addVar(
                    lb=0.0, vtype=GRB.CONTINUOUS, name=f"d_{train}_{station}"
                )

    z: Dict[PairKey, Any] = {}
    if fixed_precedence is None:
        for kind, station, i, j in universe:
            z[(kind, station, i, j)] = model.addVar(
                vtype=GRB.BINARY, name=f"z_{kind}_{i}_{j}_{station}"
            )
    model.update()

    model.setObjective(
        quicksum(
            a[(k, data.path_by_train[k][-1])] - d[(k, data.path_by_train[k][0])]
            for k in trains
        ),
        GRB.MINIMIZE,
    )

    origin_window_rows = 0
    run_time_rows = 0
    dwell_time_rows = 0
    pass_through_rows = 0
    big_m_rows = 0
    fixed_headway_rows = 0
    segment_consistency_rows = 0
    fixed_segment_conflict_rows = 0

    allow_early = as_bool(cfg.get("allow_early_departure"), True)
    fast_window = float(as_float(cfg.get("dep_window_fast"), 45.0))
    normal_window = float(as_float(cfg.get("dep_window_normal"), 60.0))
    fast_type = int(as_int(cfg.get("fast_train_type"), 350))
    if fast_window < 0 or normal_window < 0:
        raise ValueError("出发时间窗必须非负")
    for train in trains:
        first = data.path_by_train[train][0]
        expected = data.first_dep[train]
        window = fast_window if data.train_type.get(train, 300) == fast_type else normal_window
        lower = expected - window if allow_early else expected
        upper = expected + window
        model.addConstr(d[(train, first)] >= lower, name=f"origin_lb_{train}")
        model.addConstr(d[(train, first)] <= upper, name=f"origin_ub_{train}")
        origin_window_rows += 2

    for train in trains:
        seq = data.path_by_train[train]
        for upstream, downstream in zip(seq, seq[1:]):
            model.addConstr(
                a[(train, downstream)]
                >= d[(train, upstream)] + data.required_run_time(train, upstream, downstream),
                name=f"run_min_{train}_{upstream}_{downstream}",
            )
            run_time_rows += 1
        for station in seq[1:-1]:
            if data.stop_at.get((train, station), False):
                dwell_min = data.station_params.get(station, {}).get("dwell_min", 0.0)
                model.addConstr(
                    d[(train, station)] - a[(train, station)] >= dwell_min,
                    name=f"dwell_min_{train}_{station}",
                )
                dwell_time_rows += 1
            else:
                model.addConstr(
                    d[(train, station)] == a[(train, station)],
                    name=f"pass_{train}_{station}",
                )
                pass_through_rows += 1

    big_m = float(as_float(cfg.get("big_m"), 10000.0))
    if big_m <= 0:
        raise ValueError("big_m 必须为正数")
    for key in universe:
        kind, station, i, j = key
        h = data.station_params.get(station, {}).get(
            "h_dep" if kind == "dep" else "h_arr", 0.0
        )
        ti = d[(i, station)] if kind == "dep" else a[(i, station)]
        tj = d[(j, station)] if kind == "dep" else a[(j, station)]
        if fixed_precedence is not None:
            if key not in fixed:
                raise ValueError(f"Full-Fix 缺少 precedence：{key}")
            if fixed[key] == 1:
                model.addConstr(tj >= ti + h, name=f"fixed_{kind}_{i}_{j}_{station}")
            else:
                model.addConstr(ti >= tj + h, name=f"fixed_{kind}_{j}_{i}_{station}")
            fixed_headway_rows += 1
        else:
            var = z[key]
            model.addConstr(
                tj >= ti + h - big_m * (1.0 - var),
                name=f"hw1_{kind}_{i}_{j}_{station}",
            )
            model.addConstr(
                ti >= tj + h - big_m * var,
                name=f"hw0_{kind}_{i}_{j}_{station}",
            )
            big_m_rows += 2

    segment_mismatch_count = 0
    dummy = None
    for upstream, downstream in data.segments:
        present = [
            k
            for k in trains
            if data.has_event(k, upstream, "dep")
            and data.has_event(k, downstream, "arr")
            and upstream in data.path_by_train[k]
            and downstream in data.path_by_train[k]
            and data.path_by_train[k].index(downstream)
            == data.path_by_train[k].index(upstream) + 1
        ]
        for p in range(len(present)):
            for q in range(p + 1, len(present)):
                i, j = data.canonical_pair(present[p], present[q])
                dep_key = ("dep", upstream, i, j)
                arr_key = ("arr", downstream, i, j)
                if dep_key not in universe_set or arr_key not in universe_set:
                    continue
                if fixed_precedence is None:
                    model.addConstr(
                        z[dep_key] == z[arr_key],
                        name=f"segment_order_{i}_{j}_{upstream}_{downstream}",
                    )
                    segment_consistency_rows += 1
                elif fixed[dep_key] != fixed[arr_key]:
                    segment_mismatch_count += 1
                    if dummy is None:
                        dummy = model.addVar(
                            lb=0.0, ub=0.0, name="segment_consistency_dummy"
                        )
                        model.update()
                    model.addConstr(
                        dummy >= 1.0,
                        name=f"fixed_segment_conflict_{i}_{j}_{upstream}_{downstream}",
                    )
                    fixed_segment_conflict_rows += 1

    model._segment_mismatch_count = segment_mismatch_count  # type: ignore[attr-defined]
    model.update()

    num_vars = int(model.NumVars)
    num_bin_vars = int(model.NumBinVars)
    num_constrs = int(model.NumConstrs)
    model_stats: Dict[str, Any] = {
        "model_formulation": "base_MILP" if fixed_precedence is None else "fullfix_LP",
        "num_event_time_vars": len(a) + len(d),
        "num_auxiliary_cont_vars": num_vars - num_bin_vars - len(a) - len(d),
        "num_cont_vars_constructed": num_vars - num_bin_vars,
        "num_binary_vars_constructed": num_bin_vars,
        "num_precedence_binary_vars": len(z),
        "num_constraints_constructed": num_constrs,
        "num_origin_window_rows": origin_window_rows,
        "num_running_time_rows": run_time_rows,
        "num_dwell_time_rows": dwell_time_rows,
        "num_pass_through_rows": pass_through_rows,
        "num_timing_backbone_rows": (
            origin_window_rows + run_time_rows + dwell_time_rows + pass_through_rows
        ),
        "num_precedence_indices": len(universe),
        "num_departure_precedence_indices": sum(1 for key in universe if key[0] == "dep"),
        "num_arrival_precedence_indices": sum(1 for key in universe if key[0] == "arr"),
        "num_big_m_rows": big_m_rows,
        "num_fixed_one_sided_headway_rows": fixed_headway_rows,
        "num_segment_consistency_rows": segment_consistency_rows,
        "num_fixed_segment_conflict_rows": fixed_segment_conflict_rows,
    }
    return TimetableModelArtifacts(model, trains, a, d, z, fixed, universe, model_stats)

def timetable_rows(data: HSRData, artifacts: TimetableModelArtifacts) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for train in artifacts.trains:
        seq = data.path_by_train[train]
        for idx, station in enumerate(seq):
            arr = None if idx == 0 else float(artifacts.a[(train, station)].X)
            dep = None if idx == len(seq) - 1 else float(artifacts.d[(train, station)].X)
            rows.append({
                "车次": train,
                "方向": data.direction.get(train, ""),
                "车站": station,
                "到达时间": None if arr is None else round(arr, 6),
                "出发时间": None if dep is None else round(dep, 6),
                "是否停站": bool(data.stop_at.get((train, station), False)),
                "站序": int(float(data.rows_by_train[train].iloc[idx]["站序"])),
                "train_type": data.train_type.get(train, 300),
            })
    return rows


def pairwise_rows(data: HSRData, artifacts: TimetableModelArtifacts, *, directed: bool = True) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key in artifacts.universe:
        kind, station, i, j = key
        value = int(round(artifacts.z[key].X)) if artifacts.z else int(artifacts.fixed_precedence[key])
        directions = [(i, j, value, 1)]
        if directed:
            directions.append((j, i, 1 - value, 0))
        for train1, train2, directed_value, canonical_flag in directions:
            rows.append({
                "station": station,
                "train_i": train1,
                "train_j": train2,
                "kind": kind,
                "value": directed_value,
                "canonical_train_i": i,
                "canonical_train_j": j,
                "is_canonical_direction": canonical_flag,
                "车站": station,
                "车次1": train1,
                "车次2": train2,
                "变量类型": kind,
                "取值": directed_value,
            })
    return rows


# =============================================================================
# precedence 审计
# =============================================================================

def audit_precedence_assignment(
    data: HSRData,
    trains: Sequence[str],
    priors: Mapping[PairKey, int],
    *,
    count_c1_triplets: bool = True,
) -> Dict[str, Any]:
    """Audit C1--C3; optionally skip cubic C1 triple enumeration."""
    universe = set(data.event_universe(trains))
    c1_cycle_layers = 0
    c1_checked_layers = 0
    c1_transitivity_violations = 0
    c1_checked_triples = 0

    for station in data.stations:
        for kind in ("dep", "arr"):
            present = [k for k in trains if data.has_event(k, station, kind)]
            if len(present) <= 1:
                continue
            c1_checked_layers += 1
            indeg = {k: 0 for k in present}
            adj = {k: [] for k in present}
            for p in range(len(present)):
                for q in range(p + 1, len(present)):
                    i, j = data.canonical_pair(present[p], present[q])
                    key = (kind, station, i, j)
                    if key not in priors:
                        continue
                    before, after = (i, j) if int(priors[key]) == 1 else (j, i)
                    adj[before].append(after)
                    indeg[after] += 1
            queue = [k for k, deg in indeg.items() if deg == 0]
            visited = 0
            while queue:
                node = queue.pop()
                visited += 1
                for nxt in adj[node]:
                    indeg[nxt] -= 1
                    if indeg[nxt] == 0:
                        queue.append(nxt)
            if visited != len(present):
                c1_cycle_layers += 1

            if count_c1_triplets:
                for i0, j0, q0 in combinations(present, 3):
                    ordered = sorted(
                        (i0, j0, q0), key=lambda k: data.global_train_index[k]
                    )
                    i, j, q = ordered
                    key_ij = (kind, station, i, j)
                    key_iq = (kind, station, i, q)
                    key_jq = (kind, station, j, q)
                    if not all(key in priors for key in (key_ij, key_iq, key_jq)):
                        continue
                    c1_checked_triples += 1
                    z_ij = int(priors[key_ij])
                    z_iq = int(priors[key_iq])
                    z_jq = int(priors[key_jq])
                    if z_iq < z_ij + z_jq - 1 or z_iq > z_ij + z_jq:
                        c1_transitivity_violations += 1

    c2_violations = 0
    for upstream, downstream in data.segments:
        present = [
            k
            for k in trains
            if upstream in data.path_by_train[k]
            and downstream in data.path_by_train[k]
            and data.path_by_train[k].index(downstream)
            == data.path_by_train[k].index(upstream) + 1
            and data.has_event(k, upstream, "dep")
            and data.has_event(k, downstream, "arr")
        ]
        for p in range(len(present)):
            for q in range(p + 1, len(present)):
                i, j = data.canonical_pair(present[p], present[q])
                dep_key = ("dep", upstream, i, j)
                arr_key = ("arr", downstream, i, j)
                if (
                    dep_key in universe
                    and arr_key in universe
                    and dep_key in priors
                    and arr_key in priors
                ):
                    c2_violations += int(
                        int(priors[dep_key]) != int(priors[arr_key])
                    )

    c3_violations = 0
    c3_order_changes = 0
    for station in data.stations:
        present = [
            k
            for k in trains
            if data.has_event(k, station, "arr")
            and data.has_event(k, station, "dep")
        ]
        for p in range(len(present)):
            for q in range(p + 1, len(present)):
                i, j = data.canonical_pair(present[p], present[q])
                arr_key = ("arr", station, i, j)
                dep_key = ("dep", station, i, j)
                if arr_key not in priors or dep_key not in priors:
                    continue
                arr_val = int(priors[arr_key])
                dep_val = int(priors[dep_key])
                if arr_val == dep_val:
                    continue
                c3_order_changes += 1
                if (
                    arr_val == 1
                    and dep_val == 0
                    and not data.stop_at.get((i, station), False)
                ):
                    c3_violations += 1
                elif (
                    arr_val == 0
                    and dep_val == 1
                    and not data.stop_at.get((j, station), False)
                ):
                    c3_violations += 1

    return {
        "c1_checked_layers": c1_checked_layers,
        "c1_cycle_layers": c1_cycle_layers,
        "c1_checked_triples": (c1_checked_triples if count_c1_triplets else None),
        "c1_transitivity_violations": (
            c1_transitivity_violations if count_c1_triplets
            else (0 if c1_cycle_layers == 0 else None)
        ),
        "c1_audit_mode": ("FULL_TRIPLE" if count_c1_triplets else "DAG_ONLY"),
        "c2_segment_order_violations": c2_violations,
        "c3_order_changes": c3_order_changes,
        "c3_stop_compatibility_violations": c3_violations,
    }


# =============================================================================
# Independent post-solution timetable validator
# =============================================================================

def empty_validator_summary() -> Dict[str, Any]:
    return {
        "validator_ran": 0,
        "validator_pass": None,
        "validator_feasible_text": "--",
        "validator_total_violations": None,
        "validator_max_violation": None,
        "validator_recomputed_TT": None,
        "validator_reported_TT": None,
        "validator_objective_abs_diff": None,
        "validator_wall_sec": None,
        "validator_process_cpu_sec": None,
    }


def validate_timetable_csv(
    data: HSRData,
    cfg: Mapping[str, Any],
    trains: Sequence[str],
    timetable_path: Path,
    *,
    expected_objective: Optional[float],
    fixed_precedence: Optional[Mapping[PairKey, int]],
    output_dir: Path,
    validation_tag: str,
) -> Dict[str, Any]:
    """Independently verify a saved timetable against the source data.

    The validator deliberately does not inspect Gurobi constraints.  It reads
    the exported CSV and reconstructs the operational checks from ``trains``,
    ``stations`` and ``segments``.  This catches both numerical violations and
    implementation omissions in the optimization model.
    """
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    tolerance = float(as_float(cfg.get("validator_tolerance"), 1e-5) or 1e-5)
    objective_tolerance = float(
        as_float(cfg.get("validator_objective_tolerance"), 1e-3) or 1e-3
    )
    detail_limit = int(as_int(cfg.get("validator_max_detail_rows"), 20000) or 20000)
    write_details = as_bool(cfg.get("write_validation_details"), True)
    write_details_on_failure_only = _cfg_flag(cfg, "write_validation_details_on_failure_only", paper_default=True, normal_default=False)
    output_dir = ensure_dir(output_dir)

    counts: Counter[str] = Counter()
    details: List[Dict[str, Any]] = []
    max_violation = 0.0
    total_violation_count = 0

    def add_violation(category: str, magnitude: float = 1.0, **fields: Any) -> None:
        nonlocal max_violation, total_violation_count
        mag = max(0.0, float(magnitude))
        counts[category] += 1
        total_violation_count += 1
        max_violation = max(max_violation, mag)
        if len(details) < detail_limit:
            details.append({
                "category": category,
                "violation_magnitude": mag,
                **fields,
            })

    required_columns = {"车次", "车站", "到达时间", "出发时间"}
    if not timetable_path.exists():
        add_violation("missing_timetable_file", 1.0, path=str(timetable_path))
        df = pd.DataFrame(columns=sorted(required_columns))
    else:
        df = pd.read_csv(timetable_path, encoding="utf-8-sig")
        missing_columns = required_columns - set(df.columns)
        for column in sorted(missing_columns):
            add_violation("missing_csv_column", 1.0, column=column)
        for column in required_columns - missing_columns:
            if column in {"车次", "车站"}:
                df[column] = df[column].astype(str)
        if missing_columns:
            for column in missing_columns:
                df[column] = None

    selected = list(trains)
    selected_set = set(selected)
    expected_station_rows = sum(len(data.path_by_train[k]) for k in selected)

    # Duplicate and unexpected rows are checked before creating the lookup map.
    if not df.empty:
        duplicate_mask = df.duplicated(subset=["车次", "车站"], keep=False)
        if duplicate_mask.any():
            duplicate_groups = (
                df.loc[duplicate_mask]
                .groupby(["车次", "车站"], dropna=False)
                .size()
                .reset_index(name="row_count")
            )
            for rec in duplicate_groups.to_dict("records"):
                extra = max(1, int(rec["row_count"]) - 1)
                add_violation(
                    "duplicate_train_station_row",
                    float(extra),
                    train=rec["车次"],
                    station=rec["车站"],
                    row_count=rec["row_count"],
                )
        for rec in df[["车次", "车站"]].drop_duplicates().to_dict("records"):
            train = str(rec["车次"])
            station = str(rec["车站"])
            if train not in selected_set:
                add_violation("unexpected_train_row", 1.0, train=train, station=station)
            elif station not in data.path_by_train[train]:
                add_violation("unexpected_station_row", 1.0, train=train, station=station)

    row_map: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for rec in df.to_dict("records"):
        key = (str(rec.get("车次")), str(rec.get("车站")))
        row_map.setdefault(key, rec)

    arr: Dict[Tuple[str, str], float] = {}
    dep: Dict[Tuple[str, str], float] = {}

    def read_numeric(rec: Mapping[str, Any], column: str) -> Optional[float]:
        value = rec.get(column)
        if value is None or pd.isna(value) or str(value).strip() == "":
            return None
        try:
            result = float(value)
            return result if math.isfinite(result) else None
        except Exception:
            return None

    for train in selected:
        seq = data.path_by_train[train]
        for idx, station in enumerate(seq):
            rec = row_map.get((train, station))
            if rec is None:
                add_violation(
                    "missing_train_station_row", 1.0, train=train, station=station
                )
                continue
            arr_value = read_numeric(rec, "到达时间")
            dep_value = read_numeric(rec, "出发时间")
            if idx > 0:
                if arr_value is None:
                    add_violation(
                        "missing_arrival_event", 1.0, train=train, station=station
                    )
                else:
                    arr[(train, station)] = arr_value
            elif arr_value is not None:
                add_violation(
                    "unexpected_origin_arrival_event",
                    abs(arr_value),
                    train=train,
                    station=station,
                    actual=arr_value,
                )
            if idx < len(seq) - 1:
                if dep_value is None:
                    add_violation(
                        "missing_departure_event", 1.0, train=train, station=station
                    )
                else:
                    dep[(train, station)] = dep_value
            elif dep_value is not None:
                add_violation(
                    "unexpected_destination_departure_event",
                    abs(dep_value),
                    train=train,
                    station=station,
                    actual=dep_value,
                )

    # Origin windows.
    allow_early = as_bool(cfg.get("allow_early_departure"), True)
    fast_window = float(as_float(cfg.get("dep_window_fast"), 45.0) or 45.0)
    normal_window = float(as_float(cfg.get("dep_window_normal"), 60.0) or 60.0)
    fast_type = int(as_int(cfg.get("fast_train_type"), 350) or 350)
    for train in selected:
        origin = data.path_by_train[train][0]
        if (train, origin) not in dep:
            continue
        actual = dep[(train, origin)]
        expected = data.first_dep[train]
        window = fast_window if data.train_type.get(train, 300) == fast_type else normal_window
        lower = expected - window if allow_early else expected
        upper = expected + window
        if actual < lower - tolerance:
            add_violation(
                "origin_window_violation",
                lower - actual,
                train=train,
                station=origin,
                side="lower",
                actual=actual,
                required_bound=lower,
            )
        if actual > upper + tolerance:
            add_violation(
                "origin_window_violation",
                actual - upper,
                train=train,
                station=origin,
                side="upper",
                actual=actual,
                required_bound=upper,
            )

    # Running time, dwell time and pass-through checks.
    for train in selected:
        seq = data.path_by_train[train]
        for upstream, downstream in zip(seq, seq[1:]):
            if (train, upstream) not in dep or (train, downstream) not in arr:
                continue
            actual = arr[(train, downstream)] - dep[(train, upstream)]
            required = data.required_run_time(train, upstream, downstream)
            if actual < required - tolerance:
                add_violation(
                    "running_time_violation",
                    required - actual,
                    train=train,
                    upstream=upstream,
                    downstream=downstream,
                    actual=actual,
                    required=required,
                )
        for station in seq[1:-1]:
            if (train, station) not in arr or (train, station) not in dep:
                continue
            actual = dep[(train, station)] - arr[(train, station)]
            if data.stop_at.get((train, station), False):
                required = data.station_params.get(station, {}).get("dwell_min", 0.0)
                if actual < required - tolerance:
                    add_violation(
                        "dwell_time_violation",
                        required - actual,
                        train=train,
                        station=station,
                        actual=actual,
                        required=required,
                    )
            elif abs(actual) > tolerance:
                add_violation(
                    "pass_through_violation",
                    abs(actual),
                    train=train,
                    station=station,
                    actual=actual,
                    required=0.0,
                )

    # Station-event headways and, for Full-Fix, agreement with the fixed order.
    universe = data.event_universe(selected)
    for key in universe:
        kind, station, i, j = key
        time_map = dep if kind == "dep" else arr
        if (i, station) not in time_map or (j, station) not in time_map:
            continue
        ti = time_map[(i, station)]
        tj = time_map[(j, station)]
        headway = data.station_params.get(station, {}).get(
            "h_dep" if kind == "dep" else "h_arr", 0.0
        )
        separation = abs(tj - ti)
        if separation < headway - tolerance:
            add_violation(
                f"{kind}_headway_violation",
                headway - separation,
                train_i=i,
                train_j=j,
                station=station,
                actual=separation,
                required=headway,
            )
        if fixed_precedence is not None and key in fixed_precedence:
            value = int(fixed_precedence[key])
            directional_separation = (tj - ti) if value == 1 else (ti - tj)
            if directional_separation < headway - tolerance:
                add_violation(
                    "fixed_precedence_violation",
                    headway - directional_separation,
                    kind=kind,
                    train_i=i,
                    train_j=j,
                    station=station,
                    fixed_value=value,
                    actual=directional_separation,
                    required=headway,
                )

    # Segment order consistency reconstructed from event times.
    for upstream, downstream in data.segments:
        present = [
            k
            for k in selected
            if upstream in data.path_by_train[k]
            and downstream in data.path_by_train[k]
            and data.path_by_train[k].index(downstream)
            == data.path_by_train[k].index(upstream) + 1
            and (k, upstream) in dep
            and (k, downstream) in arr
        ]
        for p in range(len(present)):
            for q in range(p + 1, len(present)):
                i, j = data.canonical_pair(present[p], present[q])
                dep_i, dep_j = dep[(i, upstream)], dep[(j, upstream)]
                arr_i, arr_j = arr[(i, downstream)], arr[(j, downstream)]
                dep_order = 1 if dep_i < dep_j - tolerance else 0 if dep_j < dep_i - tolerance else None
                arr_order = 1 if arr_i < arr_j - tolerance else 0 if arr_j < arr_i - tolerance else None
                if dep_order is not None and arr_order is not None and dep_order != arr_order:
                    add_violation(
                        "segment_order_violation",
                        min(abs(dep_i - dep_j), abs(arr_i - arr_j)),
                        train_i=i,
                        train_j=j,
                        upstream=upstream,
                        downstream=downstream,
                        departure_order=dep_order,
                        arrival_order=arr_order,
                    )

    # Stop-dependent overtaking reconstructed from arrival/departure order changes.
    for station in data.stations:
        present = [
            k
            for k in selected
            if (k, station) in arr and (k, station) in dep
        ]
        for p in range(len(present)):
            for q in range(p + 1, len(present)):
                i, j = data.canonical_pair(present[p], present[q])
                ai, aj = arr[(i, station)], arr[(j, station)]
                di, dj = dep[(i, station)], dep[(j, station)]
                if ai < aj - tolerance and di > dj + tolerance:
                    if not data.stop_at.get((i, station), False):
                        add_violation(
                            "stop_overtaking_violation",
                            min(aj - ai, di - dj),
                            overtaken_train=i,
                            passing_train=j,
                            station=station,
                        )
                elif aj < ai - tolerance and dj > di + tolerance:
                    if not data.stop_at.get((j, station), False):
                        add_violation(
                            "stop_overtaking_violation",
                            min(ai - aj, dj - di),
                            overtaken_train=j,
                            passing_train=i,
                            station=station,
                        )

    recomputed_tt: Optional[float] = None
    objective_abs_diff: Optional[float] = None
    if all(
        (train, data.path_by_train[train][0]) in dep
        and (train, data.path_by_train[train][-1]) in arr
        for train in selected
    ):
        recomputed_tt = sum(
            arr[(train, data.path_by_train[train][-1])]
            - dep[(train, data.path_by_train[train][0])]
            for train in selected
        )
        if expected_objective is not None and math.isfinite(float(expected_objective)):
            objective_abs_diff = abs(recomputed_tt - float(expected_objective))
            if objective_abs_diff > objective_tolerance:
                add_violation(
                    "objective_recalculation_violation",
                    objective_abs_diff,
                    recomputed_TT=recomputed_tt,
                    reported_TT=float(expected_objective),
                    tolerance=objective_tolerance,
                )

    wall_sec = time.perf_counter() - wall_start
    cpu_sec = time.process_time() - cpu_start
    passed = total_violation_count == 0
    summary: Dict[str, Any] = {
        "validator_ran": 1,
        "validator_pass": int(passed),
        "validator_feasible_text": "Yes" if passed else "No",
        "validator_tolerance": tolerance,
        "validator_objective_tolerance": objective_tolerance,
        "validator_timetable_rows": len(df),
        "validator_expected_rows": expected_station_rows,
        "validator_total_violations": total_violation_count,
        "validator_max_violation": max_violation,
        "validator_recomputed_TT": recomputed_tt,
        "validator_reported_TT": expected_objective,
        "validator_objective_abs_diff": objective_abs_diff,
        "validator_detail_rows_written": (len(details) if write_details and ((not write_details_on_failure_only) or (not passed)) else 0),
        "validator_details_truncated": int(total_violation_count > len(details) and write_details and ((not write_details_on_failure_only) or (not passed))),
        "validator_wall_sec": wall_sec,
        "validator_process_cpu_sec": cpu_sec,
        "validator_missing_row_count": counts["missing_train_station_row"],
        "validator_missing_event_count": (
            counts["missing_arrival_event"] + counts["missing_departure_event"]
        ),
        "validator_duplicate_row_count": counts["duplicate_train_station_row"],
        "validator_origin_window_violation_count": counts["origin_window_violation"],
        "validator_running_time_violation_count": counts["running_time_violation"],
        "validator_dwell_time_violation_count": counts["dwell_time_violation"],
        "validator_pass_through_violation_count": counts["pass_through_violation"],
        "validator_departure_headway_violation_count": counts["dep_headway_violation"],
        "validator_arrival_headway_violation_count": counts["arr_headway_violation"],
        "validator_fixed_precedence_violation_count": counts["fixed_precedence_violation"],
        "validator_segment_order_violation_count": counts["segment_order_violation"],
        "validator_stop_overtaking_violation_count": counts["stop_overtaking_violation"],
        "validator_objective_violation_count": counts["objective_recalculation_violation"],
    }

    summary_path = output_dir / f"validation_{validation_tag}_summary.csv"
    details_path = output_dir / f"validation_{validation_tag}_violations.csv"
    write_csv(pd.DataFrame([summary]), summary_path)
    should_write_details = write_details and ((not write_details_on_failure_only) or (not passed))
    if should_write_details:
        detail_columns = [
            "category", "violation_magnitude", "train", "train_i", "train_j",
            "overtaken_train", "passing_train", "station", "upstream", "downstream",
            "kind", "side", "actual", "required", "required_bound", "fixed_value",
            "departure_order", "arrival_order", "recomputed_TT", "reported_TT",
            "tolerance", "path", "column", "row_count",
        ]
        detail_df = pd.DataFrame(details)
        for column in detail_columns:
            if column not in detail_df.columns:
                detail_df[column] = None
        write_csv(detail_df[detail_columns], details_path)
    elif details_path.exists():
        details_path.unlink()
    return summary


# =============================================================================
# Stage 1: Direct MILP
# =============================================================================

def run_direct(
    cfg: Mapping[str, Any],
    ctx: ExperimentContext,
    data: Optional[HSRData] = None,
) -> Dict[str, Any]:
    data = data or HSRData(cfg)
    outdir = ensure_dir(ctx.direct_dir)
    trains = data.select_trains(ctx.target_k)
    write_csv(
        pd.DataFrame({
            "selection_rank": range(1, len(trains) + 1),
            "train_id": trains,
            "selection_rule": data.train_selection_rule,
            "entry_time": [data.entry_time[k] for k in trains],
            "entry_time_source": [data.entry_time_source[k] for k in trains],
            "first_arrival_raw": [data.first_arrival[k] for k in trains],
            "first_departure": [data.first_dep[k] for k in trains],
            "file_order_index": [data.global_train_index[k] for k in trains],
            "train_type": [data.train_type[k] for k in trains],
            "intermediate_stop_count": [data.intermediate_stop_count[k] for k in trains],
        }),
        outdir / "target_train_set.csv",
    )

    stage_start = time.perf_counter()
    stage_cpu_start = time.process_time()
    started_at = now_text()
    print("\n" + "=" * 96)
    direct_threads_effective = int(
        as_int(cfg.get("direct_threads"), as_int(cfg.get("threads"), 1)) or 1
    )
    print(
        f"[1] Direct MILP | {ctx.experiment_tag} | K={len(trains)} | "
        f"TimeLimit={cfg.get('direct_time_limit')} | Threads={direct_threads_effective}"
    )
    print("=" * 96)
    build_start = time.perf_counter()
    direct_log_path = outdir / str(cfg.get("direct_log_filename") or "gurobi_direct.log")
    write_direct_log = as_bool(cfg.get("direct_write_solver_log"), True)
    if write_direct_log:
        print(f"Gurobi求解日志：{direct_log_path}")
    artifacts = build_timetable_model(
        data,
        cfg,
        trains,
        model_name=f"Direct_K{len(trains)}",
        stage="direct",
        log_file=direct_log_path if write_direct_log else None,
    )
    build_sec = time.perf_counter() - build_start
    first_solution: Dict[str, Any] = {}

    def record_first_solution(model: Model, where: int) -> None:
        if where == GRB.Callback.MIPSOL and not first_solution:
            first_solution["first_solution_solver_time_sec"] = float(model.cbGet(GRB.Callback.RUNTIME))
            first_solution["first_solution_obj_val"] = float(model.cbGet(GRB.Callback.MIPSOL_OBJ))

    solve_info = optimize_with_status_resolution(
        artifacts.model, cfg, stage="direct", callback=record_first_solution
    )
    usable = has_usable_solution(artifacts.model, keep_time_limit=True)

    post_start = time.perf_counter()
    audit: Dict[str, Any] = {}
    validation = empty_validator_summary()
    timetable_path = outdir / "timetable_direct.csv"
    pairwise_path = outdir / "pairwise_direct.csv"
    direct_priors_path = outdir / "direct_incumbent_priors_fullfix.csv"
    if usable:
        write_csv(pd.DataFrame(timetable_rows(data, artifacts)), timetable_path)
        direct_priors = {
            key: int(round(artifacts.z[key].X)) for key in artifacts.universe
        }
        # Always save the exact canonical precedence assignment of the final
        # Direct incumbent.  This file is intentionally canonical-only (one
        # row per target precedence index), so it can be fed back into the
        # full-fixing LP without ambiguity.  The optional pairwise_direct.csv
        # may contain both directions and is therefore only a human-readable
        # diagnostic output.
        write_csv(
            pd.DataFrame([
                {
                    "kind": key[0],
                    "station": key[1],
                    "train_i": key[2],
                    "train_j": key[3],
                    "value": int(value),
                }
                for key, value in direct_priors.items()
            ]),
            direct_priors_path,
        )
        if as_bool(cfg.get("write_direct_pairwise"), True):
            write_csv(pd.DataFrame(pairwise_rows(data, artifacts)), pairwise_path)
        audit = audit_precedence_assignment(data, trains, direct_priors, count_c1_triplets=not _paper_output_mode(cfg))
        validation = validate_timetable_csv(
            data,
            cfg,
            trains,
            timetable_path,
            expected_objective=safe_float(lambda: artifacts.model.ObjVal),
            fixed_precedence=None,
            output_dir=outdir,
            validation_tag="direct",
        )
    post_sec = time.perf_counter() - post_start
    total_sec = time.perf_counter() - stage_start
    total_cpu_sec = time.process_time() - stage_cpu_start

    summary = {
        "experiment_group": ctx.experiment_group,
        "experiment_tag": ctx.experiment_tag,
        "target_k": len(trains),
        "direct_threads_effective": direct_threads_effective,
        "direct_seed_effective": int(artifacts.model.Params.Seed),
        "direct_time_limit_effective": float(artifacts.model.Params.TimeLimit),
        "first_solution_solver_time_sec": first_solution.get("first_solution_solver_time_sec"),
        "first_solution_obj_val": first_solution.get("first_solution_obj_val"),
        "started_at": started_at,
        "finished_at": now_text(),
        "model_build_sec": build_sec,
        "postprocess_write_validate_sec": post_sec,
        "total_wall_sec": total_sec,
        "process_cpu_sec": total_cpu_sec,
        **solve_info,
        **solver_summary(artifacts.model, usable),
        **artifacts.model_stats,
        **audit,
        **validation,
        "gurobi_log_file": (
            relative_to_project(direct_log_path)
            if write_direct_log and direct_log_path.exists()
            else ""
        ),
        "timetable_csv": relative_to_project(timetable_path) if usable else "",
        "direct_incumbent_priors_csv": (
            relative_to_project(direct_priors_path)
            if usable and direct_priors_path.exists()
            else ""
        ),
        "pairwise_csv": (
            relative_to_project(pairwise_path)
            if usable and pairwise_path.exists()
            else ""
        ),
    }
    write_csv(pd.DataFrame([summary]), outdir / "direct_summary.csv")
    append_stage_timing(
        ctx,
        {
            "stage": "1_direct",
            "variant": ctx.experiment_tag,
            "status": summary["status_name"],
            "model_build_sec": build_sec,
            "solver_runtime_sec": summary["runtime_sec"],
            "solver_runtime_accumulated_sec": summary[
                "solver_runtime_accumulated_sec"
            ],
            "optimize_wall_sec": summary["optimize_wall_sec"],
            "process_cpu_sec": total_cpu_sec,
            "postprocess_write_sec": post_sec,
            "total_wall_sec": total_sec,
        },
    )
    artifacts.model.dispose()
    gc.collect()
    return summary


# =============================================================================
# Stage 2: 三类混合抽样 + 失败补抽
# =============================================================================

class MixedRestrictedSampler:
    METHODS = ("local", "stratified", "coverage")

    def __init__(
        self,
        data: HSRData,
        target_trains: Sequence[str],
        sample_size: Any,
        seed: int,
        *,
        local_pool_size: int,
        entry_bins: int,
        stop_groups: int,
        coverage_candidate_count: int,
        unique_subsets: bool,
        max_proposal_trials: int,
    ):
        self.data = data
        self.target_trains = list(target_trains)
        self.target_order = {k: i for i, k in enumerate(self.target_trains)}
        if isinstance(sample_size, Mapping):
            self.sample_sizes = {
                m: int(sample_size[m]) for m in self.METHODS
            }
        else:
            self.sample_sizes = {m: int(sample_size) for m in self.METHODS}
        if any(v <= 0 for v in self.sample_sizes.values()):
            raise ValueError(f"各抽样方法 sample size 必须为正：{self.sample_sizes}")
        self.sample_size = max(self.sample_sizes.values())  # legacy/display only
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.local_pool_size = min(
            len(self.target_trains),
            max(self.sample_sizes["local"], int(local_pool_size)),
        )
        self.entry_bins = max(1, int(entry_bins))
        self.stop_groups = max(1, int(stop_groups))
        self.coverage_candidate_count = max(1, int(coverage_candidate_count))
        self.unique_subsets = bool(unique_subsets)
        self.max_proposal_trials = max(100, int(max_proposal_trials))

        self.seen_signatures: set[str] = set()
        self.accepted_signatures: List[str] = []
        self.train_counts: Counter[str] = Counter()
        self.prec_counts: Counter[PairKey] = Counter()
        self.method_accepted: Counter[str] = Counter()
        self.duplicate_rejections = 0

        self.target_universe = data.event_universe(self.target_trains)
        self.pair_event_keys: Dict[Tuple[str, str], List[PairKey]] = defaultdict(list)
        for key in self.target_universe:
            _, _, i, j = key
            self.pair_event_keys[(i, j)].append(key)

        self.stop_thresholds = self._build_stop_thresholds()
        self.strata: Dict[Tuple[int, int, int], List[str]] = defaultdict(list)
        self.train_stratum: Dict[str, Tuple[int, int, int]] = {}
        for rank, train in enumerate(self.target_trains):
            time_bin = min(self.entry_bins - 1, (rank * self.entry_bins) // len(self.target_trains))
            stop_group = self._stop_group(self.data.intermediate_stop_count[train])
            key = (time_bin, int(self.data.train_type[train]), stop_group)
            self.strata[key].append(train)
            self.train_stratum[train] = key

    def _build_stop_thresholds(self) -> List[int]:
        values = sorted(self.data.intermediate_stop_count[k] for k in self.target_trains)
        thresholds: List[int] = []
        for g in range(1, self.stop_groups):
            idx = max(0, min(len(values) - 1, math.ceil(g * len(values) / self.stop_groups) - 1))
            thresholds.append(int(values[idx]))
        return thresholds

    def _stop_group(self, stop_count: int) -> int:
        for idx, threshold in enumerate(self.stop_thresholds):
            if stop_count <= threshold:
                return idx
        return self.stop_groups - 1

    def _ordered_subset(self, subset: Iterable[str]) -> List[str]:
        return sorted(set(subset), key=lambda k: self.target_order[k])

    def signature(self, subset: Sequence[str]) -> str:
        return ";".join(self._ordered_subset(subset))

    def sample_size_for(self, method: str) -> int:
        if method not in self.sample_sizes:
            raise ValueError(f"未知抽样方法：{method}")
        return int(self.sample_sizes[method])

    def _register_proposal(self, subset: Sequence[str], method: str) -> Optional[List[str]]:
        ordered = self._ordered_subset(subset)
        if len(ordered) != self.sample_size_for(method):
            return None
        sig = self.signature(ordered)
        if self.unique_subsets and sig in self.seen_signatures:
            self.duplicate_rejections += 1
            return None
        self.seen_signatures.add(sig)
        return ordered

    def _largest_remainder_allocation(self, sample_size: int) -> Dict[Tuple[int, int, int], int]:
        n_total = len(self.target_trains)
        quotas: Dict[Tuple[int, int, int], int] = {}
        fractions: List[Tuple[float, float, Tuple[int, int, int]]] = []
        assigned = 0
        for key, members in sorted(self.strata.items(), key=lambda kv: str(kv[0])):
            exact = sample_size * len(members) / n_total
            base = min(len(members), int(math.floor(exact)))
            quotas[key] = base
            assigned += base
            fractions.append((exact - base, self.rng.random(), key))
        remaining = sample_size - assigned
        eligible = [item for item in fractions if quotas[item[2]] < len(self.strata[item[2]])]
        eligible.sort(key=lambda x: (x[0], x[1]), reverse=True)
        if remaining > len(eligible):
            raise RuntimeError("分层配额无法按最大余数法填满 sample_size")
        for _, _, key in eligible[:remaining]:
            quotas[key] += 1
        return quotas

    def propose_local(self) -> Dict[str, Any]:
        sample_size = self.sample_size_for("local")
        max_start = len(self.target_trains) - self.local_pool_size
        for _ in range(self.max_proposal_trials):
            start = self.rng.randint(0, max_start) if max_start > 0 else 0
            pool = self.target_trains[start:start + self.local_pool_size]
            subset = self._register_proposal(self.rng.sample(pool, sample_size), "local")
            if subset is None:
                continue
            return {
                "subset": subset,
                "sampling_method": "local",
                "local_window_start_rank": start + 1,
                "local_window_end_rank": start + self.local_pool_size,
                "local_window_start_entry": self.data.entry_time[pool[0]],
                "local_window_end_entry": self.data.entry_time[pool[-1]],
                "stratum_allocation_json": "",
                "coverage_precedence_score": None,
                "coverage_train_score": None,
                "coverage_candidates_evaluated": None,
            }
        raise RuntimeError("local 抽样在最大尝试次数内无法生成新的唯一子集")

    def propose_stratified(self) -> Dict[str, Any]:
        sample_size = self.sample_size_for("stratified")
        for _ in range(self.max_proposal_trials):
            quotas = self._largest_remainder_allocation(sample_size)
            selected: List[str] = []
            for key, quota in quotas.items():
                if quota > 0:
                    selected.extend(self.rng.sample(self.strata[key], quota))
            subset = self._register_proposal(selected, "stratified")
            if subset is None:
                continue
            allocation = {
                f"time{key[0]}|type{key[1]}|stop{key[2]}": int(value)
                for key, value in sorted(quotas.items(), key=lambda kv: str(kv[0]))
                if value > 0
            }
            return {
                "subset": subset,
                "sampling_method": "stratified",
                "local_window_start_rank": None,
                "local_window_end_rank": None,
                "local_window_start_entry": None,
                "local_window_end_entry": None,
                "stratum_allocation_json": json.dumps(allocation, ensure_ascii=False, sort_keys=True),
                "coverage_precedence_score": None,
                "coverage_train_score": None,
                "coverage_candidates_evaluated": None,
            }
        raise RuntimeError("stratified 抽样在最大尝试次数内无法生成新的唯一子集")

    def _weighted_subset(self, sample_size: int) -> List[str]:
        keyed: List[Tuple[float, str]] = []
        for train in self.target_trains:
            weight = 1.0 / (1.0 + self.train_counts[train])
            u = max(self.rng.random(), 1e-15)
            key = u ** (1.0 / weight)
            keyed.append((key, train))
        keyed.sort(reverse=True)
        return self._ordered_subset(train for _, train in keyed[:sample_size])

    def _pair_weights(self) -> Dict[Tuple[str, str], float]:
        result: Dict[Tuple[str, str], float] = {}
        for pair, keys in self.pair_event_keys.items():
            result[pair] = sum(1.0 / (1.0 + self.prec_counts[key]) for key in keys)
        return result

    def propose_coverage(self) -> Dict[str, Any]:
        sample_size = self.sample_size_for("coverage")
        pair_weights = self._pair_weights()
        candidates: List[Tuple[float, float, float, List[str]]] = []
        local_seen: set[str] = set()
        generation_trials = 0
        max_trials = max(self.coverage_candidate_count * 20, self.max_proposal_trials)
        while len(candidates) < self.coverage_candidate_count and generation_trials < max_trials:
            generation_trials += 1
            subset = self._weighted_subset(sample_size)
            sig = self.signature(subset)
            if sig in local_seen or (self.unique_subsets and sig in self.seen_signatures):
                continue
            local_seen.add(sig)
            prec_score = 0.0
            for i, j in combinations(subset, 2):
                ci, cj = self.data.canonical_pair(i, j)
                prec_score += pair_weights.get((ci, cj), 0.0)
            train_score = sum(1.0 / (1.0 + self.train_counts[k]) for k in subset)
            candidates.append((prec_score, train_score, self.rng.random(), subset))
        if not candidates:
            raise RuntimeError("coverage-oriented 抽样未能生成新的候选子集")
        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        prec_score, train_score, _, chosen = candidates[0]
        registered = self._register_proposal(chosen, "coverage")
        if registered is None:
            raise RuntimeError("coverage-oriented 最优候选在注册时意外重复")
        return {
            "subset": registered,
            "sampling_method": "coverage",
            "local_window_start_rank": None,
            "local_window_end_rank": None,
            "local_window_start_entry": None,
            "local_window_end_entry": None,
            "stratum_allocation_json": "",
            "coverage_precedence_score": prec_score,
            "coverage_train_score": train_score,
            "coverage_candidates_evaluated": len(candidates),
        }

    def propose(self, method: str) -> Dict[str, Any]:
        if method == "local":
            return self.propose_local()
        if method == "stratified":
            return self.propose_stratified()
        if method == "coverage":
            return self.propose_coverage()
        raise ValueError(f"未知抽样方法：{method}")

    def accept(self, subset: Sequence[str], method: str) -> None:
        ordered = self._ordered_subset(subset)
        sig = self.signature(ordered)
        self.accepted_signatures.append(sig)
        self.method_accepted[method] += 1
        for train in ordered:
            self.train_counts[train] += 1
        for i, j in combinations(ordered, 2):
            ci, cj = self.data.canonical_pair(i, j)
            for key in self.pair_event_keys.get((ci, cj), []):
                self.prec_counts[key] += 1

    def get_state(self, config_signature: str) -> Dict[str, Any]:
        return {
            "config_signature": config_signature,
            "rng_state": self.rng.getstate(),
            "seen_signatures": sorted(self.seen_signatures),
            "accepted_signatures": list(self.accepted_signatures),
            "train_counts": dict(self.train_counts),
            "prec_counts": dict(self.prec_counts),
            "method_accepted": dict(self.method_accepted),
            "duplicate_rejections": self.duplicate_rejections,
        }

    def load_state(self, state: Mapping[str, Any], config_signature: str) -> None:
        if state.get("config_signature") != config_signature:
            raise RuntimeError("现有小样本断点状态与当前 config/目标列车不一致；请删除 outputs/02_samples 后重跑")
        self.rng.setstate(state["rng_state"])
        self.seen_signatures = set(state.get("seen_signatures", []))
        self.accepted_signatures = list(state.get("accepted_signatures", []))
        self.train_counts = Counter(state.get("train_counts", {}))
        self.prec_counts = Counter(state.get("prec_counts", {}))
        self.method_accepted = Counter(state.get("method_accepted", {}))
        self.duplicate_rejections = int(state.get("duplicate_rejections", 0))


def _sampling_signature(
    cfg: Mapping[str, Any],
    target_trains: Sequence[str],
    sample_sizes: Mapping[str, int],
    required_usable: int,
    method_targets: Mapping[str, int],
) -> str:
    data_dir = resolve_project_path(cfg.get("data_dir"), ".")
    file_hashes: Dict[str, str] = {}
    for name in ("trains.csv", "stations.csv", "segments.csv"):
        path = data_dir / name
        file_hashes[name] = sha256_file(path) if path.exists() else "MISSING"
    payload = {
        "target_trains": list(target_trains),
        "sample_design": {
            method: {"sample_size": int(sample_sizes[method]), "sample_count": int(method_targets[method])}
            for method in SAMPLING_METHODS
        },
        "sample_count": required_usable,
        "method_counts": dict(method_targets),
        "seed": as_int(cfg.get("seed"), 42),
        "local_pool": as_int(cfg.get("sampling_local_candidate_pool_size"), 90),
        "entry_bins": as_int(cfg.get("sampling_entry_time_bins"), 5),
        "stop_groups": as_int(cfg.get("sampling_stop_count_groups"), 3),
        "coverage_candidates": as_int(cfg.get("sampling_coverage_candidate_count"), 200),
        "train_selection_rule": cfg.get("train_selection_rule"),
        "dep_window_fast": cfg.get("dep_window_fast"),
        "dep_window_normal": cfg.get("dep_window_normal"),
        "allow_early_departure": cfg.get("allow_early_departure"),
        "sample_time_limit": cfg.get("sample_time_limit"),
        "sample_mipgap": cfg.get("sample_mipgap"),
        "sample_parallel": cfg.get("sample_parallel"),
        "sample_parallel_workers": cfg.get("sample_parallel_workers"),
        "sample_parallel_worker_basis": cfg.get("sample_parallel_worker_basis"),
        "sample_threads_per_solve": cfg.get("sample_threads_per_solve"),
        "big_m": cfg.get("big_m"),
        "data_file_sha256": file_hashes,
    }
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))

def _save_sampler_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(dict(state), f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _load_sampler_state(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def _method_targets(cfg: Mapping[str, Any], total_count: int) -> Dict[str, int]:
    """Allocate the usable restricted-instance budget among three samplers."""
    if total_count <= 0:
        raise ValueError("sample_count 必须为正整数")
    methods = ("local", "stratified", "coverage")
    mode = str(cfg.get("sampling_allocation_mode") or "auto_ratio").strip().lower()
    if mode == "manual":
        targets = {
            method: int(as_int(cfg.get(f"sampling_{method}_count"), -1) or 0)
            for method in methods
        }
        if any(value < 0 for value in targets.values()):
            raise ValueError("manual 模式下三类 sampling_*_count 均须为非负整数")
        if sum(targets.values()) != total_count:
            raise ValueError(
                f"manual 抽样数量之和必须等于 sample_count={total_count}，当前为 {targets}"
            )
        return targets
    if mode not in {"auto", "auto_ratio", "ratio"}:
        raise ValueError(
            "sampling_allocation_mode 仅支持 auto_ratio 或 manual，"
            f"当前为 {mode!r}"
        )

    ratios = {
        "local": float(as_float(cfg.get("sampling_local_ratio"), 0.30) or 0.0),
        "stratified": float(
            as_float(cfg.get("sampling_stratified_ratio"), 0.30) or 0.0
        ),
        "coverage": float(
            as_float(cfg.get("sampling_coverage_ratio"), 0.40) or 0.0
        ),
    }
    if any(value < 0 for value in ratios.values()) or sum(ratios.values()) <= 0:
        raise ValueError(f"抽样比例必须非负且总和为正，当前为 {ratios}")
    ratio_sum = sum(ratios.values())
    exact = {m: total_count * ratios[m] / ratio_sum for m in methods}
    targets = {m: int(math.floor(exact[m])) for m in methods}
    remaining = total_count - sum(targets.values())
    # Deterministic largest-remainder allocation; ties follow local/stratified/coverage.
    order = sorted(
        methods,
        key=lambda m: (exact[m] - targets[m], -methods.index(m)),
        reverse=True,
    )
    for method in order[:remaining]:
        targets[method] += 1
    return targets

def _next_sampling_method(targets: Mapping[str, int], accepted: Mapping[str, int]) -> Optional[str]:
    for method in ("local", "stratified", "coverage"):
        if int(accepted.get(method, 0)) < int(targets[method]):
            return method
    return None


_SAMPLE_WORKER_CFG: Optional[Dict[str, Any]] = None
_SAMPLE_WORKER_DATA: Optional[HSRData] = None
_SAMPLE_WORKER_ENV: Optional[Env] = None


def _sample_worker_cleanup() -> None:
    global _SAMPLE_WORKER_ENV, _SAMPLE_WORKER_DATA, _SAMPLE_WORKER_CFG
    try:
        if _SAMPLE_WORKER_ENV is not None:
            _SAMPLE_WORKER_ENV.dispose()
    except Exception:
        pass
    _SAMPLE_WORKER_ENV = None
    _SAMPLE_WORKER_DATA = None
    _SAMPLE_WORKER_CFG = None


def _sample_worker_init(cfg: Mapping[str, Any]) -> None:
    """Initialize one isolated Gurobi environment and data copy per process."""
    global _SAMPLE_WORKER_CFG, _SAMPLE_WORKER_DATA, _SAMPLE_WORKER_ENV
    _SAMPLE_WORKER_CFG = dict(cfg)
    env = Env(empty=True)
    # Suppress environment startup noise. Individual models can still write
    # their own LogFile when requested.
    env.setParam("OutputFlag", 0)
    env.start()
    _SAMPLE_WORKER_ENV = env
    _SAMPLE_WORKER_DATA = HSRData(_SAMPLE_WORKER_CFG)
    atexit.register(_sample_worker_cleanup)


def _sample_parallel_resources(
    cfg: Mapping[str, Any], task_count: int
) -> Dict[str, Any]:
    """Resolve a CPU- and memory-aware process count for restricted solves."""
    logical = int(os.cpu_count() or 1)
    physical = logical
    available_gb: Optional[float] = None
    total_gb: Optional[float] = None
    try:
        import psutil  # optional but normally present in Anaconda/Spyder

        physical = int(psutil.cpu_count(logical=False) or logical)
        logical = int(psutil.cpu_count(logical=True) or logical)
        vm = psutil.virtual_memory()
        available_gb = float(vm.available) / (1024.0 ** 3)
        total_gb = float(vm.total) / (1024.0 ** 3)
    except Exception:
        pass

    threads_per_solve = max(
        1,
        int(
            as_int(
                cfg.get("sample_threads_per_solve"),
                1,
            )
            or 1
        ),
    )
    basis = str(cfg.get("sample_parallel_worker_basis") or "physical").strip().lower()
    auto_cpu_basis = logical if basis in {"logical", "threads", "logical_cores"} else physical
    auto_worker_cap = max(1, auto_cpu_basis // threads_per_solve)

    # IMPORTANT: sample_parallel_worker_basis controls AUTO sizing only.
    # If sample_parallel_workers is explicitly specified, honor that value and
    # only apply a safety cap based on the machine's logical CPU threads.
    # This makes configurations such as 6 workers x 3 threads valid on an
    # i7-12700KF (20 logical threads), while AUTO+physical remains conservative.
    raw_workers = cfg.get("sample_parallel_workers")
    explicit_workers = not (
        raw_workers is None
        or str(raw_workers).strip().lower() in {"", "auto", "automatic"}
    )
    if explicit_workers:
        requested_workers = max(1, int(float(raw_workers)))
        worker_source = "EXPLICIT"
    else:
        requested_workers = auto_worker_cap
        worker_source = f"AUTO_{basis.upper()}"

    allow_cpu_oversubscription = as_bool(
        cfg.get("sample_allow_cpu_oversubscription"), False
    )
    logical_worker_cap = max(1, logical // threads_per_solve)
    cpu_cap = (
        max(1, task_count)
        if allow_cpu_oversubscription
        else (logical_worker_cap if explicit_workers else auto_worker_cap)
    )

    memory_guard = as_bool(cfg.get("sample_memory_guard"), True)
    memory_per_worker_gb = float(
        as_float(cfg.get("sample_memory_per_worker_gb"), 3.0) or 3.0
    )
    reserve_gb = float(as_float(cfg.get("sample_memory_reserve_gb"), 8.0) or 0.0)
    memory_cap: Optional[int] = None
    if memory_guard and available_gb is not None and memory_per_worker_gb > 0:
        usable_gb = max(0.0, available_gb - reserve_gb)
        memory_cap = max(1, int(math.floor(usable_gb / memory_per_worker_gb)))

    effective = min(max(1, task_count), requested_workers, cpu_cap)
    if memory_cap is not None:
        effective = min(effective, memory_cap)
    effective = max(1, effective)
    return {
        "parallel_workers": effective,
        "requested_workers": requested_workers,
        "worker_source": worker_source,
        "worker_basis": basis,
        "threads_per_solve": threads_per_solve,
        "physical_cpu_cores": physical,
        "logical_cpu_threads": logical,
        "auto_worker_cap": auto_worker_cap,
        "logical_worker_cap": logical_worker_cap,
        "allow_cpu_oversubscription": int(allow_cpu_oversubscription),
        "cpu_worker_cap": cpu_cap,
        "memory_guard": int(memory_guard),
        "memory_per_worker_gb": memory_per_worker_gb,
        "memory_reserve_gb": reserve_gb,
        "memory_worker_cap": memory_cap,
        "available_memory_gb_at_launch": available_gb,
        "total_memory_gb": total_gb,
    }


def _restricted_worker_error_result(
    task: Mapping[str, Any], exc: BaseException, started_at: str, wall_start: float, cpu_start: float
) -> Dict[str, Any]:
    return {
        "experiment_group": task.get("experiment_group"),
        "experiment_tag": task.get("experiment_tag"),
        "target_k": task.get("target_k"),
        "attempt_id": task.get("attempt_id"),
        "global_run_id": None,
        "sample_size": task.get("sample_size"),
        "sampling_method": task.get("sampling_method"),
        "method_attempt_id": task.get("method_attempt_id"),
        "method_usable_id": None,
        "sample_seed": task.get("sample_seed"),
        "fixed_random_seed": task.get("fixed_random_seed"),
        "train_ids": task.get("train_ids"),
        "num_trains": len(task.get("subset") or []),
        "num_precedence_labels": 0,
        "started_at": started_at,
        "finished_at": now_text(),
        "sample_generation_sec": task.get("sample_generation_sec", 0.0),
        "model_build_sec": None,
        "result_write_sec": None,
        "total_wall_sec": time.perf_counter() - wall_start,
        "process_cpu_sec": time.process_time() - cpu_start,
        "initial_status_code": -1,
        "initial_status_name": "WORKER_ERROR",
        "status_disambiguation_rerun": 0,
        "solver_runtime_accumulated_sec": 0.0,
        "optimize_wall_sec": None,
        "optimize_process_cpu_sec": None,
        "status_code": -1,
        "status_name": "WORKER_ERROR",
        "sol_count": 0,
        "is_optimal": 0,
        "has_solution": 0,
        "obj_val": None,
        "obj_bound": None,
        "mip_gap": None,
        "mip_gap_percent": None,
        "runtime_sec": 0.0,
        "node_count": None,
        "simplex_iterations": None,
        "barrier_iterations": None,
        "work_units": None,
        "num_vars": None,
        "num_constrs": None,
        "num_bin_vars": None,
        "num_cont_vars": None,
        "label_solution_source": "",
        "packing_value_candidate": None,
        "packing_value_source_candidate": "",
        "packing_value_valid": 0,
        "usable_for_evidence": 0,
        "replacement_required": 1,
        "timetable_csv": "",
        "pairwise_csv": "",
        "solver_log": task.get("solver_log_relative", ""),
        "parallel_worker_pid": os.getpid(),
        "worker_exception_type": type(exc).__name__,
        "worker_exception": str(exc),
        "worker_traceback": traceback.format_exc(),
        **dict(task.get("proposal_metadata") or {}),
    }


def _solve_restricted_parallel_task(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Build and solve one restricted MILP entirely inside one child process."""
    started_at = now_text()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    artifacts: Optional[TimetableModelArtifacts] = None
    try:
        if _SAMPLE_WORKER_CFG is None or _SAMPLE_WORKER_DATA is None or _SAMPLE_WORKER_ENV is None:
            raise RuntimeError("restricted worker was not initialized")
        cfg = _SAMPLE_WORKER_CFG
        data = _SAMPLE_WORKER_DATA
        subset = list(task["subset"])
        prefix = str(task["prefix"])
        outdir = Path(str(task["outdir"]))
        timetable_path = outdir / f"timetable_{prefix}.csv"
        pairwise_path = outdir / f"pairwise_{prefix}.csv"
        log_path = Path(str(task["solver_log_path"]))

        build_start = time.perf_counter()
        artifacts = build_timetable_model(
            data,
            cfg,
            subset,
            model_name=f"Restricted_{prefix}",
            stage="sample",
            env=_SAMPLE_WORKER_ENV,
        )
        build_sec = time.perf_counter() - build_start
        if as_bool(cfg.get("sample_write_solver_log"), True):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            _set_gurobi_log_file(artifacts.model, log_path)

        solve_info = optimize_with_status_resolution(
            artifacts.model, cfg, stage="sample"
        )
        keep_tl = as_bool(cfg.get("keep_time_limit_incumbent"), True)
        provisional_usable = has_usable_solution(
            artifacts.model, keep_time_limit=keep_tl
        )

        packing_value_candidate: Optional[float] = None
        packing_value_source_candidate = ""
        label_solution_source = ""
        packing_value_valid = False
        if provisional_usable:
            obj_val = safe_float(lambda: artifacts.model.ObjVal)
            obj_bound = safe_float(lambda: artifacts.model.ObjBound)
            is_optimal = int(artifacts.model.Status) == int(GRB.OPTIMAL)
            label_solution_source = (
                "optimal_solution" if is_optimal else "feasible_incumbent"
            )
            if is_optimal and obj_val is not None and math.isfinite(obj_val):
                packing_value_candidate = float(obj_val)
                packing_value_source_candidate = "optimal_obj_val"
                packing_value_valid = True
            elif (
                obj_bound is not None
                and math.isfinite(obj_bound)
                and obj_val is not None
                and math.isfinite(obj_val)
                and obj_bound <= obj_val + 1e-5
            ):
                packing_value_candidate = float(obj_bound)
                packing_value_source_candidate = "solver_best_bound"
                packing_value_valid = True

        usable = bool(provisional_usable and packing_value_valid)
        solver_rec = solver_summary(artifacts.model, usable)

        write_start = time.perf_counter()
        if usable:
            if as_bool(cfg.get("write_sample_timetables"), True):
                write_csv(pd.DataFrame(timetable_rows(data, artifacts)), timetable_path)
            write_csv(
                pd.DataFrame(
                    pairwise_rows(
                        data,
                        artifacts,
                        directed=not as_bool(
                            cfg.get("sample_pairwise_canonical_only"), True
                        ),
                    )
                ),
                pairwise_path,
            )
        write_sec = time.perf_counter() - write_start

        result = {
            "experiment_group": task["experiment_group"],
            "experiment_tag": task["experiment_tag"],
            "target_k": task["target_k"],
            "attempt_id": task["attempt_id"],
            "global_run_id": None,
            "sample_size": task["sample_size"],
            "sampling_method": task["sampling_method"],
            "method_attempt_id": task["method_attempt_id"],
            "method_usable_id": None,
            "sample_seed": task["sample_seed"],
            "fixed_random_seed": task["fixed_random_seed"],
            "train_ids": task["train_ids"],
            "num_trains": len(subset),
            "num_precedence_labels": len(artifacts.universe) if usable else 0,
            "started_at": started_at,
            "finished_at": now_text(),
            "sample_generation_sec": task.get("sample_generation_sec", 0.0),
            "model_build_sec": build_sec,
            "result_write_sec": write_sec,
            "total_wall_sec": time.perf_counter() - wall_start,
            "process_cpu_sec": time.process_time() - cpu_start,
            **dict(task.get("proposal_metadata") or {}),
            **solve_info,
            **solver_rec,
            **artifacts.model_stats,
            "label_solution_source": label_solution_source,
            "packing_value_candidate": packing_value_candidate,
            "packing_value_source_candidate": packing_value_source_candidate,
            "packing_value_valid": int(packing_value_valid),
            "usable_for_evidence": int(usable),
            "replacement_required": int(not usable),
            "timetable_csv": (
                relative_to_project(timetable_path)
                if usable and timetable_path.exists()
                else ""
            ),
            "pairwise_csv": (
                relative_to_project(pairwise_path)
                if usable and pairwise_path.exists()
                else ""
            ),
            "solver_log": (
                relative_to_project(log_path)
                if log_path.exists()
                else ""
            ),
            "parallel_worker_pid": os.getpid(),
            "worker_exception_type": "",
            "worker_exception": "",
            "worker_traceback": "",
        }
    except BaseException as exc:
        result = _restricted_worker_error_result(
            task, exc, started_at, wall_start, cpu_start
        )
    finally:
        if artifacts is not None:
            try:
                artifacts.model.dispose()
            except Exception:
                pass
        gc.collect()

    result_path = Path(str(task["result_json_path"]))
    try:
        write_json(result, result_path)
    except Exception:
        # Returning the result still lets the parent persist it.
        pass
    return result


def _rebase_sampler_to_results(
    sampler: MixedRestrictedSampler, rows: Sequence[Mapping[str, Any]]
) -> None:
    """Keep RNG/seen proposals, but count only successful evidence samples."""
    sampler.accepted_signatures = []
    sampler.train_counts = Counter()
    sampler.prec_counts = Counter()
    sampler.method_accepted = Counter()
    for row in sorted(rows, key=lambda x: int(x.get("attempt_id") or 0)):
        if int(row.get("usable_for_evidence") or 0) != 1:
            continue
        subset = [x for x in str(row.get("train_ids") or "").split(";") if x]
        sampler.accept(subset, str(row.get("sampling_method")))


def _save_parallel_plan(
    path: Path,
    csv_path: Path,
    config_signature: str,
    tasks: Sequence[Mapping[str, Any]],
) -> None:
    write_json(
        {
            "sampling_config_signature": config_signature,
            "created_or_updated_at": now_text(),
            "tasks": list(tasks),
        },
        path,
    )
    plan_rows = []
    for task in tasks:
        plan_rows.append({
            "attempt_id": task["attempt_id"],
            "sampling_method": task["sampling_method"],
            "method_attempt_id": task["method_attempt_id"],
            "sample_size": task["sample_size"],
            "train_ids": task["train_ids"],
            "sample_generation_sec": task.get("sample_generation_sec"),
            **dict(task.get("proposal_metadata") or {}),
        })
    write_csv(pd.DataFrame(plan_rows), csv_path)


def _load_parallel_results(
    result_dir: Path,
) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}
    if not result_dir.exists():
        return results
    for path in sorted(result_dir.glob("attempt_result_*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                row = json.load(f)
            attempt_id = int(row["attempt_id"])
            results[attempt_id] = row
        except Exception:
            # A partially written file is ignored and the task is resubmitted.
            continue
    return results


def _parallel_task_from_proposal(
    cfg: Mapping[str, Any],
    ctx: ExperimentContext,
    outdir: Path,
    proposal: Mapping[str, Any],
    attempt_id: int,
    method_attempt_id: int,
    sample_size: int,
    effective_seed: int,
    use_fixed_seed: bool,
    generation_sec: float,
    sampler: MixedRestrictedSampler,
) -> Dict[str, Any]:
    proposal_copy = dict(proposal)
    subset = list(proposal_copy.pop("subset"))
    method = str(proposal_copy.pop("sampling_method", proposal_copy.get("sampling_method", "")))
    if not method:
        method = str(proposal.get("sampling_method"))
    signature = sampler.signature(subset)
    prefix = f"attempt{attempt_id:04d}_{method}_N{sample_size}"
    solver_log_path = outdir / "solver_logs" / f"{prefix}.log"
    result_json_path = outdir / "attempt_results" / f"attempt_result_{attempt_id:04d}.json"
    return {
        "experiment_group": ctx.experiment_group,
        "experiment_tag": ctx.experiment_tag,
        "target_k": ctx.target_k,
        "attempt_id": attempt_id,
        "sample_size": sample_size,
        "sampling_method": method,
        "method_attempt_id": method_attempt_id,
        "sample_seed": effective_seed,
        "fixed_random_seed": int(use_fixed_seed),
        "subset": subset,
        "train_ids": signature,
        "prefix": prefix,
        "outdir": str(outdir),
        "solver_log_path": str(solver_log_path),
        "solver_log_relative": relative_to_project(solver_log_path),
        "result_json_path": str(result_json_path),
        "sample_generation_sec": generation_sec,
        "proposal_metadata": proposal_copy,
    }


def _run_restricted_instances_serial(
    cfg: Mapping[str, Any],
    ctx: ExperimentContext,
    data: Optional[HSRData] = None,
) -> pd.DataFrame:
    data = data or HSRData(cfg)
    outdir = ctx.sample_dir
    resume = as_bool(cfg.get("sample_resume"), True)
    clear_output = as_bool(cfg.get("sample_clear_output"), False)
    if clear_output and not resume:
        clean_dir(outdir)
    else:
        ensure_dir(outdir)

    if len(ctx.sample_counts) != 1:
        raise ValueError("主实验每次仅支持一个总样本数量")
    method_sizes = configured_method_sample_sizes(cfg)
    sample_size = max(method_sizes.values())  # backward-compatible display field
    required_usable = int(ctx.sample_counts[0])
    targets = _method_targets(cfg, required_usable)
    if sum(targets.values()) != required_usable:
        raise AssertionError("内部错误：三类抽样数量之和与 sample_count 不一致")
    for method, size in method_sizes.items():
        if size >= ctx.target_k:
            raise ValueError(f"sampling_{method}_size 必须严格小于 train_limit")
    design_label = configured_sampling_design_label(cfg, required_usable)

    target_trains = data.select_trains(ctx.target_k)
    write_csv(
        pd.DataFrame({
            "selection_rank": range(1, len(target_trains) + 1),
            "train_id": target_trains,
            "selection_rule": data.train_selection_rule,
            "entry_time": [data.entry_time[k] for k in target_trains],
            "entry_time_source": [data.entry_time_source[k] for k in target_trains],
            "first_arrival_raw": [data.first_arrival[k] for k in target_trains],
            "first_departure": [data.first_dep[k] for k in target_trains],
            "train_type": [data.train_type[k] for k in target_trains],
            "intermediate_stop_count": [
                data.intermediate_stop_count[k] for k in target_trains
            ],
        }),
        outdir / "target_train_set_for_sampling.csv",
    )

    use_fixed_seed = as_bool(cfg.get("random_seed"), True)
    configured_seed = int(as_int(cfg.get("seed"), 42) or 42)
    effective_seed = (
        configured_seed
        if use_fixed_seed
        else random.SystemRandom().randrange(0, 2**63)
    )
    sampler = MixedRestrictedSampler(
        data,
        target_trains,
        method_sizes,
        effective_seed,
        local_pool_size=int(
            as_int(cfg.get("sampling_local_candidate_pool_size"), 90) or 90
        ),
        entry_bins=int(as_int(cfg.get("sampling_entry_time_bins"), 5) or 5),
        stop_groups=int(
            as_int(cfg.get("sampling_stop_count_groups"), 3) or 3
        ),
        coverage_candidate_count=int(
            as_int(cfg.get("sampling_coverage_candidate_count"), 200) or 200
        ),
        unique_subsets=as_bool(cfg.get("sampling_unique_subsets"), True),
        max_proposal_trials=int(
            as_int(cfg.get("sampling_max_proposal_trials"), 5000) or 5000
        ),
    )
    config_signature = _sampling_signature(
        cfg, target_trains, method_sizes, required_usable, targets
    )
    state_path = outdir / "sampling_state.pkl"
    attempts_partial = outdir / "sample_attempt_manifest.partial.csv"
    usable_partial = outdir / "sample_manifest.partial.csv"

    attempt_rows: List[Dict[str, Any]] = []
    usable_rows: List[Dict[str, Any]] = []
    if resume and state_path.exists() and attempts_partial.exists() and usable_partial.exists():
        sampler.load_state(_load_sampler_state(state_path), config_signature)
        attempt_rows = pd.read_csv(
            attempts_partial, encoding="utf-8-sig"
        ).to_dict("records")
        usable_rows = pd.read_csv(
            usable_partial, encoding="utf-8-sig"
        ).to_dict("records")
        print(
            f"[RESUME] attempts={len(attempt_rows)}, usable={len(usable_rows)}, "
            f"method={dict(sampler.method_accepted)}"
        )
    elif resume and (outdir / "sample_manifest.csv").exists():
        completed = pd.read_csv(
            outdir / "sample_manifest.csv", encoding="utf-8-sig"
        )
        completion_path = outdir / "sampling_completion_record.json"
        recorded_signature = None
        if completion_path.exists():
            with completion_path.open("r", encoding="utf-8") as f:
                completion_record = json.load(f)
            recorded_signature = completion_record.get("summary", {}).get(
                "sampling_config_signature"
            )
        if len(completed) == required_usable and recorded_signature == config_signature:
            print(
                f"[SKIP] 已有完整且配置一致的 {required_usable} 个可用小样本："
                f"{outdir / 'sample_manifest.csv'}"
            )
            return completed
        if len(completed) == required_usable and recorded_signature != config_signature:
            raise RuntimeError(
                "现有小样本与当前配置签名不一致。案例目录通常会随 K/Ksub/R 自动变化；"
                "若手动复用了 instance_tag，请删除该案例的 02_samples 后重跑。"
            )
    elif clear_output:
        clean_dir(outdir)

    keep_tl = as_bool(cfg.get("keep_time_limit_incumbent"), True)
    configured_max_attempts = as_int(cfg.get("sample_max_total_attempts"), None)
    max_attempts = int(
        configured_max_attempts
        if configured_max_attempts is not None and configured_max_attempts > 0
        else max(required_usable * 5, required_usable + 100)
    )
    attempt_id = len(attempt_rows)
    usable_id = len(usable_rows)
    stage_start = time.perf_counter()
    stage_cpu_start = time.process_time()
    stage_started_at = now_text()

    print("\n" + "=" * 104)
    print(
        f"[2] Mixed restricted instances | K={ctx.target_k}, design={design_label}, "
        f"required usable R={required_usable}"
    )
    print(
        f"composition={targets}; per-attempt TimeLimit={cfg.get('sample_time_limit')} s"
    )
    print(
        f"selection={data.train_selection_rule}, fixed_seed={use_fixed_seed}, "
        f"effective_seed={effective_seed}"
    )
    print(
        "无 incumbent / infeasible / 无有效 best bound 的 attempt 不计入 R，"
        "并按原抽样类别继续补抽。"
    )
    print("=" * 104)

    while usable_id < required_usable:
        method = _next_sampling_method(targets, sampler.method_accepted)
        if method is None:
            break
        attempt_id += 1
        if attempt_id > max_attempts:
            raise RuntimeError(
                f"达到 sample_max_total_attempts={max_attempts}，仍未得到 "
                f"{required_usable} 个可用样本"
            )

        attempt_wall_start = time.perf_counter()
        attempt_cpu_start = time.process_time()
        proposal_start = time.perf_counter()
        proposal = sampler.propose(method)
        generation_sec = time.perf_counter() - proposal_start
        subset = list(proposal.pop("subset"))
        signature = sampler.signature(subset)
        method_attempt_id = (
            sum(
                1
                for row in attempt_rows
                if str(row.get("sampling_method")) == method
            )
            + 1
        )
        started_at = now_text()
        current_sample_size = len(subset)
        prefix = f"attempt{attempt_id:04d}_{method}_N{current_sample_size}"
        timetable_path = outdir / f"timetable_{prefix}.csv"
        pairwise_path = outdir / f"pairwise_{prefix}.csv"

        print("\n" + "-" * 104)
        print(
            f"Attempt {attempt_id}/{max_attempts} | method={method} | "
            f"accepted {sampler.method_accepted[method]}/{targets[method]} | "
            f"usable total {usable_id}/{required_usable}"
        )
        print(f"subset={signature}")

        build_start = time.perf_counter()
        artifacts = build_timetable_model(
            data,
            cfg,
            subset,
            model_name=f"Restricted_{prefix}",
            stage="sample",
        )
        build_sec = time.perf_counter() - build_start
        solve_info = optimize_with_status_resolution(
            artifacts.model, cfg, stage="sample"
        )
        solver_rec = solver_summary(artifacts.model, has_solution=False)
        provisional_usable = has_usable_solution(
            artifacts.model, keep_time_limit=keep_tl
        )

        packing_value_candidate: Optional[float] = None
        packing_value_source_candidate = ""
        label_solution_source = ""
        packing_value_valid = False
        if provisional_usable:
            obj_val = safe_float(lambda: artifacts.model.ObjVal)
            obj_bound = safe_float(lambda: artifacts.model.ObjBound)
            is_optimal = int(artifacts.model.Status) == int(GRB.OPTIMAL)
            label_solution_source = (
                "optimal_solution" if is_optimal else "feasible_incumbent"
            )
            if is_optimal and obj_val is not None and math.isfinite(obj_val):
                packing_value_candidate = float(obj_val)
                packing_value_source_candidate = "optimal_obj_val"
                packing_value_valid = True
            elif (
                obj_bound is not None
                and math.isfinite(obj_bound)
                and obj_val is not None
                and math.isfinite(obj_val)
                and obj_bound <= obj_val + 1e-5
            ):
                packing_value_candidate = float(obj_bound)
                packing_value_source_candidate = "solver_best_bound"
                packing_value_valid = True

        usable = bool(provisional_usable and packing_value_valid)
        # Recompute the solver summary with the final usable flag so ObjVal is retained only
        # for evidence-pool runs, while failed attempts still keep status/bound diagnostics.
        solver_rec = solver_summary(artifacts.model, usable)

        write_start = time.perf_counter()
        accepted_run_id: Optional[int] = None
        if usable:
            usable_id += 1
            accepted_run_id = usable_id
            sampler.accept(subset, method)
            if as_bool(cfg.get("write_sample_timetables"), True):
                write_csv(pd.DataFrame(timetable_rows(data, artifacts)), timetable_path)
            write_csv(
                pd.DataFrame(
                    pairwise_rows(
                        data,
                        artifacts,
                        directed=not as_bool(
                            cfg.get("sample_pairwise_canonical_only"), True
                        ),
                    )
                ),
                pairwise_path,
            )
        write_sec = time.perf_counter() - write_start
        total_attempt_sec = time.perf_counter() - attempt_wall_start
        attempt_cpu_sec = time.process_time() - attempt_cpu_start

        base_result = {
            "experiment_group": ctx.experiment_group,
            "experiment_tag": ctx.experiment_tag,
            "target_k": len(target_trains),
            "attempt_id": attempt_id,
            "global_run_id": accepted_run_id,
            "sample_size": current_sample_size,
            "sampling_method": method,
            "method_attempt_id": method_attempt_id,
            "method_usable_id": (
                sampler.method_accepted[method] if usable else None
            ),
            "sample_seed": effective_seed,
            "fixed_random_seed": int(use_fixed_seed),
            "train_ids": signature,
            "num_trains": len(subset),
            "num_precedence_labels": len(artifacts.universe) if usable else 0,
            "started_at": started_at,
            "finished_at": now_text(),
            "sample_generation_sec": generation_sec,
            "model_build_sec": build_sec,
            "result_write_sec": write_sec,
            "total_wall_sec": total_attempt_sec,
            "process_cpu_sec": attempt_cpu_sec,
            **proposal,
            **solve_info,
            **solver_rec,
            **artifacts.model_stats,
            "label_solution_source": label_solution_source,
            "packing_value_candidate": packing_value_candidate,
            "packing_value_source_candidate": packing_value_source_candidate,
            "packing_value_valid": int(packing_value_valid),
            "usable_for_evidence": int(usable),
            "replacement_required": int(not usable),
            "timetable_csv": (
                relative_to_project(timetable_path)
                if usable and timetable_path.exists()
                else ""
            ),
            "pairwise_csv": relative_to_project(pairwise_path) if usable else "",
        }
        attempt_rows.append(base_result)
        if usable:
            usable_rows.append(dict(base_result))

        artifacts.model.dispose()
        gc.collect()
        write_csv(pd.DataFrame(attempt_rows), attempts_partial)
        write_csv(pd.DataFrame(usable_rows), usable_partial)
        _save_sampler_state(state_path, sampler.get_state(config_signature))

        print(
            f"[ATTEMPT DONE] status={base_result['status_name']}, usable={usable}, "
            f"packing_source={packing_value_source_candidate or '--'}, "
            f"build={build_sec:.2f}s, solver={base_result['runtime_sec']}, "
            f"write={write_sec:.2f}s, total={total_attempt_sec:.2f}s"
        )
        if usable:
            print(
                f"[USABLE {usable_id}/{required_usable}] method={method} "
                f"({sampler.method_accepted[method]}/{targets[method]})"
            )
        else:
            print(f"[REPLACEMENT] 本 attempt 不计入 R；继续补抽 {method} 类样本。")

    if usable_id != required_usable:
        raise RuntimeError(
            f"小样本阶段异常结束：usable={usable_id}, required={required_usable}"
        )

    attempts = pd.DataFrame(attempt_rows)
    usable_manifest = (
        pd.DataFrame(usable_rows)
        .sort_values("global_run_id")
        .reset_index(drop=True)
    )
    write_csv(attempts, outdir / "sample_attempt_manifest.csv")
    write_csv(usable_manifest, outdir / "sample_manifest.csv")
    plan_columns = [
        "global_run_id",
        "attempt_id",
        "sample_size",
        "sampling_method",
        "method_usable_id",
        "sample_seed",
        "train_ids",
        "status_name",
        "is_optimal",
        "obj_val",
        "obj_bound",
        "mip_gap",
        "label_solution_source",
        "packing_value_candidate",
        "packing_value_source_candidate",
        "runtime_sec",
        "solver_runtime_accumulated_sec",
        "model_build_sec",
        "result_write_sec",
        "process_cpu_sec",
        "total_wall_sec",
    ]
    write_csv(
        usable_manifest[[c for c in plan_columns if c in usable_manifest.columns]],
        outdir / "final_sampling_plan.csv",
    )

    train_count_rows = [
        {
            "train_id": k,
            "entry_rank": idx + 1,
            "entry_time": data.entry_time[k],
            "entry_time_source": data.entry_time_source[k],
            "train_type": data.train_type[k],
            "intermediate_stop_count": data.intermediate_stop_count[k],
            "accepted_sample_count": sampler.train_counts[k],
        }
        for idx, k in enumerate(target_trains)
    ]
    write_csv(
        pd.DataFrame(train_count_rows),
        outdir / "sampling_train_inclusion_counts.csv",
    )
    coverage_rows = [
        {
            "kind": key[0],
            "station": key[1],
            "train_i": key[2],
            "train_j": key[3],
            "subset_coverage_count": sampler.prec_counts[key],
        }
        for key in sampler.target_universe
    ]
    write_csv(
        pd.DataFrame(coverage_rows),
        outdir / "sampling_precedence_coverage.csv",
    )

    method_rows = []
    for method in ("local", "stratified", "coverage"):
        sub = attempts[attempts["sampling_method"] == method]
        used = usable_manifest[usable_manifest["sampling_method"] == method]
        method_rows.append({
            "sampling_method": method,
            "sample_size": int(method_sizes[method]),
            "target_usable": targets[method],
            "attempt_count": len(sub),
            "usable_count": len(used),
            "failed_attempt_count": len(sub) - len(used),
            "optimal_count": int((used["status_name"] == "OPTIMAL").sum()),
            "time_limit_with_incumbent_count": int(
                (used["status_name"] == "TIME_LIMIT").sum()
            ),
            "solver_runtime_sum_sec": float(
                sub["solver_runtime_accumulated_sec"].fillna(0).sum()
            ),
            "solver_runtime_mean_sec": (
                float(sub["solver_runtime_accumulated_sec"].dropna().mean())
                if sub["solver_runtime_accumulated_sec"].notna().any()
                else None
            ),
            "process_cpu_sum_sec": float(sub["process_cpu_sec"].fillna(0).sum()),
            "total_wall_sum_sec": float(sub["total_wall_sec"].fillna(0).sum()),
        })
    write_csv(
        pd.DataFrame(method_rows), outdir / "sampling_method_summary.csv"
    )

    session_wall_sec = time.perf_counter() - stage_start
    session_cpu_sec = time.process_time() - stage_cpu_start
    status_series = attempts["status_name"].astype(str)
    usable_status = usable_manifest["status_name"].astype(str)
    runtime = attempts["solver_runtime_accumulated_sec"].dropna().astype(float)
    summary = {
        "experiment_group": ctx.experiment_group,
        "experiment_tag": ctx.experiment_tag,
        "target_k": len(target_trains),
        "sample_size": sample_size,
        "sample_design": design_label,
        "local_sample_size": int(method_sizes["local"]),
        "stratified_sample_size": int(method_sizes["stratified"]),
        "coverage_sample_size": int(method_sizes["coverage"]),
        "required_usable_samples": required_usable,
        "final_usable_samples": len(usable_manifest),
        "sampling_allocation_mode": str(
            cfg.get("sampling_allocation_mode") or "auto_ratio"
        ),
        "local_target": targets["local"],
        "stratified_target": targets["stratified"],
        "coverage_target": targets["coverage"],
        "total_attempts": len(attempts),
        "replacement_attempts": len(attempts) - len(usable_manifest),
        "local_usable": int(
            (usable_manifest["sampling_method"] == "local").sum()
        ),
        "stratified_usable": int(
            (usable_manifest["sampling_method"] == "stratified").sum()
        ),
        "coverage_usable": int(
            (usable_manifest["sampling_method"] == "coverage").sum()
        ),
        "optimal_count": int((usable_status == "OPTIMAL").sum()),
        "time_limit_with_solution_count": int(
            (usable_status == "TIME_LIMIT").sum()
        ),
        "optimal_value_count_for_packing": int(
            (usable_manifest["packing_value_source_candidate"] == "optimal_obj_val").sum()
        ),
        "best_bound_count_for_packing": int(
            (usable_manifest["packing_value_source_candidate"] == "solver_best_bound").sum()
        ),
        "time_limit_without_solution_attempts": int(
            (
                (status_series == "TIME_LIMIT")
                & (attempts["has_solution"].fillna(0).astype(int) == 0)
            ).sum()
        ),
        "infeasible_attempts": int((status_series == "INFEASIBLE").sum()),
        "other_failed_attempts": int(
            (
                (attempts["usable_for_evidence"].fillna(0).astype(int) == 0)
                & ~status_series.isin(["TIME_LIMIT", "INFEASIBLE"])
            ).sum()
        ),
        "nonoptimal_usable_count": int(
            (usable_manifest["is_optimal"].fillna(0).astype(int) == 0).sum()
        ),
        "solver_runtime_sum_all_attempts_sec": (
            float(runtime.sum()) if not runtime.empty else 0.0
        ),
        "solver_runtime_mean_all_attempts_sec": (
            float(runtime.mean()) if not runtime.empty else None
        ),
        "solver_runtime_median_all_attempts_sec": (
            float(runtime.median()) if not runtime.empty else None
        ),
        "solver_runtime_min_all_attempts_sec": (
            float(runtime.min()) if not runtime.empty else None
        ),
        "solver_runtime_max_all_attempts_sec": (
            float(runtime.max()) if not runtime.empty else None
        ),
        "sample_generation_sum_sec": float(
            attempts["sample_generation_sec"].fillna(0).sum()
        ),
        "model_build_sum_sec": float(attempts["model_build_sec"].fillna(0).sum()),
        "result_write_sum_sec": float(attempts["result_write_sec"].fillna(0).sum()),
        "process_cpu_sum_all_attempts_sec": float(
            attempts["process_cpu_sec"].fillna(0).sum()
        ),
        "attempt_wall_sum_all_attempts_sec": float(
            attempts["total_wall_sec"].fillna(0).sum()
        ),
        "stage_session_wall_sec": session_wall_sec,
        "stage_session_process_cpu_sec": session_cpu_sec,
        # For an uninterrupted run this is the measured stage wall time.  The
        # attempt sum remains available for resumed runs.
        "stage_total_wall_sec": session_wall_sec,
        "stage_process_cpu_sec": session_cpu_sec,
        "stage_started_at": stage_started_at,
        "stage_finished_at": now_text(),
        "fixed_random_seed": int(use_fixed_seed),
        "effective_seed": effective_seed,
        "sampling_config_signature": config_signature,
        "duplicate_proposals_rejected": sampler.duplicate_rejections,
        "stop_group_thresholds": ";".join(
            str(x) for x in sampler.stop_thresholds
        ),
    }
    write_csv(
        pd.DataFrame([summary]), outdir / "sample_solve_summary.csv"
    )
    write_json(
        {
            "summary": summary,
            "sampling_parameters": {
                "method_targets": targets,
                "method_sample_sizes": method_sizes,
                "sample_design": design_label,
                "local_candidate_pool_size": sampler.local_pool_size,
                "entry_time_bins": sampler.entry_bins,
                "stop_count_groups": sampler.stop_groups,
                "stop_group_thresholds": sampler.stop_thresholds,
                "coverage_candidate_count": sampler.coverage_candidate_count,
                "unique_subsets": sampler.unique_subsets,
            },
        },
        outdir / "sampling_completion_record.json",
    )

    for path in (attempts_partial, usable_partial):
        if path.exists():
            path.unlink()
    append_stage_timing(
        ctx,
        {
            "stage": "2_restricted_samples",
            "variant": (
                f"mixed_{targets['local']}_{targets['stratified']}_{targets['coverage']}"
            ),
            "status": "COMPLETED",
            "model_build_sec": summary["model_build_sum_sec"],
            "solver_runtime_sec": summary[
                "solver_runtime_sum_all_attempts_sec"
            ],
            "process_cpu_sec": summary["stage_process_cpu_sec"],
            "postprocess_write_sec": summary["result_write_sum_sec"],
            "total_wall_sec": summary["stage_total_wall_sec"],
        },
    )

    print("\n" + "=" * 88)
    print("小样本阶段完成并已写入文件")
    print("=" * 88)
    print(f"最终可用样本：{len(usable_manifest)}/{required_usable}")
    print(
        "Local / Stratified / Coverage："
        f"{summary['local_usable']} / {summary['stratified_usable']} / "
        f"{summary['coverage_usable']}"
    )
    print(
        f"总 attempt 数：{summary['total_attempts']}；"
        f"补抽数：{summary['replacement_attempts']}"
    )
    print(
        f"OPTIMAL：{summary['optimal_count']}；"
        f"TIME_LIMIT有incumbent：{summary['time_limit_with_solution_count']}"
    )
    print(
        "Packing value来源：optimal objective / best bound = "
        f"{summary['optimal_value_count_for_packing']} / "
        f"{summary['best_bound_count_for_packing']}"
    )
    print(
        "全部 attempt 的 Gurobi Runtime 合计："
        f"{summary['solver_runtime_sum_all_attempts_sec']:.2f}s"
    )
    print(f"小样本阶段总 wall time：{summary['stage_total_wall_sec']:.2f}s")
    print(f"完成记录：{outdir / 'sampling_completion_record.json'}")
    print("=" * 88)
    return usable_manifest


def _run_restricted_instances_parallel(
    cfg: Mapping[str, Any],
    ctx: ExperimentContext,
    data: Optional[HSRData] = None,
) -> pd.DataFrame:
    """Solve independent restricted MILPs concurrently in isolated processes.

    Subsets are generated deterministically in the same local -> stratified ->
    coverage order as the serial implementation.  They are then solved by a
    spawn-based process pool.  Each process owns one Gurobi environment and
    reuses it across its assigned tasks.
    """
    data = data or HSRData(cfg)
    outdir = ctx.sample_dir
    resume = as_bool(cfg.get("sample_resume"), True)
    clear_output = as_bool(cfg.get("sample_clear_output"), False)
    if clear_output and not resume:
        clean_dir(outdir)
    else:
        ensure_dir(outdir)
    ensure_dir(outdir / "solver_logs")
    result_dir = ensure_dir(outdir / "attempt_results")

    if len(ctx.sample_counts) != 1:
        raise ValueError("主实验每次仅支持一个总样本数量")
    method_sizes = configured_method_sample_sizes(cfg)
    sample_size = max(method_sizes.values())  # backward-compatible display field
    required_usable = int(ctx.sample_counts[0])
    targets = _method_targets(cfg, required_usable)
    if sum(targets.values()) != required_usable:
        raise AssertionError("内部错误：三类抽样数量之和与 sample_count 不一致")
    for method, size in method_sizes.items():
        if size >= ctx.target_k:
            raise ValueError(f"sampling_{method}_size 必须严格小于 train_limit")
    design_label = configured_sampling_design_label(cfg, required_usable)

    target_trains = data.select_trains(ctx.target_k)
    write_csv(
        pd.DataFrame({
            "selection_rank": range(1, len(target_trains) + 1),
            "train_id": target_trains,
            "selection_rule": data.train_selection_rule,
            "entry_time": [data.entry_time[k] for k in target_trains],
            "entry_time_source": [data.entry_time_source[k] for k in target_trains],
            "first_arrival_raw": [data.first_arrival[k] for k in target_trains],
            "first_departure": [data.first_dep[k] for k in target_trains],
            "train_type": [data.train_type[k] for k in target_trains],
            "intermediate_stop_count": [
                data.intermediate_stop_count[k] for k in target_trains
            ],
        }),
        outdir / "target_train_set_for_sampling.csv",
    )

    use_fixed_seed = as_bool(cfg.get("random_seed"), True)
    configured_seed = int(as_int(cfg.get("seed"), 42) or 42)
    effective_seed = (
        configured_seed
        if use_fixed_seed
        else random.SystemRandom().randrange(0, 2**63)
    )
    sampler = MixedRestrictedSampler(
        data,
        target_trains,
        method_sizes,
        effective_seed,
        local_pool_size=int(
            as_int(cfg.get("sampling_local_candidate_pool_size"), 90) or 90
        ),
        entry_bins=int(as_int(cfg.get("sampling_entry_time_bins"), 5) or 5),
        stop_groups=int(as_int(cfg.get("sampling_stop_count_groups"), 3) or 3),
        coverage_candidate_count=int(
            as_int(cfg.get("sampling_coverage_candidate_count"), 200) or 200
        ),
        unique_subsets=as_bool(cfg.get("sampling_unique_subsets"), True),
        max_proposal_trials=int(
            as_int(cfg.get("sampling_max_proposal_trials"), 5000) or 5000
        ),
    )
    config_signature = _sampling_signature(
        cfg, target_trains, method_sizes, required_usable, targets
    )

    completion_path = outdir / "sampling_completion_record.json"
    final_manifest_path = outdir / "sample_manifest.csv"
    if resume and final_manifest_path.exists() and completion_path.exists():
        with completion_path.open("r", encoding="utf-8") as f:
            completion_record = json.load(f)
        recorded_signature = completion_record.get("summary", {}).get(
            "sampling_config_signature"
        )
        completed = pd.read_csv(final_manifest_path, encoding="utf-8-sig")
        if len(completed) == required_usable and recorded_signature == config_signature:
            print(
                f"[SKIP] 已有完整且配置一致的 {required_usable} 个并行小样本："
                f"{final_manifest_path}"
            )
            return completed
        if recorded_signature != config_signature:
            raise RuntimeError(
                "现有并行小样本与当前配置签名不一致。请删除当前案例的 "
                "02_samples 目录，或启用 fresh_full_run=TRUE。"
            )

    plan_path = outdir / "parallel_sampling_plan.json"
    plan_csv_path = outdir / "parallel_sampling_plan.csv"
    state_path = outdir / "sampling_state.pkl"
    attempts_partial = outdir / "sample_attempt_manifest.partial.csv"
    usable_partial = outdir / "sample_manifest.partial.csv"

    tasks: List[Dict[str, Any]] = []
    if resume and plan_path.exists():
        with plan_path.open("r", encoding="utf-8") as f:
            plan_record = json.load(f)
        if plan_record.get("sampling_config_signature") != config_signature:
            raise RuntimeError(
                "parallel_sampling_plan.json 与当前 K/Ksub/R 或输入数据不一致；"
                "请删除当前案例的 02_samples 后重跑。"
            )
        tasks = [dict(x) for x in plan_record.get("tasks", [])]
        if not state_path.exists():
            raise RuntimeError(
                "存在并行抽样计划但缺少 sampling_state.pkl，无法可靠续跑；"
                "请删除当前案例的 02_samples 后重跑。"
            )
        sampler.load_state(_load_sampler_state(state_path), config_signature)
        print(f"[RESUME] 已加载并行抽样计划：{len(tasks)} 个 attempt")
    elif resume and (attempts_partial.exists() or usable_partial.exists()):
        raise RuntimeError(
            "检测到旧的串行断点文件，但没有并行抽样计划。为避免混合两种口径，"
            "请删除当前案例的 02_samples 后使用并行版重跑。"
        )

    configured_max_attempts = as_int(cfg.get("sample_max_total_attempts"), None)
    max_attempts = int(
        configured_max_attempts
        if configured_max_attempts is not None and configured_max_attempts > 0
        else max(required_usable * 5, required_usable + 100)
    )
    stage_start = time.perf_counter()
    stage_cpu_start = time.process_time()
    stage_started_at = now_text()
    resource_history: List[Dict[str, Any]] = []
    parallel_wave = 0

    sample_limit = as_float(cfg.get("sample_time_limit"), None)
    limit_text = "UNLIMITED" if sample_limit is None or sample_limit <= 0 else f"{sample_limit:g}s"
    print("\n" + "=" * 104)
    print(
        f"[2] Parallel mixed restricted instances | K={ctx.target_k}, "
        f"design={design_label}, required usable R={required_usable}"
    )
    print(f"composition={targets}; per-instance TimeLimit={limit_text}")
    print(
        f"selection={data.train_selection_rule}, fixed_seed={use_fixed_seed}, "
        f"effective_seed={effective_seed}"
    )
    print(
        "并行策略：多个独立进程同时求解；每个进程独享 Gurobi environment；"
        "每个小样本的线程数与进程数分别由配置控制。"
    )
    print("=" * 104)

    results = _load_parallel_results(result_dir)

    def add_needed_tasks() -> List[Dict[str, Any]]:
        new_tasks: List[Dict[str, Any]] = []
        method_attempt_counts = Counter(
            str(task["sampling_method"]) for task in tasks
        )
        while True:
            method = _next_sampling_method(targets, sampler.method_accepted)
            if method is None:
                break
            attempt_id = len(tasks) + 1
            if attempt_id > max_attempts:
                raise RuntimeError(
                    f"达到 sample_max_total_attempts={max_attempts}，仍未得到 "
                    f"{required_usable} 个可用样本"
                )
            generation_start = time.perf_counter()
            proposal = sampler.propose(method)
            generation_sec = time.perf_counter() - generation_start
            method_attempt_counts[method] += 1
            task = _parallel_task_from_proposal(
                cfg,
                ctx,
                outdir,
                proposal,
                attempt_id,
                method_attempt_counts[method],
                method_sizes[method],
                effective_seed,
                use_fixed_seed,
                generation_sec,
                sampler,
            )
            # Reserve the planned subset immediately. This exactly reproduces
            # the sequential coverage updates when all no-limit solves succeed.
            # After each wave the counters are rebased to successful results.
            sampler.accept(task["subset"], method)
            tasks.append(task)
            new_tasks.append(task)
        if new_tasks:
            _save_parallel_plan(
                plan_path, plan_csv_path, config_signature, tasks
            )
            _save_sampler_state(
                state_path, sampler.get_state(config_signature)
            )
        return new_tasks

    while True:
        # On a new run, this creates exactly R planned subsets. After failures,
        # it creates replacements only for the deficient sampling category.
        add_needed_tasks()
        pending = [
            task for task in tasks
            if int(task["attempt_id"]) not in results
        ]

        if pending:
            parallel_wave += 1
            resources = _sample_parallel_resources(cfg, len(pending))
            resources = {"wave": parallel_wave, **resources}
            resource_history.append(resources)
            print("\n" + "-" * 104)
            print(
                f"Parallel wave {parallel_wave}: pending={len(pending)}, "
                f"workers={resources['parallel_workers']}, "
                f"Threads/solve={resources['threads_per_solve']}, "
                f"physical/logical={resources['physical_cpu_cores']}/"
                f"{resources['logical_cpu_threads']}"
            )
            if resources["available_memory_gb_at_launch"] is not None:
                print(
                    "memory: available="
                    f"{resources['available_memory_gb_at_launch']:.1f} GB, "
                    f"reserve={resources['memory_reserve_gb']:.1f} GB, "
                    f"assumed/worker={resources['memory_per_worker_gb']:.1f} GB"
                )
            print("每个 attempt 的 Gurobi 日志写入 02_samples/solver_logs。")
            print("-" * 104)

            futures = {}
            pool_errors: List[str] = []
            spawn_context = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=int(resources["parallel_workers"]),
                mp_context=spawn_context,
                initializer=_sample_worker_init,
                initargs=(dict(cfg),),
            ) as executor:
                for task in pending:
                    future = executor.submit(_solve_restricted_parallel_task, task)
                    futures[future] = task
                completed_in_wave = 0
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        row = future.result()
                    except BaseException as exc:
                        # This mainly catches a process-level crash before the
                        # worker can persist its own structured error record.
                        now = now_text()
                        row = _restricted_worker_error_result(
                            task, exc, now, time.perf_counter(), time.process_time()
                        )
                        row["worker_exception"] = (
                            f"ProcessPool failure: {type(exc).__name__}: {exc}"
                        )
                        pool_errors.append(row["worker_exception"])
                    attempt_id = int(task["attempt_id"])
                    results[attempt_id] = dict(row)
                    write_json(
                        results[attempt_id],
                        result_dir / f"attempt_result_{attempt_id:04d}.json",
                    )
                    completed_in_wave += 1

                    partial_rows = [
                        results[key] for key in sorted(results)
                    ]
                    write_csv(pd.DataFrame(partial_rows), attempts_partial)
                    partial_usable = [
                        r for r in partial_rows
                        if int(r.get("usable_for_evidence") or 0) == 1
                    ]
                    write_csv(pd.DataFrame(partial_usable), usable_partial)

                    status = row.get("status_name")
                    usable = int(row.get("usable_for_evidence") or 0)
                    runtime = row.get("runtime_sec")
                    print(
                        f"[DONE {completed_in_wave}/{len(pending)}] "
                        f"attempt={attempt_id:04d} "
                        f"method={row.get('sampling_method')} "
                        f"status={status} usable={usable} "
                        f"solver={runtime} wall={row.get('total_wall_sec')}"
                    )

            if pool_errors and not any(
                int(results[int(task["attempt_id"])].get("usable_for_evidence") or 0) == 1
                for task in pending
            ):
                raise RuntimeError(
                    "并行进程池未完成任何可用小样本。常见原因是许可证不允许足够多的"
                    "并发环境、内存不足或子进程初始化失败。第一条错误："
                    + pool_errors[0]
                )

        all_rows = [results[key] for key in sorted(results)]
        _rebase_sampler_to_results(sampler, all_rows)
        _save_sampler_state(state_path, sampler.get_state(config_signature))
        shortages = {
            method: targets[method] - int(sampler.method_accepted[method])
            for method in ("local", "stratified", "coverage")
        }
        if all(value <= 0 for value in shortages.values()):
            break
        print(f"[REPLACEMENT WAVE NEEDED] 各类别尚缺：{shortages}")

    # Select exactly the requested number from each category and assign stable
    # global/method run identifiers in attempt order.
    selected_rows: List[Dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    selected_attempts: set[int] = set()
    all_attempt_rows = [dict(results[key]) for key in sorted(results)]
    for row in all_attempt_rows:
        method = str(row.get("sampling_method"))
        if (
            int(row.get("usable_for_evidence") or 0) == 1
            and selected_counts[method] < targets[method]
        ):
            selected_counts[method] += 1
            selected_attempts.add(int(row["attempt_id"]))
            selected_rows.append(row)

    if len(selected_rows) != required_usable:
        raise RuntimeError(
            f"并行小样本阶段异常结束：selected={len(selected_rows)}, "
            f"required={required_usable}, counts={dict(selected_counts)}"
        )

    method_ids: Counter[str] = Counter()
    selected_by_attempt: Dict[int, Dict[str, Any]] = {}
    for global_run_id, row in enumerate(selected_rows, start=1):
        method = str(row["sampling_method"])
        method_ids[method] += 1
        row["global_run_id"] = global_run_id
        row["method_usable_id"] = method_ids[method]
        row["usable_for_evidence"] = 1
        row["replacement_required"] = 0
        selected_by_attempt[int(row["attempt_id"])] = row

    for row in all_attempt_rows:
        attempt_id = int(row["attempt_id"])
        if attempt_id in selected_by_attempt:
            row.update(selected_by_attempt[attempt_id])
        else:
            row["global_run_id"] = None
            row["method_usable_id"] = None
            row["usable_for_evidence"] = 0
            row["replacement_required"] = 1

    attempts = pd.DataFrame(all_attempt_rows).sort_values("attempt_id").reset_index(drop=True)
    usable_manifest = (
        pd.DataFrame(selected_rows)
        .sort_values("global_run_id")
        .reset_index(drop=True)
    )
    write_csv(attempts, outdir / "sample_attempt_manifest.csv")
    write_csv(usable_manifest, final_manifest_path)

    plan_columns = [
        "global_run_id", "attempt_id", "sample_size", "sampling_method",
        "method_usable_id", "sample_seed", "train_ids", "status_name",
        "is_optimal", "obj_val", "obj_bound", "mip_gap",
        "label_solution_source", "packing_value_candidate",
        "packing_value_source_candidate", "runtime_sec",
        "solver_runtime_accumulated_sec", "model_build_sec",
        "result_write_sec", "process_cpu_sec", "total_wall_sec",
        "parallel_worker_pid", "solver_log",
    ]
    write_csv(
        usable_manifest[[c for c in plan_columns if c in usable_manifest.columns]],
        outdir / "final_sampling_plan.csv",
    )

    # Recompute coverage diagnostics from the final selected evidence pool only.
    _rebase_sampler_to_results(sampler, selected_rows)
    train_count_rows = [
        {
            "train_id": k,
            "entry_rank": idx + 1,
            "entry_time": data.entry_time[k],
            "entry_time_source": data.entry_time_source[k],
            "train_type": data.train_type[k],
            "intermediate_stop_count": data.intermediate_stop_count[k],
            "accepted_sample_count": sampler.train_counts[k],
        }
        for idx, k in enumerate(target_trains)
    ]
    write_csv(pd.DataFrame(train_count_rows), outdir / "sampling_train_inclusion_counts.csv")
    coverage_rows = [
        {
            "kind": key[0],
            "station": key[1],
            "train_i": key[2],
            "train_j": key[3],
            "subset_coverage_count": sampler.prec_counts[key],
        }
        for key in sampler.target_universe
    ]
    write_csv(pd.DataFrame(coverage_rows), outdir / "sampling_precedence_coverage.csv")

    method_rows = []
    for method in ("local", "stratified", "coverage"):
        sub = attempts[attempts["sampling_method"] == method]
        used = usable_manifest[usable_manifest["sampling_method"] == method]
        method_rows.append({
            "sampling_method": method,
            "sample_size": int(method_sizes[method]),
            "target_usable": targets[method],
            "attempt_count": len(sub),
            "usable_count": len(used),
            "failed_attempt_count": len(sub) - len(used),
            "optimal_count": int((used["status_name"] == "OPTIMAL").sum()),
            "time_limit_with_incumbent_count": int((used["status_name"] == "TIME_LIMIT").sum()),
            "solver_runtime_sum_sec": float(sub["solver_runtime_accumulated_sec"].fillna(0).sum()),
            "solver_runtime_mean_sec": (
                float(sub["solver_runtime_accumulated_sec"].dropna().mean())
                if sub["solver_runtime_accumulated_sec"].notna().any()
                else None
            ),
            "process_cpu_sum_sec": float(sub["process_cpu_sec"].fillna(0).sum()),
            "total_wall_sum_sec": float(sub["total_wall_sec"].fillna(0).sum()),
        })
    write_csv(pd.DataFrame(method_rows), outdir / "sampling_method_summary.csv")

    session_wall_sec = time.perf_counter() - stage_start
    parent_cpu_sec = time.process_time() - stage_cpu_start
    worker_cpu_sum = float(attempts["process_cpu_sec"].fillna(0).sum())
    status_series = attempts["status_name"].astype(str)
    usable_status = usable_manifest["status_name"].astype(str)
    runtime = attempts["solver_runtime_accumulated_sec"].dropna().astype(float)
    max_workers_used = max(
        (int(x["parallel_workers"]) for x in resource_history),
        default=0,
    )
    threads_per_solve = int(
        resource_history[0]["threads_per_solve"]
        if resource_history
        else int(as_int(cfg.get("sample_threads_per_solve"), 1) or 1)
    )
    summary = {
        "experiment_group": ctx.experiment_group,
        "experiment_tag": ctx.experiment_tag,
        "target_k": len(target_trains),
        "sample_size": sample_size,
        "sample_design": design_label,
        "local_sample_size": int(method_sizes["local"]),
        "stratified_sample_size": int(method_sizes["stratified"]),
        "coverage_sample_size": int(method_sizes["coverage"]),
        "required_usable_samples": required_usable,
        "final_usable_samples": len(usable_manifest),
        "sampling_allocation_mode": str(cfg.get("sampling_allocation_mode") or "auto_ratio"),
        "local_target": targets["local"],
        "stratified_target": targets["stratified"],
        "coverage_target": targets["coverage"],
        "total_attempts": len(attempts),
        "replacement_attempts": len(attempts) - len(usable_manifest),
        "local_usable": int((usable_manifest["sampling_method"] == "local").sum()),
        "stratified_usable": int((usable_manifest["sampling_method"] == "stratified").sum()),
        "coverage_usable": int((usable_manifest["sampling_method"] == "coverage").sum()),
        "optimal_count": int((usable_status == "OPTIMAL").sum()),
        "time_limit_with_solution_count": int((usable_status == "TIME_LIMIT").sum()),
        "optimal_value_count_for_packing": int((usable_manifest["packing_value_source_candidate"] == "optimal_obj_val").sum()),
        "best_bound_count_for_packing": int((usable_manifest["packing_value_source_candidate"] == "solver_best_bound").sum()),
        "time_limit_without_solution_attempts": int(((status_series == "TIME_LIMIT") & (attempts["has_solution"].fillna(0).astype(int) == 0)).sum()),
        "infeasible_attempts": int((status_series == "INFEASIBLE").sum()),
        "worker_error_attempts": int((status_series == "WORKER_ERROR").sum()),
        "other_failed_attempts": int(((attempts["usable_for_evidence"].fillna(0).astype(int) == 0) & ~status_series.isin(["TIME_LIMIT", "INFEASIBLE", "WORKER_ERROR"])).sum()),
        "nonoptimal_usable_count": int((usable_manifest["is_optimal"].fillna(0).astype(int) == 0).sum()),
        "solver_runtime_sum_all_attempts_sec": float(runtime.sum()) if not runtime.empty else 0.0,
        "solver_runtime_mean_all_attempts_sec": float(runtime.mean()) if not runtime.empty else None,
        "solver_runtime_median_all_attempts_sec": float(runtime.median()) if not runtime.empty else None,
        "solver_runtime_min_all_attempts_sec": float(runtime.min()) if not runtime.empty else None,
        "solver_runtime_max_all_attempts_sec": float(runtime.max()) if not runtime.empty else None,
        "sample_generation_sum_sec": float(attempts["sample_generation_sec"].fillna(0).sum()),
        "model_build_sum_sec": float(attempts["model_build_sec"].fillna(0).sum()),
        "result_write_sum_sec": float(attempts["result_write_sec"].fillna(0).sum()),
        "process_cpu_sum_all_attempts_sec": worker_cpu_sum,
        "attempt_wall_sum_all_attempts_sec": float(attempts["total_wall_sec"].fillna(0).sum()),
        "stage_session_wall_sec": session_wall_sec,
        "stage_session_process_cpu_sec": parent_cpu_sec,
        "stage_parent_process_cpu_sec": parent_cpu_sec,
        "stage_worker_process_cpu_sum_sec": worker_cpu_sum,
        # Table 9 CPU time uses aggregate child + parent process CPU time.
        "stage_process_cpu_sec": worker_cpu_sum + parent_cpu_sec,
        "stage_total_wall_sec": session_wall_sec,
        "stage_started_at": stage_started_at,
        "stage_finished_at": now_text(),
        "fixed_random_seed": int(use_fixed_seed),
        "effective_seed": effective_seed,
        "sampling_config_signature": config_signature,
        "duplicate_proposals_rejected": sampler.duplicate_rejections,
        "stop_group_thresholds": ";".join(str(x) for x in sampler.stop_thresholds),
        "parallel_mode": 1,
        "parallel_waves": parallel_wave,
        "parallel_workers_max": max_workers_used,
        "sample_threads_per_solve": threads_per_solve,
        "sample_time_limit_effective": "UNLIMITED" if sample_limit is None or sample_limit <= 0 else sample_limit,
    }
    write_csv(pd.DataFrame([summary]), outdir / "sample_solve_summary.csv")
    write_json(
        {
            "summary": summary,
            "resource_history": resource_history,
            "sampling_parameters": {
                "method_targets": targets,
                "method_sample_sizes": method_sizes,
                "sample_design": design_label,
                "local_candidate_pool_size": sampler.local_pool_size,
                "entry_time_bins": sampler.entry_bins,
                "stop_count_groups": sampler.stop_groups,
                "stop_group_thresholds": sampler.stop_thresholds,
                "coverage_candidate_count": sampler.coverage_candidate_count,
                "unique_subsets": sampler.unique_subsets,
                "parallel": True,
            },
        },
        completion_path,
    )

    for path in (attempts_partial, usable_partial):
        if path.exists():
            path.unlink()
    append_stage_timing(
        ctx,
        {
            "stage": "2_restricted_samples",
            "variant": f"parallel_mixed_{targets['local']}_{targets['stratified']}_{targets['coverage']}",
            "status": "COMPLETED",
            "model_build_sec": summary["model_build_sum_sec"],
            "solver_runtime_sec": summary["solver_runtime_sum_all_attempts_sec"],
            "process_cpu_sec": summary["stage_process_cpu_sec"],
            "postprocess_write_sec": summary["result_write_sum_sec"],
            "total_wall_sec": summary["stage_total_wall_sec"],
            "parallel_workers": max_workers_used,
            "threads_per_solve": threads_per_solve,
        },
    )

    print("\n" + "=" * 88)
    print("并行小样本阶段完成并已写入文件")
    print("=" * 88)
    print(f"最终可用样本：{len(usable_manifest)}/{required_usable}")
    print(
        "Local / Stratified / Coverage："
        f"{summary['local_usable']} / {summary['stratified_usable']} / "
        f"{summary['coverage_usable']}"
    )
    print(
        f"总 attempt 数：{summary['total_attempts']}；"
        f"补抽数：{summary['replacement_attempts']}"
    )
    print(
        f"OPTIMAL：{summary['optimal_count']}；"
        f"非最优但可用：{summary['nonoptimal_usable_count']}"
    )
    print(
        f"并行 workers 最大值：{max_workers_used}；"
        f"每个 solve 的 Threads：{threads_per_solve}"
    )
    print(
        "全部 attempt 的 Gurobi Runtime 合计："
        f"{summary['solver_runtime_sum_all_attempts_sec']:.2f}s"
    )
    print(f"小样本阶段实际 wall time：{summary['stage_total_wall_sec']:.2f}s")
    print(f"完成记录：{completion_path}")
    print("=" * 88)
    return usable_manifest


def run_restricted_instances(
    cfg: Mapping[str, Any],
    ctx: ExperimentContext,
    data: Optional[HSRData] = None,
) -> pd.DataFrame:
    """Dispatch to the parallel or legacy serial restricted-instance runner."""
    if as_bool(cfg.get("sample_parallel"), True):
        return _run_restricted_instances_parallel(cfg, ctx, data)
    return _run_restricted_instances_serial(cfg, ctx, data)


# =============================================================================
# 样本读取和观测聚合
# =============================================================================

def load_sample_manifest(ctx: ExperimentContext) -> pd.DataFrame:
    if not ctx.sample_manifest.exists():
        raise FileNotFoundError(f"未找到小样本 manifest：{ctx.sample_manifest}")
    df = pd.read_csv(ctx.sample_manifest, encoding="utf-8-sig")
    expected_total = int(ctx.sample_counts[0])
    if len(df) != expected_total:
        raise RuntimeError(f"可用小样本数量应为{expected_total}，实际为{len(df)}")
    method_sizes = dict(ctx.extra.get("method_sample_sizes", {}))
    method_counts = dict(ctx.extra.get("method_sample_counts", {}))
    if method_sizes and method_counts:
        for method in SAMPLING_METHODS:
            sub = df[df["sampling_method"].astype(str) == method]
            expected_count = int(method_counts[method])
            expected_size = int(method_sizes[method])
            if len(sub) != expected_count:
                raise RuntimeError(
                    f"{method} 可用样本应为 {expected_count}，实际为 {len(sub)}"
                )
            if "sample_size" in sub.columns and not sub.empty:
                actual_sizes = sorted(set(pd.to_numeric(sub["sample_size"], errors="coerce").dropna().astype(int)))
                if actual_sizes != [expected_size]:
                    raise RuntimeError(
                        f"{method} sample_size 应为 {expected_size}，实际为 {actual_sizes}"
                    )
    return df


def _resolve_saved_path(text: Any) -> Path:
    path = Path(str(text))
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def _paper_output_mode(cfg: Mapping[str, Any]) -> bool:
    return as_bool(cfg.get("paper_output_mode"), False)


def _cfg_flag(cfg: Mapping[str, Any], key: str, *, paper_default: bool, normal_default: bool) -> bool:
    if key in cfg and cfg.get(key) not in {None, ""}:
        return as_bool(cfg.get(key), paper_default if _paper_output_mode(cfg) else normal_default)
    return paper_default if _paper_output_mode(cfg) else normal_default


def _observation_cache_path(ctx: ExperimentContext) -> Path:
    return ctx.vote_dir / "observation_counts_cache.pkl"


def _vote_assignment_cache_path(ctx: ExperimentContext) -> Path:
    return ctx.vote_dir / "vote_priors_fullfix_cache.pkl"


def _inverse_component_cache_path(ctx: ExperimentContext, alpha: float) -> Path:
    return ctx.inverse_dir / alpha_tag(alpha) / "learned_core_components_cache.pkl"


def _save_observation_cache(path: Path, trains: Sequence[str], universe: Sequence[PairKey], counts: Mapping[PairKey, Mapping[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_arr = array("I"); n1_arr = array("I")
    for key in universe:
        rec = counts.get(key, {"N": 0, "N1": 0})
        n_arr.append(int(rec.get("N", 0))); n1_arr.append(int(rec.get("N1", 0)))
    payload = {"version":1,"trains":tuple(str(k) for k in trains),"universe_size":len(universe),"N":n_arr,"N1":n1_arr}
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("wb") as f: pickle.dump(payload,f,protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _load_observation_cache(path: Path, trains: Sequence[str], universe: Sequence[PairKey]) -> Optional[Dict[PairKey, Dict[str, int]]]:
    if not path.exists(): return None
    try:
        with path.open("rb") as f: payload=pickle.load(f)
        if int(payload.get("version",0))!=1 or tuple(str(k) for k in trains)!=tuple(payload.get("trains",())) or int(payload.get("universe_size",-1))!=len(universe): return None
        n_arr=payload.get("N"); n1_arr=payload.get("N1")
        if len(n_arr)!=len(universe) or len(n1_arr)!=len(universe): return None
        out={}
        for i,key in enumerate(universe):
            n=int(n_arr[i])
            if n>0: out[key]={"N":n,"N1":int(n1_arr[i])}
        return out
    except Exception: return None


def _save_full_assignment_cache(path: Path, trains: Sequence[str], universe: Sequence[PairKey], priors: Mapping[PairKey,int]) -> None:
    values=array("b",(int(priors[key]) for key in universe))
    payload={"version":1,"trains":tuple(str(k) for k in trains),"universe_size":len(universe),"values":values}
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("wb") as f: pickle.dump(payload,f,protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _load_full_assignment_cache(path: Path, trains: Sequence[str], universe: Sequence[PairKey]) -> Optional[Dict[PairKey,int]]:
    if not path.exists(): return None
    try:
        with path.open("rb") as f: payload=pickle.load(f)
        if int(payload.get("version",0))!=1 or tuple(str(k) for k in trains)!=tuple(payload.get("trains",())) or int(payload.get("universe_size",-1))!=len(universe): return None
        vals=payload.get("values")
        if len(vals)!=len(universe): return None
        return {key:int(vals[i]) for i,key in enumerate(universe)}
    except Exception: return None


def _save_component_assignment_cache(path: Path, component_values: Mapping[int,int]) -> None:
    ids=sorted(int(c) for c in component_values)
    payload={"version":1,"component_ids":array("I",ids),"values":array("b",(int(component_values[c]) for c in ids))}
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("wb") as f: pickle.dump(payload,f,protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _load_component_assignment_cache(path: Path) -> Optional[Dict[int,int]]:
    if not path.exists(): return None
    try:
        with path.open("rb") as f: payload=pickle.load(f)
        if int(payload.get("version",0))!=1: return None
        ids=payload.get("component_ids"); vals=payload.get("values")
        if len(ids)!=len(vals): return None
        return {int(ids[i]):int(vals[i]) for i in range(len(ids))}
    except Exception: return None


def aggregate_observations(
    data: HSRData,
    trains: Sequence[str],
    manifest: pd.DataFrame,
) -> Dict[PairKey, Dict[str, int]]:
    """Aggregate canonical precedence labels with legacy-file compatibility.

    New sample files contain one canonical row per precedence.  Older files may
    contain both directions; in that case ``is_canonical_direction`` is used
    to read only one row and avoid doubling I/O and memory.
    """
    train_set = set(trains)
    counts: Dict[PairKey, Dict[str, int]] = defaultdict(lambda: {"N": 0, "N1": 0})
    for row in manifest.itertuples(index=False):
        path = _resolve_saved_path(getattr(row, "pairwise_csv"))
        if not path.exists():
            raise FileNotFoundError(f"manifest 指向的 pairwise 文件不存在：{path}")
        header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
        columns = set(header.columns)
        canonical_schema = {
            "kind", "station", "train_i", "train_j", "value",
            "is_canonical_direction",
        }
        if canonical_schema.issubset(columns):
            df = pd.read_csv(
                path,
                encoding="utf-8-sig",
                usecols=list(canonical_schema),
            )
            df = df[df["is_canonical_direction"].astype(int) == 1]
            file_labels: Dict[PairKey, int] = {}
            for rec in df.itertuples(index=False):
                kind = str(rec.kind).strip().lower()
                station = str(rec.station)
                i = str(rec.train_i)
                j = str(rec.train_j)
                if (
                    i not in train_set
                    or j not in train_set
                    or i == j
                    or kind not in {"arr", "dep"}
                ):
                    continue
                ci, cj = data.canonical_pair(i, j)
                value = int(round(float(rec.value)))
                if (i, j) != (ci, cj):
                    value = 1 - value
                key = (kind, station, ci, cj)
                if key in file_labels and file_labels[key] != value:
                    raise ValueError(
                        f"同一小样本内canonical precedence不一致：{path}, key={key}"
                    )
                file_labels[key] = value
        else:
            df = pd.read_csv(path, encoding="utf-8-sig")
            file_labels = {}
            for rec in df.itertuples(index=False):
                rec_dict = rec._asdict()
                kind = str(
                    rec_dict.get("kind", rec_dict.get("变量类型", ""))
                ).strip().lower()
                station = str(rec_dict.get("station", rec_dict.get("车站", "")))
                i = str(rec_dict.get("train_i", rec_dict.get("车次1", "")))
                j = str(rec_dict.get("train_j", rec_dict.get("车次2", "")))
                if i not in train_set or j not in train_set or i == j or kind not in {"arr", "dep"}:
                    continue
                ci, cj = data.canonical_pair(i, j)
                raw_value = rec_dict.get("value", rec_dict.get("取值"))
                if raw_value is None or pd.isna(raw_value):
                    continue
                value = int(round(float(raw_value)))
                if (i, j) != (ci, cj):
                    value = 1 - value
                key = (kind, station, ci, cj)
                if key in file_labels and file_labels[key] != value:
                    raise ValueError(f"同一小样本内双向precedence不一致：{path}, key={key}")
                file_labels[key] = value
        for key, value in file_labels.items():
            counts[key]["N"] += 1
            counts[key]["N1"] += value
    return dict(counts)


def compute_evidence_diagnostics(
    data: HSRData,
    trains: Sequence[str],
    counts: Mapping[PairKey, Mapping[str, int]],
    restricted_count: int,
    *,
    include_details: bool = True,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Compute manuscript evidence metrics; detailed rows are optional."""
    universe = data.event_universe(trains)
    universe_size = len(universe)
    threshold = int(math.ceil(0.025 * restricted_count))
    threshold = max(1, threshold)
    rows: List[Dict[str, Any]] = []
    n_values: List[int] = []
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for kind, station, i, j in universe:
        rec = counts.get((kind, station, i, j), {"N": 0, "N1": 0})
        n = int(rec.get("N", 0))
        n1 = int(rec.get("N1", 0))
        n0 = n - n1
        n_values.append(n)
        groups[(station, kind)].append(n)
        if include_details:
            rows.append({
                "kind": kind,
                "station": station,
                "train_i": i,
                "train_j": j,
                "N": n,
                "N1": n1,
                "N0": n0,
                "repeated_support_threshold": threshold,
                "is_observed": int(n > 0),
                "has_repeated_support": int(n >= threshold),
            })

    if universe_size == 0:
        obs_ratio = relcov = effcov = p10 = 0.0
        total_observations = 0
        group_count = 0
    else:
        observed = sum(n > 0 for n in n_values)
        repeated = sum(n >= threshold for n in n_values)
        obs_ratio = observed / universe_size
        relcov = repeated / universe_size
        total_observations = sum(n_values)
        if total_observations > 0:
            entropy = 0.0
            for n in n_values:
                if n <= 0:
                    continue
                mu = n / total_observations
                entropy -= mu * math.log(mu)
            effcov = math.exp(entropy) / universe_size
        else:
            effcov = 0.0
        group_fractions = [
            sum(n >= threshold for n in values) / len(values)
            for values in groups.values()
            if values
        ]
        group_count = len(group_fractions)
        p10 = (
            float(pd.Series(group_fractions, dtype=float).quantile(0.10))
            if group_fractions
            else 0.0
        )

    diagnostics = {
        "target_k": len(trains),
        "restricted_count": restricted_count,
        "precedence_universe_size": universe_size,
        "observed_precedence_count": sum(n > 0 for n in n_values),
        "observation_ratio": obs_ratio,
        "repeated_support_threshold": threshold,
        "repeated_supported_precedence_count": sum(
            n >= threshold for n in n_values
        ),
        "RelCov_2p5": relcov,
        "EffCov": effcov,
        "P10_StrCov_2p5": p10,
        "station_event_group_count": group_count,
        "total_observation_mass": total_observations,
        "mean_observations_per_index": (
            total_observations / universe_size if universe_size else 0.0
        ),
        "median_observations_per_index": (
            float(pd.Series(n_values, dtype=float).median()) if n_values else 0.0
        ),
        "min_observations_per_index": min(n_values) if n_values else 0,
        "max_observations_per_index": max(n_values) if n_values else 0,
    }
    return diagnostics, pd.DataFrame(rows)


# =============================================================================
# Stage 3: Vote
# =============================================================================

def fallback_order_value(data: HSRData, i: str, j: str, rule: str) -> int:
    resolved = normalize_pair_resolution_rule(rule, "pair_resolution_rule")
    if resolved == "first_departure":
        key_i = (data.first_dep[i], data.global_train_index[i])
        key_j = (data.first_dep[j], data.global_train_index[j])
    elif resolved == "entry_time":
        key_i = (data.entry_time[i], data.global_train_index[i])
        key_j = (data.entry_time[j], data.global_train_index[j])
    elif resolved == "file_order":
        key_i = (data.global_train_index[i],)
        key_j = (data.global_train_index[j],)
    elif resolved == "train_id":
        key_i = (train_id_sort_key(i), data.global_train_index[i])
        key_j = (train_id_sort_key(j), data.global_train_index[j])
    else:
        raise AssertionError(resolved)
    return 1 if key_i < key_j else 0


def run_vote(
    cfg: Mapping[str, Any],
    ctx: ExperimentContext,
    data: Optional[HSRData] = None,
) -> pd.DataFrame:
    data = data or HSRData(cfg)
    outdir = ensure_dir(ctx.vote_dir)
    stage_start = time.perf_counter()
    stage_cpu_start = time.process_time()
    trains = data.select_trains(ctx.target_k)
    manifest = load_sample_manifest(ctx)
    paper_mode = _paper_output_mode(cfg)
    write_evidence_details = _cfg_flag(cfg, "write_evidence_index_details", paper_default=False, normal_default=True)
    cache_observations = _cfg_flag(cfg, "cache_observation_counts", paper_default=True, normal_default=False)
    write_vote_details = _cfg_flag(cfg, "write_vote_precedence_details", paper_default=False, normal_default=True)
    write_vote_assignment_csv = _cfg_flag(cfg, "write_vote_assignment_csv", paper_default=False, normal_default=True)
    need_vote_rows = write_vote_assignment_csv

    aggregation_start = time.perf_counter()
    counts = aggregate_observations(data, trains, manifest)
    aggregation_sec = time.perf_counter() - aggregation_start
    universe = data.event_universe(trains)
    diagnostics, observation_df = compute_evidence_diagnostics(
        data, trains, counts, len(manifest), include_details=write_evidence_details
    )
    if write_evidence_details:
        write_csv(observation_df, outdir / "evidence_observation_counts.csv")
    elif (outdir / "evidence_observation_counts.csv").exists():
        (outdir / "evidence_observation_counts.csv").unlink()
    write_csv(pd.DataFrame([diagnostics]), outdir / "evidence_diagnostics.csv")
    if cache_observations:
        _save_observation_cache(_observation_cache_path(ctx), trains, universe, counts)

    tie_rule = normalize_pair_resolution_rule(
        cfg.get("vote_tie_rule", "first_departure"), "vote_tie_rule"
    )
    unobserved_rule = normalize_pair_resolution_rule(
        cfg.get("vote_unobserved_rule", "first_departure"),
        "vote_unobserved_rule",
    )

    rows: List[Dict[str, Any]] = []
    tie_count = 0
    unobserved_count = 0
    priors_dict: Dict[PairKey, int] = {}
    for key in universe:
        kind, station, i, j = key
        rec = counts.get(key, {"N": 0, "N1": 0})
        n, n1 = int(rec["N"]), int(rec["N1"])
        n0 = n - n1
        if n == 0:
            value = fallback_order_value(data, i, j, unobserved_rule)
            reason = f"unobserved_{unobserved_rule}"
            unobserved_count += 1
        elif n1 == n0:
            value = fallback_order_value(data, i, j, tie_rule)
            reason = f"tie_{tie_rule}"
            tie_count += 1
        else:
            value = int(n1 > n0)
            reason = "majority_vote"
        priors_dict[key] = value
        if need_vote_rows:
            vote_row = {"kind": kind, "station": station, "train_i": i, "train_j": j, "value": value}
            if write_vote_details:
                vote_row.update({
                    "N": n, "N1": n1, "N0": n0,
                    "vote_share_i_before_j": (n1 / n) if n else None,
                    "decision_reason": reason,
                })
            rows.append(vote_row)

    write_start = time.perf_counter()
    priors = pd.DataFrame(rows) if need_vote_rows else pd.DataFrame()
    if write_vote_assignment_csv:
        write_csv(priors, outdir / "vote_priors_fullfix.csv")
    elif (outdir / "vote_priors_fullfix.csv").exists():
        (outdir / "vote_priors_fullfix.csv").unlink()
    _save_full_assignment_cache(_vote_assignment_cache_path(ctx), trains, universe, priors_dict)
    full_vote_audit = _cfg_flag(cfg, "vote_full_c1_triplet_audit", paper_default=False, normal_default=True)
    audit = audit_precedence_assignment(data, trains, priors_dict, count_c1_triplets=full_vote_audit)
    write_sec = time.perf_counter() - write_start
    total_sec = time.perf_counter() - stage_start
    cpu_sec = time.process_time() - stage_cpu_start
    summary = {
        "experiment_group": ctx.experiment_group,
        "experiment_tag": ctx.experiment_tag,
        "target_k": len(trains),
        "universe_size": len(universe),
        "tie_count": tie_count,
        "unobserved_count": unobserved_count,
        "all_precedence_fixed": 1,
        "paper_output_mode": int(paper_mode),
        "evidence_index_details_written": int(write_evidence_details),
        "vote_precedence_details_written": int(write_vote_details),
        "vote_assignment_csv_written": int(write_vote_assignment_csv),
        "vote_assignment_cache": relative_to_project(_vote_assignment_cache_path(ctx)),
        "observation_cache_written": int(cache_observations),
        "tie_rule": tie_rule,
        "unobserved_rule": unobserved_rule,
        "observation_aggregation_sec": aggregation_sec,
        "result_write_and_audit_sec": write_sec,
        "total_wall_sec": total_sec,
        "process_cpu_sec": cpu_sec,
        **diagnostics,
        **audit,
    }
    write_csv(pd.DataFrame([summary]), outdir / "vote_summary.csv")
    append_stage_timing(
        ctx,
        {
            "stage": "3_vote",
            "variant": "VOTE",
            "status": "COMPLETED",
            "model_build_sec": None,
            "solver_runtime_sec": 0.0,
            "process_cpu_sec": cpu_sec,
            "postprocess_write_sec": aggregation_sec + write_sec,
            "total_wall_sec": total_sec,
        },
    )
    print(
        f"[3] Vote完成：|U|={len(universe)}, tie={tie_count}, "
        f"unobserved={unobserved_count}, ObsRatio={diagnostics['observation_ratio']:.4f}, "
        f"time={total_sec:.2f}s"
    )
    return priors


# =============================================================================
# Stage 4: Packing LP
# =============================================================================

def run_packing(
    cfg: Mapping[str, Any],
    ctx: ExperimentContext,
    data: Optional[HSRData] = None,
) -> Dict[str, Any]:
    """Solve the packing primal/dual using exact values or valid best bounds.

    For an optimal restricted solve, ``ObjVal`` is used.  For a time-limited
    solve with a feasible incumbent, the incumbent still supplies precedence
    labels, but ``ObjBound`` is used in the packing LP.  This is the convention
    required for the packing value to remain a valid lower bound.
    """
    data = data or HSRData(cfg)
    outdir = ensure_dir(ctx.packing_dir)
    stage_start = time.perf_counter()
    stage_cpu_start = time.process_time()
    trains = data.select_trains(ctx.target_k)
    manifest = load_sample_manifest(ctx)

    sample_sets: Dict[int, List[str]] = {}
    values: Dict[int, float] = {}
    source_by_run: Dict[int, str] = {}
    used_rows: List[Dict[str, Any]] = []
    for rec in manifest.to_dict("records"):
        if int(rec.get("has_solution", 0) or 0) != 1:
            continue
        run_id = int(rec["global_run_id"])
        is_optimal = int(rec.get("is_optimal", 0) or 0) == 1
        obj_val = rec.get("obj_val")
        obj_bound = rec.get("obj_bound")
        value: Optional[float] = None
        source = ""
        if is_optimal and obj_val is not None and not pd.isna(obj_val):
            value = float(obj_val)
            source = "optimal_obj_val"
        elif obj_bound is not None and not pd.isna(obj_bound):
            value = float(obj_bound)
            source = "solver_best_bound"
        if value is None or not math.isfinite(value):
            raise RuntimeError(
                f"Restricted run {run_id} 没有可用于 Packing-LP 的有效最优值或 best bound"
            )
        if (
            not is_optimal
            and obj_val is not None
            and not pd.isna(obj_val)
            and value > float(obj_val) + 1e-5
        ):
            raise RuntimeError(
                f"Restricted run {run_id} 的 best bound={value} 高于 incumbent={obj_val}，"
                "无法作为经审计的最小化下界使用"
            )
        subset = [x for x in str(rec["train_ids"]).split(";") if x]
        sample_sets[run_id] = subset
        values[run_id] = value
        source_by_run[run_id] = source
        used_rows.append({
            "global_run_id": run_id,
            "attempt_id": rec.get("attempt_id"),
            "sampling_method": rec.get("sampling_method"),
            "status_name": rec.get("status_name"),
            "is_optimal": int(is_optimal),
            "label_solution_source": rec.get("label_solution_source"),
            "obj_val": obj_val,
            "obj_bound": obj_bound,
            "mip_gap": rec.get("mip_gap"),
            "runtime_sec": rec.get("runtime_sec"),
            "packing_value_used": value,
            "packing_value_source": source,
            "train_ids": rec.get("train_ids"),
        })

    if not sample_sets:
        raise RuntimeError("没有可用于 Packing-LP 的受限实例")
    if len(sample_sets) != len(manifest):
        raise RuntimeError(
            f"Packing-LP 应使用全部 {len(manifest)} 个可用样本，实际仅 {len(sample_sets)} 个"
        )
    write_csv(pd.DataFrame(used_rows), outdir / "packing_runs_used.csv")

    primal_build_start = time.perf_counter()
    primal = Model(f"PackingPrimal_{ctx.experiment_tag}")
    configure_model(primal, cfg, "packing")
    lambdas = {
        r: primal.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"lambda_{r}")
        for r in sample_sets
    }
    primal.setObjective(
        quicksum(values[r] * lambdas[r] for r in sample_sets), GRB.MAXIMIZE
    )
    for train in trains:
        relevant = [r for r, subset in sample_sets.items() if train in subset]
        primal.addConstr(
            quicksum(lambdas[r] for r in relevant) <= 1.0,
            name=f"cover_{train}",
        )
    primal_build_sec = time.perf_counter() - primal_build_start
    primal_opt = optimize_with_status_resolution(primal, cfg, stage="packing")
    if int(primal.Status) != int(GRB.OPTIMAL):
        raise RuntimeError(
            f"Packing primal 必须求到 OPTIMAL，当前={status_name(primal.Status)}"
        )
    lb_primal = float(primal.ObjVal)

    selected_rows = [
        {
            "global_run_id": r,
            "lambda": float(var.X),
            "restricted_value": values[r],
            "packing_value_source": source_by_run[r],
            "weighted_value": float(var.X) * values[r],
            "train_ids": ";".join(sample_sets[r]),
        }
        for r, var in lambdas.items()
        if float(var.X) > 1e-9
    ]
    write_csv(
        pd.DataFrame(selected_rows), outdir / "packing_selected_runs.csv"
    )

    dual_build_start = time.perf_counter()
    dual = Model(f"PackingDual_{ctx.experiment_tag}")
    configure_model(dual, cfg, "packing")
    eta = {
        k: dual.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"eta_{k}")
        for k in trains
    }
    dual.setObjective(quicksum(eta[k] for k in trains), GRB.MINIMIZE)
    for run_id, subset in sample_sets.items():
        dual.addConstr(
            quicksum(eta[k] for k in subset) >= values[run_id],
            name=f"sample_{run_id}",
        )
    dual_build_sec = time.perf_counter() - dual_build_start
    dual_opt = optimize_with_status_resolution(dual, cfg, stage="packing")
    if int(dual.Status) != int(GRB.OPTIMAL):
        raise RuntimeError(
            f"Packing dual 必须求到 OPTIMAL，当前={status_name(dual.Status)}"
        )
    lb_dual = float(dual.ObjVal)

    duality_diff = abs(lb_primal - lb_dual)
    if duality_diff > max(
        1e-6, 1e-9 * max(1.0, abs(lb_primal), abs(lb_dual))
    ):
        raise RuntimeError(
            f"Packing primal/dual 不一致：{lb_primal} vs {lb_dual}"
        )

    prices = pd.DataFrame([
        {
            "train_id": train,
            "eta": float(eta[train].X),
            "entry_rank": idx + 1,
            "entry_time": data.entry_time[train],
        }
        for idx, train in enumerate(trains)
    ])
    write_csv(prices, outdir / "packing_train_prices.csv")

    total_sec = time.perf_counter() - stage_start
    cpu_sec = time.process_time() - stage_cpu_start
    eta_series = prices["eta"].astype(float)
    num_exact = sum(source == "optimal_obj_val" for source in source_by_run.values())
    num_bound = sum(source == "solver_best_bound" for source in source_by_run.values())
    summary = {
        "experiment_group": ctx.experiment_group,
        "experiment_tag": ctx.experiment_tag,
        "target_k": len(trains),
        "num_runs_used": len(sample_sets),
        "num_optimal_values_used": num_exact,
        "num_best_bounds_used": num_bound,
        "num_nonoptimal_runs_used": num_bound,
        # Valid because every coefficient is either an exact restricted optimum
        # or a solver-certified lower bound for that restricted minimization.
        "packing_bound_certified": 1,
        "packing_uses_only_exact_optima": int(num_bound == 0),
        "packing_value_source": "optimal_obj_val_or_solver_best_bound",
        "packing_lb_primal": lb_primal,
        "packing_lb_dual": lb_dual,
        "duality_abs_diff": duality_diff,
        "primal_status": status_name(primal.Status),
        "dual_status": status_name(dual.Status),
        "primal_build_sec": primal_build_sec,
        "dual_build_sec": dual_build_sec,
        "primal_runtime_sec": safe_float(lambda: primal.Runtime),
        "dual_runtime_sec": safe_float(lambda: dual.Runtime),
        "primal_solver_runtime_accumulated_sec": primal_opt[
            "solver_runtime_accumulated_sec"
        ],
        "dual_solver_runtime_accumulated_sec": dual_opt[
            "solver_runtime_accumulated_sec"
        ],
        "nonzero_eta_count": int((eta_series > 1e-9).sum()),
        "eta_mean": float(eta_series.mean()),
        "eta_median": float(eta_series.median()),
        "eta_max": float(eta_series.max()),
        "eta_p10": float(eta_series.quantile(0.10)),
        "eta_p90": float(eta_series.quantile(0.90)),
        "total_wall_sec": total_sec,
        "process_cpu_sec": cpu_sec,
    }
    write_csv(pd.DataFrame([summary]), outdir / "packing_summary.csv")
    append_stage_timing(
        ctx,
        {
            "stage": "4_packing",
            "variant": "primal_and_dual",
            "status": "OPTIMAL",
            "model_build_sec": primal_build_sec + dual_build_sec,
            "solver_runtime_sec": (
                float(summary["primal_runtime_sec"] or 0.0)
                + float(summary["dual_runtime_sec"] or 0.0)
            ),
            "solver_runtime_accumulated_sec": (
                float(summary["primal_solver_runtime_accumulated_sec"] or 0.0)
                + float(summary["dual_solver_runtime_accumulated_sec"] or 0.0)
            ),
            "process_cpu_sec": cpu_sec,
            "postprocess_write_sec": None,
            "total_wall_sec": total_sec,
        },
    )
    primal.dispose()
    dual.dispose()
    gc.collect()
    print(
        f"[4] Packing LP完成：LB={lb_primal:.6f}, runs={len(sample_sets)}, "
        f"exact/bound={num_exact}/{num_bound}, time={total_sec:.2f}s"
    )
    return summary


def load_train_prices(ctx: ExperimentContext) -> Dict[str, float]:
    path = ctx.packing_dir / "packing_train_prices.csv"
    if not path.exists():
        raise FileNotFoundError(f"未找到packing dual train prices：{path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    return {str(row.train_id): float(row.eta) for row in df.itertuples(index=False)}


# =============================================================================
# Stage 5 structural quotient partition (no precedence fixing)
# =============================================================================

class _CoreDSU:
    """Small union--find helper used to collapse exact precedence equalities."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


@dataclass
class PrecedenceCorePartition:
    forced: Dict[PairKey, int]
    component_by_key: Dict[PairKey, int]
    component_members: Dict[int, List[PairKey]]
    summary: Dict[str, Any]


def _event_key(kind: str, station: str, train: str) -> Tuple[str, str, str]:
    return (str(kind), str(station), str(train))



def build_inverse_structural_partition(
    cfg: Mapping[str, Any],
    ctx: ExperimentContext,
    data: HSRData,
    trains: Sequence[str],
) -> PrecedenceCorePartition:
    """Build the Stage-5 quotient partition in memory with no fixed precedence.

    This is *not* Stage 4B preprocessing. No precedence direction is fixed and
    no time-window propagation is performed. The only variable aggregation is
    exact equality substitution already implied by the Full inverse model.

    C2 and C3 are handled as two separate structural families:

    1) C2 segment-order equalities identify the corresponding upstream-departure
       and downstream-arrival precedence variables;
    2) the equality case of C3 identifies arrival/departure precedence only when
       neither train stops at the station. The remaining C3 relations stay as
       explicit inequalities in the inverse model.

    These substitutions reduce the number of binary variables but do not fix any
    precedence to 0 or 1. Structural ablation is intentionally handled outside
    the main-experiment configuration.
    """
    universe = data.event_universe(trains)
    universe_size = len(universe)
    if universe_size == 0:
        raise RuntimeError("目标实例没有 precedence，无法构造 inverse partition")

    key_to_index = {key: idx for idx, key in enumerate(universe)}
    dsu = _CoreDSU(universe_size)
    c2_equality_count = 0
    c3_equality_count = 0

    # C2: exact segment-order equality substitution.
    # This block is independent of the C3 block below.
    for upstream, downstream in data.segments:
        present = [
            k
            for k in trains
            if data.has_event(k, upstream, "dep")
            and data.has_event(k, downstream, "arr")
            and upstream in data.path_by_train[k]
            and downstream in data.path_by_train[k]
            and data.path_by_train[k].index(downstream)
            == data.path_by_train[k].index(upstream) + 1
        ]
        for p0 in range(len(present)):
            for q0 in range(p0 + 1, len(present)):
                i, j = data.canonical_pair(present[p0], present[q0])
                dep_idx = key_to_index.get(("dep", upstream, i, j))
                arr_idx = key_to_index.get(("arr", downstream, i, j))
                if dep_idx is not None and arr_idx is not None:
                    dsu.union(dep_idx, arr_idx)
                    c2_equality_count += 1

    # C3 equality case: exact substitution when neither train stops.
    # This is distinct from C2 and from the explicit one-sided C3 inequalities.
    for station in data.stations:
        present = [
            k
            for k in trains
            if data.has_event(k, station, "arr")
            and data.has_event(k, station, "dep")
        ]
        for p0 in range(len(present)):
            for q0 in range(p0 + 1, len(present)):
                i, j = data.canonical_pair(present[p0], present[q0])
                arr_idx = key_to_index.get(("arr", station, i, j))
                dep_idx = key_to_index.get(("dep", station, i, j))
                if arr_idx is None or dep_idx is None:
                    continue
                stop_i = bool(data.stop_at.get((i, station), False))
                stop_j = bool(data.stop_at.get((j, station), False))
                if not stop_i and not stop_j:
                    dsu.union(arr_idx, dep_idx)
                    c3_equality_count += 1

    root_to_component: Dict[int, int] = {}
    component_by_key: Dict[PairKey, int] = {}
    component_members: Dict[int, List[PairKey]] = defaultdict(list)
    for idx0, key in enumerate(universe):
        root = dsu.find(idx0)
        comp = root_to_component.get(root)
        if comp is None:
            comp = len(root_to_component)
            root_to_component[root] = comp
        component_by_key[key] = comp
        component_members[comp].append(key)

    summary = {
        "partition_mode": "NO4B_FULL_STRUCTURAL_QUOTIENT",
        "use_C1": 1,
        "use_C2": 1,
        "use_C3": 1,
        "structure_variant": "Full",
        "target_k": len(trains),
        "universe_size": universe_size,
        "forced_precedence_count": 0,
        "unresolved_event_core_count": universe_size,
        "unresolved_quotient_core_count": len(component_members),
        "inverse_binary_reduction_percent": (
            100.0 * (universe_size - len(component_members)) / universe_size
        ),
        "C2_equalities_collapsed": c2_equality_count,
        "C3_equalities_collapsed": c3_equality_count,
    }
    return PrecedenceCorePartition(
        forced={},
        component_by_key=component_by_key,
        component_members=dict(component_members),
        summary=summary,
    )



def load_complete_io_priors(
    cfg: Mapping[str, Any],
    ctx: ExperimentContext,
    data: HSRData,
    trains: Sequence[str],
    alpha: float,
) -> Tuple[Dict[PairKey, int], Dict[str, int]]:
    partition = build_inverse_structural_partition(cfg, ctx, data, trains)
    inverse_alpha_dir = ctx.inverse_dir / alpha_tag(alpha)
    if as_bool(cfg.get("inverse_temporal_cuts"), False):
        summary_path = inverse_alpha_dir / "inverse_summary.csv"
        summary_rec = read_first_row(summary_path)
        cert_raw = summary_rec.get("inverse_temporal_certified")
        try:
            cert_ok = int(round(float(cert_raw))) == 1
        except Exception:
            cert_ok = False
        if not cert_ok:
            raise RuntimeError(
                "当前IO precedence尚未通过同时间窗 temporal certification，"
                f"脚本6拒绝Full-Fix：{summary_path}"
            )
    core_path = inverse_alpha_dir / "learned_core_precedence.csv"
    component_cache_path = _inverse_component_cache_path(ctx, float(alpha))
    component_values = _load_component_assignment_cache(component_cache_path)
    learned: Dict[PairKey, int] = {}
    if component_values is not None:
        missing_components = set(partition.component_members) - set(component_values)
        extra_components = set(component_values) - set(partition.component_members)
        if missing_components or extra_components:
            raise RuntimeError(
                "learned component cache与当前quotient partition不一致："
                f"missing={len(missing_components)}, extra={len(extra_components)}"
            )
        for comp, members in partition.component_members.items():
            value = int(component_values[int(comp)])
            for key in members:
                learned[key] = value
    else:
        if not core_path.exists():
            raise FileNotFoundError(f"未找到逆优化core结果或compact cache：{core_path}; {component_cache_path}")
        core_df = pd.read_csv(core_path, encoding="utf-8-sig", usecols=["kind", "station", "train_i", "train_j", "value"])
        for row in core_df.itertuples(index=False):
            key = (str(row.kind), str(row.station), str(row.train_i), str(row.train_j))
            value = int(round(float(row.value)))
            if key in learned and learned[key] != value:
                raise RuntimeError(f"learned_core_precedence.csv 内部冲突：{key}")
            learned[key] = value
    expected_core = set(partition.component_by_key)
    missing = expected_core - set(learned)
    extra = set(learned) - expected_core
    overlap = set(learned) & set(partition.forced)
    if missing or extra or overlap:
        raise RuntimeError(
            "forced/core 合并失败："
            f"missing_core={len(missing)}, extra_core={len(extra)}, overlap={len(overlap)}"
        )
    merged = dict(partition.forced)
    merged.update(learned)
    return merged, {
        "forced_precedence_pairs": len(partition.forced),
        "learned_core_precedence_pairs": len(learned),
        "learned_core_component_count": len(partition.component_members),
    }


# =============================================================================
# Stage 5: 五组 alpha 逆优化（仅未决 quotient core）
# =============================================================================

def _add_rank_acyclicity(
    model: Model,
    data: HSRData,
    trains: Sequence[str],
    z: Mapping[PairKey, Any],
) -> Tuple[Dict[Tuple[str, str, str], Any], int]:
    rank_vars: Dict[Tuple[str, str, str], Any] = {}
    count = 0
    for station in data.stations:
        for kind in ("dep", "arr"):
            present = [k for k in trains if data.has_event(k, station, kind)]
            if len(present) <= 1:
                continue
            m_rank = float(len(present))
            for train in present:
                rank_vars[(kind, station, train)] = model.addVar(
                    lb=0.0, ub=float(len(present) - 1), vtype=GRB.CONTINUOUS,
                    name=f"rank_{kind}_{train}_{station}",
                )
            model.update()
            for p in range(len(present)):
                for q in range(p + 1, len(present)):
                    i, j = data.canonical_pair(present[p], present[q])
                    var = z[(kind, station, i, j)]
                    ri = rank_vars[(kind, station, i)]
                    rj = rank_vars[(kind, station, j)]
                    model.addConstr(ri + 1.0 <= rj + m_rank * (1.0 - var), name=f"C1a_{kind}_{i}_{j}_{station}")
                    model.addConstr(rj + 1.0 <= ri + m_rank * var, name=f"C1b_{kind}_{i}_{j}_{station}")
                    count += 2
    return rank_vars, count


def _add_inverse_c2(model: Model, data: HSRData, trains: Sequence[str], z: Mapping[PairKey, Any]) -> int:
    count = 0
    universe = set(z)
    for upstream, downstream in data.segments:
        present = [
            k for k in trains
            if upstream in data.path_by_train[k] and downstream in data.path_by_train[k]
            and data.path_by_train[k].index(downstream) == data.path_by_train[k].index(upstream) + 1
            and data.has_event(k, upstream, "dep") and data.has_event(k, downstream, "arr")
        ]
        for p in range(len(present)):
            for q in range(p + 1, len(present)):
                i, j = data.canonical_pair(present[p], present[q])
                dep_key = ("dep", upstream, i, j)
                arr_key = ("arr", downstream, i, j)
                if dep_key in universe and arr_key in universe:
                    model.addConstr(z[dep_key] == z[arr_key], name=f"C2_{i}_{j}_{upstream}_{downstream}")
                    count += 1
    return count


def _add_inverse_c3_stop_overtaking(model: Model, data: HSRData, trains: Sequence[str], z: Mapping[PairKey, Any]) -> int:
    count = 0
    universe = set(z)
    for station in data.stations:
        present = [k for k in trains if data.has_event(k, station, "arr") and data.has_event(k, station, "dep")]
        for p in range(len(present)):
            for q in range(p + 1, len(present)):
                i, j = data.canonical_pair(present[p], present[q])
                arr_key = ("arr", station, i, j)
                dep_key = ("dep", station, i, j)
                if arr_key not in universe or dep_key not in universe:
                    continue
                if not data.stop_at.get((i, station), False):
                    model.addConstr(z[dep_key] >= z[arr_key], name=f"C3_i_{i}_{j}_{station}")
                    count += 1
                if not data.stop_at.get((j, station), False):
                    model.addConstr(z[dep_key] <= z[arr_key], name=f"C3_j_{i}_{j}_{station}")
                    count += 1
    return count


def _load_vote_priors(
    ctx: ExperimentContext, data: HSRData, trains: Sequence[str], universe: Sequence[PairKey]
) -> Dict[PairKey, int]:
    cache = _load_full_assignment_cache(_vote_assignment_cache_path(ctx), trains, universe)
    if cache is not None:
        return cache
    path = ctx.vote_dir / "vote_priors_fullfix.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=["kind","station","train_i","train_j","value"])
    return {
        (str(row.kind), str(row.station), str(row.train_i), str(row.train_j)): int(row.value)
        for row in df.itertuples(index=False)
    }


def _core_value_or_var(
    key: PairKey,
    partition: PrecedenceCorePartition,
    y: Mapping[int, Any],
) -> Any:
    if key in partition.forced:
        return int(partition.forced[key])
    return y[int(partition.component_by_key[key])]


def _add_rank_acyclicity_core(
    model: Model,
    data: HSRData,
    trains: Sequence[str],
    universe: Sequence[PairKey],
    partition: PrecedenceCorePartition,
    y: Mapping[int, Any],
) -> Tuple[Dict[Tuple[str, str, str], Any], Dict[str, int]]:
    """C1 rank model with one-sided rows for fixed pairs and two-sided rows for core pairs."""
    rank_vars: Dict[Tuple[str, str, str], Any] = {}
    fixed_rows = 0
    free_rows = 0
    for station in data.stations:
        for kind in ("dep", "arr"):
            present = [k for k in trains if data.has_event(k, station, kind)]
            if len(present) <= 1:
                continue
            m_rank = float(len(present))
            for train in present:
                rank_vars[(kind, station, train)] = model.addVar(
                    lb=0.0,
                    ub=float(len(present) - 1),
                    vtype=GRB.CONTINUOUS,
                    name=f"rank_{kind}_{train}_{station}",
                )
            model.update()
            for p in range(len(present)):
                for q in range(p + 1, len(present)):
                    i, j = data.canonical_pair(present[p], present[q])
                    key = (kind, station, i, j)
                    ri = rank_vars[(kind, station, i)]
                    rj = rank_vars[(kind, station, j)]
                    if key in partition.forced:
                        if int(partition.forced[key]) == 1:
                            model.addConstr(
                                ri + 1.0 <= rj,
                                name=f"C1F_{kind}_{i}_{j}_{station}",
                            )
                        else:
                            model.addConstr(
                                rj + 1.0 <= ri,
                                name=f"C1F_{kind}_{j}_{i}_{station}",
                            )
                        fixed_rows += 1
                    else:
                        var = y[int(partition.component_by_key[key])]
                        model.addConstr(
                            ri + 1.0 <= rj + m_rank * (1.0 - var),
                            name=f"C1a_{kind}_{i}_{j}_{station}",
                        )
                        model.addConstr(
                            rj + 1.0 <= ri + m_rank * var,
                            name=f"C1b_{kind}_{i}_{j}_{station}",
                        )
                        free_rows += 2
    return rank_vars, {
        "fixed_one_sided_rank_rows": fixed_rows,
        "free_two_sided_rank_rows": free_rows,
        "total_rank_rows": fixed_rows + free_rows,
    }



# -----------------------------------------------------------------------------
# Stage-5 C1 lazy separation
# -----------------------------------------------------------------------------

def _prepare_c1_lazy_layers(
    data: HSRData,
    trains: Sequence[str],
) -> List[Tuple[str, str, Tuple[str, ...]]]:
    """Return station-event tournament layers used by the lazy C1 checker."""
    layers: List[Tuple[str, str, Tuple[str, ...]]] = []
    for station in data.stations:
        for kind in ("dep", "arr"):
            present = tuple(k for k in trains if data.has_event(k, station, kind))
            if len(present) >= 3:
                layers.append((kind, station, present))
    return layers


def _c1_lazy_edge_assignment(
    data: HSRData,
    partition: PrecedenceCorePartition,
    component_values: Mapping[int, int],
    kind: str,
    station: str,
    before: str,
    after: str,
) -> Tuple[PairKey, int, Optional[int]]:
    """Translate a directed tournament edge into its canonical binary value."""
    i, j = data.canonical_pair(before, after)
    key: PairKey = (kind, station, i, j)
    value = 1 if before == i else 0
    if key in partition.forced:
        if int(partition.forced[key]) != int(value):
            raise RuntimeError(
                f"C1 lazy方向与固定值冲突：key={key}, "
                f"forced={partition.forced[key]}, requested={value}"
            )
        return key, int(value), None
    comp = int(partition.component_by_key[key])
    if int(component_values[comp]) != int(value):
        raise RuntimeError(
            f"C1 lazy component取值解码错误：component={comp}, "
            f"candidate={component_values[comp]}, edge={value}, key={key}"
        )
    return key, int(value), comp


def _find_tournament_triangles_binary(
    data: HSRData,
    partition: PrecedenceCorePartition,
    component_values: Mapping[int, int],
    kind: str,
    station: str,
    present: Sequence[str],
    max_triangles: int = 1,
) -> List[Tuple[str, str, str]]:
    """Find directed 3-cycles in a complete 0/1 tournament assignment.

    A Hamiltonian path is built by insertion in O(n^2). Backward edges along
    that path expose directed triangles. Every returned triple is defensively
    verified against the actual event-level precedence assignment before it is
    used as a C1 cut.

    Only a bounded number of triangles is returned. C1 exactness comes from
    repeated separation within the same Gurobi solve; all O(n^3) triples are
    never enumerated up front.
    """
    max_triangles = max(1, int(max_triangles))

    def before(a: str, b: str) -> bool:
        i, j = data.canonical_pair(a, b)
        key: PairKey = (kind, station, i, j)
        if key in partition.forced:
            value = int(partition.forced[key])
        else:
            value = int(component_values[int(partition.component_by_key[key])])
        return bool(value == 1 if a == i else value == 0)

    path: List[str] = []
    for train in present:
        pos = 0
        while pos < len(path) and not before(train, path[pos]):
            pos += 1
        path.insert(pos, train)

    triangles: List[Tuple[str, str, str]] = []
    seen: set[Tuple[str, str, str]] = set()
    n = len(path)
    for gap in range(2, n):
        for p0 in range(0, n - gap):
            q0 = p0 + gap
            if not before(path[q0], path[p0]):
                continue
            a = path[p0]
            b = path[q0 - 1]
            c = path[q0]
            if before(a, b) and before(b, c) and before(c, a):
                tri = (a, b, c)
                if tri not in seen:
                    seen.add(tri)
                    triangles.append(tri)
                    if len(triangles) >= max_triangles:
                        return triangles
    return triangles


def _find_tournament_triangle_lazy(
    data: HSRData,
    partition: PrecedenceCorePartition,
    component_values: Mapping[int, int],
    kind: str,
    station: str,
    present: Sequence[str],
) -> Optional[Tuple[str, str, str]]:
    """Backward-compatible one-triangle wrapper."""
    triangles = _find_tournament_triangles_binary(
        data,
        partition,
        component_values,
        kind,
        station,
        present,
        max_triangles=1,
    )
    return triangles[0] if triangles else None


def _c1_cycle_signature(
    data: HSRData,
    partition: PrecedenceCorePartition,
    component_values: Mapping[int, int],
    kind: str,
    station: str,
    triangle: Tuple[str, str, str],
) -> Tuple[Tuple[Tuple[int, int], ...], List[Tuple[PairKey, int, Optional[int]]]]:
    """Return the quotient-component no-good signature for one directed triangle."""
    a, b, c = triangle
    assignments: Dict[int, int] = {}
    edge_rows: List[Tuple[PairKey, int, Optional[int]]] = []
    for before, after in ((a, b), (b, c), (c, a)):
        key, value, comp = _c1_lazy_edge_assignment(
            data,
            partition,
            component_values,
            kind,
            station,
            before,
            after,
        )
        edge_rows.append((key, value, comp))
        if comp is not None:
            if comp in assignments and assignments[comp] != value:
                raise RuntimeError(
                    f"一个C1三车环要求同一component相反取值：component={comp}"
                )
            assignments[comp] = int(value)

    if not assignments:
        raise RuntimeError(
            "固定precedence自身形成C1三车环；当前NO4B流程不应出现该情况。"
        )
    return tuple(sorted(assignments.items())), edge_rows


def _c1_signature_expr(
    y: Mapping[int, Any],
    signature: Sequence[Tuple[int, int]],
) -> LinExpr:
    """No-good form of an exact triangle-transitivity cut."""
    expr = LinExpr()
    for comp, value in signature:
        if int(value) == 1:
            expr += 1.0 - y[int(comp)]
        else:
            expr += y[int(comp)]
    return expr


def _make_inverse_c1_lazy_callback(
    cfg: Mapping[str, Any],
    data: HSRData,
    partition: PrecedenceCorePartition,
    y: Mapping[int, Any],
    component_ids: Sequence[int],
    layers: Sequence[Tuple[str, str, Tuple[str, ...]]],
) -> Tuple[Any, Dict[str, Any]]:
    """Exact integer-only lazy C1 separator retained for compatibility."""
    max_per_callback = int(
        as_int(cfg.get("inverse_c1_lazy_max_cuts_per_callback"), 50) or 50
    )
    max_triangles_per_layer = int(
        as_int(cfg.get("inverse_c1_lazy_max_triangles_per_layer"), 4) or 4
    )
    if max_per_callback <= 0 or max_triangles_per_layer <= 0:
        raise ValueError("C1 lazy cut数量参数必须为正整数")

    var_list = [y[int(comp)] for comp in component_ids]
    comp_list = [int(comp) for comp in component_ids]
    state: Dict[str, Any] = {
        "callback_calls": 0,
        "mipsol_calls": 0,
        "mipnode_calls": 0,
        "candidate_layers_checked": 0,
        "cycle_cuts_added": 0,
        "user_cuts_added": 0,
        "duplicate_cycle_signatures": 0,
        "duplicate_usercut_signatures": 0,
        "max_cuts_per_callback": max_per_callback,
        "seen_signatures": set(),
        "seen_usercut_signatures": set(),
        "cut_records": [],
        "usercut_records": [],
        "error": "",
    }

    def callback(model: Model, where: int) -> None:
        if where != GRB.Callback.MIPSOL:
            return
        state["callback_calls"] += 1
        state["mipsol_calls"] += 1
        try:
            values_raw = model.cbGetSolution(var_list)
            component_values = {
                comp: int(float(value) >= 0.5)
                for comp, value in zip(comp_list, values_raw)
            }
            cuts_this_call = 0
            for kind, station, present in layers:
                state["candidate_layers_checked"] += 1
                triangles = _find_tournament_triangles_binary(
                    data,
                    partition,
                    component_values,
                    kind,
                    station,
                    present,
                    max_triangles=max_triangles_per_layer,
                )
                for triangle in triangles:
                    a, b, c = triangle
                    signature, _ = _c1_cycle_signature(
                        data,
                        partition,
                        component_values,
                        kind,
                        station,
                        triangle,
                    )
                    is_new_signature = signature not in state["seen_signatures"]
                    if is_new_signature:
                        state["seen_signatures"].add(signature)
                    else:
                        state["duplicate_cycle_signatures"] += 1

                    model.cbLazy(_c1_signature_expr(y, signature) >= 1.0)
                    state["cycle_cuts_added"] += 1
                    cuts_this_call += 1
                    if is_new_signature:
                        state["cut_records"].append({
                            "cut_index": len(state["cut_records"]) + 1,
                            "kind": kind,
                            "station": station,
                            "train_a": a,
                            "train_b": b,
                            "train_c": c,
                            "free_component_count": len(signature),
                            "cut_signature": ";".join(
                                f"{comp}:{value}" for comp, value in signature
                            ),
                            "edge_1": f"{a}->{b}",
                            "edge_2": f"{b}->{c}",
                            "edge_3": f"{c}->{a}",
                        })
                    if cuts_this_call >= max_per_callback:
                        return
        except Exception as exc:
            state["error"] = repr(exc)
            try:
                model.terminate()
            except Exception:
                pass

    return callback, state


def _make_inverse_c1_branchcut_callback(
    cfg: Mapping[str, Any],
    data: HSRData,
    partition: PrecedenceCorePartition,
    y: Mapping[int, Any],
    component_ids: Sequence[int],
    layers: Sequence[Tuple[str, str, Tuple[str, ...]]],
) -> Tuple[Any, Dict[str, Any]]:
    """One-shot C1 branch-and-cut.

    MIPNODE adds valid user cuts for violated triangle-transitivity structure.
    MIPSOL adds exact lazy triangle cuts, so every accepted incumbent satisfies
    C1. No rank variables, no rank Big-M rows, no repaired complete Start, and
    no outer inverse rounds are introduced.
    """
    lazy_max = int(
        as_int(cfg.get("inverse_c1_lazy_max_cuts_per_callback"), 200) or 200
    )
    lazy_tri_per_layer = int(
        as_int(cfg.get("inverse_c1_lazy_max_triangles_per_layer"), 12) or 12
    )
    user_enabled = as_bool(cfg.get("inverse_c1_usercuts_enabled"), True)
    user_root_only = as_bool(cfg.get("inverse_c1_usercut_root_only"), True)
    user_max_root_passes = int(
        as_int(cfg.get("inverse_c1_usercut_max_root_passes"), 30) or 30
    )
    user_node_interval = int(
        as_int(cfg.get("inverse_c1_usercut_node_interval"), 100) or 100
    )
    user_max = int(
        as_int(cfg.get("inverse_c1_usercut_max_cuts_per_callback"), 200) or 200
    )
    user_tri_per_layer = int(
        as_int(cfg.get("inverse_c1_usercut_max_triangles_per_layer"), 20) or 20
    )
    violation_tol = float(
        as_float(cfg.get("inverse_c1_usercut_violation_tol"), 1e-6) or 1e-6
    )
    history_limit = int(as_int(cfg.get("inverse_c1_cut_history_limit"), 50000) or 50000)
    if _paper_output_mode(cfg) and not _cfg_flag(cfg, "write_callback_cut_history", paper_default=False, normal_default=True):
        history_limit = 1

    for name, value in (
        ("inverse_c1_lazy_max_cuts_per_callback", lazy_max),
        ("inverse_c1_lazy_max_triangles_per_layer", lazy_tri_per_layer),
        ("inverse_c1_usercut_max_root_passes", user_max_root_passes),
        ("inverse_c1_usercut_node_interval", user_node_interval),
        ("inverse_c1_usercut_max_cuts_per_callback", user_max),
        ("inverse_c1_usercut_max_triangles_per_layer", user_tri_per_layer),
        ("inverse_c1_cut_history_limit", history_limit),
    ):
        if value <= 0:
            raise ValueError(f"{name} 必须为正整数")
    if violation_tol < 0.0:
        raise ValueError("inverse_c1_usercut_violation_tol 不能为负数")

    var_list = [y[int(comp)] for comp in component_ids]
    comp_list = [int(comp) for comp in component_ids]
    state: Dict[str, Any] = {
        "callback_calls": 0,
        "mipsol_calls": 0,
        "mipnode_calls": 0,
        "mipnode_root_calls": 0,
        "mipnode_usercut_passes": 0,
        "candidate_layers_checked": 0,
        "usercut_layers_checked": 0,
        "cycle_cuts_added": 0,
        "user_cuts_added": 0,
        "usercut_candidate_triangles": 0,
        "duplicate_cycle_signatures": 0,
        "duplicate_usercut_signatures": 0,
        "max_cuts_per_callback": lazy_max,
        "usercut_max_cuts_per_callback": user_max,
        "usercut_root_only": int(user_root_only),
        "usercut_max_root_passes": user_max_root_passes,
        "seen_signatures": set(),
        "seen_usercut_signatures": set(),
        "cut_records": [],
        "usercut_records": [],
        "max_usercut_violation": 0.0,
        "error": "",
    }

    def append_record(target: List[Dict[str, Any]], record: Dict[str, Any]) -> None:
        if len(target) < history_limit:
            target.append(record)

    def callback(model: Model, where: int) -> None:
        if where not in {GRB.Callback.MIPSOL, GRB.Callback.MIPNODE}:
            return
        state["callback_calls"] += 1
        try:
            if where == GRB.Callback.MIPSOL:
                state["mipsol_calls"] += 1
                values_raw = model.cbGetSolution(var_list)
                component_values = {
                    comp: int(float(value) >= 0.5)
                    for comp, value in zip(comp_list, values_raw)
                }
                cuts_this_call = 0
                for kind, station, present in layers:
                    state["candidate_layers_checked"] += 1
                    triangles = _find_tournament_triangles_binary(
                        data,
                        partition,
                        component_values,
                        kind,
                        station,
                        present,
                        max_triangles=lazy_tri_per_layer,
                    )
                    for triangle in triangles:
                        a, b, c = triangle
                        signature, _ = _c1_cycle_signature(
                            data,
                            partition,
                            component_values,
                            kind,
                            station,
                            triangle,
                        )
                        is_new = signature not in state["seen_signatures"]
                        if is_new:
                            state["seen_signatures"].add(signature)
                        else:
                            state["duplicate_cycle_signatures"] += 1
                        model.cbLazy(_c1_signature_expr(y, signature) >= 1.0)
                        state["cycle_cuts_added"] += 1
                        cuts_this_call += 1
                        if is_new:
                            append_record(state["cut_records"], {
                                "cut_index": len(state["cut_records"]) + 1,
                                "source": "MIPSOL_LAZY",
                                "kind": kind,
                                "station": station,
                                "train_a": a,
                                "train_b": b,
                                "train_c": c,
                                "free_component_count": len(signature),
                                "cut_signature": ";".join(
                                    f"{comp}:{value}" for comp, value in signature
                                ),
                                "edge_1": f"{a}->{b}",
                                "edge_2": f"{b}->{c}",
                                "edge_3": f"{c}->{a}",
                            })
                        if cuts_this_call >= lazy_max:
                            return
                return

            state["mipnode_calls"] += 1
            if not user_enabled:
                return
            node_status = int(model.cbGet(GRB.Callback.MIPNODE_STATUS))
            if node_status != int(GRB.OPTIMAL):
                return
            node_count = float(model.cbGet(GRB.Callback.MIPNODE_NODCNT))
            is_root = node_count < 0.5
            if is_root:
                state["mipnode_root_calls"] += 1
                if state["mipnode_usercut_passes"] >= user_max_root_passes:
                    return
            elif user_root_only:
                return
            elif user_node_interval > 1 and int(node_count) % user_node_interval != 0:
                return

            rel_raw = model.cbGetNodeRel(var_list)
            rel_values = {
                comp: float(value)
                for comp, value in zip(comp_list, rel_raw)
            }
            rounded_values = {
                comp: int(float(value) >= 0.5)
                for comp, value in rel_values.items()
            }
            if is_root:
                state["mipnode_usercut_passes"] += 1

            cuts_this_call = 0
            for kind, station, present in layers:
                state["usercut_layers_checked"] += 1
                triangles = _find_tournament_triangles_binary(
                    data,
                    partition,
                    rounded_values,
                    kind,
                    station,
                    present,
                    max_triangles=user_tri_per_layer,
                )
                for triangle in triangles:
                    state["usercut_candidate_triangles"] += 1
                    a, b, c = triangle
                    signature, _ = _c1_cycle_signature(
                        data,
                        partition,
                        rounded_values,
                        kind,
                        station,
                        triangle,
                    )
                    mismatch = 0.0
                    for comp, expected in signature:
                        value = float(rel_values[int(comp)])
                        mismatch += (1.0 - value) if int(expected) == 1 else value
                    violation = 1.0 - mismatch
                    if violation <= violation_tol:
                        continue
                    if signature in state["seen_usercut_signatures"]:
                        state["duplicate_usercut_signatures"] += 1
                        continue

                    model.cbCut(_c1_signature_expr(y, signature) >= 1.0)
                    state["seen_usercut_signatures"].add(signature)
                    state["user_cuts_added"] += 1
                    state["max_usercut_violation"] = max(
                        float(state["max_usercut_violation"]),
                        float(violation),
                    )
                    cuts_this_call += 1
                    append_record(state["usercut_records"], {
                        "cut_index": len(state["usercut_records"]) + 1,
                        "source": "MIPNODE_USERCUT",
                        "node_count": node_count,
                        "kind": kind,
                        "station": station,
                        "train_a": a,
                        "train_b": b,
                        "train_c": c,
                        "free_component_count": len(signature),
                        "cut_signature": ";".join(
                            f"{comp}:{value}" for comp, value in signature
                        ),
                        "fractional_mismatch_lhs": mismatch,
                        "violation": violation,
                        "edge_1": f"{a}->{b}",
                        "edge_2": f"{b}->{c}",
                        "edge_3": f"{c}->{a}",
                    })
                    if cuts_this_call >= user_max:
                        return
        except Exception as exc:
            state["error"] = repr(exc)
            try:
                model.terminate()
            except Exception:
                pass

    return callback, state


# -----------------------------------------------------------------------------
# Stage-5 sample-based partial Start / variable hints
# -----------------------------------------------------------------------------

def _apply_inverse_sample_guidance(
    cfg: Mapping[str, Any],
    outdir: Path,
    y: Mapping[int, Any],
    component_ids: Sequence[int],
    component_coefficients: Mapping[int, float],
    component_weighted_evidence: Mapping[int, float],
    component_raw_observations: Mapping[int, int],
) -> Dict[str, Any]:
    """Guide, but never fix, the inverse quotient-core binaries.

    Preferred directions come from the same weighted mismatch objective used by
    inverse optimization.  Confidence is the normalized absolute objective
    margin ``|beta_c| / weighted_evidence_c``.  Only the most confident fraction
    receives a MIP Start; a wider set may receive Gurobi variable hints.
    """
    start_enabled = as_bool(cfg.get("inverse_sample_start_enabled"), False)
    hint_enabled = as_bool(cfg.get("inverse_sample_hint_enabled"), False)
    start_fraction = float(
        as_float(cfg.get("inverse_sample_start_fraction"), 0.25) or 0.0
    )
    start_min_conf = float(
        as_float(cfg.get("inverse_sample_start_min_confidence"), 0.60) or 0.0
    )
    hint_min_conf = float(
        as_float(cfg.get("inverse_sample_hint_min_confidence"), 0.10) or 0.0
    )
    hint_max_priority = int(
        as_int(cfg.get("inverse_sample_hint_max_priority"), 100) or 100
    )
    if not 0.0 <= start_fraction <= 1.0:
        raise ValueError("inverse_sample_start_fraction 必须位于[0,1]")
    if not 0.0 <= start_min_conf <= 1.0:
        raise ValueError("inverse_sample_start_min_confidence 必须位于[0,1]")
    if not 0.0 <= hint_min_conf <= 1.0:
        raise ValueError("inverse_sample_hint_min_confidence 必须位于[0,1]")
    if hint_max_priority <= 0:
        raise ValueError("inverse_sample_hint_max_priority 必须为正整数")

    rows: List[Dict[str, Any]] = []
    eligible: List[Tuple[float, float, int, int, int]] = []
    # tuple = confidence, weighted_evidence, raw_observations, comp, preferred
    for comp in component_ids:
        coefficient = float(component_coefficients.get(int(comp), 0.0))
        evidence = float(component_weighted_evidence.get(int(comp), 0.0))
        raw_n = int(component_raw_observations.get(int(comp), 0))
        if evidence > 0.0 and abs(coefficient) > 1e-15:
            confidence = min(1.0, abs(coefficient) / evidence)
            preferred = 1 if coefficient < 0.0 else 0
            eligible.append(
                (confidence, evidence, raw_n, int(comp), int(preferred))
            )
        else:
            confidence = 0.0
            preferred = -1
        rows.append({
            "component_id": int(comp),
            "objective_coefficient": coefficient,
            "weighted_evidence": evidence,
            "raw_observations": raw_n,
            "preferred_value": None if preferred < 0 else preferred,
            "confidence": confidence,
            "is_partial_start": 0,
            "is_var_hint": 0,
            "hint_priority": 0,
        })

    eligible.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    start_pool = [item for item in eligible if item[0] >= start_min_conf]
    if start_enabled and start_fraction > 0.0 and start_pool:
        start_count = max(1, int(math.ceil(start_fraction * len(start_pool))))
    else:
        start_count = 0
    start_components = {item[3] for item in start_pool[:start_count]}

    row_by_comp = {int(row["component_id"]): row for row in rows}
    hint_count = 0
    for confidence, evidence, raw_n, comp, preferred in eligible:
        var = y[int(comp)]
        if hint_enabled and confidence >= hint_min_conf:
            priority = max(
                1,
                min(
                    hint_max_priority,
                    int(round(1 + confidence * (hint_max_priority - 1))),
                ),
            )
            var.VarHintVal = float(preferred)
            var.VarHintPri = int(priority)
            row_by_comp[comp]["is_var_hint"] = 1
            row_by_comp[comp]["hint_priority"] = priority
            hint_count += 1
        if comp in start_components:
            var.Start = float(preferred)
            row_by_comp[comp]["is_partial_start"] = 1

    write_csv(
        pd.DataFrame(rows),
        outdir / "inverse_sample_guidance.csv",
    )
    return {
        "sample_guidance_enabled": int(start_enabled or hint_enabled),
        "sample_partial_start_enabled": int(start_enabled),
        "sample_hint_enabled": int(hint_enabled),
        "sample_partial_start_fraction": start_fraction,
        "sample_partial_start_min_confidence": start_min_conf,
        "sample_hint_min_confidence": hint_min_conf,
        "sample_guidance_observed_component_count": len(eligible),
        "sample_partial_start_component_count": len(start_components),
        "sample_hint_component_count": hint_count,
        "sample_guidance_csv": relative_to_project(
            outdir / "inverse_sample_guidance.csv"
        ),
    }


def _verify_c2_collapsed(
    data: HSRData,
    trains: Sequence[str],
    partition: PrecedenceCorePartition,
) -> int:
    checked = 0
    universe = set(partition.forced) | set(partition.component_by_key)
    for upstream, downstream in data.segments:
        present = [
            k
            for k in trains
            if upstream in data.path_by_train[k]
            and downstream in data.path_by_train[k]
            and data.path_by_train[k].index(downstream)
            == data.path_by_train[k].index(upstream) + 1
            and data.has_event(k, upstream, "dep")
            and data.has_event(k, downstream, "arr")
        ]
        for p in range(len(present)):
            for q in range(p + 1, len(present)):
                i, j = data.canonical_pair(present[p], present[q])
                dep_key = ("dep", upstream, i, j)
                arr_key = ("arr", downstream, i, j)
                if dep_key not in universe or arr_key not in universe:
                    continue
                checked += 1
                dep_fixed = partition.forced.get(dep_key)
                arr_fixed = partition.forced.get(arr_key)
                if dep_fixed is not None or arr_fixed is not None:
                    if dep_fixed is None or arr_fixed is None or int(dep_fixed) != int(arr_fixed):
                        raise RuntimeError(
                            f"C2 partition不一致：{dep_key} vs {arr_key}"
                        )
                elif int(partition.component_by_key[dep_key]) != int(
                    partition.component_by_key[arr_key]
                ):
                    raise RuntimeError(
                        f"C2 equality未被quotient合并：{dep_key} vs {arr_key}"
                    )
    return checked


def _add_core_relation(
    model: Model,
    lhs: Any,
    sense: str,
    rhs: Any,
    name: str,
) -> int:
    lhs_is_num = isinstance(lhs, (int, float))
    rhs_is_num = isinstance(rhs, (int, float))
    if lhs_is_num and rhs_is_num:
        valid = float(lhs) >= float(rhs) if sense == ">=" else float(lhs) <= float(rhs)
        if not valid:
            raise RuntimeError(f"预处理固定值违反{name}: {lhs} {sense} {rhs}")
        return 0
    if sense == ">=":
        model.addConstr(lhs >= rhs, name=name)
    else:
        model.addConstr(lhs <= rhs, name=name)
    return 1


def _add_inverse_c3_core(
    model: Model,
    data: HSRData,
    trains: Sequence[str],
    partition: PrecedenceCorePartition,
    y: Mapping[int, Any],
) -> Dict[str, int]:
    rows = 0
    fixed_checks = 0
    skipped_same_component = 0
    universe = set(partition.forced) | set(partition.component_by_key)
    for station in data.stations:
        present = [
            k
            for k in trains
            if data.has_event(k, station, "arr")
            and data.has_event(k, station, "dep")
        ]
        for p in range(len(present)):
            for q in range(p + 1, len(present)):
                i, j = data.canonical_pair(present[p], present[q])
                arr_key = ("arr", station, i, j)
                dep_key = ("dep", station, i, j)
                if arr_key not in universe or dep_key not in universe:
                    continue
                stop_i = bool(data.stop_at.get((i, station), False))
                stop_j = bool(data.stop_at.get((j, station), False))
                if stop_i and stop_j:
                    continue
                arr_expr = _core_value_or_var(arr_key, partition, y)
                dep_expr = _core_value_or_var(dep_key, partition, y)
                if (
                    arr_key not in partition.forced
                    and dep_key not in partition.forced
                    and partition.component_by_key[arr_key]
                    == partition.component_by_key[dep_key]
                ):
                    skipped_same_component += 1
                    continue
                if isinstance(arr_expr, (int, float)) and isinstance(dep_expr, (int, float)):
                    fixed_checks += 1
                if not stop_i:
                    rows += _add_core_relation(
                        model,
                        dep_expr,
                        ">=",
                        arr_expr,
                        f"C3_i_{i}_{j}_{station}",
                    )
                if not stop_j:
                    rows += _add_core_relation(
                        model,
                        dep_expr,
                        "<=",
                        arr_expr,
                        f"C3_j_{i}_{j}_{station}",
                    )
    return {
        "C3_rows_instantiated": rows,
        "C3_fixed_checks": fixed_checks,
        "C3_same_component_equalities_skipped": skipped_same_component,
    }



def _core_candidate_values_and_priors(
    partition: PrecedenceCorePartition,
    component_ids: Sequence[int],
    y: Mapping[int, Any],
) -> Tuple[Dict[int, int], Dict[PairKey, int]]:
    """Read the current integer master incumbent and expand it to event level."""
    component_values = {
        int(comp): int(round(float(y[int(comp)].X)))
        for comp in component_ids
    }
    priors: Dict[PairKey, int] = dict(partition.forced)
    for comp in component_ids:
        value = int(component_values[int(comp)])
        for key in partition.component_members[int(comp)]:
            priors[key] = value
    return component_values, priors



def _run_temporal_feasibility_check_graph(
    cfg: Mapping[str, Any],
    ctx: ExperimentContext,
    data: HSRData,
    trains: Sequence[str],
    universe: Sequence[PairKey],
    partition: PrecedenceCorePartition,
    priors: Mapping[PairKey, int],
    component_values: Mapping[int, int],
    *,
    alpha: float,
    check_round: int,
) -> Dict[str, Any]:
    """Exact feasibility check for the current fixed precedence assignment.

    The current timetable formulation contains only origin lower/upper bounds,
    minimum running/dwell inequalities, pass-through equalities, and fixed
    station-event headway inequalities.  After contracting each pass-through
    equality, every remaining timing row has the form ``t_v >= t_u + c``.

    For the current data, all non-contracted edges have strictly positive
    weights.  Hence a directed cycle is immediately infeasible.  If the graph
    is acyclic, longest-path propagation gives the tightest lower bounds implied
    by the fixed order; feasibility is equivalent to every propagated lower
    bound respecting the finite origin upper bounds.  A conflicting path/cycle
    directly identifies the learned quotient components for an exact no-good
    cut.  A zero-weight directed cycle is reported as not certified by this
    propagation checker; no secondary infeasibility-analysis model is invoked.
    """
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    tolerance = float(
        as_float(cfg.get("inverse_temporal_graph_tolerance"), 1e-9) or 1e-9
    )
    alpha_dir = ensure_dir(ctx.inverse_dir / alpha_tag(alpha))
    check_dir = ensure_dir(
        alpha_dir / "temporal_checks" / f"round_{int(check_round):03d}"
    )
    chain_path = check_dir / "temporal_graph_conflict_chain.csv"
    bounds_path = check_dir / "temporal_graph_bounds.csv"
    feasible_tt_path = check_dir / "temporal_feasible_timetable.csv"

    # Original event nodes.
    events: List[Tuple[str, str, str]] = []
    event_to_id: Dict[Tuple[str, str, str], int] = {}

    def event_id(kind: str, station: str, train: str) -> int:
        key = _event_key(kind, station, train)
        idx = event_to_id.get(key)
        if idx is None:
            idx = len(events)
            event_to_id[key] = idx
            events.append(key)
        return idx

    for train in trains:
        seq = data.path_by_train[train]
        for pos, station in enumerate(seq):
            if pos > 0:
                event_id("arr", station, train)
            if pos < len(seq) - 1:
                event_id("dep", station, train)

    # Contract exact pass-through equalities a=d before building inequalities.
    dsu = _CoreDSU(len(events))
    for train in trains:
        seq = data.path_by_train[train]
        for station in seq[1:-1]:
            if not data.stop_at.get((train, station), False):
                dsu.union(
                    event_to_id[_event_key("arr", station, train)],
                    event_to_id[_event_key("dep", station, train)],
                )

    root_to_node: Dict[int, int] = {}
    event_node = array("i", [0]) * len(events)
    for eid in range(len(events)):
        root = dsu.find(eid)
        node = root_to_node.get(root)
        if node is None:
            node = len(root_to_node)
            root_to_node[root] = node
        event_node[eid] = int(node)
    node_count = len(root_to_node)
    del dsu, root_to_node

    lower = [0.0] * node_count
    upper = [math.inf] * node_count
    allow_early = as_bool(cfg.get("allow_early_departure"), True)
    fast_window = float(as_float(cfg.get("dep_window_fast"), 45.0) or 0.0)
    normal_window = float(as_float(cfg.get("dep_window_normal"), 60.0) or 0.0)
    fast_type = int(as_int(cfg.get("fast_train_type"), 350) or 350)
    for train in trains:
        origin = data.path_by_train[train][0]
        eid = event_to_id[_event_key("dep", origin, train)]
        node = int(event_node[eid])
        window = (
            fast_window
            if data.train_type.get(train, 300) == fast_type
            else normal_window
        )
        lb = data.first_dep[train] - window if allow_early else data.first_dep[train]
        ub = data.first_dep[train] + window
        lower[node] = max(lower[node], float(lb))
        upper[node] = min(upper[node], float(ub))

    # Compact edge storage: adjacency contains only integer edge indices.
    adjacency: List[array] = [array("i") for _ in range(node_count)]
    indegree = array("i", [0]) * node_count
    edge_u = array("i")
    edge_v = array("i")
    edge_w = array("d")
    # >=0: index in ``universe``; <0: -(backbone_meta_index+1)
    edge_ref = array("i")
    # -2 backbone, -1 forced precedence, >=0 learned component
    edge_component = array("i")
    edge_value = bytearray()
    backbone_meta: List[Tuple[str, Tuple[str, str, str], Tuple[str, str, str]]] = []
    immediate_positive_self_edge: Optional[int] = None
    has_nonpositive_noncontracted_edge = False

    def append_edge(
        from_event: Tuple[str, str, str],
        to_event: Tuple[str, str, str],
        weight: float,
        *,
        ref: int,
        component: int,
        value: int,
    ) -> Optional[int]:
        nonlocal immediate_positive_self_edge, has_nonpositive_noncontracted_edge
        u = int(event_node[event_to_id[from_event]])
        v = int(event_node[event_to_id[to_event]])
        w = float(weight)
        if u == v and w <= tolerance:
            return None
        eid = len(edge_v)
        edge_u.append(u)
        edge_v.append(v)
        edge_w.append(w)
        edge_ref.append(int(ref))
        edge_component.append(int(component))
        edge_value.append(int(value) if value in (0, 1) else 255)
        if u == v:
            if w > tolerance:
                immediate_positive_self_edge = eid
            return eid
        adjacency[u].append(eid)
        indegree[v] += 1
        if w <= tolerance:
            has_nonpositive_noncontracted_edge = True
        return eid

    # Train timing backbone.
    for train in trains:
        seq = data.path_by_train[train]
        for upstream, downstream in zip(seq, seq[1:]):
            from_event = _event_key("dep", upstream, train)
            to_event = _event_key("arr", downstream, train)
            meta_idx = len(backbone_meta)
            backbone_meta.append(("RUN_MIN", from_event, to_event))
            append_edge(
                from_event,
                to_event,
                data.required_run_time(train, upstream, downstream),
                ref=-(meta_idx + 1),
                component=-2,
                value=255,
            )
        for station in seq[1:-1]:
            if data.stop_at.get((train, station), False):
                from_event = _event_key("arr", station, train)
                to_event = _event_key("dep", station, train)
                meta_idx = len(backbone_meta)
                backbone_meta.append(("DWELL_MIN", from_event, to_event))
                append_edge(
                    from_event,
                    to_event,
                    data.station_params.get(station, {}).get("dwell_min", 0.0),
                    ref=-(meta_idx + 1),
                    component=-2,
                    value=255,
                )

    # Current complete precedence assignment.
    if len(universe) != len(priors):
        missing = set(universe) - set(priors)
        extra = set(priors) - set(universe)
        raise RuntimeError(
            "Temporal graph requires one value for every precedence: "
            f"universe={len(universe)}, priors={len(priors)}, "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    for key_idx, key in enumerate(universe):
        kind, station, i, j = key
        value = int(priors[key])
        h = float(
            data.station_params.get(station, {}).get(
                "h_dep" if kind == "dep" else "h_arr", 0.0
            )
        )
        if value == 1:
            before, after = i, j
        else:
            before, after = j, i
        component = (
            -1
            if key in partition.forced
            else int(partition.component_by_key[key])
        )
        append_edge(
            _event_key(kind, station, before),
            _event_key(kind, station, after),
            h,
            ref=int(key_idx),
            component=component,
            value=value,
        )

    def edge_rows(edge_ids: Sequence[int]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for order, edge_id in enumerate(edge_ids, start=1):
            ref = int(edge_ref[edge_id])
            comp = int(edge_component[edge_id])
            row: Dict[str, Any] = {
                "edge_order": order,
                "edge_weight": float(edge_w[edge_id]),
                "edge_type": "",
                "from_event": "",
                "to_event": "",
                "kind": "",
                "station": "",
                "train_i": "",
                "train_j": "",
                "value": None,
                "source": "",
                "component_id": None,
            }
            if ref >= 0:
                kind, station, i, j = universe[ref]
                value = int(edge_value[edge_id])
                before, after = (i, j) if value == 1 else (j, i)
                row.update({
                    "edge_type": "FIXED_HEADWAY",
                    "from_event": f"{kind}:{station}:{before}",
                    "to_event": f"{kind}:{station}:{after}",
                    "kind": kind,
                    "station": station,
                    "train_i": i,
                    "train_j": j,
                    "value": value,
                    "source": (
                        "PREPROCESS_FIXED" if comp < 0 else "INVERSE_CORE"
                    ),
                    "component_id": None if comp < 0 else comp,
                })
            else:
                edge_type, from_event, to_event = backbone_meta[-ref - 1]
                row.update({
                    "edge_type": edge_type,
                    "from_event": ":".join(from_event),
                    "to_event": ":".join(to_event),
                    "source": "TIMING_BACKBONE",
                })
            rows.append(row)
        return rows

    def cut_from_edges(edge_ids: Sequence[int]) -> Dict[int, int]:
        assignments: Dict[int, int] = {}
        for edge_id in edge_ids:
            comp = int(edge_component[edge_id])
            if comp < 0:
                continue
            value = int(edge_value[edge_id])
            expected = int(component_values[comp])
            if expected != value:
                raise RuntimeError(
                    f"Temporal graph component value mismatch: component={comp}, "
                    f"expected={expected}, edge={value}"
                )
            if comp in assignments and assignments[comp] != value:
                raise RuntimeError(
                    f"Temporal graph conflict asks opposite values for component={comp}"
                )
            assignments[comp] = value
        return assignments

    conflict_kind = ""
    conflict_edges: List[int] = []
    conflict_node: Optional[int] = None
    graph_not_certified = False

    if immediate_positive_self_edge is not None:
        conflict_kind = "POSITIVE_SELF_CYCLE"
        conflict_edges = [int(immediate_positive_self_edge)]
    else:
        # Kahn topological sort.  With positive non-contracted edge weights, any
        # remaining directed cycle is a strict temporal contradiction.
        queue = deque(i for i in range(node_count) if int(indegree[i]) == 0)
        topo: List[int] = []
        while queue:
            u = queue.popleft()
            topo.append(u)
            for edge_id in adjacency[u]:
                v = int(edge_v[edge_id])
                indegree[v] -= 1
                if int(indegree[v]) == 0:
                    queue.append(v)

        if len(topo) != node_count:
            leftover = bytearray(
                1 if int(indegree[i]) > 0 else 0 for i in range(node_count)
            )
            color = bytearray(node_count)
            parent_node = array("i", [-1]) * node_count
            parent_edge = array("i", [-1]) * node_count
            found_cycle: List[int] = []

            for start in range(node_count):
                if not leftover[start] or color[start] != 0:
                    continue
                color[start] = 1
                stack_nodes: List[int] = [start]
                stack_pos: List[int] = [0]
                while stack_nodes and not found_cycle:
                    u = stack_nodes[-1]
                    pos = stack_pos[-1]
                    if pos >= len(adjacency[u]):
                        color[u] = 2
                        stack_nodes.pop()
                        stack_pos.pop()
                        continue
                    edge_id = int(adjacency[u][pos])
                    stack_pos[-1] = pos + 1
                    v = int(edge_v[edge_id])
                    if not leftover[v]:
                        continue
                    if color[v] == 0:
                        color[v] = 1
                        parent_node[v] = u
                        parent_edge[v] = edge_id
                        stack_nodes.append(v)
                        stack_pos.append(0)
                    elif color[v] == 1:
                        reverse_path: List[int] = []
                        cur = u
                        while cur != v:
                            pe = int(parent_edge[cur])
                            if pe < 0:
                                raise RuntimeError(
                                    "Temporal graph cycle reconstruction failed"
                                )
                            reverse_path.append(pe)
                            cur = int(parent_node[cur])
                        found_cycle = list(reversed(reverse_path)) + [edge_id]
                if found_cycle:
                    break

            if not found_cycle:
                raise RuntimeError(
                    "Temporal graph Kahn detected a cycle but DFS could not recover it"
                )
            cycle_weight = sum(float(edge_w[e]) for e in found_cycle)
            if cycle_weight > tolerance:
                conflict_kind = "POSITIVE_DIRECTED_CYCLE"
                conflict_edges = found_cycle
            else:
                # A zero-weight cycle can be feasible.  This graph checker does
                # not claim infeasibility in that rare case.
                graph_not_certified = True
                conflict_kind = "ZERO_WEIGHT_CYCLE_NOT_CERTIFIED"
        else:
            parent_edge = array("i", [-1]) * node_count
            for u in topo:
                # A component may itself carry a finite origin upper bound.
                if lower[u] > upper[u] + tolerance:
                    conflict_node = u
                    break
                base = lower[u]
                for edge_id in adjacency[u]:
                    v = int(edge_v[edge_id])
                    candidate = base + float(edge_w[edge_id])
                    if candidate > lower[v] + tolerance:
                        lower[v] = candidate
                        parent_edge[v] = int(edge_id)
                        if lower[v] > upper[v] + tolerance:
                            conflict_node = v
                            break
                if conflict_node is not None:
                    break

            if conflict_node is not None:
                conflict_kind = "LOWER_BOUND_EXCEEDS_UPPER_BOUND"
                rev: List[int] = []
                cur = int(conflict_node)
                seen_nodes = set()
                while int(parent_edge[cur]) >= 0:
                    if cur in seen_nodes:
                        raise RuntimeError(
                            "Temporal graph parent chain unexpectedly contains a cycle"
                        )
                    seen_nodes.add(cur)
                    pe = int(parent_edge[cur])
                    rev.append(pe)
                    cur = int(edge_u[pe])
                conflict_edges = list(reversed(rev))

    feasible = (
        not graph_not_certified
        and not conflict_kind
        and conflict_node is None
    )
    certified_infeasible = (
        not graph_not_certified
        and bool(conflict_kind)
        and not feasible
    )
    cut_assignments = (
        cut_from_edges(conflict_edges) if certified_infeasible else {}
    )

    if conflict_edges:
        write_csv(pd.DataFrame(edge_rows(conflict_edges)), chain_path)
    else:
        write_csv(
            pd.DataFrame(
                columns=[
                    "edge_order", "edge_weight", "edge_type", "from_event",
                    "to_event", "kind", "station", "train_i", "train_j",
                    "value", "source", "component_id",
                ]
            ),
            chain_path,
        )

    if as_bool(cfg.get("inverse_temporal_graph_write_bounds"), True):
        rows = []
        for eid, (kind, station, train) in enumerate(events):
            node = int(event_node[eid])
            rows.append({
                "kind": kind,
                "station": station,
                "train_id": train,
                "graph_node": node,
                "lower_bound": lower[node],
                "upper_bound": (
                    upper[node] if math.isfinite(upper[node]) else None
                ),
                "finite_upper": int(math.isfinite(upper[node])),
            })
        write_csv(pd.DataFrame(rows), bounds_path)

    if feasible and as_bool(
        cfg.get("inverse_temporal_save_feasible_timetable"), True
    ):
        rows = []
        for train in trains:
            seq = data.path_by_train[train]
            for pos, station in enumerate(seq):
                arr = None
                dep = None
                if pos > 0:
                    arr_eid = event_to_id[_event_key("arr", station, train)]
                    arr = lower[int(event_node[arr_eid])]
                if pos < len(seq) - 1:
                    dep_eid = event_to_id[_event_key("dep", station, train)]
                    dep = lower[int(event_node[dep_eid])]
                rows.append({
                    "车次": train,
                    "方向": data.direction.get(train, ""),
                    "车站": station,
                    "到达时间": None if arr is None else round(float(arr), 6),
                    "出发时间": None if dep is None else round(float(dep), 6),
                    "是否停站": bool(data.stop_at.get((train, station), False)),
                    "站序": int(float(data.rows_by_train[train].iloc[pos]["站序"])),
                    "train_type": data.train_type.get(train, 300),
                })
        write_csv(pd.DataFrame(rows), feasible_tt_path)

    wall_sec = time.perf_counter() - wall_start
    cpu_sec = time.process_time() - cpu_start
    if feasible:
        final_status_code = int(GRB.OPTIMAL)
        graph_status_name = "GRAPH_FEASIBLE"
    elif certified_infeasible:
        final_status_code = int(GRB.INFEASIBLE)
        graph_status_name = f"GRAPH_INFEASIBLE_{conflict_kind}"
    else:
        final_status_code = int(GRB.SUBOPTIMAL)
        graph_status_name = conflict_kind or "GRAPH_NOT_CERTIFIED"

    free_precedence_edges = sum(
        1 for e in conflict_edges if int(edge_component[e]) >= 0
    )
    forced_precedence_edges = sum(
        1
        for e in conflict_edges
        if int(edge_ref[e]) >= 0 and int(edge_component[e]) == -1
    )
    summary = {
        "alpha": float(alpha),
        "check_round": int(check_round),
        "check_mode": "graph",
        "first_status_code": final_status_code,
        "final_status_code": final_status_code,
        "status_name": graph_status_name,
        "sol_count": 1 if feasible else 0,
        "temporal_feasible": int(feasible),
        "runtime_sec": wall_sec,
        "wall_sec": wall_sec,
        "process_cpu_sec": cpu_sec,
        "time_limit": None,
        "threads": 1,
        "write_solver_log": 0,
        "console_log": 0,
        "fixed_precedence_count": len(priors),
        "conflict_edge_count": len(conflict_edges),
        "conflict_precedence_edge_count": forced_precedence_edges + free_precedence_edges,
        "conflict_forced_precedence_edge_count": forced_precedence_edges,
        "conflict_free_precedence_edge_count": free_precedence_edges,
        "conflict_has_bound_violation_node": int(conflict_node is not None),
        "cut_component_count": len(cut_assignments),
        "solver_log_file": "",
        "conflict_chain_file": relative_to_project(chain_path),
        "feasible_timetable_file": (
            relative_to_project(feasible_tt_path)
            if feasible_tt_path.exists()
            else ""
        ),
        "graph_event_count": len(events),
        "graph_contracted_node_count": node_count,
        "graph_edge_count": len(edge_v),
        "graph_nonpositive_noncontracted_edge_exists": int(
            has_nonpositive_noncontracted_edge
        ),
        "graph_conflict_kind": conflict_kind,
        "graph_conflict_edge_count": len(conflict_edges),
        "graph_conflict_free_component_count": len(cut_assignments),
        "graph_not_certified": int(graph_not_certified),
        "graph_conflict_chain_file": relative_to_project(chain_path),
        "graph_bounds_file": (
            relative_to_project(bounds_path) if bounds_path.exists() else ""
        ),
    }
    write_csv(
        pd.DataFrame([summary]),
        check_dir / "temporal_check_summary.csv",
    )
    return {**summary, "cut_assignments": cut_assignments}


def _run_temporal_feasibility_check(
    cfg: Mapping[str, Any],
    ctx: ExperimentContext,
    data: HSRData,
    trains: Sequence[str],
    partition: PrecedenceCorePartition,
    priors: Mapping[PairKey, int],
    component_values: Mapping[int, int],
    *,
    alpha: float,
    check_round: int,
    universe: Optional[Sequence[PairKey]] = None,
) -> Dict[str, Any]:
    """Run graph-based Stage-5 temporal certification only."""
    if universe is None:
        universe = data.event_universe(trains)
    return _run_temporal_feasibility_check_graph(
        cfg, ctx, data, trains, universe, partition, priors, component_values,
        alpha=alpha, check_round=check_round,
    )


def _add_exact_temporal_nogood_cut(
    model: Model,
    y: Mapping[int, Any],
    cut_assignments: Mapping[int, int],
    *,
    cut_index: int,
) -> Any:
    """Exclude one graph-certified infeasible conjunction of core directions."""
    if not cut_assignments:
        raise ValueError("空 temporal cut 不允许加入 inverse master")
    expr = LinExpr()
    for comp, value in sorted(cut_assignments.items()):
        if int(value) == 1:
            expr += 1.0 - y[int(comp)]
        else:
            expr += y[int(comp)]
    constr = model.addConstr(
        expr >= 1.0,
        name=f"TEMP_FEAS_CUT_{int(cut_index):04d}",
    )
    model.update()
    return constr


def _set_partial_start_after_temporal_cut(
    y: Mapping[int, Any],
    component_values: Mapping[int, int],
    cut_assignments: Mapping[int, int],
) -> None:
    """Reuse the previous incumbent without fixing the rest of the core.

    Values outside the conflict cut are supplied only as Start values.  Values
    inside the cut are left undefined so Gurobi may choose which one(s) to
    change.  All variables remain decision variables in the re-optimization.
    """
    cut_components = {int(c) for c in cut_assignments}
    for comp, var in y.items():
        if int(comp) in cut_components:
            var.Start = GRB.UNDEFINED
        else:
            var.Start = float(component_values[int(comp)])


# -----------------------------------------------------------------------------
# Stage-5 cross-round restart archive
# -----------------------------------------------------------------------------

def _configure_inverse_restart_archive(
    cfg: Mapping[str, Any],
    model: Model,
) -> Dict[str, Any]:
    """Keep a small native Gurobi solution pool for cross-round warm starts.

    PoolSearchMode deliberately remains 0 by default: we do not spend extra
    search effort enumerating alternatives.  We only retain alternative
    solutions that normal inverse branch-and-bound happens to discover.
    """
    enabled = as_bool(cfg.get("inverse_restart_archive_enabled"), False)
    pool_solutions = int(
        as_int(cfg.get("inverse_restart_pool_solutions"), 30) or 30
    )
    archive_max = int(
        as_int(cfg.get("inverse_restart_archive_max_solutions"), 200) or 200
    )
    pool_search_mode = int(
        as_int(cfg.get("inverse_restart_pool_search_mode"), 0) or 0
    )
    if pool_solutions <= 0:
        raise ValueError("inverse_restart_pool_solutions 必须为正整数")
    if archive_max <= 0:
        raise ValueError("inverse_restart_archive_max_solutions 必须为正整数")
    if pool_search_mode not in {0, 1, 2}:
        raise ValueError("inverse_restart_pool_search_mode 仅支持0/1/2")
    if enabled:
        model.setParam("PoolSolutions", pool_solutions)
        model.setParam("PoolSearchMode", pool_search_mode)
    return {
        "enabled": int(enabled),
        "pool_solutions": pool_solutions,
        "archive_max_solutions": archive_max,
        "pool_search_mode": pool_search_mode,
    }


def _snapshot_inverse_solution_pool(
    model: Model,
    y: Mapping[int, Any],
    component_ids: Sequence[int],
    *,
    master_round: int,
    max_solutions: int,
) -> List[Dict[str, Any]]:
    """Copy the current native solution pool before the model is modified.

    Binary assignments are stored as compact ``bytes`` in component_ids order.
    This is important because adding a new temporal cut can invalidate/discard
    Gurobi's current pool, while we still want to reuse surviving old solutions.
    """
    sol_count = int(safe_int(lambda: model.SolCount, 0) or 0)
    if sol_count <= 0:
        return []
    take = min(sol_count, max(1, int(max_solutions)))
    var_list = [y[int(comp)] for comp in component_ids]
    rows: List[Dict[str, Any]] = []
    old_solution_number = int(model.Params.SolutionNumber)
    try:
        for sol_num in range(take):
            model.setParam("SolutionNumber", int(sol_num))
            raw = model.getAttr(GRB.Attr.Xn, var_list)
            values = bytes(1 if float(v) >= 0.5 else 0 for v in raw)
            obj = safe_float(lambda: model.PoolObjVal)
            rows.append({
                "master_round_found": int(master_round),
                "pool_solution_number": int(sol_num),
                "objective": obj,
                "values": values,
            })
    finally:
        try:
            model.setParam("SolutionNumber", old_solution_number)
        except Exception:
            model.setParam("SolutionNumber", 0)
    return rows


def _merge_inverse_solution_archive(
    archive: List[Dict[str, Any]],
    archive_seen: set[bytes],
    new_entries: Sequence[Mapping[str, Any]],
    *,
    archive_max: int,
) -> int:
    """Merge newly discovered solutions, deduplicate, and retain best objective."""
    added = 0
    for rec in new_entries:
        blob = bytes(rec["values"])
        if blob in archive_seen:
            continue
        archive_seen.add(blob)
        archive.append(dict(rec))
        added += 1
    archive.sort(
        key=lambda r: (
            float("inf") if r.get("objective") is None else float(r["objective"]),
            int(r.get("master_round_found") or 0),
        )
    )
    if len(archive) > int(archive_max):
        removed = archive[int(archive_max):]
        del archive[int(archive_max):]
        # Values trimmed from the bounded archive may be rediscovered later.
        for rec in removed:
            archive_seen.discard(bytes(rec["values"]))
    return added


def _archive_solution_satisfies_temporal_cuts(
    values: bytes,
    component_position: Mapping[int, int],
    cut_signatures: Sequence[Sequence[Tuple[int, int]]],
) -> bool:
    """Return True iff the binary vector satisfies every accumulated no-good cut."""
    for signature in cut_signatures:
        # A no-good cut is violated exactly when every listed assignment matches.
        violates = True
        for comp, required in signature:
            pos = component_position.get(int(comp))
            if pos is None or int(values[int(pos)]) != int(required):
                violates = False
                break
        if violates:
            return False
    return True


def _best_cut_feasible_archive_solution(
    archive: Sequence[Mapping[str, Any]],
    component_position: Mapping[int, int],
    cut_signatures: Sequence[Sequence[Tuple[int, int]]],
) -> Tuple[Optional[Dict[str, Any]], int]:
    survivors: List[Mapping[str, Any]] = []
    for rec in archive:
        if _archive_solution_satisfies_temporal_cuts(
            bytes(rec["values"]), component_position, cut_signatures
        ):
            survivors.append(rec)
    if not survivors:
        return None, 0
    best = min(
        survivors,
        key=lambda r: float("inf")
        if r.get("objective") is None else float(r["objective"]),
    )
    return dict(best), len(survivors)


def _set_full_start_from_archive_solution(
    y: Mapping[int, Any],
    component_ids: Sequence[int],
    values: bytes,
) -> None:
    """Supply one complete binary warm start from a cut-feasible archived solution."""
    if len(values) != len(component_ids):
        raise ValueError("archive warm-start vector length mismatch")
    for pos, comp in enumerate(component_ids):
        y[int(comp)].Start = float(int(values[pos]))


def _clear_inverse_binary_start(
    y: Mapping[int, Any],
) -> None:
    for var in y.values():
        var.Start = GRB.UNDEFINED


def run_inverse_one_alpha(
    cfg: Mapping[str, Any],
    ctx: ExperimentContext,
    data: HSRData,
    *,
    alpha: float,
    trains: Sequence[str],
    universe: Sequence[PairKey],
    obs: Mapping[PairKey, Mapping[str, int]],
    eta: Mapping[str, float],
    vote_priors: Mapping[PairKey, int],
    partition: PrecedenceCorePartition,
) -> Dict[str, Any]:
    outdir = ensure_dir(ctx.inverse_dir / alpha_tag(alpha))
    paper_mode = _paper_output_mode(cfg)
    write_inverse_details = _cfg_flag(cfg, "write_inverse_precedence_details", paper_default=False, normal_default=True)
    write_inverse_full_assignment = _cfg_flag(cfg, "write_inverse_full_assignment_csv", paper_default=False, normal_default=True)
    # An old accepted solution must never survive a new run that fails to prove
    # optimality.  Delete deployable artefacts before model construction.
    for stale_name in (
        "learned_core_precedence.csv",
        "learned_core_components.csv",
        "learned_priors_fullfix.csv",
        "learned_core_components_cache.pkl",
    ):
        stale_path = outdir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    stage_start = time.perf_counter()
    stage_cpu_start = time.process_time()
    eta_values = [float(eta.get(k, 0.0)) for k in trains]
    eta_bar = sum(eta_values) / len(eta_values) if eta_values else 0.0
    epsilon = float(as_float(cfg.get("dual_price_epsilon"), 1e-9) or 1e-9)
    min_obs = int(as_int(cfg.get("min_obs"), 1) or 1)
    # Backward-compatible inverse acceptance policy.
    # - inverse_require_optimal=TRUE: only GRB.OPTIMAL may be deployed.
    # - otherwise, an OPTIMAL solve (including termination at the configured
    #   MIPGap) or a TIME_LIMIT solve with an incumbent may be deployed.
    require_optimal = as_bool(cfg.get("inverse_require_optimal"), False)
    accept_time_limit_incumbent = as_bool(
        cfg.get("inverse_accept_time_limit_incumbent"),
        not require_optimal,
    )
    if require_optimal:
        accept_time_limit_incumbent = False
    inverse_target_mipgap = float(
        as_float(cfg.get("inverse_mipgap"), as_float(cfg.get("mipgap"), 0.0))
        or 0.0
    )

    forced_conflict_indices = 0
    forced_conflicting_labels = 0
    for key, value in partition.forced.items():
        rec = obs.get(key, {"N": 0, "N1": 0})
        n, n1 = int(rec.get("N", 0)), int(rec.get("N1", 0))
        conflicts = (n - n1) if int(value) == 1 else n1
        if conflicts > 0:
            forced_conflict_indices += 1
            forced_conflicting_labels += conflicts
    if forced_conflicting_labels > 0 and as_bool(
        cfg.get("precedence_core_fail_on_observation_conflict"), False
    ):
        raise RuntimeError(
            "固定precedence与受限实例标签冲突："
            f"indices={forced_conflict_indices}, labels={forced_conflicting_labels}。"
            "当前NO4B主流程不应出现固定precedence。"
        )

    build_start = time.perf_counter()
    model = Model(f"InverseCore_{ctx.experiment_tag}_{alpha_tag(alpha)}")

    # Persist one complete Gurobi log per alpha.  The log file is configured
    # before all other solver parameters so parameter changes, presolve, root
    # relaxation, incumbent/bound trajectories, and the final status are kept
    # even if the run is interrupted.
    inverse_write_solver_log = as_bool(
        cfg.get("inverse_write_solver_log"), True
    )
    inverse_log_filename = str(
        cfg.get("inverse_log_filename") or "gurobi_inverse.log"
    ).strip() or "gurobi_inverse.log"
    inverse_log_filename = (
        inverse_log_filename
        .replace("{alpha}", str(alpha))
        .replace("{alpha_tag}", alpha_tag(alpha))
    )
    inverse_log_path = outdir / inverse_log_filename
    if inverse_write_solver_log:
        inverse_log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if inverse_log_path.exists():
                inverse_log_path.unlink()
        except Exception:
            pass
        _set_gurobi_log_file(model, inverse_log_path)

    configure_model(model, cfg, "inverse")

    # Keep large branch-and-bound trees from exhausting RAM. This changes only
    # node storage, not the mathematical model.
    inverse_nodefile_start_gb = float(
        as_float(cfg.get("inverse_nodefile_start_gb"), 0.5) or 0.0
    )
    if inverse_nodefile_start_gb > 0.0:
        model.setParam("NodefileStart", inverse_nodefile_start_gb)
    inverse_nodefile_dir_raw = str(cfg.get("inverse_nodefile_dir") or "").strip()
    inverse_nodefile_dir = ""
    if inverse_nodefile_dir_raw:
        nodefile_path = Path(inverse_nodefile_dir_raw)
        if not nodefile_path.is_absolute():
            nodefile_path = (project_root() / nodefile_path).resolve()
        nodefile_path.mkdir(parents=True, exist_ok=True)
        inverse_nodefile_dir = str(nodefile_path)
        model.setParam("NodefileDir", inverse_nodefile_dir)

    restart_archive_cfg = _configure_inverse_restart_archive(cfg, model)
    component_ids = sorted(partition.component_members)
    y = {
        comp: model.addVar(vtype=GRB.BINARY, name=f"y_core_{comp}")
        for comp in component_ids
    }
    model.update()

    c1_mode = str(cfg.get("inverse_c1_mode") or "branch_cut_triangle").strip().lower()
    rank_vars: Dict[Tuple[str, str, str], Any] = {}
    c1_stats: Dict[str, int] = {
        "fixed_one_sided_rank_rows": 0,
        "free_two_sided_rank_rows": 0,
        "total_rank_rows": 0,
    }
    c1_lazy_layers: List[Tuple[str, str, Tuple[str, ...]]] = []
    c1_lazy_callback = None
    c1_lazy_state: Dict[str, Any] = {
        "callback_calls": 0,
        "mipsol_calls": 0,
        "mipnode_calls": 0,
        "mipnode_root_calls": 0,
        "mipnode_usercut_passes": 0,
        "candidate_layers_checked": 0,
        "usercut_layers_checked": 0,
        "cycle_cuts_added": 0,
        "user_cuts_added": 0,
        "usercut_candidate_triangles": 0,
        "duplicate_cycle_signatures": 0,
        "duplicate_usercut_signatures": 0,
        "max_cuts_per_callback": 0,
        "seen_signatures": set(),
        "seen_usercut_signatures": set(),
        "cut_records": [],
        "usercut_records": [],
        "max_usercut_violation": 0.0,
        "error": "",
    }
    if ctx.use_c1:
        if c1_mode in {"rank", "rank_big_m", "legacy"}:
            rank_vars, c1_stats = _add_rank_acyclicity_core(
                model, data, trains, universe, partition, y
            )
            c1_mode = "rank"
        elif c1_mode in {"branch_cut", "branch_cut_triangle", "triangle_branch_cut", "usercut_lazy"}:
            c1_mode = "branch_cut_triangle"
            c1_lazy_layers = _prepare_c1_lazy_layers(data, trains)
            model.setParam("LazyConstraints", 1)
            # Required so callback user cuts can be mapped reliably after presolve.
            model.setParam("PreCrush", 1)
            c1_lazy_callback, c1_lazy_state = _make_inverse_c1_branchcut_callback(
                cfg,
                data,
                partition,
                y,
                component_ids,
                c1_lazy_layers,
            )
        elif c1_mode in {"lazy", "lazy_cycle", "lazy_triangle"}:
            c1_mode = "lazy_cycle"
            c1_lazy_layers = _prepare_c1_lazy_layers(data, trains)
            model.setParam("LazyConstraints", 1)
            c1_lazy_callback, c1_lazy_state = _make_inverse_c1_lazy_callback(
                cfg,
                data,
                partition,
                y,
                component_ids,
                c1_lazy_layers,
            )
        else:
            raise ValueError(
                "inverse_c1_mode 仅支持 branch_cut_triangle、lazy_cycle 或 rank"
            )
    else:
        c1_mode = "disabled"
    c2_checked = _verify_c2_collapsed(data, trains, partition) if ctx.use_c2 else 0
    c3_stats = (
        _add_inverse_c3_core(model, data, trains, partition, y)
        if ctx.use_c3
        else {
            "C3_rows_instantiated": 0,
            "C3_fixed_checks": 0,
            "C3_same_component_equalities_skipped": 0,
        }
    )

    component_coefficients: Dict[int, float] = defaultdict(float)
    component_weighted_evidence: Dict[int, float] = defaultdict(float)
    component_raw_observations: Dict[int, int] = defaultdict(int)
    objective_constant = 0.0
    observed_in_objective = 0
    fixed_observed_indices = 0
    free_observed_indices = 0
    for key in universe:
        rec = obs.get(key, {"N": 0, "N1": 0})
        n, n1 = int(rec["N"]), int(rec["N1"])
        if n < min_obs:
            continue
        _, _, i, j = key
        pair_price = (
            float(eta.get(i, 0.0)) + float(eta.get(j, 0.0))
        ) / (2.0 * (eta_bar + epsilon))
        weight = 1.0 + float(alpha) * pair_price
        base = weight * n1
        coefficient = weight * (n - 2 * n1)
        observed_in_objective += 1
        if key in partition.forced:
            objective_constant += base + coefficient * int(partition.forced[key])
            fixed_observed_indices += 1
        else:
            objective_constant += base
            comp = int(partition.component_by_key[key])
            component_coefficients[comp] += coefficient
            component_weighted_evidence[comp] += weight * n
            component_raw_observations[comp] += n
            free_observed_indices += 1
    if observed_in_objective == 0:
        raise RuntimeError("逆优化目标为空")

    objective = LinExpr()
    objective += float(objective_constant)
    for comp, coefficient in component_coefficients.items():
        if abs(coefficient) > 0.0:
            objective += float(coefficient) * y[comp]
    model.setObjective(objective, GRB.MINIMIZE)
    model.update()

    sample_guidance_stats = _apply_inverse_sample_guidance(
        cfg,
        outdir,
        y,
        component_ids,
        component_coefficients,
        component_weighted_evidence,
        component_raw_observations,
    )
    model.update()
    build_sec = time.perf_counter() - build_start

    temporal_cuts_enabled = as_bool(
        cfg.get("inverse_temporal_cuts"), False
    )
    _temporal_max_cuts_raw = as_int(
        cfg.get("inverse_temporal_max_cuts"), 20
    )
    temporal_max_cuts = (
        20 if _temporal_max_cuts_raw is None else int(_temporal_max_cuts_raw)
    )
    if temporal_max_cuts < 0:
        raise ValueError("inverse_temporal_max_cuts 不能为负数")
    temporal_use_partial_start = as_bool(
        cfg.get("inverse_temporal_use_partial_start"), True
    )
    restart_fallback_partial = as_bool(
        cfg.get("inverse_restart_fallback_partial_start"), True
    )
    component_position = {
        int(comp): pos for pos, comp in enumerate(component_ids)
    }
    inverse_solution_archive: List[Dict[str, Any]] = []
    inverse_solution_archive_seen: set[bytes] = set()
    accumulated_temporal_cut_signatures: List[Tuple[Tuple[int, int], ...]] = []
    restart_history_rows: List[Dict[str, Any]] = []
    next_round_start_source = (
        "SAMPLE_GUIDANCE"
        if int(sample_guidance_stats.get("sample_partial_start_component_count") or 0) > 0
        else "NONE"
    )
    next_round_start_obj: Optional[float] = None

    # Remove stale temporal-cut diagnostics from a previous run of this alpha.
    temporal_root = outdir / "temporal_checks"
    if temporal_root.exists():
        shutil.rmtree(temporal_root)
    for stale_name in (
        "inverse_round_history.csv",
        "temporal_cut_history.csv",
        "c1_lazy_cut_history.csv",
        "c1_usercut_history.csv",
        "inverse_restart_history.csv",
    ):
        stale_path = outdir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    actual_threads = int(model.Params.Threads)
    actual_mipfocus = int(model.Params.MIPFocus)

    print("\n" + "=" * 104)
    print(
        f"[5] Ordinary inverse + structural quotient "
        f"{alpha_tag(alpha)} | K={len(trains)} | "
        f"C1/C2/C3={ctx.use_c1}/{ctx.use_c2}/{ctx.use_c3}"
    )
    print(
        f"alpha={alpha}, |U|={len(universe):,}, "
        f"fixed precedence=0, "
        f"event precedence={len(partition.component_by_key):,}, "
        f"binary components={len(component_ids):,}"
    )
    print(
        f"objective indices={observed_in_objective:,} "
        f"(fixed/free={fixed_observed_indices:,}/{free_observed_indices:,})"
    )
    print(
        "acceptance policy: "
        f"target MIPGap={inverse_target_mipgap:.6g}; "
        f"require OPTIMAL={require_optimal}; "
        f"accept TIME_LIMIT incumbent={accept_time_limit_incumbent}"
    )
    print(
        "solver controls: "
        f"Threads={actual_threads}; MIPFocus={actual_mipfocus}; "
        "NoRelHeurTime/NoRelHeurWork=NOT USED"
    )
    print(
        "sample guidance: "
        f"partial Start={sample_guidance_stats['sample_partial_start_component_count']:,}; "
        f"VarHint={sample_guidance_stats['sample_hint_component_count']:,}; "
        f"observed components={sample_guidance_stats['sample_guidance_observed_component_count']:,}"
    )
    print(
        "cross-round restart archive: "
        f"enabled={bool(restart_archive_cfg['enabled'])}; "
        f"PoolSolutions={restart_archive_cfg['pool_solutions']}; "
        f"PoolSearchMode={restart_archive_cfg['pool_search_mode']}; "
        f"archive max={restart_archive_cfg['archive_max_solutions']}"
    )
    print(
        "C1 implementation: "
        f"mode={c1_mode}; rank_vars={len(rank_vars):,}; "
        f"rank_rows={c1_stats['total_rank_rows']:,}; "
        f"triangle_layers={len(c1_lazy_layers):,}; "
        f"MIPNODE usercuts={as_bool(cfg.get('inverse_c1_usercuts_enabled'), True)}; "
        f"MIPSOL lazy=ON"
    )
    print(
        "node storage: "
        f"NodefileStart={inverse_nodefile_start_gb:g} GB; "
        f"NodefileDir={inverse_nodefile_dir or 'Gurobi default'}"
    )
    print(
        "temporal certification: "
        f"enabled={temporal_cuts_enabled}; "
        f"mode={cfg.get('inverse_temporal_check_mode', 'graph')}; "
        f"max cuts={temporal_max_cuts}; "
        f"partial previous-incumbent start={temporal_use_partial_start}"
    )
    print(f"patch version: {TEMPORAL_CUT_PATCH_VERSION}")
    print("=" * 104)

    gap_tolerance = max(
        1e-12, 1e-9 * max(1.0, abs(inverse_target_mipgap))
    )
    accepted = False
    acceptance_reason = "NOT_SOLVED"
    candidate_acceptance_reason = "NOT_SOLVED"
    temporal_certified: Optional[bool] = None
    temporal_final_status = "NOT_RUN"
    temporal_check_count = 0
    temporal_cut_count = 0
    temporal_forced_only_failure = 0
    temporal_duplicate_cut_failure = 0
    temporal_check_runtime_total = 0.0
    temporal_check_wall_total = 0.0
    inverse_solver_runtime_total = 0.0
    inverse_optimize_wall_total = 0.0
    inverse_optimize_cpu_total = 0.0
    round_history_rows: List[Dict[str, Any]] = []
    cut_history_rows: List[Dict[str, Any]] = []
    seen_cut_signatures: set[Tuple[Tuple[int, int], ...]] = set()
    last_solve_info: Dict[str, Any] = {}
    final_mip_gap: Optional[float] = None
    gap_target_reached = False
    status_code = int(GRB.LOADED)
    sol_count = 0
    has_any_solution = False

    # At most max_cuts infeasible candidates can be excluded; therefore the
    # number of master solves is at most max_cuts + 1.
    for master_round in range(1, temporal_max_cuts + 2):
        round_start_source = str(next_round_start_source)
        round_start_obj = next_round_start_obj
        print("\n" + "-" * 104)
        print(
            f"[5][{alpha_tag(alpha)}] inverse master round {master_round} | "
            f"existing temporal cuts={temporal_cut_count} | "
            f"Start={round_start_source}"
        )
        print("-" * 104)

        last_solve_info = optimize_with_status_resolution(
            model,
            cfg,
            stage="inverse",
            callback=c1_lazy_callback,
        )
        if c1_lazy_state.get("error"):
            raise RuntimeError(
                "C1 lazy callback failed: " + str(c1_lazy_state.get("error"))
            )
        inverse_solver_runtime_total += float(
            last_solve_info.get("solver_runtime_accumulated_sec") or 0.0
        )
        inverse_optimize_wall_total += float(
            last_solve_info.get("optimize_wall_sec") or 0.0
        )
        inverse_optimize_cpu_total += float(
            last_solve_info.get("optimize_process_cpu_sec") or 0.0
        )

        status_code = int(model.Status)
        sol_count = int(safe_int(lambda: model.SolCount, 0) or 0)
        has_any_solution = has_usable_solution(
            model, keep_time_limit=True
        )
        final_mip_gap = (
            safe_float(lambda: model.MIPGap)
            if has_any_solution
            else None
        )
        gap_target_reached = bool(
            has_any_solution
            and final_mip_gap is not None
            and float(final_mip_gap)
            <= inverse_target_mipgap + gap_tolerance
        )

        if status_code == int(GRB.OPTIMAL) and sol_count > 0:
            candidate_accepted = True
            candidate_acceptance_reason = (
                "TARGET_MIPGAP_REACHED"
                if inverse_target_mipgap > 0
                else "OPTIMAL_STATUS"
            )
        elif (
            status_code == int(GRB.TIME_LIMIT)
            and sol_count > 0
            and accept_time_limit_incumbent
            and not require_optimal
        ):
            candidate_accepted = True
            candidate_acceptance_reason = "TIME_LIMIT_INCUMBENT"
        else:
            candidate_accepted = False
            if (
                status_code == int(GRB.TIME_LIMIT)
                and sol_count <= 0
            ):
                candidate_acceptance_reason = (
                    "REJECTED_TIME_LIMIT_NO_INCUMBENT"
                )
            elif status_code == int(GRB.TIME_LIMIT):
                candidate_acceptance_reason = (
                    "REJECTED_TIME_LIMIT_BY_POLICY"
                )
            elif require_optimal:
                candidate_acceptance_reason = "REJECTED_NOT_OPTIMAL"
            else:
                candidate_acceptance_reason = (
                    "REJECTED_STATUS_OR_NO_SOLUTION"
                )

        pool_entries_this_round: List[Dict[str, Any]] = []
        archive_added_this_round = 0
        if bool(restart_archive_cfg.get("enabled")) and has_any_solution:
            pool_entries_this_round = _snapshot_inverse_solution_pool(
                model, y, component_ids,
                master_round=master_round,
                max_solutions=int(restart_archive_cfg["pool_solutions"]),
            )
            archive_added_this_round = _merge_inverse_solution_archive(
                inverse_solution_archive,
                inverse_solution_archive_seen,
                pool_entries_this_round,
                archive_max=int(restart_archive_cfg["archive_max_solutions"]),
            )

        round_row: Dict[str, Any] = {
            "master_round": master_round,
            "start_source_used": round_start_source,
            "start_objective_used": round_start_obj,
            "solution_pool_count_captured": len(pool_entries_this_round),
            "solution_archive_added": archive_added_this_round,
            "solution_archive_size": len(inverse_solution_archive),
            "temporal_cuts_before_solve": temporal_cut_count,
            "status_code": status_code,
            "status_name": status_name(status_code),
            "sol_count": sol_count,
            "obj_val": (
                safe_float(lambda: model.ObjVal)
                if has_any_solution
                else None
            ),
            "obj_bound": safe_float(lambda: model.ObjBound),
            "mip_gap": final_mip_gap,
            "gap_target_reached": int(gap_target_reached),
            "candidate_accepted_by_mip_policy": int(candidate_accepted),
            "candidate_acceptance_reason": candidate_acceptance_reason,
            "solver_runtime_accumulated_sec_this_round": (
                last_solve_info.get("solver_runtime_accumulated_sec")
            ),
            "optimize_wall_sec_this_round": (
                last_solve_info.get("optimize_wall_sec")
            ),
            "temporal_check_status": "NOT_RUN",
            "temporal_feasible": None,
            "cut_added": 0,
            "cut_component_count": 0,
        }

        print(
            f"inverse candidate: status={status_name(status_code)}, "
            f"SolCount={sol_count}, gap={final_mip_gap}, "
            f"target reached={gap_target_reached}, "
            f"accepted by MIP policy={candidate_accepted}"
        )

        if not candidate_accepted:
            accepted = False
            acceptance_reason = candidate_acceptance_reason
            round_history_rows.append(round_row)
            break

        if not temporal_cuts_enabled:
            accepted = True
            acceptance_reason = candidate_acceptance_reason
            temporal_certified = None
            temporal_final_status = "DISABLED"
            round_history_rows.append(round_row)
            break

        component_values, candidate_priors = (
            _core_candidate_values_and_priors(
                partition, component_ids, y
            )
        )
        temporal_check_count += 1
        print(
            f"[Temporal check] round={temporal_check_count}: "
            "fix complete precedence and test SAME target windows..."
        )
        temporal_result = _run_temporal_feasibility_check(
            cfg,
            ctx,
            data,
            trains,
            partition,
            candidate_priors,
            component_values,
            alpha=alpha,
            check_round=temporal_check_count,
            universe=universe,
        )
        temporal_check_runtime_total += float(
            temporal_result.get("runtime_sec") or 0.0
        )
        temporal_check_wall_total += float(
            temporal_result.get("wall_sec") or 0.0
        )
        temporal_final_status = str(
            temporal_result.get("status_name") or "UNKNOWN"
        )
        is_temporal_feasible = bool(
            int(temporal_result.get("temporal_feasible") or 0)
        )
        round_row["temporal_check_status"] = temporal_final_status
        round_row["temporal_feasible"] = int(is_temporal_feasible)
        round_row["temporal_conflict_edge_count"] = (
            temporal_result.get("conflict_edge_count")
        )
        round_row["temporal_conflict_free_precedence_edge_count"] = (
            temporal_result.get("conflict_free_precedence_edge_count")
        )

        if is_temporal_feasible:
            temporal_certified = True
            accepted = True
            acceptance_reason = (
                f"{candidate_acceptance_reason}_TEMPORAL_CERTIFIED"
            )
            round_history_rows.append(round_row)
            print(
                "[Temporal check] PASS: current inverse precedence is "
                "realizable under the unchanged target windows."
            )
            break

        if int(temporal_result.get("final_status_code", -1)) != int(
            GRB.INFEASIBLE
        ):
            temporal_certified = False
            accepted = False
            acceptance_reason = (
                "TEMPORAL_CHECK_NOT_CERTIFIED_"
                f"{temporal_final_status}"
            )
            round_history_rows.append(round_row)
            print(
                "[Temporal check] no infeasibility certificate was obtained; "
                "stop without exporting Full-Fix priors."
            )
            break

        temporal_certified = False
        cut_assignments = {
            int(k): int(v)
            for k, v in dict(
                temporal_result.get("cut_assignments") or {}
            ).items()
        }

        if not cut_assignments:
            temporal_forced_only_failure = 1
            accepted = False
            acceptance_reason = (
                "TEMPORAL_INFEASIBLE_WITH_NO_FREE_CORE_IN_CONFLICT"
            )
            round_history_rows.append(round_row)
            print(
                "[Temporal check] INFEASIBLE, but the conflict chain contains no "
                "learned-core component that can be cut.  This points to "
                "the timing backbone rather than a "
                "repairable inverse combination."
            )
            break

        if temporal_cut_count >= temporal_max_cuts:
            accepted = False
            acceptance_reason = "TEMPORAL_MAX_CUTS_REACHED"
            round_history_rows.append(round_row)
            print(
                f"[Temporal check] INFEASIBLE and max cuts "
                f"({temporal_max_cuts}) reached."
            )
            break

        signature = tuple(sorted(cut_assignments.items()))
        if signature in seen_cut_signatures:
            temporal_duplicate_cut_failure = 1
            accepted = False
            acceptance_reason = "TEMPORAL_DUPLICATE_CONFLICT_CUT"
            round_history_rows.append(round_row)
            print(
                "[Temporal check] duplicate temporal cut signature detected; "
                "stop to avoid an infinite loop."
            )
            break
        seen_cut_signatures.add(signature)

        temporal_cut_count += 1
        accumulated_temporal_cut_signatures.append(signature)
        _add_exact_temporal_nogood_cut(
            model,
            y,
            cut_assignments,
            cut_index=temporal_cut_count,
        )

        # Preferred next-round Start: among every archived solution discovered
        # so far, choose the best one that already satisfies ALL accumulated
        # temporal no-good cuts.  The just-rejected best incumbent itself cannot
        # survive the newly generated cut, but an alternative solution from the
        # previous search often can.
        archive_best = None
        archive_survivor_count = 0
        if bool(restart_archive_cfg.get("enabled")):
            archive_best, archive_survivor_count = (
                _best_cut_feasible_archive_solution(
                    inverse_solution_archive,
                    component_position,
                    accumulated_temporal_cut_signatures,
                )
            )

        restart_source = "NONE"
        restart_obj = None
        restart_origin_round = None
        restart_pool_solution_number = None
        if archive_best is not None:
            _set_full_start_from_archive_solution(
                y, component_ids, bytes(archive_best["values"])
            )
            restart_source = "ARCHIVE_BEST_CUT_FEASIBLE"
            restart_obj = archive_best.get("objective")
            restart_origin_round = archive_best.get("master_round_found")
            restart_pool_solution_number = archive_best.get(
                "pool_solution_number"
            )
        elif restart_fallback_partial and temporal_use_partial_start:
            _set_partial_start_after_temporal_cut(
                y, component_values, cut_assignments
            )
            restart_source = "PREVIOUS_INCUMBENT_PARTIAL_REPAIR"
            restart_obj = round_row.get("obj_val")
            restart_origin_round = master_round
        else:
            _clear_inverse_binary_start(y)

        model.update()
        next_round_start_source = restart_source
        next_round_start_obj = (
            None if restart_obj is None else float(restart_obj)
        )
        restart_history_rows.append({
            "after_master_round": master_round,
            "temporal_cut_index": temporal_cut_count,
            "archive_size": len(inverse_solution_archive),
            "archive_survivor_count_after_all_cuts": archive_survivor_count,
            "next_round_start_source": restart_source,
            "next_round_start_objective": restart_obj,
            "start_origin_master_round": restart_origin_round,
            "start_origin_pool_solution_number": restart_pool_solution_number,
            "fallback_partial_enabled": int(
                restart_fallback_partial and temporal_use_partial_start
            ),
        })
        write_csv(
            pd.DataFrame(restart_history_rows),
            outdir / "inverse_restart_history.csv",
        )
        print(
            f"[Restart] next round Start={restart_source}; "
            f"archive survivors={archive_survivor_count}; "
            f"Start obj={restart_obj}"
        )

        round_row["cut_added"] = 1
        round_row["cut_component_count"] = len(cut_assignments)
        round_history_rows.append(round_row)

        cut_history_rows.append({
            "cut_index": temporal_cut_count,
            "master_round_found": master_round,
            "temporal_check_round": temporal_check_count,
            "cut_component_count": len(cut_assignments),
            "candidate_obj_val": round_row.get("obj_val"),
            "candidate_mip_gap": final_mip_gap,
            "conflict_edge_count": temporal_result.get(
                "conflict_edge_count"
            ),
            "conflict_precedence_edge_count": temporal_result.get(
                "conflict_precedence_edge_count"
            ),
            "conflict_forced_precedence_edge_count": temporal_result.get(
                "conflict_forced_precedence_edge_count"
            ),
            "conflict_free_precedence_edge_count": temporal_result.get(
                "conflict_free_precedence_edge_count"
            ),
            "cut_signature": ";".join(
                f"{comp}:{value}"
                for comp, value in signature
            ),
            "conflict_chain_file": temporal_result.get(
                "conflict_chain_file"
            ),
        })
        write_csv(
            pd.DataFrame(cut_history_rows),
            outdir / "temporal_cut_history.csv",
        )
        print(
            f"[Temporal check] INFEASIBLE -> added exact cut "
            f"#{temporal_cut_count} over {len(cut_assignments)} "
            "quotient components; re-optimize the SAME inverse master."
        )

    write_callback_history = _cfg_flag(cfg, "write_callback_cut_history", paper_default=False, normal_default=True)
    write_inverse_histories = _cfg_flag(cfg, "write_inverse_diagnostic_histories", paper_default=False, normal_default=True)
    c1_lazy_history_path = outdir / "c1_lazy_cut_history.csv"
    c1_usercut_history_path = outdir / "c1_usercut_history.csv"
    if write_callback_history:
        c1_lazy_records = list(c1_lazy_state.get("cut_records") or [])
        if c1_lazy_records: write_csv(pd.DataFrame(c1_lazy_records), c1_lazy_history_path)
        c1_usercut_records = list(c1_lazy_state.get("usercut_records") or [])
        if c1_usercut_records: write_csv(pd.DataFrame(c1_usercut_records), c1_usercut_history_path)
    else:
        for path in (c1_lazy_history_path, c1_usercut_history_path):
            if path.exists(): path.unlink()
    if write_inverse_histories:
        write_csv(pd.DataFrame(round_history_rows), outdir / "inverse_round_history.csv")
        if restart_history_rows: write_csv(pd.DataFrame(restart_history_rows), outdir / "inverse_restart_history.csv")
        if cut_history_rows: write_csv(pd.DataFrame(cut_history_rows), outdir / "temporal_cut_history.csv")
    else:
        for name in ("inverse_round_history.csv","inverse_restart_history.csv","temporal_cut_history.csv"):
            path=outdir/name
            if path.exists(): path.unlink()

    # Preserve the final-round fields for backward compatibility while making
    # the accumulated multi-round effort explicit.
    solve_info = dict(last_solve_info)
    solve_info["solver_runtime_accumulated_sec"] = (
        inverse_solver_runtime_total
    )
    solve_info["optimize_wall_sec"] = inverse_optimize_wall_total
    solve_info["optimize_process_cpu_sec"] = (
        inverse_optimize_cpu_total
    )

    print(
        f"inverse final: status={status_name(status_code)}, "
        f"SolCount={sol_count}, final gap={final_mip_gap}, "
        f"temporal certified={temporal_certified}, "
        f"cuts={temporal_cut_count}, accepted={accepted}, "
        f"reason={acceptance_reason}"
    )
    if c1_mode in {"branch_cut_triangle", "lazy_cycle"}:
        print(
            "C1 callback final: "
            f"MIPNODE={int(c1_lazy_state.get('mipnode_calls') or 0):,}, "
            f"user cuts={int(c1_lazy_state.get('user_cuts_added') or 0):,}, "
            f"MIPSOL={int(c1_lazy_state.get('mipsol_calls') or 0):,}, "
            f"lazy cuts={int(c1_lazy_state.get('cycle_cuts_added') or 0):,}, "
            f"max usercut violation={float(c1_lazy_state.get('max_usercut_violation') or 0.0):.6g}"
        )

    post_start = time.perf_counter()
    raw_mismatch: Optional[float] = None
    dual_weighted_component: Optional[float] = None
    weighted_recomputed: Optional[float] = None
    reconstruction_diff: Optional[float] = None
    vote_difference_count: Optional[int] = None
    audit: Dict[str, Any] = {}
    full_path = outdir / "learned_priors_fullfix.csv"
    core_path = outdir / "learned_core_precedence.csv"
    component_path = outdir / "learned_core_components.csv"
    if accepted:
        component_values = {comp: int(round(y[comp].X)) for comp in component_ids}
        _save_component_assignment_cache(_inverse_component_cache_path(ctx, alpha), component_values)
        priors_dict: Dict[PairKey, int] = dict(partition.forced)
        core_rows: List[Dict[str, Any]] = [] if write_inverse_details else []
        component_rows: List[Dict[str, Any]] = [] if write_inverse_details else []
        for comp in component_ids:
            value = component_values[comp]
            members = partition.component_members[comp]
            representative = members[0]
            if write_inverse_details:
                component_rows.append({
                    "component_id": comp, "value": value, "member_count": len(members),
                    "representative_kind": representative[0], "representative_station": representative[1],
                    "representative_train_i": representative[2], "representative_train_j": representative[3],
                    "objective_coefficient": component_coefficients.get(comp, 0.0),
                })
            for key in members:
                priors_dict[key] = value
                if write_inverse_details:
                    kind, station, i, j = key
                    core_rows.append({"kind":kind,"station":station,"train_i":i,"train_j":j,"value":value,"component_id":comp,"component_size":len(members)})
        if write_inverse_details:
            write_csv(pd.DataFrame(core_rows), core_path)
            write_csv(pd.DataFrame(component_rows), component_path)
        else:
            for path in (core_path, component_path):
                if path.exists(): path.unlink()

        raw_value = dual_value = weighted_value = 0.0
        vote_diff = 0
        full_rows = [] if write_inverse_full_assignment else None
        for key in universe:
            kind, station, i, j = key
            rec = obs.get(key, {"N": 0, "N1": 0})
            n, n1 = int(rec["N"]), int(rec["N1"])
            value = int(priors_dict[key])
            pair_price = (float(eta.get(i,0.0))+float(eta.get(j,0.0))) / (2.0*(eta_bar+epsilon))
            weight = 1.0 + float(alpha)*pair_price
            mismatch = float((n-n1) if value==1 else n1)
            in_objective = int(n >= min_obs)
            if in_objective:
                raw_value += mismatch
                dual_value += pair_price*mismatch
                weighted_value += weight*mismatch
            if key in vote_priors and value != int(vote_priors[key]): vote_diff += 1
            if full_rows is not None:
                full_rows.append({"kind":kind,"station":station,"train_i":i,"train_j":j,"value":value,"precedence_source":("PREPROCESS_FIXED" if key in partition.forced else "INVERSE_CORE"),"component_id":(None if key in partition.forced else partition.component_by_key[key]),"N":n,"N1":n1,"N0":n-n1,"eta_i":float(eta.get(i,0.0)),"eta_j":float(eta.get(j,0.0)),"p_pack":pair_price,"alpha":alpha,"weight":weight,"mismatch_count":mismatch,"in_objective":in_objective})
        if full_rows is not None:
            write_csv(pd.DataFrame(full_rows), full_path)
        elif full_path.exists(): full_path.unlink()
        raw_mismatch = raw_value
        dual_weighted_component = dual_value
        weighted_recomputed = weighted_value
        model_obj = safe_float(lambda: model.ObjVal)
        reconstruction_diff = abs(float(model_obj)-weighted_value) if model_obj is not None else None
        vote_difference_count = vote_diff
        if paper_mode and ctx.use_c1 and ctx.use_c2 and ctx.use_c3:
            audit = {"c1_checked_layers":None,"c1_cycle_layers":0,"c1_checked_triples":None,"c1_transitivity_violations":0,"c2_segment_order_violations":0,"c3_order_changes":None,"c3_stop_compatibility_violations":0,"c1_audit_mode":"MODEL_ENFORCED"}
        else:
            audit = audit_precedence_assignment(data, trains, priors_dict, count_c1_triplets=True)
    post_sec = time.perf_counter() - post_start
    total_sec = time.perf_counter() - stage_start
    cpu_sec = time.process_time() - stage_cpu_start

    solver_rec = solver_summary(model, has_any_solution)
    summary = {
        "experiment_group": ctx.experiment_group,
        "experiment_tag": ctx.experiment_tag,
        "target_k": len(trains),
        "alpha": alpha,
        "alpha_tag": alpha_tag(alpha),
        "min_obs": min_obs,
        "inverse_target_mipgap": inverse_target_mipgap,
        "inverse_write_solver_log": int(inverse_write_solver_log),
        "inverse_console_log": int(as_bool(cfg.get("inverse_console_log"), True)),
        "inverse_log_filename": inverse_log_filename if inverse_write_solver_log else "",
        "gurobi_log_file": (
            relative_to_project(inverse_log_path)
            if inverse_write_solver_log else ""
        ),
        "inverse_require_optimal": int(require_optimal),
        "inverse_accept_time_limit_incumbent": int(accept_time_limit_incumbent),
        "inverse_gap_target_reached": int(gap_target_reached),
        "inverse_acceptance_reason": acceptance_reason,
        "inverse_candidate_acceptance_reason": candidate_acceptance_reason,
        "accepted_for_fullfix": int(accepted),
        "temporal_cut_patch_version": TEMPORAL_CUT_PATCH_VERSION,
        "inverse_c1_mode": c1_mode,
        "inverse_c1_lazy_layer_count": len(c1_lazy_layers),
        "inverse_c1_lazy_mipsol_calls": int(c1_lazy_state.get("mipsol_calls") or 0),
        "inverse_c1_lazy_candidate_layers_checked": int(
            c1_lazy_state.get("candidate_layers_checked") or 0
        ),
        "inverse_c1_lazy_cycle_cuts_added": int(
            c1_lazy_state.get("cycle_cuts_added") or 0
        ),
        "inverse_c1_lazy_unique_cycle_signatures": len(
            c1_lazy_state.get("seen_signatures") or set()
        ),
        "inverse_c1_lazy_duplicate_cycle_signatures": int(
            c1_lazy_state.get("duplicate_cycle_signatures") or 0
        ),
        "inverse_c1_lazy_cut_history_csv": (relative_to_project(outdir / "c1_lazy_cut_history.csv") if write_callback_history else ""),
        "inverse_c1_mipnode_calls": int(c1_lazy_state.get("mipnode_calls") or 0),
        "inverse_c1_mipnode_root_calls": int(c1_lazy_state.get("mipnode_root_calls") or 0),
        "inverse_c1_mipnode_usercut_passes": int(
            c1_lazy_state.get("mipnode_usercut_passes") or 0
        ),
        "inverse_c1_usercut_layers_checked": int(
            c1_lazy_state.get("usercut_layers_checked") or 0
        ),
        "inverse_c1_usercut_candidate_triangles": int(
            c1_lazy_state.get("usercut_candidate_triangles") or 0
        ),
        "inverse_c1_user_cuts_added": int(
            c1_lazy_state.get("user_cuts_added") or 0
        ),
        "inverse_c1_usercut_unique_signatures": len(
            c1_lazy_state.get("seen_usercut_signatures") or set()
        ),
        "inverse_c1_usercut_duplicate_signatures": int(
            c1_lazy_state.get("duplicate_usercut_signatures") or 0
        ),
        "inverse_c1_max_usercut_violation": float(
            c1_lazy_state.get("max_usercut_violation") or 0.0
        ),
        "inverse_c1_usercut_history_csv": (relative_to_project(outdir / "c1_usercut_history.csv") if write_callback_history else ""),
        "inverse_nodefile_start_gb": inverse_nodefile_start_gb,
        "inverse_nodefile_dir": inverse_nodefile_dir,
        **sample_guidance_stats,
        "inverse_temporal_check_mode": str(
            cfg.get("inverse_temporal_check_mode") or "graph"
        ),
        "inverse_temporal_cuts_enabled": int(temporal_cuts_enabled),
        "inverse_temporal_certified": (
            None if temporal_certified is None else int(temporal_certified)
        ),
        "inverse_temporal_final_status": temporal_final_status,
        "inverse_temporal_check_count": temporal_check_count,
        "inverse_temporal_cut_count": temporal_cut_count,
        "inverse_temporal_max_cuts": temporal_max_cuts,
        "inverse_temporal_forced_only_failure": temporal_forced_only_failure,
        "inverse_temporal_duplicate_cut_failure": temporal_duplicate_cut_failure,
        "inverse_temporal_check_solver_runtime_sec": temporal_check_runtime_total,
        "inverse_temporal_check_wall_sec": temporal_check_wall_total,
        "inverse_master_solver_runtime_all_rounds_sec": inverse_solver_runtime_total,
        "inverse_master_optimize_wall_all_rounds_sec": inverse_optimize_wall_total,
        "inverse_master_round_count": len(round_history_rows),
        "inverse_threads_actual": actual_threads,
        "inverse_mipfocus_actual": actual_mipfocus,
        "inverse_norel_used": 0,
        "inverse_round_history_csv": relative_to_project(
            outdir / "inverse_round_history.csv"
        ),
        "temporal_cut_history_csv": relative_to_project(
            outdir / "temporal_cut_history.csv"
        ),
        "use_C1": int(ctx.use_c1),
        "use_C2": int(ctx.use_c2),
        "use_C3": int(ctx.use_c3),
        "universe_size": len(universe),
        "preprocess_fixed_precedence_count": len(partition.forced),
        "unresolved_event_core_count": len(partition.component_by_key),
        "unresolved_quotient_core_count": len(component_ids),
        "inverse_binary_reduction_percent": 100.0
        * (len(universe) - len(component_ids))
        / len(universe),
        "forced_observation_conflict_indices": forced_conflict_indices,
        "forced_observation_conflicting_labels": forced_conflicting_labels,
        "observed_indices_in_objective": observed_in_objective,
        "fixed_observed_indices_in_objective": fixed_observed_indices,
        "free_observed_indices_in_objective": free_observed_indices,
        "unobserved_indices": len(universe)
        - sum(1 for key in universe if int(obs.get(key, {}).get("N", 0)) > 0),
        "tied_observed_indices": sum(
            1
            for key in universe
            if int(obs.get(key, {}).get("N", 0)) > 0
            and 2 * int(obs.get(key, {}).get("N1", 0))
            == int(obs.get(key, {}).get("N", 0))
        ),
        "raw_mismatch_count": raw_mismatch,
        "dual_weighted_mismatch_component": dual_weighted_component,
        "weighted_mismatch_objective": solver_rec.get("obj_val"),
        "weighted_mismatch_recomputed": weighted_recomputed,
        "weighted_objective_reconstruction_abs_diff": reconstruction_diff,
        "different_from_vote_count": vote_difference_count,
        "num_C1_constraints": c1_stats["total_rank_rows"],
        "num_C1_fixed_one_sided_rows": c1_stats["fixed_one_sided_rank_rows"],
        "num_C1_free_two_sided_rows": c1_stats["free_two_sided_rank_rows"],
        "num_C1_lazy_cycle_cuts": int(c1_lazy_state.get("cycle_cuts_added") or 0),
        "num_C2_constraints": 0,
        "num_C2_equalities_collapsed": c2_checked,
        "num_C3_constraints": c3_stats["C3_rows_instantiated"],
        "num_C3_fixed_checks": c3_stats["C3_fixed_checks"],
        "num_C3_equalities_collapsed": c3_stats[
            "C3_same_component_equalities_skipped"
        ],
        "num_rank_vars": len(rank_vars),
        "eta_bar": eta_bar,
        "objective_fixed_constant_and_free_bases": objective_constant,
        "model_build_sec": build_sec,
        "postprocess_write_sec": post_sec,
        "total_wall_sec": total_sec,
        "process_cpu_sec": cpu_sec,
        **solve_info,
        **solver_rec,
        **audit,
        "forced_priors_csv": "",
        "inverse_partition_mode": "NO4B_STRUCTURAL_QUOTIENT",
        "paper_output_mode": int(paper_mode),
        "inverse_precedence_details_written": int(write_inverse_details),
        "inverse_full_assignment_csv_written": int(write_inverse_full_assignment),
        "learned_core_priors_csv": (relative_to_project(core_path) if accepted and write_inverse_details and core_path.exists() else ""),
        "learned_core_components_csv": (relative_to_project(component_path) if accepted and write_inverse_details and component_path.exists() else ""),
        "learned_core_components_cache": (relative_to_project(_inverse_component_cache_path(ctx, alpha)) if accepted else ""),
        "learned_priors_csv": (relative_to_project(full_path) if accepted and write_inverse_full_assignment and full_path.exists() else ""),
    }
    write_csv(pd.DataFrame([summary]), outdir / "inverse_summary.csv")
    append_stage_timing(
        ctx,
        {
            "stage": "5_inverse",
            "variant": alpha_tag(alpha),
            "status": summary["status_name"],
            "model_build_sec": build_sec,
            "solver_runtime_sec": summary["runtime_sec"],
            "solver_runtime_accumulated_sec": summary[
                "solver_runtime_accumulated_sec"
            ],
            "optimize_wall_sec": summary["optimize_wall_sec"],
            "process_cpu_sec": cpu_sec,
            "postprocess_write_sec": post_sec,
            "total_wall_sec": total_sec,
        },
    )
    model.dispose()
    gc.collect()
    return summary

def run_inverse_all_alphas(
    cfg: Mapping[str, Any],
    ctx: ExperimentContext,
    data: Optional[HSRData] = None,
) -> pd.DataFrame:
    whole_start = time.perf_counter()
    whole_cpu_start = time.process_time()
    data = data or HSRData(cfg)
    trains = data.select_trains(ctx.target_k)
    manifest = load_sample_manifest(ctx)
    universe = data.event_universe(trains)
    obs_start = time.perf_counter()
    cache_enabled = _cfg_flag(cfg, "cache_observation_counts", paper_default=True, normal_default=False)
    obs = _load_observation_cache(_observation_cache_path(ctx), trains, universe) if cache_enabled else None
    obs_source = "BINARY_CACHE" if obs is not None else "PAIRWISE_CSV"
    if obs is None:
        obs = aggregate_observations(data, trains, manifest)
        if cache_enabled:
            _save_observation_cache(_observation_cache_path(ctx), trains, universe, obs)
    obs_sec = time.perf_counter() - obs_start
    eta = load_train_prices(ctx)
    partition = build_inverse_structural_partition(cfg, ctx, data, trains)
    vote_priors = _load_vote_priors(ctx, data, trains, universe)
    alphas = parse_float_list(cfg.get("alpha_list"))
    if not alphas:
        raise ValueError("alpha_list 不能为空")
    if any(alpha < 0 for alpha in alphas):
        raise ValueError(f"alpha必须非负：{alphas}")
    if len(set(alphas)) != len(alphas):
        raise ValueError(f"alpha_list 不能包含重复值：{alphas}")

    summaries: List[Dict[str, Any]] = []
    continue_on_error = as_bool(cfg.get("continue_on_error"), True)
    for idx, alpha in enumerate(alphas, 1):
        print(f"\n##### alpha {idx}/{len(alphas)} = {alpha} #####")
        try:
            summaries.append(
                run_inverse_one_alpha(
                    cfg,
                    ctx,
                    data,
                    alpha=alpha,
                    trains=trains,
                    universe=universe,
                    obs=obs,
                    eta=eta,
                    vote_priors=vote_priors,
                    partition=partition,
                )
            )
        except Exception as exc:
            failure = {
                "experiment_group": ctx.experiment_group,
                "experiment_tag": ctx.experiment_tag,
                "target_k": len(trains),
                "alpha": alpha,
                "alpha_tag": alpha_tag(alpha),
                "status_name": "ERROR",
                "error": repr(exc),
            }
            summaries.append(failure)
            write_csv(
                pd.DataFrame([failure]),
                ensure_dir(ctx.inverse_dir / alpha_tag(alpha))
                / "inverse_summary.csv",
            )
            if not continue_on_error:
                raise

    summary_df = pd.DataFrame(summaries)
    summary_df["shared_observation_aggregation_sec"] = obs_sec
    summary_df["shared_observation_source"] = obs_source
    write_csv(summary_df, ensure_dir(ctx.inverse_dir) / "inverse_all_alpha_summary.csv")

    write_alpha_differences = _cfg_flag(cfg, "write_alpha_precedence_differences", paper_default=False, normal_default=True)
    if write_alpha_differences:
        component_maps: Dict[str, Dict[int, int]] = {}
        for alpha in alphas:
            values = _load_component_assignment_cache(_inverse_component_cache_path(ctx, alpha))
            if values is not None:
                component_maps[alpha_tag(alpha)] = values
        diff_rows: List[Dict[str, Any]] = []
        tags = list(component_maps)
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                a_tag, b_tag = tags[i], tags[j]
                keys = set(component_maps[a_tag]) & set(component_maps[b_tag])
                diff_rows.append({
                    "alpha_a": a_tag, "alpha_b": b_tag, "common_components": len(keys),
                    "different_component_count": sum(component_maps[a_tag][key] != component_maps[b_tag][key] for key in keys),
                })
        write_csv(pd.DataFrame(diff_rows), ctx.inverse_dir / "alpha_precedence_differences.csv")
    elif (ctx.inverse_dir / "alpha_precedence_differences.csv").exists():
        (ctx.inverse_dir / "alpha_precedence_differences.csv").unlink()

    total_wall = time.perf_counter() - whole_start
    total_cpu = time.process_time() - whole_cpu_start
    append_stage_timing(
        ctx,
        {
            "stage": "5_inverse_all_alpha",
            "variant": ";".join(alpha_tag(alpha) for alpha in alphas),
            "status": "COMPLETED",
            "model_build_sec": None,
            "solver_runtime_sec": None,
            "process_cpu_sec": total_cpu,
            "postprocess_write_sec": obs_sec,
            "total_wall_sec": total_wall,
        },
    )
    return summary_df


# =============================================================================
# Stage 6: Vote + 五组 IO Full-Fix
# =============================================================================

def load_priors_file(path: Path) -> Dict[PairKey, int]:
    if not path.exists():
        raise FileNotFoundError(f"未找到Full-Fix priors：{path}")
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=["kind", "station", "train_i", "train_j", "value"])
    return {
        (str(row.kind), str(row.station), str(row.train_i), str(row.train_j)): int(round(float(row.value)))
        for row in df.itertuples(index=False)
    }


def run_fullfix_one(
    cfg: Mapping[str, Any],
    ctx: ExperimentContext,
    data: HSRData,
    *,
    variant: str,
    method: str,
    priors_path: Optional[Path],
    alpha: Optional[float],
    merge_preprocess_and_inverse: bool = False,
) -> Dict[str, Any]:
    outdir = ensure_dir(ctx.fullfix_dir / variant)
    paper_mode = _paper_output_mode(cfg)
    write_merged_priors = _cfg_flag(cfg, "write_fullfix_merged_priors", paper_default=False, normal_default=True)
    # Clear solution artefacts that would otherwise survive an infeasible or
    # interrupted rerun of the same variant.
    for stale_name in (
        "timetable_fullfix.csv",
        "merged_fullfix_priors.csv",
        "pairwise_fixed_canonical.csv",
    ):
        stale_path = outdir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    stage_start = time.perf_counter()
    stage_cpu_start = time.process_time()
    trains = data.select_trains(ctx.target_k)
    prep_start = time.perf_counter()
    source_counts = {
        "forced_precedence_pairs": 0,
        "learned_core_precedence_pairs": 0,
        "learned_core_component_count": 0,
    }
    prior_source_files: List[str] = []
    inverse_source_summary: Dict[str, Any] = {}
    if merge_preprocess_and_inverse:
        if alpha is None:
            raise ValueError("IO Full-Fix合并模式必须提供alpha")
        priors, source_counts = load_complete_io_priors(
            cfg, ctx, data, trains, float(alpha)
        )
        inverse_dir = ctx.inverse_dir / alpha_tag(float(alpha))
        inverse_summary_path = inverse_dir / "inverse_summary.csv"
        inverse_rec = read_first_row(inverse_summary_path)
        inverse_source_summary = {
            "inverse_source_summary_csv": relative_to_project(inverse_summary_path),
            "inverse_source_status_name": inverse_rec.get("status_name"),
            "inverse_source_target_mipgap": inverse_rec.get("inverse_target_mipgap"),
            "inverse_source_final_mip_gap": inverse_rec.get("mip_gap"),
            "inverse_source_gap_target_reached": inverse_rec.get(
                "inverse_gap_target_reached"
            ),
            "inverse_source_acceptance_reason": inverse_rec.get(
                "inverse_acceptance_reason"
            ),
            "inverse_source_accepted_for_fullfix": inverse_rec.get(
                "accepted_for_fullfix"
            ),
            "inverse_source_temporal_certified": inverse_rec.get(
                "inverse_temporal_certified"
            ),
            "inverse_source_temporal_cut_count": inverse_rec.get(
                "inverse_temporal_cut_count"
            ),
            "inverse_source_temporal_check_count": inverse_rec.get(
                "inverse_temporal_check_count"
            ),
        }
        inverse_core_csv = inverse_dir / "learned_core_precedence.csv"
        inverse_core_cache = _inverse_component_cache_path(ctx, float(alpha))
        inverse_core_source = inverse_core_csv if inverse_core_csv.exists() else inverse_core_cache
        prior_source_files = [relative_to_project(inverse_core_source)]
    else:
        if priors_path is None:
            raise ValueError("非core合并模式必须提供priors_path")
        universe_for_cache = data.event_universe(trains)
        if priors_path.suffix.lower() == ".pkl":
            priors = _load_full_assignment_cache(priors_path, trains, universe_for_cache)
            if priors is None:
                raise RuntimeError(f"无法读取compact precedence cache：{priors_path}")
        else:
            priors = load_priors_file(priors_path)
        prior_source_files = [relative_to_project(priors_path)]

    universe = data.event_universe(trains)
    universe_set = set(universe)
    missing = [key for key in universe if key not in priors]
    extra = [key for key in priors if key not in universe_set]
    prep_sec = time.perf_counter() - prep_start
    if missing:
        write_csv(
            pd.DataFrame(
                missing, columns=["kind", "station", "train_i", "train_j"]
            ),
            outdir / "missing_priors.csv",
        )
        raise RuntimeError(
            f"{variant} 缺少 {len(missing)} 个 precedence，不能称为 Full-Fix LP"
        )
    if extra:
        write_csv(
            pd.DataFrame(
                extra, columns=["kind", "station", "train_i", "train_j"]
            ),
            outdir / "extra_priors.csv",
        )

    if write_merged_priors:
        merged_rows = []
        forced_keys: set[PairKey] = set()
        if merge_preprocess_and_inverse:
            forced_keys = set()
        for key in universe:
            kind, station, i, j = key
            merged_rows.append({"kind":kind,"station":station,"train_i":i,"train_j":j,"value":int(priors[key]),"source":("PREPROCESS_FIXED" if key in forced_keys else ("INVERSE_CORE" if merge_preprocess_and_inverse else "VOTE"))})
        write_csv(pd.DataFrame(merged_rows), outdir / "merged_fullfix_priors.csv")
    elif (outdir / "merged_fullfix_priors.csv").exists():
        (outdir / "merged_fullfix_priors.csv").unlink()

    build_start = time.perf_counter()
    artifacts = build_timetable_model(
        data,
        cfg,
        trains,
        model_name=f"FullFix_{variant}",
        stage="forward",
        fixed_precedence=priors,
    )
    build_sec = time.perf_counter() - build_start
    print("\n" + "=" * 104)
    print(
        f"[6] Full-Fix {method} | variant={variant} | K={len(trains)} | "
        f"fixed={len(priors)} | binaries=0"
    )
    if merge_preprocess_and_inverse:
        print(
            f"inverse-only precedence: "
            f"event pairs={source_counts['learned_core_precedence_pairs']:,}, "
            f"structural components={source_counts['learned_core_component_count']:,}"
        )
    print("=" * 104)
    solve_info = optimize_with_status_resolution(
        artifacts.model, cfg, stage="forward"
    )
    usable = has_usable_solution(artifacts.model, keep_time_limit=True)

    post_start = time.perf_counter()
    timetable_path = outdir / "timetable_fullfix.csv"
    validation = empty_validator_summary()
    if usable:
        write_csv(pd.DataFrame(timetable_rows(data, artifacts)), timetable_path)
        if as_bool(cfg.get("write_fullfix_pairwise"), False):
            write_csv(
                pd.DataFrame(pairwise_rows(data, artifacts, directed=False)),
                outdir / "pairwise_fixed_canonical.csv",
            )
        validation = validate_timetable_csv(
            data,
            cfg,
            trains,
            timetable_path,
            expected_objective=safe_float(lambda: artifacts.model.ObjVal),
            fixed_precedence=priors,
            output_dir=outdir,
            validation_tag=variant,
        )
    audit = audit_precedence_assignment(data, trains, priors, count_c1_triplets=not paper_mode)
    post_sec = time.perf_counter() - post_start
    total_sec = time.perf_counter() - stage_start
    cpu_sec = time.process_time() - stage_cpu_start

    summary = {
        "experiment_group": ctx.experiment_group,
        "experiment_tag": ctx.experiment_tag,
        "method": method,
        "variant": variant,
        "alpha": alpha,
        "target_k": len(trains),
        "prior_merge_mode": (
            "inverse_only"
            if merge_preprocess_and_inverse
            else "single_file"
        ),
        "prior_source_files": ";".join(prior_source_files),
        **inverse_source_summary,
        "preprocess_forced_precedence_pairs": source_counts[
            "forced_precedence_pairs"
        ],
        "inverse_learned_core_precedence_pairs": source_counts[
            "learned_core_precedence_pairs"
        ],
        "inverse_learned_core_component_count": source_counts[
            "learned_core_component_count"
        ],
        "expected_precedence_pairs": len(universe),
        "loaded_prior_pairs": len(priors),
        "missing_prior_pairs": len(missing),
        "extra_prior_pairs": len(extra),
        "fixed_precedence_pairs": len(universe),
        "remaining_forward_binaries": 0,
        "remaining_precedence_big_m_constraints": 0,
        "segment_precedence_mismatch_count": int(
            getattr(artifacts.model, "_segment_mismatch_count", 0)
        ),
        "prior_read_and_check_sec": prep_sec,
        "model_build_sec": build_sec,
        "postprocess_write_validate_sec": post_sec,
        "total_wall_sec": total_sec,
        "process_cpu_sec": cpu_sec,
        **solve_info,
        **solver_summary(artifacts.model, usable),
        **artifacts.model_stats,
        **audit,
        **validation,
        "forward_runtime_sec": safe_float(lambda: artifacts.model.Runtime),
        "paper_output_mode": int(paper_mode),
        "fullfix_merged_priors_written": int(write_merged_priors),
        "merged_fullfix_priors_csv": (relative_to_project(outdir / "merged_fullfix_priors.csv") if write_merged_priors and (outdir / "merged_fullfix_priors.csv").exists() else ""),
        "timetable_csv": relative_to_project(timetable_path) if usable else "",
    }
    write_csv(pd.DataFrame([summary]), outdir / "fullfix_summary.csv")
    append_stage_timing(
        ctx,
        {
            "stage": "6_fullfix",
            "variant": variant,
            "status": summary["status_name"],
            "model_build_sec": build_sec,
            "solver_runtime_sec": summary["runtime_sec"],
            "solver_runtime_accumulated_sec": summary[
                "solver_runtime_accumulated_sec"
            ],
            "optimize_wall_sec": summary["optimize_wall_sec"],
            "process_cpu_sec": cpu_sec,
            "postprocess_write_sec": post_sec,
            "total_wall_sec": total_sec,
        },
    )
    artifacts.model.dispose()
    gc.collect()
    return summary


def run_fullfix_all_alphas(
    cfg: Mapping[str, Any],
    ctx: ExperimentContext,
    data: Optional[HSRData] = None,
) -> pd.DataFrame:
    whole_start = time.perf_counter()
    whole_cpu_start = time.process_time()
    data = data or HSRData(cfg)
    alphas = parse_float_list(cfg.get("alpha_list"))
    tasks: List[Dict[str, Any]] = []
    if as_bool(cfg.get("run_vote_fullfix"), True):
        vote_csv = ctx.vote_dir / "vote_priors_fullfix.csv"
        vote_cache = _vote_assignment_cache_path(ctx)
        vote_source = vote_csv if vote_csv.exists() else vote_cache
        tasks.append({"variant":"VOTE","method":"VOTE+LP","priors_path":vote_source,"alpha":None,"merge_core":False})
    for alpha in alphas:
        tag = alpha_tag(alpha)
        tasks.append({
            "variant": f"IO_{tag}",
            "method": f"IO+LP ({tag})",
            "priors_path": None,
            "alpha": alpha,
            "merge_core": True,
        })

    summaries: List[Dict[str, Any]] = []
    continue_on_error = as_bool(cfg.get("continue_on_error"), True)
    for idx, task in enumerate(tasks, 1):
        variant = str(task["variant"])
        method = str(task["method"])
        prior_path = task["priors_path"]
        alpha = task["alpha"]
        merge_core = bool(task["merge_core"])
        print(f"\n##### Full-Fix {idx}/{len(tasks)}: {variant} #####")
        required_paths: List[Path]
        if merge_core:
            core_csv = ctx.inverse_dir / alpha_tag(float(alpha)) / "learned_core_precedence.csv"
            core_cache = _inverse_component_cache_path(ctx, float(alpha))
            required_paths = []
            if not core_csv.exists() and not core_cache.exists():
                required_paths.append(core_csv)
        else:
            required_paths = [Path(prior_path)]
        missing_paths = [path for path in required_paths if not path.exists()]
        if missing_paths:
            failure = {
                "experiment_group": ctx.experiment_group,
                "experiment_tag": ctx.experiment_tag,
                "method": method,
                "variant": variant,
                "alpha": alpha,
                "status_name": "SKIPPED_NO_PRIORS",
                "error": "missing " + ";".join(str(path) for path in missing_paths),
            }
            summaries.append(failure)
            write_csv(
                pd.DataFrame([failure]),
                ensure_dir(ctx.fullfix_dir / variant) / "fullfix_summary.csv",
            )
            continue
        try:
            summaries.append(
                run_fullfix_one(
                    cfg,
                    ctx,
                    data,
                    variant=variant,
                    method=method,
                    priors_path=prior_path,
                    alpha=alpha,
                    merge_preprocess_and_inverse=merge_core,
                )
            )
        except Exception as exc:
            failure = {
                "experiment_group": ctx.experiment_group,
                "experiment_tag": ctx.experiment_tag,
                "method": method,
                "variant": variant,
                "alpha": alpha,
                "status_name": "ERROR",
                "error": repr(exc),
            }
            summaries.append(failure)
            write_csv(
                pd.DataFrame([failure]),
                ensure_dir(ctx.fullfix_dir / variant) / "fullfix_summary.csv",
            )
            if not continue_on_error:
                raise
    summary_df = pd.DataFrame(summaries)
    write_csv(
        summary_df,
        ensure_dir(ctx.fullfix_dir) / "fullfix_all_alpha_summary.csv",
    )
    total_wall = time.perf_counter() - whole_start
    total_cpu = time.process_time() - whole_cpu_start
    append_stage_timing(
        ctx,
        {
            "stage": "6_fullfix_all_alpha",
            "variant": ";".join(str(task["variant"]) for task in tasks),
            "status": "COMPLETED",
            "model_build_sec": None,
            "solver_runtime_sec": None,
            "process_cpu_sec": total_cpu,
            "postprocess_write_sec": None,
            "total_wall_sec": total_wall,
        },
    )
    return summary_df


# =============================================================================
# Stage 90: 汇总
# =============================================================================

def _record_value(record: Mapping[str, Any], key: str, default: Any = None) -> Any:
    value = record.get(key, default)
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return value


def _number(record: Mapping[str, Any], key: str, default: Any = None) -> Any:
    value = _record_value(record, key, default)
    if value is None:
        return default
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except Exception:
        return default


def _paper_status(record: Mapping[str, Any]) -> str:
    status = str(_record_value(record, "status_name", "") or "")
    has_solution = bool(int(_number(record, "has_solution", 0) or 0))
    if status == "OPTIMAL":
        return "Opt"
    if status == "TIME_LIMIT":
        return "TL-F" if has_solution else "TL-NF"
    if status == "INFEASIBLE":
        return "Inf."
    if status == "INF_OR_UNBD":
        return "Inf./Unbd."
    if status == "UNBOUNDED":
        return "Unbd."
    return status or "--"


def _gap_from_lb(tt: Any, lower_bound: Any) -> Optional[float]:
    try:
        tt_f = float(tt)
        lb_f = float(lower_bound)
        if not math.isfinite(tt_f) or not math.isfinite(lb_f) or lb_f == 0:
            return None
        return 100.0 * (tt_f - lb_f) / lb_f
    except Exception:
        return None


def _row_for_alpha(df: pd.DataFrame, alpha: float) -> Dict[str, Any]:
    if df.empty or "alpha" not in df.columns:
        return {}
    numeric = pd.to_numeric(df["alpha"], errors="coerce")
    matched = df[(numeric - float(alpha)).abs() <= 1e-10]
    if matched.empty:
        return {}
    return {
        key: (None if pd.isna(value) else value)
        for key, value in matched.iloc[0].to_dict().items()
    }


def _row_for_variant(df: pd.DataFrame, variant: str) -> Dict[str, Any]:
    if df.empty or "variant" not in df.columns:
        return {}
    matched = df[df["variant"].astype(str) == str(variant)]
    if matched.empty:
        return {}
    return {
        key: (None if pd.isna(value) else value)
        for key, value in matched.iloc[0].to_dict().items()
    }


def write_effective_parameters(
    cfg: Mapping[str, Any],
    ctx: ExperimentContext,
    data: HSRData,
    validation: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist all effective parameters and input fingerprints before solving."""
    outdir = ensure_dir(ctx.parameter_dir)
    data_dir = data.data_dir
    method_targets = _method_targets(cfg, int(ctx.sample_counts[0]))
    method_sizes = configured_method_sample_sizes(cfg)
    sample_design = configured_sampling_design_label(cfg, int(ctx.sample_counts[0]))
    selected = data.select_trains(ctx.target_k)

    try:
        import gurobipy as gp  # local import keeps the module surface small

        gurobi_version = ".".join(str(x) for x in gp.gurobi.version())
    except Exception:
        gurobi_version = "unknown"

    file_rows: List[Dict[str, Any]] = []
    for filename in ("trains.csv", "stations.csv", "segments.csv", "config.csv", "case_plan.csv", "generation_config.csv"):
        if filename in ("trains.csv", "stations.csv", "segments.csv"):
            file_path = data_dir / filename
        elif filename == "generation_config.csv":
            file_path = project_root() / "trainset_tools" / filename
        else:
            file_path = project_root() / filename
        file_rows.append(
            {
                "file": filename,
                "path": str(file_path),
                "exists": int(file_path.exists()),
                "size_bytes": file_path.stat().st_size if file_path.exists() else None,
                "sha256": sha256_file(file_path) if file_path.exists() else None,
            }
        )
    script_path = Path(__file__).resolve()
    file_rows.append(
        {
            "file": script_path.name,
            "path": str(script_path),
            "exists": 1,
            "size_bytes": script_path.stat().st_size,
            "sha256": sha256_file(script_path),
        }
    )
    write_csv(pd.DataFrame(file_rows), outdir / "input_and_script_fingerprints.csv")

    config_rows = [
        {"key": key, "value": value, "source": ("case_plan.csv / derived" if key in {"train_limit", "sample_size", "sampling_local_size", "sampling_stratified_size", "sampling_coverage_size", "sampling_local_candidate_pool_size", "sampling_local_count", "sampling_stratified_count", "sampling_coverage_count", "sample_count", "instance_tag", "data_dir", "create_case_subfolder"} else "config.csv")}
        for key, value in cfg.items()
        if not str(key).startswith("_")
    ]
    effective = {
        "effective_target_k": ctx.target_k,
        "effective_sample_size": int(ctx.sample_sizes[0]),
        "effective_sample_design": sample_design,
        "effective_local_sample_size": int(method_sizes["local"]),
        "effective_stratified_sample_size": int(method_sizes["stratified"]),
        "effective_coverage_sample_size": int(method_sizes["coverage"]),
        "effective_sample_count": int(ctx.sample_counts[0]),
        "effective_main_alpha": configured_main_alpha(cfg),
        "effective_local_count": method_targets["local"],
        "effective_stratified_count": method_targets["stratified"],
        "effective_coverage_count": method_targets["coverage"],
        "effective_case_tag": ctx.experiment_tag,
        "effective_output_dir": str(ctx.output_dir),
        "effective_train_selection_rule": data.train_selection_rule,
        "effective_selected_first_train": selected[0] if selected else None,
        "effective_selected_last_train": selected[-1] if selected else None,
        "effective_event_universe_size": len(data.event_universe(selected)),
        "effective_dep_window_fast": as_float(cfg.get("dep_window_fast"), 45.0),
        "effective_dep_window_normal": as_float(cfg.get("dep_window_normal"), 60.0),
        "effective_segments_source": str(data_dir / "segments.csv"),
        "effective_packing_value_rule": "OPTIMAL:ObjVal; nonoptimal incumbent:ObjBound",
        "python_version": sys.version.replace("\n", " "),
        "gurobi_version": gurobi_version,
        "platform": platform.platform(),
        "created_at": now_text(),
    }
    if validation:
        for key, value in validation.items():
            effective[f"validation_{key}"] = value
    config_rows.extend(
        {"key": key, "value": value, "source": "effective"}
        for key, value in effective.items()
    )
    write_csv(pd.DataFrame(config_rows), outdir / "effective_parameters.csv")
    write_csv(
        pd.DataFrame(
            {
                "selection_rank": range(1, len(selected) + 1),
                "train_id": selected,
                "entry_time": [data.entry_time[k] for k in selected],
                "entry_time_source": [data.entry_time_source[k] for k in selected],
                "first_departure": [data.first_dep[k] for k in selected],
                "train_type": [data.train_type[k] for k in selected],
                "intermediate_stop_count": [
                    data.intermediate_stop_count[k] for k in selected
                ],
            }
        ),
        outdir / "selected_target_trains.csv",
    )
    write_json(
        {
            "effective": effective,
            "method_targets": method_targets,
            "validation": dict(validation or {}),
            "raw_config": {
                key: value for key, value in cfg.items() if not str(key).startswith("_")
            },
        },
        outdir / "effective_parameters.json",
    )
    return effective


def _write_latex_rows(
    table_dir: Path,
    table6: Mapping[str, Any],
    table7: Mapping[str, Any],
    table9: Mapping[str, Any],
) -> None:
    """Write convenience LaTeX rows; CSV remains the authoritative output."""
    def fmt(value: Any, digits: int = 3) -> str:
        if value is None:
            return "--"
        try:
            if pd.isna(value):
                return "--"
        except Exception:
            pass
        if isinstance(value, str):
            return value
        try:
            number = float(value)
            if not math.isfinite(number):
                return "--"
            if abs(number - round(number)) <= 1e-9:
                return str(int(round(number)))
            return f"{number:.{digits}f}"
        except Exception:
            return str(value)

    lines = [
        "% Auto-generated convenience rows. CSV files are authoritative.",
        "% Table 6", 
        " & ".join(
            fmt(table6.get(key))
            for key in (
                "K", "base_continuous", "base_binary", "base_constraints",
                "base_big_m_rows", "lp_continuous", "lp_binary",
                "lp_constraints", "lp_big_m_rows", "binary_reduction_percent",
            )
        ) + r" \\",
        "% Table 7",
        " & ".join(
            fmt(table7.get(key))
            for key in (
                "K", "direct_status", "direct_TT", "direct_best_bound",
                "direct_GapMIP_percent", "direct_time_sec", "direct_feasible",
                "packing_LB", "IO_TT", "IO_GapPack_percent", "IO_LP_time_sec",
                "IO_feasible",
            )
        ) + r" \\",
        "% Table 9",
        " & ".join(
            fmt(table9.get(key))
            for key in (
                "K", "restricted_CPU_time_sec", "restricted_wall_time_sec",
                "packing_LP_time_sec", "learning_time_sec", "target_LP_time_sec",
                "end_to_end_wall_time_sec", "optimal_restricted_solves",
            )
        ) + r" \\",
    ]
    (table_dir / "latex_rows_table6_table7_table9.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def aggregate_experiment(cfg: Mapping[str, Any], ctx: ExperimentContext) -> pd.DataFrame:
    stage_start = time.perf_counter()
    ensure_dir(ctx.aggregate_dir)
    table_dir = ensure_dir(ctx.aggregate_dir / "chapter6_table_rows")

    direct = read_first_row(ctx.direct_dir / "direct_summary.csv")
    sample = read_first_row(ctx.sample_dir / "sample_solve_summary.csv")
    vote = read_first_row(ctx.vote_dir / "vote_summary.csv")
    diagnostics = read_first_row(ctx.vote_dir / "evidence_diagnostics.csv")
    packing = read_first_row(ctx.packing_dir / "packing_summary.csv")
    core: Dict[str, Any] = {}
    packing_lb = _number(packing, "packing_lb_primal")

    inverse_path = ctx.inverse_dir / "inverse_all_alpha_summary.csv"
    inverse_df = (
        pd.read_csv(inverse_path, encoding="utf-8-sig")
        if inverse_path.exists()
        else pd.DataFrame()
    )
    fullfix_path = ctx.fullfix_dir / "fullfix_all_alpha_summary.csv"
    fullfix_df = (
        pd.read_csv(fullfix_path, encoding="utf-8-sig")
        if fullfix_path.exists()
        else pd.DataFrame()
    )

    main_alpha = configured_main_alpha(cfg)
    method_sizes = configured_method_sample_sizes(cfg)
    method_counts = _method_targets(cfg, int(ctx.sample_counts[0]))
    sample_design = configured_sampling_design_label(cfg, int(ctx.sample_counts[0]))
    k_sub_design = configured_k_sub_design_text(cfg)
    sample_design_fields = {
        "sample_design": sample_design,
        "K_sub_design": k_sub_design,
        "local_K_sub": int(method_sizes["local"]),
        "local_R": int(method_counts["local"]),
        "stratified_K_sub": int(method_sizes["stratified"]),
        "stratified_R": int(method_counts["stratified"]),
        "coverage_K_sub": int(method_sizes["coverage"]),
        "coverage_R": int(method_counts["coverage"]),
        "R_total": int(ctx.sample_counts[0]),
    }
    main_inverse = _row_for_alpha(inverse_df, main_alpha)
    main_fullfix = _row_for_alpha(fullfix_df, main_alpha)
    vote_fullfix = _row_for_variant(fullfix_df, "VOTE")

    # Long-format method results retained for detailed auditing.
    rows: List[Dict[str, Any]] = []
    if direct:
        direct_tt = _number(direct, "obj_val")
        rows.append(
            {
                "experiment_tag": ctx.experiment_tag,
                "target_k": ctx.target_k,
                "sample_size": int(ctx.sample_sizes[0]),
                "sample_count": int(ctx.sample_counts[0]),
                **sample_design_fields,
                "method": "Direct MILP",
                "variant": "DIRECT",
                "alpha": None,
                "status": _record_value(direct, "status_name"),
                "paper_status": _paper_status(direct),
                "TT": direct_tt,
                "LB_pack": packing_lb,
                "Gap_pack_percent": _gap_from_lb(direct_tt, packing_lb),
                "solver_runtime_sec": _number(direct, "runtime_sec"),
                "total_wall_sec": _number(direct, "total_wall_sec"),
                "mip_gap_percent": _number(direct, "mip_gap_percent"),
                "obj_bound": _number(direct, "obj_bound"),
                "validator_pass": _number(direct, "validator_pass"),
            }
        )
    for rec in fullfix_df.to_dict("records"):
        alpha = _record_value(rec, "alpha")
        inv = (
            _row_for_alpha(inverse_df, float(alpha))
            if alpha is not None and not pd.isna(alpha)
            else {}
        )
        tt = _number(rec, "obj_val")
        rows.append(
            {
                "experiment_tag": ctx.experiment_tag,
                "target_k": ctx.target_k,
                "sample_size": int(ctx.sample_sizes[0]),
                "sample_count": int(ctx.sample_counts[0]),
                **sample_design_fields,
                "method": _record_value(rec, "method"),
                "variant": _record_value(rec, "variant"),
                "alpha": alpha,
                "status": _record_value(rec, "status_name"),
                "paper_status": _paper_status(rec),
                "TT": tt,
                "LB_pack": packing_lb,
                "Gap_pack_percent": _gap_from_lb(tt, packing_lb),
                "solver_runtime_sec": _number(rec, "runtime_sec"),
                "total_wall_sec": _number(rec, "total_wall_sec"),
                "validator_pass": _number(rec, "validator_pass"),
                "fixed_precedence_pairs": _number(rec, "fixed_precedence_pairs"),
                "remaining_forward_binaries": _number(rec, "remaining_forward_binaries"),
                "inverse_status": _record_value(inv, "status_name"),
                "inverse_weighted_objective": _number(inv, "weighted_mismatch_objective"),
                "inverse_raw_mismatch": _number(inv, "raw_mismatch_count"),
                "inverse_runtime_sec": _number(inv, "runtime_sec"),
                "inverse_total_wall_sec": _number(inv, "total_wall_sec"),
                "different_from_vote_count": _number(inv, "different_from_vote_count"),
            }
        )
    results = pd.DataFrame(rows)
    write_csv(results, ctx.aggregate_dir / "experiment_results_long.csv")
    if not results.empty:
        wide = results.pivot_table(
            index=["experiment_tag", "target_k", "sample_size", "sample_count"],
            columns="variant",
            values=["TT", "Gap_pack_percent", "solver_runtime_sec", "total_wall_sec"],
            aggfunc="first",
        )
        wide.columns = [f"{a}__{b}" for a, b in wide.columns]
        write_csv(wide.reset_index(), ctx.aggregate_dir / "experiment_results_wide.csv")

    base_binary = _number(direct, "num_binary_vars_constructed")
    lp_binary = _number(main_fullfix, "num_binary_vars_constructed")
    binary_reduction = None
    if base_binary is not None and base_binary > 0 and lp_binary is not None:
        binary_reduction = 100.0 * (base_binary - lp_binary) / base_binary
    table6 = {
        "experiment_tag": ctx.experiment_tag,
        "K": ctx.target_k,
        "K_sub": int(ctx.sample_sizes[0]),
        "R": int(ctx.sample_counts[0]),
        **sample_design_fields,
        "main_alpha": main_alpha,
        "base_continuous": _number(direct, "num_cont_vars_constructed"),
        "base_binary": base_binary,
        "base_constraints": _number(direct, "num_constraints_constructed"),
        "base_big_m_rows": _number(direct, "num_big_m_rows"),
        "lp_continuous": _number(main_fullfix, "num_cont_vars_constructed"),
        "lp_binary": lp_binary,
        "lp_constraints": _number(main_fullfix, "num_constraints_constructed"),
        "lp_big_m_rows": _number(main_fullfix, "num_big_m_rows"),
        "binary_reduction_percent": binary_reduction,
        "lp_one_sided_headway_rows": _number(main_fullfix, "num_fixed_one_sided_headway_rows"),
        "base_segment_consistency_rows": _number(direct, "num_segment_consistency_rows"),
    }
    write_csv(pd.DataFrame([table6]), table_dir / "table6_model_size_row.csv")

    direct_tt = _number(direct, "obj_val")
    io_tt = _number(main_fullfix, "obj_val")
    table7 = {
        "experiment_tag": ctx.experiment_tag,
        "K": ctx.target_k,
        "K_sub": int(ctx.sample_sizes[0]),
        "R": int(ctx.sample_counts[0]),
        **sample_design_fields,
        "main_alpha": main_alpha,
        "direct_status": _paper_status(direct),
        "direct_status_raw": _record_value(direct, "status_name", "--"),
        "direct_TT": direct_tt,
        "direct_best_bound": _number(direct, "obj_bound"),
        "direct_GapMIP_percent": _number(direct, "mip_gap_percent"),
        "direct_time_sec": _number(direct, "solver_runtime_accumulated_sec", _number(direct, "runtime_sec")),
        "direct_last_solver_runtime_sec": _number(direct, "runtime_sec"),
        "direct_total_wall_sec": _number(direct, "total_wall_sec"),
        "direct_feasible": _record_value(direct, "validator_feasible_text", "--"),
        "direct_validator_pass": _number(direct, "validator_pass"),
        "packing_LB": packing_lb,
        "packing_bound_certified": _number(packing, "packing_bound_certified"),
        "packing_exact_values_used": _number(packing, "num_optimal_values_used"),
        "packing_best_bounds_used": _number(packing, "num_best_bounds_used"),
        "IO_status": _paper_status(main_fullfix),
        "IO_status_raw": _record_value(main_fullfix, "status_name", "--"),
        "IO_TT": io_tt,
        "IO_GapPack_percent": _gap_from_lb(io_tt, packing_lb),
        "IO_LP_time_sec": _number(main_fullfix, "solver_runtime_accumulated_sec", _number(main_fullfix, "runtime_sec")),
        "IO_last_solver_runtime_sec": _number(main_fullfix, "runtime_sec"),
        "IO_total_wall_sec": _number(main_fullfix, "total_wall_sec"),
        "IO_feasible": _record_value(main_fullfix, "validator_feasible_text", "--"),
        "IO_validator_pass": _number(main_fullfix, "validator_pass"),
    }
    write_csv(pd.DataFrame([table7]), table_dir / "table7_main_results_row.csv")

    vote_tt = _number(vote_fullfix, "obj_val")
    table8 = {
        "experiment_tag": ctx.experiment_tag,
        "K": ctx.target_k,
        "K_sub": int(ctx.sample_sizes[0]),
        "R": int(ctx.sample_counts[0]),
        **sample_design_fields,
        "vote_status": _paper_status(vote_fullfix),
        "vote_status_raw": _record_value(vote_fullfix, "status_name", "--"),
        "vote_TT": vote_tt,
        "vote_time_sec": _number(vote_fullfix, "solver_runtime_accumulated_sec", _number(vote_fullfix, "runtime_sec")),
        "vote_last_solver_runtime_sec": _number(vote_fullfix, "runtime_sec"),
        "vote_total_wall_sec": _number(vote_fullfix, "total_wall_sec"),
        "vote_validator_pass": _number(vote_fullfix, "validator_pass"),
        "TS_P_best_TT": None,
        "TS_P_mean_TT": None,
        "TS_P_SD": None,
        "TS_P_feasible_runs": None,
        "TS_P_status": "NOT_RUN_IN_MAIN_PIPELINE",
        "ALNS_R_best_TT": None,
        "ALNS_R_mean_TT": None,
        "ALNS_R_SD": None,
        "ALNS_R_feasible_runs": None,
        "ALNS_R_status": "NOT_RUN_IN_MAIN_PIPELINE",
        "IO_TT": io_tt,
        "IO_GapPack_percent": _gap_from_lb(io_tt, packing_lb),
        "main_alpha": main_alpha,
    }
    write_csv(pd.DataFrame([table8]), table_dir / "table8_vote_io_partial_row.csv")

    restricted_cpu = _number(sample, "stage_process_cpu_sec")
    restricted_wall = _number(sample, "stage_total_wall_sec")
    packing_wall = _number(packing, "total_wall_sec")
    core_wall = 0.0
    observation_wall = _number(main_inverse, "shared_observation_aggregation_sec", 0.0) or 0.0
    learning_model_wall = _number(main_inverse, "total_wall_sec")
    learning_time = (
        core_wall + observation_wall + learning_model_wall
        if learning_model_wall is not None
        else None
    )
    target_lp_wall = _number(main_fullfix, "total_wall_sec")
    end_to_end = None
    if all(value is not None for value in (restricted_wall, packing_wall, learning_time, target_lp_wall)):
        end_to_end = float(restricted_wall) + float(packing_wall) + float(learning_time) + float(target_lp_wall)
    optimal_solves = _number(sample, "optimal_count")
    table9 = {
        "experiment_tag": ctx.experiment_tag,
        "K": ctx.target_k,
        "K_sub": int(ctx.sample_sizes[0]),
        "R": int(ctx.sample_counts[0]),
        **sample_design_fields,
        "main_alpha": main_alpha,
        "restricted_CPU_time_sec": restricted_cpu,
        "restricted_wall_time_sec": restricted_wall,
        "restricted_aggregate_solver_time_sec": _number(sample, "solver_runtime_sum_all_attempts_sec"),
        "restricted_attempt_wall_sum_sec": _number(sample, "attempt_wall_sum_all_attempts_sec"),
        "packing_LP_time_sec": packing_wall,
        "precedence_core_preprocessing_time_sec": 0.0,
        "precedence_core_forced_pairs": 0,
        "precedence_core_event_core_pairs": _number(main_inverse, "unresolved_event_core_count"),
        "precedence_core_quotient_binaries": _number(main_inverse, "unresolved_quotient_core_count"),
        "inverse_binary_reduction_percent": _number(main_inverse, "inverse_binary_reduction_percent"),
        "packing_solver_time_sec": (
            (_number(packing, "primal_runtime_sec", 0.0) or 0.0)
            + (_number(packing, "dual_runtime_sec", 0.0) or 0.0)
        ),
        "observation_aggregation_time_sec": observation_wall,
        "learning_model_time_sec": learning_model_wall,
        "learning_time_sec": learning_time,
        "learning_solver_time_sec": _number(main_inverse, "solver_runtime_accumulated_sec", _number(main_inverse, "runtime_sec")),
        "target_LP_time_sec": target_lp_wall,
        "target_LP_solver_time_sec": _number(main_fullfix, "solver_runtime_accumulated_sec", _number(main_fullfix, "runtime_sec")),
        "end_to_end_wall_time_sec": end_to_end,
        "optimal_restricted_solves": optimal_solves,
        "required_restricted_solves": int(ctx.sample_counts[0]),
        "nonoptimal_incumbent_solves": _number(sample, "time_limit_with_solution_count"),
        "timing_scope": "Proposed main-alpha path only; excludes Direct, Vote, and other alpha runs",
    }
    write_csv(pd.DataFrame([table9]), table_dir / "table9_runtime_row.csv")

    alpha_rows: List[Dict[str, Any]] = []
    for rec in inverse_df.to_dict("records"):
        alpha = _record_value(rec, "alpha")
        if alpha is None or pd.isna(alpha):
            continue
        forward = _row_for_alpha(fullfix_df, float(alpha))
        tt = _number(forward, "obj_val")
        alpha_rows.append(
            {
                "experiment_tag": ctx.experiment_tag,
                "K": ctx.target_k,
                "K_sub": int(ctx.sample_sizes[0]),
                "R": int(ctx.sample_counts[0]),
                **sample_design_fields,
                "alpha": float(alpha),
                "Raw_mismatch": _number(rec, "raw_mismatch_count"),
                "Weighted_mismatch": _number(rec, "weighted_mismatch_objective"),
                "Dual_weighted_component_audit": _number(rec, "dual_weighted_mismatch_component"),
                "Weighted_mismatch_recomputed": _number(rec, "weighted_mismatch_recomputed"),
                "Objective_reconstruction_abs_diff": _number(rec, "weighted_objective_reconstruction_abs_diff"),
                "Learning_time_sec": _number(rec, "solver_runtime_accumulated_sec", _number(rec, "runtime_sec")),
                "Learning_total_wall_sec": _number(rec, "total_wall_sec"),
                "Learning_status": _paper_status(rec),
                "Learning_status_raw": _record_value(rec, "status_name", "--"),
                "Reduced_LP_status": _paper_status(forward),
                "Reduced_LP_status_raw": _record_value(forward, "status_name", "--"),
                "TT": tt,
                "GapPack_percent": _gap_from_lb(tt, packing_lb),
                "LP_time_sec": _number(forward, "solver_runtime_accumulated_sec", _number(forward, "runtime_sec")),
                "LP_total_wall_sec": _number(forward, "total_wall_sec"),
                "Validator_pass": _number(forward, "validator_pass"),
            }
        )
    alpha_df = pd.DataFrame(alpha_rows).sort_values("alpha") if alpha_rows else pd.DataFrame()
    write_csv(alpha_df, table_dir / "table10_alpha_sensitivity_rows.csv")

    evidence_offline = restricted_wall
    full_offline_before_target = None
    if all(value is not None for value in (restricted_wall, packing_wall, learning_time)):
        full_offline_before_target = float(restricted_wall) + float(packing_wall) + float(learning_time)
    table12 = {
        "experiment_tag": ctx.experiment_tag,
        "K": ctx.target_k,
        "K_sub": int(ctx.sample_sizes[0]),
        "R": int(ctx.sample_counts[0]),
        **sample_design_fields,
        "Obs_ratio": _number(diagnostics or vote, "observation_ratio"),
        "RelCov_2p5": _number(diagnostics or vote, "RelCov_2p5"),
        "EffCov": _number(diagnostics or vote, "EffCov"),
        "P10_StrCov_2p5": _number(diagnostics or vote, "P10_StrCov_2p5"),
        "Repeated_support_threshold": _number(diagnostics or vote, "repeated_support_threshold"),
        "Evidence_generation_wall_time_sec": evidence_offline,
        "Full_offline_before_target_LP_sec": full_offline_before_target,
        "Optimal_solves": optimal_solves,
        "Required_solves": int(ctx.sample_counts[0]),
        "Packing_LB": packing_lb,
        "Learning_time_sec": _number(main_inverse, "solver_runtime_accumulated_sec", _number(main_inverse, "runtime_sec")),
        "Learning_total_wall_sec": learning_time,
        "TT": io_tt,
        "GapPack_percent": _gap_from_lb(io_tt, packing_lb),
        "LP_time_sec": _number(main_fullfix, "solver_runtime_accumulated_sec", _number(main_fullfix, "runtime_sec")),
        "LP_total_wall_sec": target_lp_wall,
        "formal_table12_row": int(
            ctx.target_k == 344 and as_bool(cfg.get("nested_pool_certified"), False)
        ),
        "note": (
            "Formal Table 12 additionally requires a certified nested master pool; "
            "independent main-experiment runs are diagnostics only"
        ),
    }
    write_csv(pd.DataFrame([table12]), table_dir / "table12_current_R_diagnostics_row.csv")

    combined = {
        "experiment_tag": ctx.experiment_tag,
        "K": ctx.target_k,
        "K_sub": int(ctx.sample_sizes[0]),
        "R": int(ctx.sample_counts[0]),
        **sample_design_fields,
        "main_alpha": main_alpha,
        "direct_status": table7["direct_status"],
        "direct_TT": direct_tt,
        "packing_LB": packing_lb,
        "IO_status": table7["IO_status"],
        "IO_TT": io_tt,
        "IO_GapPack_percent": table7["IO_GapPack_percent"],
        "IO_validator_pass": table7["IO_validator_pass"],
        "optimal_restricted_solves": optimal_solves,
        "packing_exact_values_used": _number(packing, "num_optimal_values_used"),
        "packing_best_bounds_used": _number(packing, "num_best_bounds_used"),
        "precedence_core_forced_pairs": 0,
        "precedence_core_event_core_pairs": _number(main_inverse, "unresolved_event_core_count"),
        "precedence_core_quotient_binaries": _number(main_inverse, "unresolved_quotient_core_count"),
        "inverse_binary_reduction_percent": _number(main_inverse, "inverse_binary_reduction_percent"),
        "end_to_end_main_method_wall_sec": end_to_end,
        "Obs_ratio": table12["Obs_ratio"],
        "RelCov_2p5": table12["RelCov_2p5"],
        "EffCov": table12["EffCov"],
        "P10_StrCov_2p5": table12["P10_StrCov_2p5"],
    }
    write_csv(pd.DataFrame([combined]), ctx.aggregate_dir / "main_experiment_summary.csv")
    _write_latex_rows(table_dir, table6, table7, table9)

    write_json(
        {
            "context": {
                "experiment_group": ctx.experiment_group,
                "experiment_tag": ctx.experiment_tag,
                "target_k": ctx.target_k,
                "sample_size": int(ctx.sample_sizes[0]),
                "sample_count": int(ctx.sample_counts[0]),
                **sample_design_fields,
                "main_alpha": main_alpha,
                "alpha_list": parse_float_list(cfg.get("alpha_list")),
                "train_selection_rule": normalize_train_selection_rule(
                    cfg.get("train_selection_rule", "entry_time")
                ),
                "output_dir": str(ctx.output_dir),
            },
            "sample_summary": sample,
            "vote_summary": vote,
            "evidence_diagnostics": diagnostics,
            "packing_summary": packing,
            "table_files": sorted(path.name for path in table_dir.glob("*")),
        },
        ctx.aggregate_dir / "experiment_manifest.json",
    )

    total_sec = time.perf_counter() - stage_start
    append_stage_timing(
        ctx,
        {
            "stage": "90_aggregate",
            "variant": "chapter6_rows",
            "status": "COMPLETED",
            "model_build_sec": None,
            "solver_runtime_sec": 0.0,
            "postprocess_write_sec": total_sec,
            "total_wall_sec": total_sec,
        },
    )
    timing_path = ctx.timing_dir / "stage_timing_log.csv"
    if timing_path.exists():
        write_csv(
            pd.read_csv(timing_path, encoding="utf-8-sig"),
            ctx.aggregate_dir / "stage_timing_summary.csv",
        )
    collect_all_case_rows(cfg, ctx.output_root)
    print(f"[90] 汇总完成：{ctx.aggregate_dir}")
    return results


def collect_all_case_rows(cfg: Mapping[str, Any], output_root: Optional[Path] = None) -> Path:
    """Collect table-ready rows from all completed case folders under outputs."""
    root = output_root or resolve_project_path(cfg.get("output_root"), "outputs")
    root = root.resolve()
    outdir = ensure_dir(root / "_ALL_CASES")
    patterns = {
        "table6_all_cases.csv": "table6_model_size_row.csv",
        "table7_all_cases.csv": "table7_main_results_row.csv",
        "table8_vote_io_partial_all_cases.csv": "table8_vote_io_partial_row.csv",
        "table9_all_cases.csv": "table9_runtime_row.csv",
        "table10_alpha_sensitivity_all_cases.csv": "table10_alpha_sensitivity_rows.csv",
        "table12_R_diagnostics_all_cases.csv": "table12_current_R_diagnostics_row.csv",
    }
    for output_name, leaf_name in patterns.items():
        frames: List[pd.DataFrame] = []
        for path in sorted(root.glob(f"*/90_aggregate/chapter6_table_rows/{leaf_name}")):
            if "_ALL_CASES" in path.parts:
                continue
            try:
                frame = pd.read_csv(path, encoding="utf-8-sig")
            except pd.errors.EmptyDataError:
                continue
            frame.insert(0, "source_case_dir", path.parents[2].name)
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not combined.empty:
            sort_cols = [col for col in ("K", "K_sub", "R", "alpha") if col in combined.columns]
            if sort_cols:
                combined = combined.sort_values(sort_cols, kind="stable")
        write_csv(combined, outdir / output_name)
    write_json(
        {
            "collected_at": now_text(),
            "output_root": str(root),
            "files": sorted(path.name for path in outdir.glob("*.csv")),
        },
        outdir / "collection_manifest.json",
    )
    return outdir


# =============================================================================
# Context and pipeline control
# =============================================================================

def standalone_context(cfg: Mapping[str, Any]) -> ExperimentContext:
    k_value = as_int(cfg.get("train_limit"), 100)
    k = int(100 if k_value is None else k_value)
    sizes, counts = configured_sample_design(cfg)
    if len(sizes) != 1 or len(counts) != 1:
        raise ValueError("主实验每次必须且只能配置一组总样本设计")
    sample_size, sample_count = int(sizes[0]), int(counts[0])
    method_sizes = configured_method_sample_sizes(cfg)
    method_counts = _method_targets(cfg, sample_count)
    design_label = configured_sampling_design_label(cfg, sample_count)
    output_root = resolve_project_path(cfg.get("output_root"), "outputs")
    case_tag = case_tag_from_config(cfg, k, sample_size, sample_count)
    create_subfolder = as_bool(cfg.get("create_case_subfolder"), True)
    output_dir = output_root / case_tag if create_subfolder else output_root
    return ExperimentContext(
        experiment_group=str(cfg.get("experiment_group") or "MAIN_EXPERIMENT"),
        experiment_tag=case_tag,
        target_k=k,
        output_root=output_root,
        output_dir=output_dir,
        sample_sizes=(sample_size,),
        sample_counts=(sample_count,),
        use_c1=True,
        use_c2=True,
        use_c3=True,
        extra={
            "method_sample_sizes": method_sizes,
            "method_sample_counts": method_counts,
            "sample_design": design_label,
            "k_sub_design": configured_k_sub_design_text(cfg),
        },
    )


def validate_config(
    cfg: Mapping[str, Any], ctx: ExperimentContext, data: HSRData
) -> Dict[str, Any]:
    if len(ctx.sample_counts) != 1:
        raise ValueError("主实验每次只能运行一个总样本设计")
    method_sizes = configured_method_sample_sizes(cfg)
    sample_size = max(method_sizes.values())
    sample_count = int(ctx.sample_counts[0])
    if ctx.target_k <= 1 or ctx.target_k > len(data.all_trains):
        raise ValueError(
            f"train_limit={ctx.target_k} 必须位于 2..{len(data.all_trains)}"
        )
    for method, size in method_sizes.items():
        if size <= 1 or size >= ctx.target_k:
            raise ValueError(
                f"sampling_{method}_size={size} 必须位于 2..{ctx.target_k - 1}"
            )
    if sample_count <= 0:
        raise ValueError("sample_count 必须为正整数")

    targets = _method_targets(cfg, sample_count)
    if as_bool(cfg.get("sampling_unique_subsets"), True):
        for method in SAMPLING_METHODS:
            maximum_unique = math.comb(ctx.target_k, method_sizes[method])
            if int(targets[method]) > maximum_unique:
                raise ValueError(
                    f"sampling_unique_subsets=TRUE时，K={ctx.target_k}, method={method}, "
                    f"K_sub={method_sizes[method]}最多只有{maximum_unique}个不同子集，"
                    f"但需要{targets[method]}个"
                )
    alphas = parse_float_list(cfg.get("alpha_list"))
    if not alphas:
        raise ValueError("alpha_list 不能为空")
    if any(alpha < 0 for alpha in alphas):
        raise ValueError(f"alpha必须非负：{alphas}")
    if len(set(alphas)) != len(alphas):
        raise ValueError(f"alpha_list 不能重复：{alphas}")
    main_alpha = configured_main_alpha(cfg)
    if not any(abs(alpha - main_alpha) <= 1e-10 for alpha in alphas):
        raise ValueError(
            f"main_alpha={main_alpha} 必须包含在 alpha_list={alphas} 中"
        )
    # The main pipeline always uses the Full structural model: C1+C2+C3.
    if not (ctx.use_c1 and ctx.use_c2 and ctx.use_c3):
        raise ValueError("主实验固定使用 Full=C1+C2+C3；结构消融请使用独立脚本。")

    fast_window = as_float(cfg.get("dep_window_fast"), 45.0)
    normal_window = as_float(cfg.get("dep_window_normal"), 60.0)
    if fast_window is None or normal_window is None or fast_window < 0 or normal_window < 0:
        raise ValueError("dep_window_fast 和 dep_window_normal 必须为非负数")
    if as_float(cfg.get("big_m"), 10000.0) is None or float(cfg.get("big_m", 10000.0)) <= 0:
        raise ValueError("big_m 必须为正数")

    inverse_c1_mode = str(
        cfg.get("inverse_c1_mode") or "branch_cut_triangle"
    ).strip().lower()
    if inverse_c1_mode not in {
        "branch_cut", "branch_cut_triangle", "triangle_branch_cut", "usercut_lazy",
        "lazy", "lazy_cycle", "lazy_triangle",
        "rank", "rank_big_m", "legacy",
    }:
        raise ValueError(
            "inverse_c1_mode 仅支持 branch_cut_triangle、lazy_cycle 或 rank"
        )
    for key, default in (
        ("inverse_c1_lazy_max_cuts_per_callback", 200),
        ("inverse_c1_lazy_max_triangles_per_layer", 12),
        ("inverse_c1_usercut_max_root_passes", 30),
        ("inverse_c1_usercut_node_interval", 100),
        ("inverse_c1_usercut_max_cuts_per_callback", 200),
        ("inverse_c1_usercut_max_triangles_per_layer", 20),
        ("inverse_c1_cut_history_limit", 50000),
    ):
        value = int(as_int(cfg.get(key), default) or 0)
        if value <= 0:
            raise ValueError(f"{key} 必须为正整数")
    usercut_tol = float(
        as_float(cfg.get("inverse_c1_usercut_violation_tol"), 1e-6) or 0.0
    )
    if usercut_tol < 0.0:
        raise ValueError("inverse_c1_usercut_violation_tol 不能为负数")
    nodefile_start = float(
        as_float(cfg.get("inverse_nodefile_start_gb"), 0.5) or 0.0
    )
    if nodefile_start < 0.0:
        raise ValueError("inverse_nodefile_start_gb 不能为负数")

    trains = data.select_trains(ctx.target_k)
    universe_size = len(data.event_universe(trains))
    if universe_size <= 0:
        raise ValueError("目标实例没有任何precedence index，无法执行学习流程")

    parallel_enabled = as_bool(cfg.get("sample_parallel"), True)
    parallel_resources = (
        _sample_parallel_resources(cfg, sample_count)
        if parallel_enabled
        else {
            "parallel_workers": 1,
            "threads_per_solve": int(as_int(cfg.get("threads"), 1) or 1),
            "worker_source": "SERIAL",
            "worker_basis": "serial",
            "physical_cpu_cores": None,
            "logical_cpu_threads": None,
            "available_memory_gb_at_launch": None,
        }
    )

    return {
        "target_k": len(trains),
        "sample_size": sample_size,
        "sample_design": configured_sampling_design_label(cfg, sample_count),
        "local_sample_size": method_sizes["local"],
        "stratified_sample_size": method_sizes["stratified"],
        "coverage_sample_size": method_sizes["coverage"],
        "required_usable_samples": sample_count,
        "method_targets": targets,
        "alpha_list": alphas,
        "main_alpha": main_alpha,
        "event_universe_size": universe_size,
        "dep_window_fast": fast_window,
        "dep_window_normal": normal_window,
        "segments_csv": str(data.data_dir / "segments.csv"),
        "packing_value_rule": "OPTIMAL uses ObjVal; nonoptimal incumbent uses ObjBound",
        "direct_threads_effective": int(
            as_int(cfg.get("direct_threads"), as_int(cfg.get("threads"), 1)) or 1
        ),
        "inverse_threads_effective": int(
            as_int(cfg.get("inverse_threads"), as_int(cfg.get("threads"), 1)) or 1
        ),
        "inverse_target_mipgap": float(
            as_float(cfg.get("inverse_mipgap"), as_float(cfg.get("mipgap"), 0.0))
            or 0.0
        ),
        "inverse_require_optimal": int(as_bool(cfg.get("inverse_require_optimal"), False)),
        "inverse_accept_time_limit_incumbent": int(
            as_bool(
                cfg.get("inverse_accept_time_limit_incumbent"),
                not as_bool(cfg.get("inverse_require_optimal"), False),
            )
            and not as_bool(cfg.get("inverse_require_optimal"), False)
        ),
        "inverse_mipfocus_effective": (
            as_int(cfg.get("inverse_mipfocus"), None)
        ),
        "inverse_norel_used": 0,
        "inverse_c1_mode": (
            "rank"
            if inverse_c1_mode in {"rank", "rank_big_m", "legacy"}
            else (
                "branch_cut_triangle"
                if inverse_c1_mode in {
                    "branch_cut", "branch_cut_triangle",
                    "triangle_branch_cut", "usercut_lazy"
                }
                else "lazy_cycle"
            )
        ),
        "inverse_sample_start_enabled": 0,
        "inverse_sample_hint_enabled": 0,
        "inverse_temporal_cuts_enabled": 0,
        "precedence_core_enabled": 0,
        "inverse_partition_mode": "NO4B_STRUCTURAL_QUOTIENT",
        "sample_threads_per_solve_effective": int(
            parallel_resources.get("threads_per_solve", as_int(cfg.get("sample_threads_per_solve"), 1) or 1)
        ),
        "sample_parallel": int(parallel_enabled),
        "sample_parallel_resources": parallel_resources,
        "sample_time_limit_effective": (
            "UNLIMITED"
            if as_float(cfg.get("sample_time_limit"), None) is None
            or float(as_float(cfg.get("sample_time_limit"), 0.0) or 0.0) <= 0
            else float(as_float(cfg.get("sample_time_limit"), 0.0) or 0.0)
        ),
    }


def run_after_direct(cfg: Mapping[str, Any], ctx: ExperimentContext) -> pd.DataFrame:
    """Run Stages 2--6 and aggregation while preserving Stage-1 outputs.

    This entry point deliberately ignores ``fresh_full_run`` so that an
    already-completed Direct run under ``01_direct`` is never deleted.
    Restricted-instance resume/clear behaviour is still controlled by the
    existing ``sample_resume`` and ``sample_clear_output`` settings.
    """
    pipeline_start = time.perf_counter()
    pipeline_cpu_start = time.process_time()
    ensure_dir(ctx.output_dir)

    data = HSRData(cfg)
    validation = validate_config(cfg, ctx, data)
    direct_summary_path = ctx.direct_dir / "direct_summary.csv"
    require_direct = as_bool(cfg.get("after_direct_require_direct_result"), True)
    if require_direct and not direct_summary_path.exists():
        raise FileNotFoundError(
            "RUN_AFTER_DIRECT 未找到已完成的 Direct 摘要："
            f"{direct_summary_path}。请先运行 1.solve_direct.py，"
            "或将 after_direct_require_direct_result 设为 FALSE。"
        )

    write_effective_parameters(cfg, ctx, data, validation)
    print("RUN_AFTER_DIRECT：保留 01_direct，不执行也不清空脚本1结果。")
    run_restricted_instances(cfg, ctx, data)
    run_vote(cfg, ctx, data)
    run_packing(cfg, ctx, data)
    legacy_core_dir = ctx.output_dir / "04B_precedence_core"
    if legacy_core_dir.exists():
        shutil.rmtree(legacy_core_dir, ignore_errors=True)
    run_inverse_all_alphas(cfg, ctx, data)
    run_fullfix_all_alphas(cfg, ctx, data)
    result = aggregate_experiment(cfg, ctx)

    append_stage_timing(
        ctx,
        {
            "stage": "RUN_AFTER_DIRECT",
            "variant": ctx.experiment_tag,
            "status": "COMPLETED",
            "model_build_sec": None,
            "solver_runtime_sec": None,
            "process_cpu_sec": time.process_time() - pipeline_cpu_start,
            "postprocess_write_sec": None,
            "total_wall_sec": time.perf_counter() - pipeline_start,
        },
    )
    timing_path = ctx.timing_dir / "stage_timing_log.csv"
    if timing_path.exists():
        write_csv(
            pd.read_csv(timing_path, encoding="utf-8-sig"),
            ensure_dir(ctx.aggregate_dir) / "stage_timing_summary.csv",
        )
    collect_all_case_rows(cfg, ctx.output_root)
    return result


def run_core_1_to_6(cfg: Mapping[str, Any], ctx: ExperimentContext) -> pd.DataFrame:
    pipeline_start = time.perf_counter()
    pipeline_cpu_start = time.process_time()
    if as_bool(cfg.get("fresh_full_run"), True):
        if ctx.output_dir.resolve() == ctx.output_root.resolve():
            raise RuntimeError(
                "fresh_full_run=TRUE时必须使用create_case_subfolder=TRUE，"
                "以免清空其他案例结果"
            )
        clean_dir(ctx.output_dir)
    else:
        ensure_dir(ctx.output_dir)
    data = HSRData(cfg)
    validation = validate_config(cfg, ctx, data)
    write_effective_parameters(cfg, ctx, data, validation)

    run_direct(cfg, ctx, data)
    run_restricted_instances(cfg, ctx, data)
    run_vote(cfg, ctx, data)
    run_packing(cfg, ctx, data)
    legacy_core_dir = ctx.output_dir / "04B_precedence_core"
    if legacy_core_dir.exists():
        shutil.rmtree(legacy_core_dir, ignore_errors=True)
    run_inverse_all_alphas(cfg, ctx, data)
    run_fullfix_all_alphas(cfg, ctx, data)
    result = aggregate_experiment(cfg, ctx)

    append_stage_timing(
        ctx,
        {
            "stage": "RUN_MAIN_EXPERIMENT",
            "variant": ctx.experiment_tag,
            "status": "COMPLETED",
            "model_build_sec": None,
            "solver_runtime_sec": None,
            "process_cpu_sec": time.process_time() - pipeline_cpu_start,
            "postprocess_write_sec": None,
            "total_wall_sec": time.perf_counter() - pipeline_start,
        },
    )
    timing_path = ctx.timing_dir / "stage_timing_log.csv"
    if timing_path.exists():
        write_csv(
            pd.read_csv(timing_path, encoding="utf-8-sig"),
            ensure_dir(ctx.aggregate_dir) / "stage_timing_summary.csv",
        )
    collect_all_case_rows(cfg, ctx.output_root)
    return result
