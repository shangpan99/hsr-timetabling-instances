import csv
import datetime
import json
import platform
import shutil
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

from .data import Config, Source, write_csv, write_json, key_label
from .generate import compose, assign_times, audit_instances, signature
from .report import detailed_profile, comparisons
from .validate import validate_group
from .constraints import copy_bounds
from .unique import GENERATOR_VERSION
from train_numbering import TrainNumberRegistry, REGISTRY_PATH, NUMBERING_VERSION


def snapshot(cfg, run_dir):
    rows = [dict(r) for r in cfg.rows]
    data_dir = run_dir / "_source"; data_dir.mkdir()
    for r in rows:
        if r["section"] == "input" and r["parameter"] in ("trains", "stations", "segments", "experiment_config"):
            key = "input." + r["parameter"]; p = cfg.path_value(key)
            if p:
                shutil.copy2(p, data_dir / p.name)
                r["value"] = "_source/" + p.name
    write_csv(run_dir / "config_used.csv", rows)
    return Config(run_dir / "config_used.csv")


def experiment_config(source, destination, k):
    path = source.paths["experiment_config"]
    if path is None: return
    with path.open(encoding="utf-8-sig", newline="") as f: rows = list(csv.reader(f))
    updates = {"train_limit": str(k), "data_dir": ".", "fast_train_type": source.cfg.get("model.fast_type"),
               "dep_window_fast": source.cfg.get("model.fast_departure_window"),
               "dep_window_normal": source.cfg.get("model.normal_departure_window"),
               "allow_early_departure": source.cfg.get("model.allow_early_departure")}
    found = set()
    for r in rows:
        if r and r[0].strip() in updates:
            key = r[0].strip(); found.add(key)
            while len(r) < 2: r.append("")
            r[1] = updates[key]
    rows.extend([[k, v, ""] for k, v in updates.items() if k not in found])
    with destination.open("w", encoding="utf-8-sig", newline="") as f: csv.writer(f).writerows(rows)


def catalog(source, path):
    rows = []
    def add(layer, name, level, meaning, file, control):
        rows.append(dict(layer=layer, feature=name, level=level, definition=meaning, output_file=file, config=control))
    for name, meaning in [
        ("type_direction", "选中模板的车型、方向"), ("route", "入口出口、完整径路、经过站和区间"),
        ("stop_pattern", "完整停站向量、中间停站数、可停站数、停站比例"),
        ("time_components", "逐项停站及区间时长、总运行时间、总停站时间、旅行时间、时间构成"),
        ("model_slack", "原始参考时长减模型下限；不自动修正负余量"),
        ("window_widths", "适用的提前和推迟幅度；绝对窗口端点随整列车平移")]:
        add("车", name, "S", meaning, "trains_detail.csv", "columns.*;model.*")
    add("车站", "infrastructure", "S", "站名、已有设施字段、安全间隔、停站下上界字段与启用状态", "stations_detail.csv", "model.*")
    add("车—车站", "events", "S/A", "完整事件、缺失标记、站角色；同一列车各有效事件施加同一个平移量", "train_station.csv", "time.*")
    add("区间", "technical", "S", "有向区间及分车型纯运行与起停附加时间；原有额外字段保留", "segments_detail.csv", "input.segments")
    add("车—区间", "duration_state", "S", "原区间时长、两端停站状态、适用下限与余量、可选上界", "train_segment.csv", "model.*")
    for f in source.counts:
        add("组级构成", f, "B" if f in source.cfg.list("features.count_families") else "C",
            "每类数量按K/N换算；原始逐车字段继承不意味着组级比例不变", "comparison_counts.csv", "features.count_families;tolerance.count_extra")
    for f in source.reference_time:
        add("组级时间", f, "B" if f in source.cfg.list("features.time_families") else "C",
            "按平移后时刻计算；进入时刻采用首站到达，缺失时采用首站出发", "comparison_counts.csv", "features.time_families")
    for f in source.duration_groups:
        add("组级时长", f, "B/C", "均值及累计分布按config启用；小样本条件分组只记录；极值和分位数记录", "duration_distributions.csv;comparison_durations.csv", "features.mean_families;features.cdf_families;features.conditional_min_events")
    add("车站", "conditional_stop_rate", "B/C", "中间停站/(中间停站+中间通过)，首末站不入分母", "stations_detail.csv", "features.station_stop_rates")
    add("车站/区间", "downstream_time_bins", "C", "平移后各站到发、各区间进入的分时数量", "station_time_bins.csv;segment_time_bins.csv", "time.bin_minutes")
    add("车—车", "space_and_time", "S/C", "模板空间关系继承；新车对数量、相对时刻、顺序、小间隔重新评估，包括同模板副本对", "train_train.csv;train_train_summary.json", "output.*pair_details")
    add("车—车", "adjacent_headways", "C", "逐站事件相邻间隔、下限余量、并列时刻明确标记；不把共站等同冲突", "station_neighbours.csv", "output.near_gap_minutes")
    add("生成规则", "exact_K_and_provenance", "A", "K<N无放回；K=N每模板一份；K>N全部保留再复制；编号唯一、来源可追溯", "lineage.csv;source_multiplicities.csv", "generation.*;copy.*")
    write_csv(path, rows)


def run(cfg_path, sizes=None, groups=None, backend=None, destination=None, model_values=None):
    wall_start = time.perf_counter()
    original_cfg = Config(cfg_path)
    overrides = {}
    if sizes is not None: overrides["generation.target_sizes"] = ";".join(map(str, sizes))
    if groups is not None: overrides["generation.groups_per_size"] = str(groups)
    if backend is not None: overrides["solver.backend"] = backend
    overrides.update(model_values or {})
    for key, value in overrides.items():
        if not any(r["section"] + "." + r["parameter"] == key for r in original_cfg.rows):
            section, parameter = key.split(".", 1)
            original_cfg.rows.append(dict(section=section, parameter=parameter, value=str(value), unit="", description="Inherited from shared config.csv"))
        original_cfg.values[key] = value
        for r in original_cfg.rows:
            if r["section"] + "." + r["parameter"] == key: r["value"] = value
    sizes = [int(x) for x in original_cfg.list("generation.target_sizes")]
    count = original_cfg.int("generation.groups_per_size", 5)
    if not sizes or any(k <= 0 for k in sizes) or len(set(sizes)) != len(sizes) or count <= 0:
        raise ValueError("Use distinct positive target sizes and a positive group count")
    if destination is None:
        raise ValueError("Use generate_trainsets.py to publish complete flat-layout instances")
    run_dir = Path(destination)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = snapshot(original_cfg, run_dir); source = Source(cfg)
    max_copies = max(3, max(copy_bounds(source, k)[1] - 1 for k in sizes))
    registry = TrainNumberRegistry.load(REGISTRY_PATH).ensure(
        [t.source for t in source.trains], max_copies)
    registry.save(REGISTRY_PATH)
    registry.save(run_dir / "train_number_registry.csv")
    unique_mode = cfg.get("solver.composition_mode") == "unique_score"
    write_json(run_dir / "manifest.json", {"version": "1.0.0", "source_train_count": source.n,
                                           "generator_version": GENERATOR_VERSION if unique_mode else "legacy",
                                           "source_sha256": source.fingerprints, "python": sys.version, "platform": platform.platform(),
                                           "target_sizes": sizes, "requested_groups_per_size": count,
                                           "operating_start": source.horizon_start, "operating_end": source.horizon_end,
                                           "unit": "minutes", "status": "RUNNING",
                                           "numbering_version": NUMBERING_VERSION,
                                           "number_registry_sha256": registry.fingerprint})
    write_csv(run_dir / "input_audit.csv", source.audit, ["train", "field", "action", "resolved_value"])
    catalog(source, run_dir / "feature_catalog.csv")
    print(f"[source] N={source.n}; horizon={source.horizon_start:g}..{source.horizon_end:g} minutes", flush=True)
    print("[source] writing six-layer original-data profile", flush=True)
    baseline = detailed_profile(source, [(t.index, 0.0) for t in source.trains], [t.source for t in source.trains], run_dir / "baseline", True)
    log_rows = []; summary_rows = []; diversity = []; source_usage = Counter(); statuses = []
    start = time.perf_counter(); max_attempts = cfg.int("generation.max_attempts_per_group", 8)
    if max_attempts <= 0: raise ValueError("max_attempts_per_group must be positive")
    for k in sizes:
        accepted = []; seen = set(); exclusions = []; failed = False
        for group in range(1, count + 1):
            group_start = time.perf_counter()
            deadline = group_start + cfg.float("solver.time_limit_seconds", 60) if unique_mode else None
            success = False; last = "SEARCH_EXHAUSTED"
            candidate_exclusions = list(exclusions)
            for attempt in range(1, max_attempts + 1):
                seed = (cfg.int("generation.seed", 40) + 1000003 * k + 10007 * group + 101 * attempt) % 2000000000
                print(f"[K{k} group {group}/{count}] composition, attempt {attempt}", flush=True)
                m, result = compose(source, k, seed, candidate_exclusions, deadline=deadline)
                composition_result = dict(result)
                log_rows.append({"K": k, "group": group, "attempt": attempt, "seed": seed, "stage": "composition", **result})
                if m is None:
                    if result["status"] == "INFEASIBLE" and not candidate_exclusions:
                        last = "COMPOSITION_INFEASIBLE"
                    elif (result["status"] == "INFEASIBLE" and not cfg.bool("generation.allow_time_only_difference", True)
                          and accepted and len(candidate_exclusions) == len(exclusions)):
                        last = "NO_MORE_COMPOSITIONS"
                    elif result["status"] == "UNKNOWN_LIMIT":
                        last = "UNKNOWN_LIMIT"
                    else:
                        last = "SEARCH_EXHAUSTED"
                    # A slower machine must not advance to another seed merely
                    # because a proof took longer. Only conclusive outcomes retry.
                    if result["status"] == "INFEASIBLE" or (unique_mode and result["status"] == "UNKNOWN_LIMIT"): break
                    continue
                print(f"[K{k} group {group}/{count}] assigning whole-train shifts", flush=True)
                time_seed = seed if unique_mode else seed + 1
                instances, result = assign_times(source, m, time_seed, deadline=deadline)
                log_rows.append({"K": k, "group": group, "attempt": attempt, "seed": time_seed, "stage": "time_assignment", **result})
                if instances is None:
                    if unique_mode and result["status"] != "INFEASIBLE":
                        last = "UNKNOWN_LIMIT"; break
                    candidate_exclusions.append(m); last = "SEARCH_EXHAUSTED"; continue
                sig = signature(source, instances, cfg.bool("generation.allow_time_only_difference", True))
                if sig in seen:
                    last = "DISTINCT_GROUP_NOT_FOUND"; continue
                errors, _ = audit_instances(source, k, instances)
                if errors: raise RuntimeError("Internal generated-instance check: " + " | ".join(errors))
                folder = run_dir / f"._pending_K{k}_G{group:02d}"; folder.mkdir(parents=True)
                ids, _ = registry.assign([source.trains[i].source for i, _shift in instances])
                write_csv(folder / "trains.csv", source.output_rows(instances, ids), source.fields)
                for key in ("stations", "segments"): shutil.copy2(source.paths[key], folder / (key + ".csv"))
                details = detailed_profile(source, instances, ids, folder)
                comparisons(source, instances, folder)
                write_csv(folder / "source_multiplicities.csv", [{"source_id": t.source, "occurrences": n, "extra_copies": max(0, n - 1)} for t, n in zip(source.trains, m)])
                errors, reread = validate_group(source, folder, k)
                if errors: raise RuntimeError("Re-read CSV verification: " + " | ".join(errors))
                validation = {"status": "PASS", "S_template_inheritance": "PASS", "A_generation_rules": "PASS", "B_configured_targets": "PASS", "C_statistics": "RECORDED", "independent_csv_reread": True, "normalized_signature": sig, "seed": seed, "operational_timetable_feasibility": "NOT_TESTED"}
                if unique_mode:
                    validation.update(generator_version=GENERATOR_VERSION,
                                      composition_unique=composition_result.get("unique", False),
                                      timing_unique=result.get("unique", False),
                                      composition_model_sha256=composition_result.get("model_sha256"),
                                      timing_model_sha256=result.get("model_sha256"))
                    if not validation["composition_unique"] or not validation["timing_unique"]:
                        raise RuntimeError("Unique-score generation requires both uniqueness checks")
                write_json(folder / "validation.json", validation)
                final_folder = folder.with_name(f"K{k}_G{group:02d}"); folder.rename(final_folder)
                for old_group, old_m, old_instances in accepted:
                    edit = sum(abs(a - b) for a, b in zip(m, old_m)) // 2
                    diversity.append({"K": k, "group_a": old_group, "group_b": group, "composition_replacement_count": edit,
                                      "same_composition": m == old_m, "same_up_to_common_shift": signature(source, instances) == signature(source, old_instances)})
                accepted.append((group, m, instances)); seen.add(sig)
                if not cfg.bool("generation.allow_time_only_difference", True): exclusions.append(m)
                for t, n in zip(source.trains, m): source_usage[(k, t.source)] += n
                summary_rows.append({"K": k, "group": group, "status": "PASS", "distinct_sources": details["distinct_sources"],
                                     "max_occurrences": details["max_source_occurrences"], "type_350": details["type_counts"].get("350", 0),
                                     "mean_travel": details["travel"]["mean"], "mean_run": details["run"]["mean"], "mean_dwell": details["dwell"]["mean"],
                                     "min_shift": details["shift_min"], "max_shift": details["shift_max"],
                                     "latest_event": details["event_max"], "folder": str(final_folder.relative_to(run_dir))})
                summary_rows[-1]["total_seconds"] = time.perf_counter() - group_start
                print(f"[K{k} group {group}] PASS: template, composition, time-range and CSV re-read checks", flush=True)
                success = True; break
            write_csv(run_dir / "solve_log.csv", log_rows)
            write_csv(run_dir / "generation_summary.csv", summary_rows)
            if not success:
                statuses.append({"K": k, "requested": count, "generated": len(accepted), "status": last})
                print(f"[K{k}] stopped: {last}; generated {len(accepted)}/{count}; no tolerance was relaxed", flush=True)
                failed = True; break
        if not failed: statuses.append({"K": k, "requested": count, "generated": len(accepted), "status": "COMPLETE"})
        elif unique_mode:
            break  # The publisher requires a complete run; do not waste later groups.
    write_csv(run_dir / "run_status.csv", statuses)
    write_csv(run_dir / "group_diversity.csv", diversity, ["K", "group_a", "group_b", "composition_replacement_count", "same_composition", "same_up_to_common_shift"])
    write_csv(run_dir / "source_usage_all_groups.csv", [{"K": k, "source_id": tid, "total_occurrences": n} for (k, tid), n in sorted(source_usage.items())])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(status="COMPLETE" if all(r["status"] == "COMPLETE" for r in statuses) else "PARTIAL_OR_FAILED",
                    generation_seconds=time.perf_counter() - start, total_wall_seconds=time.perf_counter() - wall_start)
    write_json(run_dir / "manifest.json", manifest)
    print("[output] " + str(run_dir), flush=True)
    return run_dir, statuses
