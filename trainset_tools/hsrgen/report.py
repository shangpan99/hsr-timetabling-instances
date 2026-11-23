import csv
import math
from collections import Counter, defaultdict
from itertools import combinations

from .data import write_csv, write_json, number, key_label
from .constraints import quantile, composition_requirements, time_requirements, quota


def stats(values):
    if not values: return {"n": 0, "mean": "", "min": "", "p10": "", "p25": "", "median": "", "p75": "", "p90": "", "p95": "", "max": ""}
    return dict(n=len(values), mean=sum(values) / len(values), min=min(values),
                p10=quantile(values, .1), p25=quantile(values, .25), median=quantile(values, .5),
                p75=quantile(values, .75), p90=quantile(values, .9), p95=quantile(values, .95), max=max(values))


def multiplicities(source, instances):
    m = [0] * source.n
    for i, _ in instances: m[i] += 1
    return m


def samples_for(source, m):
    out = defaultdict(lambda: defaultdict(list))
    for t, n in zip(source.trains, m):
        if n:
            for f, groups in t.samples.items():
                for key, vs in groups.items(): out[f][key].extend(vs * n)
    return out


def pair_reports(source, instances, ids, folder, details=False):
    if not hasattr(source, "_pair_cache"):
        source._pair_cache = ({}, [set(t.segments) for t in source.trains],
                              [{s: j for j, s in enumerate(t.stations)} for t in source.trains])
    spatial_cache, paths, positions = source._pair_cache
    spatial = Counter(); types_shared = Counter(); overlap_sizes = Counter()
    relation_summary = Counter(); minimum_gaps = []; pair_rows = 0
    fh = None; writer = None
    if details:
        fh = (folder / "train_train.csv").open("w", encoding="utf-8-sig", newline="")
    try:
        for ai, bi in combinations(range(len(instances)), 2):
            a, sa = instances[ai]; b, sb = instances[bi]; ta, tb = source.trains[a], source.trains[b]
            cachekey = tuple(sorted((a, b)))
            if cachekey not in spatial_cache:
                shared_stations = tuple(s for s in ta.stations if s in positions[b])
                shared_segments = tuple(s for s in ta.segments if s in paths[b])
                relation = ("same_path" if ta.stations == tb.stations else
                            "contains" if paths[a] >= paths[b] or paths[b] >= paths[a] else
                            "overlap" if shared_segments else "station_only" if shared_stations else "disjoint")
                spatial_cache[cachekey] = shared_stations, shared_segments, relation
            shared_stations, shared_segments, relation = spatial_cache[cachekey]
            spatial[relation] += 1; overlap_sizes[len(shared_segments)] += 1; pair_rows += 1
            stop_combos = Counter(); arrival_differences = []; departure_differences = []
            overtakes = []; inversions = []; ties = 0
            for s in shared_stations:
                ja, jb = positions[a][s], positions[b][s]
                if 0 < ja < len(ta.stations) - 1 and 0 < jb < len(tb.stations) - 1:
                    stop_combos[f"{int(ta.stops[ja])}-{int(tb.stops[jb])}"] += 1
                av, bv, ad, bd = ta.arrivals[ja], tb.arrivals[jb], ta.departures[ja], tb.departures[jb]
                da = (av + sa - bv - sb) if av is not None and bv is not None else None
                dd = (ad + sa - bd - sb) if ad is not None and bd is not None else None
                if da is not None: arrival_differences.append((s, da))
                if dd is not None: departure_differences.append((s, dd))
                if da is not None and dd is not None:
                    if da * dd < -1e-9: overtakes.append(s)
                    if abs(da) < 1e-8 or abs(dd) < 1e-8: ties += 1
            for u, v in shared_segments:
                ja, jb = positions[a][u], positions[b][u]
                da = ta.departures[ja] + sa - tb.departures[jb] - sb
                db = ta.arrivals[ja + 1] + sa - tb.arrivals[jb + 1] - sb
                if da * db < -1e-9: inversions.append((u, v))
            gaps = [abs(d) for _, d in arrival_differences + departure_differences]
            min_gap = min(gaps) if gaps else None
            if shared_segments:
                relation_summary["pairs_sharing_segment"] += 1
                types_shared[tuple(sorted((ta.kind, tb.kind)))] += 1
            if min_gap is not None:
                minimum_gaps.append(min_gap)
                if min_gap < source.cfg.float("output.near_gap_minutes", 5): relation_summary["pairs_close_at_common_station"] += 1
            relation_summary["station_order_reversal_events"] += len(overtakes)
            relation_summary["segment_order_inversion_events"] += len(inversions)
            relation_summary["pairs_with_same_source"] += int(a == b)
            relation_summary["simultaneous_station_order_cases"] += ties
            if details:
                row = {"train_a": ids[ai], "train_b": ids[bi], "source_a": ta.source, "source_b": tb.source,
                       "type_a": ta.kind, "type_b": tb.kind, "direction_a": ta.direction, "direction_b": tb.direction,
                       "path_relation": relation, "shared_station_count": len(shared_stations), "shared_segment_count": len(shared_segments),
                       "shared_stations": key_label(shared_stations), "shared_segments": key_label(shared_segments),
                       "intermediate_stop_combinations": key_label(stop_combos),
                       "arrival_difference_a_minus_b": key_label(arrival_differences),
                       "departure_difference_a_minus_b": key_label(departure_differences),
                       "min_same_event_gap": number(min_gap), "station_order_reversals": key_label(overtakes),
                       "segment_order_inversions": key_label(inversions), "order_tie_count": ties}
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(row)); writer.writeheader()
                writer.writerow(row)
    finally:
        if fh: fh.close()
    summary = {"total_pairs": pair_rows, **dict(relation_summary),
               "share_sharing_segment": relation_summary["pairs_sharing_segment"] / pair_rows if pair_rows else None,
               "path_relations": dict(spatial), "shared_segment_count_distribution": dict(overlap_sizes),
               "type_pairs_on_shared_segments": {key_label(k): v for k, v in types_shared.items()},
               "min_gap_at_common_stations": stats(minimum_gaps),
               "note": "C级描述；共站/接近/参考顺序反转不等于模型冲突或可行运行图的认证。"}
    write_json(folder / "train_train_summary.json", summary)
    return summary


def detailed_profile(source, instances, ids, folder, baseline=False):
    folder.mkdir(parents=True, exist_ok=True)
    m = multiplicities(source, instances); counts = source.counts_for(m); samples = samples_for(source, m)
    tr, ts, sg = [], [], []; events = defaultdict(list); station_bins = Counter(); segment_bins = Counter()
    lineage = []; copy_counts = Counter()
    for (i, shift), gid in zip(instances, ids):
        t = source.trains[i]; early, late = source.origin_window(t)
        copy_index = copy_counts[t.source]; copy_counts[t.source] += 1
        line = {"generated_id": gid, "source_id": t.source, "copy_index": copy_index, "source_index": i, "shift_minutes": shift,
                "train_type": t.kind, "source_entry": t.entry, "generated_entry": t.entry + shift,
                "source_first_departure": t.departures[0], "generated_first_departure": t.departures[0] + shift,
                "generated_last_arrival": t.arrivals[-1] + shift,
                "origin_window_lower": t.departures[0] + shift - early,
                "origin_window_upper": t.departures[0] + shift + late}
        lineage.append(line)
        tr.append({**line, "direction": t.direction, "origin": t.od[0], "destination": t.od[1],
                   "route": key_label(t.stations), "segments": key_label(t.segments), "stop_pattern": key_label(t.pattern),
                   "stop_vector": key_label(t.stops), "station_count": len(t.stations), "segment_count": len(t.segments),
                   "intermediate_stops": t.stop_count, "available_intermediate_stations": len(t.stations) - 2,
                   "stop_ratio": t.stop_count / (len(t.stations) - 2) if len(t.stations) > 2 else "",
                   "travel_time": t.travel, "total_dwell": sum(t.samples["dwell"]["all"]),
                   "total_run": sum(t.samples["run"]["all"]), "dwell_share": t.samples["dwell_share"]["all"][0],
                   "relative_arrivals": key_label([None if v is None else v - t.entry for v in t.arrivals]),
                   "relative_departures": key_label([None if v is None else v - t.entry for v in t.departures])})
        for original in t.train_station:
            row = dict(original); row["generated_id"] = gid
            row["source_id"] = t.source; j = row["position"]; s = row["station"]
            for event, col in (("arrival", "arrivals"), ("departure", "departures")):
                v = getattr(t, col)[j]
                row[event] = number(v + shift) if v is not None else ""
                if v is not None:
                    events[(s, event)].append((v + shift, gid, t.source))
                    station_bins[(s, event, math.floor((v + shift) / source.bin_minutes))] += 1
            row["origin_departure_window_lower"] = line["origin_window_lower"] if j == 0 else ""
            row["origin_departure_window_upper"] = line["origin_window_upper"] if j == 0 else ""
            ts.append(row)
        for original in t.train_segment:
            row = dict(original); row["generated_id"] = gid
            row["departure"] = float(row["departure"]) + shift; row["arrival"] = float(row["arrival"]) + shift
            segment_bins[(row["from"], row["to"], math.floor(row["departure"] / source.bin_minutes))] += 1
            sg.append(row)
    write_csv(folder / "trains_detail.csv", tr)
    write_csv(folder / "lineage.csv", lineage)
    write_csv(folder / "train_station.csv", ts)
    write_csv(folder / "train_segment.csv", sg)
    station_out = []
    for r in source.station_rows:
        s = r["station"]; st = counts["station_stop"][s]; ps = counts["station_pass"][s]
        dv = [v for (station, kind), vs in samples["dwell_station_type"].items() if station == s for v in vs]
        station_out.append({**r, "dwell_max_enabled": source.cfg.bool("model.dwell_upper_enabled"),
                            "visit_count": counts["station_visit"][s], "origin_count": counts["station_origin"][s],
                            "destination_count": counts["station_destination"][s], "intermediate_stop_count": st,
                            "intermediate_pass_count": ps, "intermediate_stop_rate": st / (st + ps) if st + ps else "",
                            "type_counts": key_label({kind: counts["station_type"][(s, kind)] for kind in source.counts["train_type"]}),
                            **{"dwell_" + key: value for key, value in stats(dv).items()}})
    write_csv(folder / "stations_detail.csv", station_out)
    write_csv(folder / "segments_detail.csv", [{**r, "train_count": counts["segment"][(r["from"], r["to"])],
                                               "run_max_enabled": source.cfg.bool("model.run_upper_enabled")}
                                              for r in source.segment_rows])
    write_csv(folder / "station_time_bins.csv", [{"station": s, "event": e, "bin_start": b * source.bin_minutes, "count": n}
                                                for (s, e, b), n in sorted(station_bins.items())])
    write_csv(folder / "segment_time_bins.csv", [{"from": u, "to": v, "bin_start": b * source.bin_minutes, "count": n}
                                                for (u, v, b), n in sorted(segment_bins.items())])
    neighbours = []; event_gaps = []; small_gaps = 0
    for (s, event), items in sorted(events.items()):
        items.sort()
        field = source.station_map[s].get("h_arr" if event == "arrival" else "h_dep", "")
        headway = float(field) if field != "" else None
        rank = 0
        for j, (v, gid, source_id) in enumerate(items):
            gap = v - items[j - 1][0] if j else None
            if j == 0 or gap > 1e-8: rank = j + 1
            if gap is not None:
                event_gaps.append(gap); small_gaps += int(gap < source.cfg.float("output.near_gap_minutes", 5))
            neighbours.append({"station": s, "event": event, "time": v, "train": gid, "source_id": source_id,
                               "time_rank": rank, "previous_train": items[j - 1][1] if j else "",
                               "next_train": items[j + 1][1] if j + 1 < len(items) else "", "gap": number(gap),
                               "minimum_headway": number(headway), "headway_slack": number(gap - headway) if gap is not None and headway is not None else "",
                               "order_status": "simultaneous_unordered" if ((gap is not None and gap < 1e-8) or
                                              (j + 1 < len(items) and abs(items[j + 1][0] - v) < 1e-8)) else "ordered"})
    write_csv(folder / "station_neighbours.csv", neighbours)
    distribution_rows = []
    for f, groups in source.duration_groups.items():
        for key, original in groups.items():
            vs = samples[f].get(key, [])
            enough = key == "all" or len(original) >= source.cfg.int("features.conditional_min_events", 10)
            mean_b = enough and f in source.cfg.list("features.mean_families")
            cdf_b = enough and f in source.cfg.list("features.cdf_families")
            distribution_rows.append({"family": f, "group": key_label(key), "mean_level": "B" if mean_b else "C",
                                      "cdf_level": "B" if cdf_b else "C", "status": "PRESENT" if vs else "NOT_PRESENT",
                                      **stats(vs)})
    write_csv(folder / "duration_distributions.csv", distribution_rows)
    pair = pair_reports(source, instances, ids, folder,
                        source.cfg.bool("output.source_pair_details", True) if baseline else source.cfg.bool("output.generated_pair_details"))
    summary = {"trains": len(instances), "distinct_sources": sum(n > 0 for n in m), "max_source_occurrences": max(m),
               "type_counts": dict(counts["train_type"]), "stop_count_counts": dict(counts["stop_count"]),
               "entry_min": min(source.trains[i].entry + d for i, d in instances),
               "event_max": max(source.trains[i].end + d for i, d in instances),
               "shift_min": min(d for _, d in instances), "shift_max": max(d for _, d in instances),
               "reference_run_below_model_min": sum(float(r["run_slack"]) < -1e-8 for r in sg if r["run_slack"] != ""),
               "reference_dwell_below_model_min": sum(float(r["dwell_slack"]) < -1e-8 for r in ts if r["dwell_slack"] != ""),
               "travel": stats(samples["travel"]["all"]), "run": stats(samples["run"]["all"]), "dwell": stats(samples["dwell"]["all"]),
               "adjacent_event_gaps": stats(event_gaps), "small_adjacent_event_gaps": small_gaps,
               "pairs": pair, "operational_timetable_feasibility": "NOT_TESTED"}
    write_json(folder / "summary.json", summary)
    return summary


def comparisons(source, instances, folder):
    k = len(instances); m = multiplicities(source, instances); counts = source.counts_for(m)
    rows = []
    for family, original in source.counts.items():
        for key, n in original.items():
            target = n * k / source.n; actual = counts[family][key]
            lo, hi = quota(target, source.cfg.int("tolerance.count_extra", 0))
            b = family in source.cfg.list("features.count_families")
            rows.append({"family": family, "category": key_label(key), "level": "B" if b else "C", "original_count": n,
                         "target_count": target, "generated_count": actual, "lower": lo if b else "", "upper": hi if b else "",
                         "original_per_train": n / source.n, "generated_per_train": actual / k,
                         "difference_pp": 100 * (actual / k - n / source.n), "status": "PASS" if b and lo <= actual <= hi else "FAIL" if b else "RECORDED"})
    time_counts = source.time_counts(instances)
    for f, key, lo, hi, n in time_requirements(source, k):
        actual = time_counts[f][key]
        rows.append({"family": f, "category": key_label(key), "level": "B", "original_count": n,
                     "target_count": n * k / source.n, "generated_count": actual, "lower": lo, "upper": hi if math.isfinite(hi) else "",
                     "original_per_train": n / source.n, "generated_per_train": actual / k,
                     "difference_pp": 100 * (actual / k - n / source.n), "status": "PASS" if lo <= actual <= hi else "FAIL"})
    write_csv(folder / "comparison_counts.csv", rows)
    checks = []
    for q in composition_requirements(source, k):
        checks.append({"level": "B" if q.rule != "coverage" else "A", "family": q.family, "category": key_label(q.key),
                       "rule": q.rule, "reference_n": q.reference_n, "reference_target": q.target,
                       "linear_lhs": q.value(m), "lower": q.lower if math.isfinite(q.lower) else "",
                       "upper": q.upper if math.isfinite(q.upper) else "", "status": "PASS" if q.satisfied(m) else "FAIL"})
    write_csv(folder / "constraint_checks.csv", checks)
    generated = samples_for(source, m); durations = []
    for f, groups in source.duration_groups.items():
        for key, values in groups.items():
            vs = generated[f].get(key, []); a, b = stats(values), stats(vs)
            durations.append({"family": f, "category": key_label(key), "original_n": len(values), "generated_n": len(vs),
                              "original_mean": a["mean"], "generated_mean": b["mean"],
                              "mean_difference_percent": 100 * (b["mean"] / a["mean"] - 1) if vs and a["mean"] else "",
                              "original_median": a["median"], "generated_median": b["median"],
                              "original_p90": a["p90"], "generated_p90": b["p90"],
                              "original_min": a["min"], "generated_min": b["min"], "original_max": a["max"], "generated_max": b["max"]})
    write_csv(folder / "comparison_durations.csv", durations)
