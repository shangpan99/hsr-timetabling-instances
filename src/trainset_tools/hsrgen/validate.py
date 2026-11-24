"""Re-open emitted CSV files and verify them against the source templates."""
from collections import Counter, defaultdict
from pathlib import Path

from .data import Config, Source, read_csv, parse_time, truth, sha, write_csv
from .generate import audit_instances, signature
from train_numbering import TrainNumberRegistry


def validate_group(source, folder, k):
    folder = Path(folder); errors = []
    fields, rows = read_csv(folder / "trains.csv")
    _, lineage = read_csv(folder / "lineage.csv")
    by_id = defaultdict(list)
    idcol = source.cols["id"]
    for r in rows: by_id[r[idcol]].append(r)
    if len(by_id) != k or len(lineage) != k: errors.append("A: generated count / lineage count")
    if len(set(r["generated_id"] for r in lineage)) != len(lineage): errors.append("A: duplicate generated IDs in lineage")
    if set(by_id) != {r["generated_id"] for r in lineage}: errors.append("A: IDs do not match lineage")
    registry_path = folder.parent / "train_number_registry.csv"
    if registry_path.exists():
        expected_ids, copy_indices = TrainNumberRegistry.load(registry_path).assign([r["source_id"] for r in lineage])
        if expected_ids != [r["generated_id"] for r in lineage]:
            errors.append("A: train numbers do not match the fixed numbering registry")
        if any(int(r.get("copy_index", -1)) != i for r, i in zip(lineage, copy_indices)):
            errors.append("A: incorrect copy index in lineage")
    if fields != source.fields: errors.append("A: unexpected columns / column order")
    instances = []
    changed = {idcol, source.cols["type"], source.cols["arrival"], source.cols["departure"], *source.extra_times}
    for line in lineage:
        gid = line["generated_id"]; sid = line["source_id"]
        if sid not in source.by_id:
            errors.append("A: unknown source " + sid); continue
        t = source.by_id[sid]; shift = float(line["shift_minutes"]); instances.append((t.index, shift))
        rr = sorted(by_id.get(gid, []), key=lambda r: float(r[source.cols["sequence"]]))
        if len(rr) != len(t.rows): errors.append("S: route row count " + gid); continue
        for j, (row, original) in enumerate(zip(rr, t.rows)):
            for col in source.fields:
                if col not in changed and row.get(col, "") != original.get(col, ""):
                    errors.append("S: modified template field " + gid + " " + col)
            if row[source.cols["type"]] != t.kind: errors.append("S: incorrect resolved type " + gid)
            for col, before in ((source.cols["arrival"], t.arrivals[j]), (source.cols["departure"], t.departures[j])):
                after = parse_time(row[col], source.fmt)
                if (before is None) != (after is None): errors.append("S: missing-event mask " + gid)
                elif before is not None and abs(after - before - shift) > 1e-7: errors.append("S: nonuniform shift " + gid)
            for col in source.extra_times:
                before, after = parse_time(original.get(col, ""), source.fmt), parse_time(row.get(col, ""), source.fmt)
                if (before is None) != (after is None) or (before is not None and abs(after - before - shift) > 1e-7):
                    errors.append("S: additional timestamp shift " + gid)
        early, late = source.origin_window(t)
        for key, expected in (("generated_entry", t.entry + shift), ("generated_first_departure", t.departures[0] + shift),
                              ("generated_last_arrival", t.arrivals[-1] + shift),
                              ("origin_window_lower", t.departures[0] + shift - early),
                              ("origin_window_upper", t.departures[0] + shift + late)):
            if abs(float(line[key]) - expected) > 1e-7: errors.append("S/A: lineage event or window " + gid)
    for name in ("stations", "segments"):
        if sha(folder / (name + ".csv")) != source.fingerprints[name]: errors.append("S: changed fixed " + name)
    if len(instances) == k:
        other, _ = audit_instances(source, k, instances); errors.extend(other)
    return list(dict.fromkeys(errors)), instances


def validate_run(run_dir):
    run_dir = Path(run_dir).resolve(); source = Source(Config(run_dir / "config_used.csv"))
    results = []; seen = defaultdict(set)
    folders = list(run_dir.glob("K*_G*")) + list(run_dir.glob("K*/group_*"))
    for folder in sorted(folders):
        if not folder.is_dir(): continue
        k = int(folder.name.split("_")[0][1:]) if "_G" in folder.name else int(folder.parent.name[1:])
        try:
            errors, instances = validate_group(source, folder, k)
            if instances:
                sig = signature(source, instances, source.cfg.bool("generation.allow_time_only_difference", True))
                if sig in seen[k]: errors.append("A: duplicate group / common translation")
                seen[k].add(sig)
            results.append({"group": str(folder.relative_to(run_dir)), "status": "PASS" if not errors else "FAIL", "errors": " | ".join(errors)})
        except Exception as exc:
            results.append({"group": str(folder.relative_to(run_dir)), "status": "ERROR", "errors": str(exc)})
    write_csv(run_dir / "independent_validation.csv", results, ["group", "status", "errors"])
    if not results: raise ValueError("No generated groups found in " + str(run_dir))
    return results
