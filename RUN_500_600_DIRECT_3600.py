# -*- coding: utf-8 -*-
"""Reuse K500/K600 G01..G10; run Direct only, with TimeLimit=3600 seconds.

Place this file beside the original experiment's config.csv and hsr_pipeline.py.
    python RUN_500_600_DIRECT_3600.py --dry-run
    python RUN_500_600_DIRECT_3600.py
    python RUN_500_600_DIRECT_3600.py --workers 20
    python RUN_500_600_DIRECT_3600.py --resume latest --workers 4
    python RUN_500_600_DIRECT_3600.py --new-batch

No generation, feasibility solve, or stages 2..6 are called. By default,
resume the latest batch and skip completed cases (including batches from v1).
Unfinished cases restart their 3600-second search; solver search trees are
not checkpointed. Model construction and validation are outside TimeLimit.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

SCRIPT = Path(__file__).resolve()
TIME_LIMIT = 3600
DEFAULT_WORKERS = 1
# Press Run/F5 in Spyder to resume the latest existing batch without arguments.
DEFAULT_RESUME = "latest"
CASE_KEYS = [(k, group) for k in (500, 600) for group in range(1, 11)]
INPUT_FILES = ("trains.csv", "stations.csv", "segments.csv")
SUMMARY_FIELDS = [
    "case_id", "K", "group", "run_state", "solver_status", "solution_found",
    "validated_solution", "conclusion", "time_limit_sec", "threads",
    "sol_count", "obj_val", "obj_bound", "mip_gap_percent", "solver_runtime_sec",
    "solver_runtime_accumulated_sec", "model_build_sec", "total_wall_sec",
    "task_wall_sec", "input_sha256", "source_dir", "result_dir", "error",
]


def local_imports(project):
    # A Spyder console may still have modules loaded from a different copy.
    if str(project) in sys.path:
        sys.path.remove(str(project))
    sys.path.insert(0, str(project))
    for name in list(sys.modules):
        if (name in ("config_loader", "case_manager", "hsr_pipeline", "feasibility_core")
                or name in ("trainset_tools", "hsrgen")
                or name.startswith(("trainset_tools.", "hsrgen."))):
            sys.modules.pop(name, None)


def resolve_path(value, project):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project / path).resolve()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def file_lock(path):
    """OS releases this lock even when a process is killed; no stale PID files."""
    stream = path.open("a+b")
    try:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"该批次或案例仍有进程运行，请等待其停止后续跑：{path}") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()


def classify_result(summary):
    found = int(summary.get("sol_count") or 0) > 0
    valid = found and str(summary.get("validator_pass", "")).lower() in ("1", "1.0", "true")
    status = summary.get("status_name", "UNKNOWN")
    if valid:
        conclusion = "找到可行解（校验通过）"
    elif found:
        conclusion = "找到候选解，但运行图校验未通过"
    elif status == "TIME_LIMIT":
        conclusion = "达到时限，未找到可行解（不等于不可行）"
    elif status == "INFEASIBLE":
        conclusion = "求解器已证明不可行"
    else:
        conclusion = f"{status}，无可行解记录"
    return dict(solution_found=int(found), validated_solution=int(valid), conclusion=conclusion)


def run_worker(task_path):
    task = json.loads(Path(task_path).read_text(encoding="utf-8"))
    # Protect orphan workers too: killing the parent must not allow a second
    # solver to overwrite a still-running worker's files during resume.
    with file_lock(Path(task["result_dir"]) / ".worker.lock"):
        return solve_task(task_path)


def solve_task(task_path):
    task = json.loads(Path(task_path).read_text(encoding="utf-8"))
    result_dir = Path(task["result_dir"])
    result = dict(task["summary_row"])
    started = time.perf_counter()
    try:
        project = Path(task["project_dir"])
        local_imports(project)
        import gurobipy as gp
        import hsr_pipeline as hp
        from case_manager import input_fingerprint

        cfg = task["config"]
        if float(cfg["direct_time_limit"]) != TIME_LIMIT:
            raise ValueError("本脚本只运行 Direct TimeLimit=3600 秒")
        actual_hash, _ = input_fingerprint(cfg["data_dir"], task["K"])
        if actual_hash != task["input_sha256"]:
            raise ValueError("本次输入快照已改变，停止求解")
        ctx = hp.ExperimentContext(
            experiment_group=str(cfg.get("experiment_group") or "MAIN_EXPERIMENT"),
            experiment_tag=task["case_id"], target_k=task["K"],
            output_root=Path(task["run_dir"]), output_dir=result_dir,
            sample_sizes=(), sample_counts=(),
        )
        result.update(python_version=platform.python_version(),
                      gurobi_version=".".join(map(str, gp.gurobi.version())), pid=os.getpid())
        # Use the experiment's own model and its existing solver-seed policy.
        summary = hp.run_direct(cfg, ctx)
        result.update(
            run_state="COMPLETED", solver_status=summary["status_name"],
            solver_runtime_sec=summary.get("runtime_sec"),
            **{key: summary.get(key) for key in (
                "sol_count", "obj_val", "obj_bound", "mip_gap_percent",
                "solver_runtime_accumulated_sec", "model_build_sec", "total_wall_sec",
            )}, **classify_result(summary),
        )
    except BaseException as exc:
        result.update(run_state="ERROR", solver_status="ERROR", conclusion="执行失败",
                      error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        traceback.print_exc()
    result["task_wall_sec"] = time.perf_counter() - started
    write_json(result_dir / "direct_result.json", result)
    return int(result["run_state"] == "ERROR")


def prepare(args):
    project = Path(args.project_dir).expanduser().resolve()
    for name in ("hsr_pipeline.py", "config_loader.py", "case_manager.py"):
        if not (project / name).is_file():
            raise FileNotFoundError(f"缺少 {project / name}；请将脚本放到原主实验目录，或指定 --project-dir")
    local_imports(project)
    from config_loader import load_config
    from case_manager import input_fingerprint, instance_root, positive_int

    cfg_path = resolve_path(args.config, project)
    common = load_config(cfg_path)
    workers = min(positive_int(args.workers, "workers"), len(CASE_KEYS))
    threads = positive_int(common.get("direct_threads") or common.get("threads") or 1, "direct_threads")
    source = resolve_path(args.instances_root, project) if args.instances_root else instance_root()
    output = resolve_path(args.output_root, project)
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("结果目录必须与已有案例目录分开")
    cases, missing = [], []
    for k, group in CASE_KEYS:
        case_id = f"K{k}_G{group:02d}"
        folder = source / case_id
        missing.extend(str(folder / name) for name in INPUT_FILES if not (folder / name).is_file())
        cases.append(dict(case_id=case_id, K=k, group=group, source_dir=str(folder)))
    if missing:
        raise FileNotFoundError(
            "未找到本次需要的已有案例。请用 --instances-root 指向完成主实验时使用的案例目录；"
            "本脚本不会重新生成案例。\n" + "\n".join(missing))
    # Validate all 20 cases before creating outputs or launching any solver.
    for case in cases:
        case["input_sha256"], _ = input_fingerprint(case["source_dir"], case["K"])
    common = dict(common, direct_time_limit=TIME_LIMIT, direct_threads=threads)
    print(f"Direct：K500 × 10 + K600 × 10 = 20；每个 TimeLimit={TIME_LIMIT}s", flush=True)
    print(f"项目：{project}\n配置：{cfg_path}\n已有案例：{source}\n结果根目录：{output}", flush=True)
    print(f"并发进程={workers}；每个 Direct 线程={threads}；重新求解，不续接旧搜索", flush=True)
    print(f"沿用参数：dep_window_fast={common.get('dep_window_fast')}；"
          f"dep_window_normal={common.get('dep_window_normal')}；direct_mipgap={common.get('direct_mipgap')}", flush=True)
    for case in cases:
        print(f"  {case['case_id']}  K={case['K']}  SHA256={case['input_sha256'][:16]}", flush=True)
    return project, cfg_path, common, source, output, cases, workers


def save_summary(run_dir, rows, state, started):
    from case_manager import write_csv_atomic
    ordered = sorted(rows.values(), key=lambda row: (row["K"], row["group"]))
    write_csv_atomic(run_dir / "direct_3600_summary.csv",
                     [{key: row.get(key) for key in SUMMARY_FIELDS} for row in ordered], SUMMARY_FIELDS)
    completed = [row for row in ordered if row["run_state"] == "COMPLETED"]
    totals = dict(
        state=state, requested_cases=len(rows), completed_cases=len(completed),
        feasible_cases=sum(row.get("validated_solution") == 1 for row in completed),
        time_limit_without_incumbent_cases=sum(
            row.get("solver_status") == "TIME_LIMIT" and row.get("solution_found") == 0 for row in completed),
        validation_failed_cases=sum(
            row.get("solution_found") == 1 and row.get("validated_solution") != 1 for row in completed),
        error_cases=sum(row["run_state"] == "ERROR" for row in ordered),
        batch_wall_time_sec=time.perf_counter() - started,
    )
    write_json(run_dir / "batch_summary.json", totals)
    return totals


def prepare_new_batch(args):
    project, cfg_path, common, source, output, cases, workers = prepare(args)
    if args.dry_run:
        print("20 组输入检查通过；未求解，未写入结果。", flush=True)
        return None
    from case_manager import input_fingerprint
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output / f"K500_K600_T3600_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(cfg_path, run_dir / "source_config.csv")
    if (source / "config_used.csv").is_file():
        shutil.copy2(source / "config_used.csv", run_dir / "generation_config_used.csv")
    write_json(run_dir / "run_metadata.json", dict(
        created_at=datetime.now().astimezone().isoformat(), project_dir=str(project),
        source_config=str(cfg_path), instances_root=str(source), workers=workers,
        direct_time_limit=TIME_LIMIT, threads_per_case=common["direct_threads"],
        python=sys.version, platform=platform.platform(), cases=cases,
        code_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (SCRIPT, project / "hsr_pipeline.py", project / "config_loader.py", project / "case_manager.py")},
    ))
    rows, pending = {}, []
    # Snapshot all cases before any solver starts; changing the original inputs
    # during a long batch must not change the later cases in this batch.
    for case in cases:
        case_id = case["case_id"]
        result_dir = run_dir / case_id
        snapshot = result_dir / "inputs"
        snapshot.mkdir(parents=True)
        for name in INPUT_FILES:
            shutil.copy2(Path(case["source_dir"]) / name, snapshot / name)
        copied_hash, _ = input_fingerprint(snapshot, case["K"])
        if copied_hash != case["input_sha256"]:
            raise ValueError(f"{case_id}: 复制期间输入发生变化；未启动任何求解")
        cfg = dict(common, data_dir=str(snapshot), output_root=str(run_dir),
                   instance_tag=case_id, train_limit=case["K"], fresh_full_run=False,
                   direct_write_solver_log=True, direct_log_filename="gurobi_direct.log")
        write_json(result_dir / "effective_config.json", cfg)
        row = dict(case, run_state="PENDING", conclusion="等待启动", time_limit_sec=TIME_LIMIT,
                   threads=cfg["direct_threads"], result_dir=str(result_dir), error="")
        rows[case_id] = row
        task = dict(case, project_dir=str(project), config=cfg, run_dir=str(run_dir),
                    result_dir=str(result_dir), summary_row=row)
        task_path = result_dir / "direct_task.json"
        write_json(task_path, task)
        pending.append((case_id, task_path))
    return project, run_dir, workers, rows, pending


def completed_result(task, result_dir):
    """Read terminal solver output, including a crash just before result.json."""
    summary_path = result_dir / "01_direct" / "direct_summary.csv"
    try:
        with summary_path.open(encoding="utf-8-sig", newline="") as stream:
            summaries = list(csv.DictReader(stream))
        if len(summaries) != 1:
            return None
        summary = summaries[0]
        if int(summary["target_k"]) != task["K"] or summary["experiment_tag"] != task["case_id"]:
            return None
        terminal = {"OPTIMAL", "TIME_LIMIT", "INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED",
                    "CUTOFF", "ITERATION_LIMIT", "NODE_LIMIT", "SOLUTION_LIMIT",
                    "SUBOPTIMAL", "USER_OBJ_LIMIT", "WORK_LIMIT", "MEM_LIMIT"}
        if summary["status_name"] not in terminal or not summary.get("finished_at"):
            return None
        classification = classify_result(summary)
        if classification["solution_found"]:
            if not classification["validated_solution"]:
                return None
            if not (result_dir / "01_direct" / "timetable_direct.csv").is_file():
                return None
        saved = {}
        try:
            saved = json.loads((result_dir / "direct_result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        if not isinstance(saved, dict):
            saved = {}
        if saved and (saved.get("case_id") != task["case_id"]
                      or saved.get("input_sha256") != task["input_sha256"]
                      or float(saved.get("time_limit_sec", 0)) != TIME_LIMIT):
            return None
        # A terminal, fully written direct_summary can survive even when the
        # parent was interrupted before it collected the worker result.
        row = dict(task["summary_row"], **{key: saved[key] for key in SUMMARY_FIELDS if key in saved})
        row.update(run_state="COMPLETED", solver_status=summary["status_name"], error="",
                   solver_runtime_sec=summary.get("runtime_sec"), **classification,
                   **{key: summary.get(key) for key in (
                       "sol_count", "obj_val", "obj_bound", "mip_gap_percent",
                       "solver_runtime_accumulated_sec", "model_build_sec", "total_wall_sec")})
        return row
    except (OSError, ValueError, KeyError, TypeError):
        return None


def resume_directory(args, project):
    if args.resume != "latest":
        folder = resolve_path(args.resume, project)
    else:
        output = resolve_path(args.output_root, project)
        candidates = sorted(p for p in output.glob("K500_K600_T3600_*")
                            if p.is_dir() and (p / "run_metadata.json").is_file())
        if not candidates:
            raise FileNotFoundError(f"没有可续跑批次：{output}；也可以 --resume 指定完整批次目录")
        folder = candidates[-1]
    if not (folder / "run_metadata.json").is_file():
        raise FileNotFoundError(f"不是本脚本的批次目录：{folder}")
    return folder


def prepare_resume(args, project, run_dir):
    local_imports(project)
    from case_manager import input_fingerprint, positive_int

    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("direct_time_limit") != TIME_LIMIT:
        raise ValueError("只能续跑原 TimeLimit=3600 的批次")
    # The updated runner is allowed; the model and configuration parser must
    # match the original batch. Use stored task settings, never today's config.
    code_hashes = {name.replace("\\", "/").rsplit("/", 1)[-1]: value
                   for name, value in metadata.get("code_sha256", {}).items()}
    for name in ("hsr_pipeline.py", "config_loader.py", "case_manager.py"):
        if hashlib.sha256((project / name).read_bytes()).hexdigest() != code_hashes.get(name):
            raise ValueError(f"{name} 与原批次版本不一致；请使用原主实验代码，仅替换本续跑脚本")
    expected = {f"K{k}_G{g:02d}" for k, g in CASE_KEYS}
    original_cases = metadata.get("cases", [])
    if len(original_cases) != 20 or {c["case_id"] for c in original_cases} != expected:
        raise ValueError("原批次清单不是 K500、K600 各10组")
    original_cases = {c["case_id"]: c for c in original_cases}
    workers = min(positive_int(args.workers, "workers"), 20)
    rows, tasks, pending = {}, {}, []
    for k, group in CASE_KEYS:
        case_id = f"K{k}_G{group:02d}"
        result_dir = run_dir / case_id
        if result_dir.resolve().parent != run_dir.resolve():
            raise ValueError(f"案例目录不能指向批次以外：{result_dir}")
        lock = result_dir / ".worker.lock"
        if lock.exists():
            with file_lock(lock):
                pass
        task_path = result_dir / "direct_task.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        cfg = task["config"]
        effective = json.loads((result_dir / "effective_config.json").read_text(encoding="utf-8"))
        if cfg != effective:
            raise ValueError(f"{case_id}: 已保存的参数不一致，停止续跑")
        if (task.get("case_id") != case_id or task.get("K") != k or task.get("group") != group
                or cfg.get("train_limit") != k or cfg.get("instance_tag") != case_id
                or cfg.get("direct_time_limit") != TIME_LIMIT):
            raise ValueError(f"{case_id}: 已保存的案例/时限不匹配")
        sha, _ = input_fingerprint(result_dir / "inputs", k)
        if sha != task["input_sha256"] or sha != original_cases[case_id]["input_sha256"]:
            raise ValueError(f"{case_id}: 输入快照已改变，停止续跑")
        done = completed_result(task, result_dir)
        row = done or dict(task["summary_row"], run_state="PENDING", conclusion="等待重新启动", error="")
        row.update(case_id=case_id, K=k, group=group, result_dir=str(result_dir),
                   input_sha256=sha, time_limit_sec=TIME_LIMIT, threads=cfg["direct_threads"])
        rows[case_id] = row
        if done:
            print(f"[SKIP] {case_id}: {row['conclusion']}", flush=True)
        else:
            print(f"[RESUME] {case_id}: 未完成或结果无效，重新跑3600秒", flush=True)
            cfg = dict(cfg, data_dir=str(result_dir / "inputs"), output_root=str(run_dir))
            tasks[case_id] = dict(task, project_dir=str(project), run_dir=str(run_dir),
                                  result_dir=str(result_dir), config=cfg, summary_row=row)
            pending.append((case_id, task_path))
    print(f"续跑批次：{run_dir}\n已完成并跳过 {20-len(pending)}；待运行 {len(pending)}；并发 {workers}", flush=True)
    if args.dry_run:
        print("续跑检查完成；未求解，未修改结果。", flush=True)
        return None
    # Preserve failed/interrupted attempts before writing fresh solver outputs.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for case_id, task_path in pending:
        result_dir = run_dir / case_id
        old_outputs = [result_dir / name for name in (
            "01_direct", "00_timing", "direct_result.json", "worker_console.log")
            if (result_dir / name).exists()]
        if old_outputs:
            archive = result_dir / "_previous_attempts" / stamp
            if not archive.resolve().is_relative_to(result_dir.resolve()):
                raise ValueError(f"旧结果备份路径不在案例目录内：{archive}")
            archive.mkdir(parents=True, exist_ok=False)
            for name in ("direct_task.json", "effective_config.json"):
                shutil.copy2(result_dir / name, archive / name)
            for path in old_outputs:
                if path.is_symlink() or path.resolve().parent != result_dir.resolve():
                    raise ValueError(f"旧结果路径不在案例目录内：{path}")
                path.rename(archive / path.name)
        write_json(task_path, tasks[case_id])
        write_json(result_dir / "effective_config.json", tasks[case_id]["config"])
    write_json(run_dir / f"resume_{stamp}.json", dict(
        resumed_at=datetime.now().astimezone().isoformat(), workers=workers,
        skipped_cases=[key for key, row in rows.items() if row["run_state"] == "COMPLETED"],
        restarted_cases=[key for key, _ in pending]))
    return project, run_dir, workers, rows, pending


def execute_batch(project, run_dir, workers, rows, pending):
    previous_wall = 0.0
    try:
        previous_wall = float(json.loads((run_dir / "batch_summary.json").read_text(encoding="utf-8"))
                              .get("batch_wall_time_sec", 0))
    except (OSError, ValueError, TypeError):
        pass
    started = time.perf_counter() - previous_wall
    active = {}
    save_summary(run_dir, rows, "RUNNING", started)
    print(f"本次结果：{run_dir}", flush=True)
    try:
        while pending or active:
            while pending and len(active) < workers:
                case_id, task_path = pending[0]
                console = (run_dir / case_id / "worker_console.log").open("w", encoding="utf-8")
                try:
                    process = subprocess.Popen(
                        [sys.executable, "-u", str(SCRIPT), "--worker", str(task_path)],
                        cwd=project, stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT,
                        env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1"),
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                except Exception as exc:
                    console.close()
                    rows[case_id].update(run_state="ERROR", conclusion="启动失败", error=str(exc))
                    write_json(run_dir / case_id / "direct_result.json", rows[case_id])
                else:
                    active[case_id] = (process, console)
                    rows[case_id].update(run_state="RUNNING", conclusion="正在求解")
                    print(f"[START] {case_id} PID={process.pid}", flush=True)
                pending.pop(0)
                save_summary(run_dir, rows, "RUNNING", started)
            for case_id, (process, console) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                console.close()
                del active[case_id]
                try:
                    result = json.loads((run_dir / case_id / "direct_result.json").read_text(encoding="utf-8"))
                    if result.get("case_id") != case_id or result.get("run_state") not in ("COMPLETED", "ERROR"):
                        raise ValueError("子进程结果案例或状态不正确")
                    if code != 0 and result["run_state"] != "ERROR":
                        raise ValueError(f"子进程异常退出：{code}")
                    rows[case_id].update(result)
                except Exception as exc:
                    rows[case_id].update(run_state="ERROR", conclusion="子进程异常，请查看 worker_console.log",
                                         error=f"exit={code}; {exc}")
                print(f"[DONE] {case_id}: {rows[case_id]['conclusion']} "
                      f"solver_time={rows[case_id].get('solver_runtime_sec')}s", flush=True)
                save_summary(run_dir, rows, "RUNNING", started)
            if active:
                time.sleep(0.3)
    except BaseException:
        for case_id, (process, console) in active.items():
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            console.close()
            rows[case_id].update(run_state="INTERRUPTED", conclusion="本次运行被中断")
        for case_id, _ in pending:
            rows[case_id].update(run_state="NOT_STARTED", conclusion="中断时尚未启动")
        save_summary(run_dir, rows, "INTERRUPTED", started)
        print(f"已停止本次子进程；已有结果保留于：{run_dir}", flush=True)
        raise
    totals = save_summary(run_dir, rows, "FINISHED", started)
    print(f"完成 {totals['completed_cases']}/20；可行解 {totals['feasible_cases']}；"
          f"限时无解 {totals['time_limit_without_incumbent_cases']}；错误 {totals['error_cases']}", flush=True)
    print(f"汇总：{run_dir / 'direct_3600_summary.csv'}", flush=True)
    return int(totals["error_cases"] > 0 or totals["validation_failed_cases"] > 0)


def run_batch(args):
    if args.resume:
        project = Path(args.project_dir).expanduser().resolve()
        run_dir = resume_directory(args, project)
        lock = nullcontext() if args.dry_run else file_lock(run_dir / ".batch.lock")
        with lock:
            prepared = prepare_resume(args, project, run_dir)
            return 0 if prepared is None else execute_batch(*prepared)
    prepared = prepare_new_batch(args)
    if prepared is None:
        return 0
    with file_lock(prepared[1] / ".batch.lock"):
        return execute_batch(*prepared)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-dir", default=str(SCRIPT.parent), help="原主实验目录；默认脚本所在目录")
    parser.add_argument("--config", default="config.csv", help="相对于项目目录的原主实验配置")
    parser.add_argument("--instances-root", help="已有案例根目录；默认读取 generation.output_root")
    parser.add_argument("--output-root", default="outputs_direct_3600", help="独立结果根目录")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="并发案例数，默认1；20表示同时跑20组")
    parser.add_argument("--dry-run", action="store_true", help="仅检查全部20组数据和参数，不求解")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", metavar="BATCH_DIR",
                      help="续跑指定的旧批次目录；默认latest，兼容旧版输出")
    mode.add_argument("--new-batch", dest="resume", action="store_const", const=None,
                      help="明确新开一批，重新运行20组")
    parser.set_defaults(resume=DEFAULT_RESUME)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return run_worker(args.worker)
    try:
        return run_batch(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
