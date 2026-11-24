# HSR corridor timetabling: scripts and inputs

Companion code and inputs for **Solving Large-Scale High-Speed Rail Corridor Timetabling via Primal–Dual Precedence Learning**.

The package contains 50 cases: K = 200, 300, 400, 500, 600, with groups G01–G10 at each size. Each case includes 23 stations and 22 segments. Methods include Direct MILP, Vote+LP, packing LP, IO+LP, TS and ALNS. Only scripts, settings, input data and documentation are distributed; runtime results are created locally.

## Cite

Please cite the accompanying manuscript and the [code repository](https://github.com/shangpan99/hsr-timetabling-instances). The archival DOI should be added here when assigned.

## Directory layout

```text
hsr-timetabling-instances/
├── README.md
├── LICENSE
├── AUTHORS
├── requirements.txt
├── .gitignore
├── settings/
│   ├── config.csv
│   ├── case_plan.csv
│   └── generation_config.csv
├── data/
│   ├── instances/Kxxx_Gyy/{trains.csv,stations.csv,segments.csv}
│   ├── reference/{trains_344_original.csv,stations.csv,segments.csv}
│   └── train_number_registry.csv
├── src/
│   ├── hsr_pipeline.py
│   ├── case_manager.py
│   ├── config_loader.py
│   ├── resume_manager.py
│   ├── train_numbering.py
│   └── trainset_tools/
│       ├── __init__.py
│       ├── feasibility_core.py
│       └── hsrgen/
│           ├── __init__.py
│           ├── main.py
│           ├── data.py
│           ├── generate.py
│           ├── constraints.py
│           ├── solver.py
│           ├── validate.py
│           ├── unique.py
│           └── report.py
└── scripts/
    ├── run_main.py
    ├── run_method.py
    ├── 1_run_direct.py
    ├── run_ts_alns.py
    ├── generate_instances.py
    ├── collect_results.py
    └── stages/
        ├── check_inputs.py
        ├── check_feasibility.py
        ├── 2_solve_restricted.py
        ├── 3_build_vote_fullfix.py
        ├── 4_solve_packing_lp.py
        ├── 5_solve_inverse.py
        ├── 6_solve_fullfix.py
        └── aggregate_results.py
```

`src/` contains reusable implementation modules. Run the entry points under `scripts/`. Each entry point locates the repository from its own file path, so it also works when invoked by absolute path from another working directory.

## Installation

Use Python 3.10 or newer. From the repository root:

```text
python -m pip install -r requirements.txt
```

MILP/LP experiments, feasibility checks and case generation require Gurobi and a license supporting the model sizes. TS/ALNS use their bundled Python decoder. Their nine supporting Python modules are embedded in `scripts/run_ts_alns.py`; no external baseline source directory or C++ accelerator is required. Dependency versions in `requirements.txt` are compatibility floors.

## Settings and paths

| File / setting | Purpose | Default |
| --- | --- | --- |
| `settings/config.csv` | Model, solver and sampling parameters | `key,value,description` format |
| `settings/case_plan.csv` | Enabled cases and restricted-subset designs | All 50 cases enabled |
| `settings/generation_config.csv` | Reference inputs and generation parameters | `section,parameter,value,unit,description` format |
| `config.csv`: `output_root` | Root for all runtime results, relative to the repository root | `results` |
| `generation_config.csv`: `generation.output_root` | Shared destination for generated cases and input directory for all experiments, relative to `settings/` | `../data/instances` |
| `generation_config.csv`: `input.trains`, `input.stations`, `input.segments` | Reference data, relative to `settings/` | `../data/reference/...` |

The generation and experiment entry points use the same case directory. There is no separate regenerated-case directory to switch to. Absolute paths are also supported. Running from another working directory does not change the interpretation of these settings. The TS/ALNS `--out-dir` option is relative to the selected repository root.

Direct budgets are independent, in seconds per case:

```csv
direct_time_limit_K200_400,1800,K200/K300/K400每个案例的Direct求解时限秒
direct_time_limit_K500_600,3600,K500/K600每个案例的Direct求解时限秒
```

Both values must be positive finite numbers. `run_main.py` and `1_run_direct.py` use the same rules. Their `--dry-run` output displays the effective Direct limit. Changing one group's budget does not change the other group's budget or its case signature. These settings do not change the limits of other methods.

Each target uses 300 restricted instances: 60 local subsets of size 0.10K, 120 stratified subsets of size 0.20K and 120 coverage subsets of size 0.20K. Sampling uses seed 32. Restricted-instance solves have a 600-second limit; the packing, inverse and full-fixing limits are 1800 seconds. The released main-method configuration uses `alpha_list=1` and `main_alpha=1`.

## Check inputs and preview execution

These commands read inputs and settings without optimizing timetables or writing results:

```text
python -B scripts/stages/check_inputs.py
python -B scripts/run_main.py --dry-run
python -B scripts/1_run_direct.py --dry-run
python -B scripts/stages/check_feasibility.py --selected-only --dry-run
python -B scripts/run_ts_alns.py --check
python -B scripts/generate_instances.py --check
```

`NOT_CHECKED` is expected before the first local feasibility check. No precomputed feasibility certificates are distributed. The experiment entry points automatically run missing or stale checks during a real run; `--dry-run` only previews this decision and never calls the feasibility solver.

## Run experiments

From the repository root, run the full pipeline directly:

```text
python scripts/run_main.py
```

The full pipeline runs Direct, restricted-instance sampling and solves, Vote precedence construction, packing LP, inverse optimization, full-fixing LPs and aggregation. Both Vote+LP and IO+LP are solved in the full-fixing stage; the Vote stage itself only constructs the precedence assignment. Each executed aggregation also refreshes cross-case tables. TS/ALNS run separately. To select fewer cases, edit the `enabled` column of `settings/case_plan.csv` before checking feasibility and solving.

Alternative entry points, with the same automatic checks:

```text
python scripts/run_method.py
python scripts/1_run_direct.py
```

`run_method.py` runs stages 2–6 and aggregation without Direct. With the distributed `after_direct_require_direct_result=FALSE`, it can run independently; the Direct fields in its aggregate tables are empty until Direct has been run. `1_run_direct.py` runs only Direct for enabled cases, including K500/K600. Both share the same inputs and output layout as `run_main.py`.

With the default `require_feasibility_pass=TRUE`, these three entries and the individual restricted/Vote/packing/inverse/full-fix entries validate inputs and configuration, then check operational feasibility before starting a case. A saved `FEASIBLE` record is reused only when its input and model fingerprints match. Missing, stale, unsuccessful or interrupted records trigger a new check. If that check returns `INFEASIBLE`, `UNKNOWN` or `ERROR`, no experiment stage starts for that case; the status is saved and `continue_on_error` controls whether to proceed to other selected cases. This check is outside the Direct and other experiment solver time budgets.

The two standalone check scripts are optional: `check_inputs.py` performs input/configuration checks only; `check_feasibility.py --selected-only` explicitly rechecks the enabled cases even if a prior passing record exists. Both automatic and manual feasibility checks share the same implementation and `results/feasibility/feasibility_results.csv`. Unchanged passing records are not rewritten. Altering only Direct's time limit does not invalidate the feasibility certificate, although the experiment-result compatibility rules below still apply. Aggregation/collection do not perform feasibility solves. TS/ALNS retain their own input and result validators and do not require this Gurobi feasibility certificate.

The main and method entry points can resume completed stages when inputs and effective settings match. If the inputs or parameters changed, select a new `output_root`; alternatively, `fresh_full_run=TRUE` resets selected case results when running the full main pipeline. It does not reset results for a Direct-only run.

TS and ALNS are run separately:

```text
python scripts/run_ts_alns.py
```

The default is 50 cases × 2 methods × 5 seeds = 500 searches. Seeds are 6–10. Search limits are 1800 seconds for K200–K400 and 3600 seconds for K500/K600; stopping patience is 100 non-improving feasible iterations for TS and 80 for ALNS. Change `K_VALUES` at the top of the script to select sizes; this selection is independent of `case_plan.csv`. `--methods TS` or `--methods ALNS` selects a method, and `--seeds` selects from the configured seed group. `--diagnostic-seconds` runs a short diagnostic in a separate diagnostic results folder.

## Generate and overwrite cases

The paper's 50 inputs are included. To regenerate and replace them:

```text
python scripts/generate_instances.py
```

To replace only K200_G01:

```text
python scripts/generate_instances.py --sizes 200 --groups 1
```

`--groups N` means groups G01 through GN for the selected sizes. Generation uses the same `data/instances/` directory that all experiment scripts read. Once all requested cases have been generated and checked, their `trains.csv`, `stations.csv` and `segments.csv` are replaced, and the numbering registry is updated. Unselected cases remain in place. If generation is incomplete, existing case inputs are retained. Generation logs, source snapshots and diagnostics are saved under `results/generation/<run-id>/`.

After regeneration, experiment entries detect the changed inputs and automatically recheck operational feasibility before solving. Input fingerprints also detect incompatible experiment results: select a new result root or use the documented fresh-main reset before reusing an existing case-result directory. Generation checks construction constraints; it does not replace the operational feasibility solve.

## Output locations

The `results/` directory is created at runtime and is not included in the distributed folder:

```text
results/
├── Kxxx_Gyy/
│   ├── 00_parameters/
│   ├── 00_timing/
│   ├── 01_direct/
│   ├── 02_samples/
│   ├── 03_vote/
│   ├── 04_packing/
│   ├── 05_inverse/
│   ├── 06_fullfix/
│   └── 90_aggregate/
├── _ALL_CASES/                    # Cross-case main-method tables
├── TS_ALNS/                      # Baseline runs and comparisons
├── feasibility/feasibility_results.csv
├── generation/<run-id>/          # Generation logs, snapshots and diagnostics
└── run_status.csv
```

Run `python scripts/collect_results.py` to collect existing single-case tables across cases, or use `--dry-run` to preview its destination without writing files. This scans the result folders, independently of the currently enabled rows in `case_plan.csv`; it does not rerun solvers or refresh stale single-case tables. `python scripts/run_ts_alns.py --summary` updates baseline summaries.

In the table below, `C` means `results/Kxxx_Gyy/` for each enabled case. Paths assume the distributed settings.

| Entry point under `scripts/` | Main output | Repeated execution |
| --- | --- | --- |
| `run_main.py` | `C/01_direct/` through `C/06_fullfix/`, then `C/90_aggregate/` and `results/_ALL_CASES/` | Reuses complete, compatible stages; resumes missing or invalid stages |
| `run_method.py` | `C/02_samples/` through `C/06_fullfix/`, then `C/90_aggregate/` and `results/_ALL_CASES/` | Same restart rules; preserves Direct results |
| `1_run_direct.py` | `C/01_direct/` | Explicitly reruns Direct and updates its files |
| `run_ts_alns.py` | `results/TS_ALNS/Kxxx_Gyy/`; summary CSVs in `results/TS_ALNS/` | Skips verified completed case/method/seed runs; resumes unfinished runs |
| `generate_instances.py` | Replaces selected inputs in `data/instances/`; reports in `results/generation/<run-id>/` | Regenerates selected cases; creates a new reports folder |
| `collect_results.py` | `results/_ALL_CASES/` | Rewrites cross-case summary tables |
| `stages/check_inputs.py` | Console only | Reads and checks inputs/configuration |
| `stages/check_feasibility.py` | `results/feasibility/feasibility_results.csv` | Rechecks selected cases and replaces their rows; keeps other rows |
| `stages/2_solve_restricted.py` | `C/02_samples/` | Reuses eligible subsets with `sample_resume=TRUE`; solves missing subsets and refreshes stage files |
| `stages/3_build_vote_fullfix.py` | `C/03_vote/` | Rebuilds Vote precedence; does not solve the Vote+LP timetable |
| `stages/4_solve_packing_lp.py` | `C/04_packing/` | Reruns packing LP |
| `stages/5_solve_inverse.py` | `C/05_inverse/alpha_1/` and its parent summaries | Reruns inverse optimization for the configured alpha values |
| `stages/6_solve_fullfix.py` | `C/06_fullfix/VOTE/`, `C/06_fullfix/IO_alpha_1/` and their parent summary | Reruns the Vote+LP and IO+LP timetable solves |
| `stages/aggregate_results.py` | `C/90_aggregate/` and `results/_ALL_CASES/` | Rewrites single-case and cross-case tables from the currently saved stage results |

Solving and per-case aggregation entries also update `C/00_parameters/`, `C/case_signature.json` and `C/resume_state.json`. Executed computational and aggregation stages append timing records to `C/00_timing/stage_timing_log.csv`. `results/run_status.csv` records the latest batch of the main, method, Direct or stage entry; it is replaced on a subsequent batch, rather than storing all run history. A completely skipped case retains its stage files. Timetable CSVs are produced only when a usable solution is available; inspect the corresponding summary's solver status and `has_solution` field.

### Shared results and reruns

The integrated and individual entry points write to the same case and stage directories. There are no separate `run_main` and `run_direct` result copies. With unchanged inputs/settings and `fresh_full_run=FALSE`:

- **Direct first, then main:** main reuses the completed Direct result and runs the remaining stages.
- **Main first, then Direct:** Direct is explicitly solved again in the same `01_direct/` directory. The method result files in stages 2–6 are retained. Its single-case aggregate is marked for refresh; a subsequent main run skips Direct and stages 2–6 and updates aggregation.
- **Main again, without changes:** all verified completed stages are skipped. A normal time-limit result can count as complete even if it is not proven optimal.
- **Method after main:** Direct is retained and verified completed method stages are skipped.
- **An individual method stage is rerun:** its shared checkpoint is updated; dependent downstream stages are marked pending. Continue with main or method to recompute those dependent stages. An interruption is recorded before old stage results can be reused.

After a standalone Direct rerun, `python scripts/stages/aggregate_results.py` also refreshes both individual and cross-case tables without running method solvers. Direct by itself does not update these tables. Main's aggregation calls the cross-case collection function automatically; a separate `collect_results.py` run is only needed to rebuild cross-case tables from the existing single-case tables (for example, if all main stages were skipped and the cross-case tables need to be recreated).

Changing inputs or experiment parameters (including an effective Direct time limit) blocks reuse of incompatible case results. Choose a new `output_root`, or explicitly set `fresh_full_run=TRUE` and run main to clear and rebuild the selected **entire case folders**. The reset does not clear the separate `TS_ALNS/`, `feasibility/` or `generation/` folders, and does not apply to Direct-only or method-only entries. Restore `fresh_full_run=FALSE` afterward if subsequent main runs should resume. Do not run overlapping entries concurrently for the same cases and output directory.

## License

Code, settings and documentation use the MIT License. Input CSVs under `data/instances/` and `data/reference/`, and `data/train_number_registry.csv`, use CC BY 4.0. See `LICENSE` for the full terms. Third-party dependencies retain their own licenses.

## Input format

CSV files are UTF-8; readers accept an optional UTF-8 BOM. Keep the existing headers. Times are numeric minutes from midnight; for example, 373 means 06:13. Blank event times mean an absent endpoint event and must not be replaced by zero. Reference input events lie between 370 and 1429 minutes. This construction range is not a terminal upper bound on optimized timetable events.

### Per-case inputs

`instances/Kxxx_Gyy/` identifies train count `xxx` and group `yy`. Each directory contains three files.

#### trains.csv

One row represents one train's visit to one station; it is not one row per train.

| Column | Meaning |
| --- | --- |
| `车次` | Train identifier, unique within a case |
| `方向` | Direction; the released corridor trains operate in one direction |
| `车站` | Station name, matching `stations.csv` |
| `到达时间` | Reference arrival time in minutes; may be blank at the origin |
| `出发时间` | Reference departure time in minutes; may be blank at the destination |
| `是否停站` | `TRUE`/`FALSE` stopping flag |
| `站序` | Station order along the train route |
| `所属路局` | Source railway-bureau metadata |
| `train_type` | Explicit speed class, 300 or 350; copied train IDs must not be used to infer speed |

The reference times define input information and departure windows. They are not saved optimized timetable solutions.

#### stations.csv

| Column | Meaning |
| --- | --- |
| `station` | Station identifier/name |
| `h_arr` | Minimum arrival headway in minutes |
| `h_dep` | Minimum departure headway in minutes |
| `dwell_min` | Minimum station dwell in minutes |
| `dwell_max` | Retained source field; not enforced as an upper bound by the main model |

#### segments.csv

Each row is one directed adjacent segment. `from` and `to` identify its endpoints. For each speed class `v` in {300, 350}, `tau_v` gives basic running time, `eta_start_v` the starting supplement and `eta_end_v` the ending supplement, all in minutes. The model applies supplements according to stopping states.

### Reference inputs and configurations

`data/reference/trains_344_original.csv` contains the 344 source reference trains used to construct the targets. The source data may omit `train_type`; `generation_config.csv` explicitly selects the historical `legacy_id` policy for these reference records. Every distributed target row includes its explicit `train_type`.

`data/train_number_registry.csv` maps `source_id` and `copy_index` to `train_id`. Copy index 0 retains the original train number. Positive indices map to distinct copy numbers; the registry is input to repeatable numbering.

`settings/config.csv` uses `key,value,description`. `settings/case_plan.csv` contains the selected targets and subset sizes/counts. `settings/generation_config.csv` uses `section,parameter,value,unit,description` and contains source paths, generation seed, statistical controls, operating range and generation solver settings.

Restricted instances are train subsets generated during stage 2. Their sampling rules, sizes, allocation and seed are supplied as inputs. Their solved timetables, precedence observations and lower bounds are runtime products and are not included.

## 中文使用说明

三个配置文件集中放在 `settings/`，算法和公共模块放在 `src/`，运行入口放在 `scripts/`。`AUTHORS` 按当前要求保留为空文件。

所有命令均可在项目根目录执行；也可以从其他工作目录使用脚本的绝对路径运行。

```text
python -m pip install -r requirements.txt
python scripts/run_main.py
```

只运行方法流程用 `python scripts/run_method.py`；只运行 Direct 用 `python scripts/1_run_direct.py`；TS/ALNS 用 `python scripts/run_ts_alns.py`。主实验与 Direct 通过 `settings/case_plan.csv` 选择案例，TS/ALNS 通过脚本顶部的 `K_VALUES` 选择规模。

`settings/config.csv` 中的 `direct_time_limit_K200_400` 默认 1800 秒，`direct_time_limit_K500_600` 默认 3600 秒，两组分别修改。主流程和 Direct 单独入口共用规则。

**运行 `python scripts/generate_instances.py` 会重新生成并覆盖 `data/instances/` 中对应案例的三份输入 CSV。** `--sizes 200 --groups 1` 只替换 K200_G01。全部请求案例生成和校验成功后才覆盖，未完成时保留现有案例。生成后的案例就是主脚本和 benchmark 下一次读取的输入，不需要手动搬移或切换目录。

生成位置与实验读取位置统一取自 `settings/generation_config.csv` 的 `generation.output_root`，默认 `../data/instances`，相对 `settings/` 解析。其 `input.*` 路径也相对 `settings/` 解析。`settings/config.csv` 的 `output_root` 默认 `results`，相对项目根目录解析。

主实验输出位于 `results/Kxxx_Gyy/`，TS/ALNS 位于 `results/TS_ALNS/`，可行性记录位于 `results/feasibility/`，生成日志和统计位于 `results/generation/<run-id>/`。发布文件夹不附带这些结果。只预览路径和配置时，主入口使用 `--dry-run`，生成器和 TS/ALNS 使用 `--check`，均不求解或写入结果。

重新生成输入后，求解入口会根据输入指纹自动重新检查可行性，不需要手动先运行检查脚本。旧实验结果是否可复用仍由输入与参数指纹检查；若输入或实验参数已改变，可修改 `output_root` 使用新的结果目录。目录中每个案例的 `trains.csv` 是参考时刻等模型输入，不是求解后的运行图。

**正常运行：**直接运行 `run_main.py`，它会先核对输入和配置，再复用或自动生成可行性检查记录，通过后执行实验。`run_method.py`、`1_run_direct.py` 及原本要求可行性通过的五个分步入口共用此逻辑。`check_inputs.py` 和 `check_feasibility.py` 保留为可选的独立检查入口。输入、模型参数和通过记录不变时，不会重复求解可行性模型；没有记录、记录失效或上次未通过时才重新检查。检查未通过的案例不进入实验阶段，可行性检查时间不计入 Direct 的 1800/3600 秒预算。

TS/ALNS 独立运行，保留自身的数据与解校验，不依赖上述 Gurobi 可行性记录。使用已附带的 50 组输入时，不必运行生成脚本。主流程按 Direct → 子实例求解 → Vote 定序 → Packing LP → 逆优化 → Full-fix LP → 单案例汇总执行，每次执行单案例汇总还会刷新跨案例总表。`collect_results.py` 用于单独重建跨案例总表。Vote+LP 和 IO+LP 的最终求解都在 Full-fix 阶段，`3_build_vote_fullfix.py` 只构造 Vote 定序。

**共用结果与覆盖：**`run_main.py` 包含 `1_run_direct.py` 对应的计算阶段，两者共用 `results/Kxxx_Gyy/01_direct/`。主入口默认续跑、跳过有效完成阶段；单独运行 Direct 会重新求解并更新该目录的结果文件，保留阶段 2–6 的文件。随后运行主入口，会复用刚完成的 Direct，仅刷新受影响的汇总。也可以直接运行 `stages/aggregate_results.py` 更新两级汇总。单独重跑其他阶段后，主入口会补跑其受影响的后续阶段。

`run_method.py` 默认可以独立运行，覆盖的是主流程中的阶段 2–6 和单案例汇总；它保留已有 Direct。此处的“包含阶段”不等于“每次全部覆盖”：是否重算取决于入口和续跑状态。只有主入口在 `fresh_full_run=TRUE` 时才清空所选案例的整个结果目录；修改 Direct 时限等参数后，不能直接混用旧配置结果。各入口、主要输出及重复运行规则见上面的输出位置表。
