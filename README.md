# HSR corridor timetabling: scripts and inputs

Companion code and input instances for **Solving Large-Scale High-Speed Rail Corridor Timetabling via Primal–Dual Precedence Learning**.

This distribution contains 50 target instances on the Beijing–Shanghai corridor: 200, 300, 400, 500 and 600 trains, with 10 cases per size. Each case has 23 stations and 22 adjacent segments. It includes Direct MILP, Vote+LP, packing LP, IO+LP, TS, ALNS, target-instance generation and feasibility-checking code.

Only source code, input data, configurations and documentation are supplied. Precomputed solutions, learned precedences, restricted-instance solve results, logs, plots, tables and feasibility certificates are excluded. Running the programs creates new outputs locally. Original experiment inputs are preserved byte for byte.

中文使用说明：[中文使用说明](#chinese-guide)。

## Contents

| Path | Purpose |
| --- | --- |
| `instances/K200_G01/` … `instances/K600_G10/` | 50 target inputs; exactly `trains.csv`, `stations.csv`, `segments.csv` per case |
| `trainset_tools/data/` | 344 original reference trains and corridor station/segment inputs |
| `train_number_registry.csv` | Stable identifiers for source trains and their copies |
| `case_plan.csv` | Main-method case selection and restricted-subset designs; all 50 cases enabled |
| `config.csv` | Shared model and algorithm settings |
| `hsr_pipeline.py` | Direct, restricted, vote, packing, inverse and full-fixing implementations |
| `RUN_1_TO_6.py` | Main pipeline including Direct MILP |
| `RUN_2_TO_6.py` | Main pipeline without Direct MILP |
| `RUN_TS80_ALNS_FIXED.py` | TS and ALNS, all 50 cases and seeds 6–10 |
| `RUN_500_600_DIRECT_3600.py` | Optional separate Direct-only batch for the 20 large cases |
| `trainset_tools/` | Target generation, structural validation and operational feasibility checks |
| `verify_inputs.py` | Dependency-free file-integrity, input-structure and Python-syntax checks |
| `MANIFEST.csv` | Release file hashes, source-relative provenance and packaging changes |
| `LICENSE` | MIT terms for software, configuration and documentation; CC BY 4.0 terms for input data |

The numbered stage scripts can also be run individually. `90.aggregate_all_results.py` and `91.COLLECT_ALL_CASES.py` summarize results produced by a local run; no such result files are included.

## Environment

Use Python 3.10 or newer and install the dependencies:

```text
python -m pip install -r requirements.txt
```

Gurobi with a license supporting the full model sizes is required for MILP/LP solving and target generation. TS/ALNS use the supplied forward decoder; their embedded code is included in the single script. The packaged Windows path uses the Python decoder. The optional C++ accelerator is not supplied; the script falls back to Python when unavailable. The checked-in requirements specify compatibility floors, not an exact reconstruction of the historical environment.

Run the following commands from this directory after extraction. All runtime paths are relative to the package.

## Check inputs without running experiments

```text
python -B verify_inputs.py
python -B 00.CHECK_CONFIG_AND_DATA.py
python -B RUN_1_TO_6.py --dry-run
python -B RUN_TS80_ALNS_FIXED.py --check
python -B trainset_tools/generate_trainsets.py --check
```

The first command needs only Python's standard library. The other commands use the installed project dependencies. These checks do not optimize a timetable or create experiment outputs. `NOT_CHECKED` in a dry run means that no locally generated operational feasibility certificate exists yet; certificates are deliberately excluded from this distribution.

## Run the main experiments

```text
python trainset_tools/check_feasibility.py --selected-only
python RUN_1_TO_6.py
```

The first command performs operational feasibility solves and writes a new `instances/feasibility_results.csv`. It is required before the main pipeline. The second runs Direct, restricted-instance sampling/solving, Vote+LP, packing LP, structured inverse optimization and full fixing. Outputs are written under `outputs/Kxxx_Gyy/`.

For IO+LP and Vote+LP without Direct, replace the second command with:

```text
python RUN_2_TO_6.py
```

To select fewer main-method cases, set `enabled=FALSE` for other rows in `case_plan.csv` before both commands. Do not change the three input CSV files to select cases.

Each target uses 300 restricted instances: 60 local subsets of size 0.10K, 120 stratified subsets of size 0.20K and 120 coverage subsets of size 0.20K. The sampling seed is 32; the main packing-dual coefficient is 1. Restricted solves have a 600-second limit. The shared Direct limit is 1800 seconds, overridden to 3600 seconds for K=500 and K=600. Those overrides also apply to `1.solve_direct.py` and `RUN_1_TO_6.py`. Other main solve-stage limits remain 1800 seconds.

The optional `RUN_500_600_DIRECT_3600.py` starts or resumes its own Direct-only output batch. Use `python RUN_500_600_DIRECT_3600.py --new-batch` for its first run, or `--new-batch --dry-run` to check the inputs. Subsequent invocations can use `--resume latest`. The original default attempts to resume and therefore needs an existing batch. This script is not necessary after a complete `RUN_1_TO_6.py` run.

## Run TS and ALNS

```text
python RUN_TS80_ALNS_FIXED.py
```

This runs 50 cases × 2 methods × 5 seeds = 500 searches and writes `TS_ALNS/`. Seeds are 6, 7, 8, 9 and 10. The limit is 1800 seconds for K=200–400 and 3600 seconds for K=500–600. The original stagnation limits are TS 100 and ALNS 80 iterations after a feasible solution is found.

TS/ALNS case selection is independent of `case_plan.csv`: edit `K_VALUES` near the top of the script to choose train counts. `--methods TS` or `--methods ALNS` selects one method. `--seeds` selects from the configured seed group. Keep previous output in a different directory when changing experiment settings.

## Regenerate target inputs (optional)

The 50 ready-to-use target inputs are already included. To regenerate them from the original reference records:

```text
python trainset_tools/generate_trainsets.py
```

The release entry point uses the original `hsrgen` algorithms and generation settings, writes into a separate empty `regenerated_instances/` directory and preserves the distributed `instances/` and numbering registry. Use `--output-dir NEW_DIRECTORY` for another destination. `--sizes 200 --groups 1` generates the first group at that size. The default creates all 50 groups with the configured 600-second generation budget per group. Generation creates its own diagnostics and reports locally; these are not part of this distribution.

Generation validates construction constraints. Operational feasibility of the target timetable is a separate step. `trainset_tools/check_feasibility.py` checks the published `instances/` directory used by the main pipeline.

## Data and scope

See [Input format](#input-format) for field definitions and [Manuscript experiment map](#manuscript-experiment-map) for the manuscript-to-code mapping. The reference event times in `trains.csv` are optimization inputs, not optimized timetable outputs. The distributed instance files, rather than regenerated files from a different environment, define the released benchmark.

The package covers the experiments in the supplied manuscript. Unused seed-search launchers, alpha-sensitivity batches, partial-fixing experiments, historical archives and manuscript/figure files are omitted. The original optimization code is retained. Packaging changes are limited to relative output settings, size-dependent Direct budgets, enabling all five TS/ALNS sizes and a separate-destination generation entry point. `MANIFEST.csv` records individual changes and hashes; it excludes itself to avoid a circular checksum.

Use `verify_inputs.py` to check the original distribution before editing files. It will intentionally report a checksum mismatch after a tracked input, configuration or script is changed. Full experiments can require substantial runtime and memory; adjust the configured worker/thread counts to the machine when needed and record those changes for runtime comparisons.

## License

Software, configurations, documentation and release metadata are licensed under the MIT License. Input CSV files in `instances/` and `trainset_tools/data/`, together with `train_number_registry.csv`, are licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). See [LICENSE](LICENSE) for the scope and full terms. External dependencies retain their own licenses.

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

`trainset_tools/data/trains_344_original.csv` contains the 344 source reference trains used to construct the targets. The source data may omit `train_type`; `generation_config.csv` explicitly selects the historical `legacy_id` policy for these reference records. Every distributed target row includes its explicit `train_type`.

`train_number_registry.csv` maps `source_id` and `copy_index` to `train_id`. Copy index 0 retains the original train number. Positive indices map to distinct copy numbers; the registry is input to repeatable numbering.

`config.csv` uses `key,value,description`. `case_plan.csv` contains the selected targets and subset sizes/counts. `trainset_tools/generation_config.csv` uses `section,parameter,value,unit,description` and contains source paths, generation seed, statistical controls, operating range and generation solver settings.

Restricted instances are train subsets generated during stage 2. Their sampling rules, sizes, allocation and seed are supplied as inputs. Their solved timetables, precedence observations and lower bounds are runtime products and are not included.

## Manuscript experiment map

| Manuscript component | Entry point / implementation | Inputs |
| --- | --- | --- |
| Computational experiments: 50 targets | `case_plan.csv`, `instances/` | Five sizes, ten groups each |
| Direct MILP | `1.solve_direct.py`, `hsr_pipeline.py` | Per-case three CSVs; Direct budgets in `config.csv` |
| Restricted-instance evidence | `2.solve_restricted_instances.py` | Sampling seed 32; local/stratified/coverage design in the case plan |
| Vote+LP | `3.build_vote_fullfix.py` | Target inputs and newly computed stage-2 observations |
| Packing lower bound and dual train prices | `4.solve_packing_lp.py` | Newly computed restricted-instance bounds |
| Structured precedence learning | `5.solve_inverse_all_alpha.py` | Observations, packing duals; `alpha_list=1` |
| IO+LP full fixing | `6.solve_fullfix_all_alpha.py` | Target input and newly learned complete precedence |
| Runtime decomposition and aggregation | `RUN_1_TO_6.py`, `RUN_2_TO_6.py`, `90.aggregate_all_results.py`, `91.COLLECT_ALL_CASES.py` | Stage timing/results generated by the local run |
| Large-case Direct rerun | `RUN_500_600_DIRECT_3600.py` | K500/K600 inputs; 3600 seconds each |
| Appendix A.1 station and segment parameters | `stations.csv`, `segments.csv` in each case | Preserved experimental infrastructure inputs |
| Appendix A.2 target-instance construction | `trainset_tools/generate_trainsets.py`, `trainset_tools/hsrgen/` | 344 references, generation configuration and numbering registry |
| Appendices A.3–A.4 multi-scale pool design and train-subset sampling | `hsr_pipeline.py` stage 2 | 60 local (0.10K), 120 stratified (0.20K), 120 coverage (0.20K) |
| Appendix B structured-learning solution procedure | `hsr_pipeline.py` inverse stage | Structural aggregation and dynamic triangle constraints |
| Appendix C heuristic baselines (TS/ALNS) | `RUN_TS80_ALNS_FIXED.py` | Same target cases; seeds 6, 7, 8, 9, 10 |
| Appendix D instance-level runtime decomposition | `RUN_1_TO_6.py`, `RUN_2_TO_6.py`, `90.aggregate_all_results.py`, `91.COLLECT_ALL_CASES.py` | Stage timings and aggregate runtime records generated by the local run |
| Operational feasibility checks | `trainset_tools/check_feasibility.py` | Selected target inputs and shared model configuration |

Later pipeline stages depend on files produced by earlier stages. They do not require precomputed research outputs from the original workspace. The full pipeline regenerates these intermediate files.

The standalone TS/ALNS file contains all nine of its Python support modules as source strings; no external baseline package is required. Some historical module comments use earlier appendix labels. Use the current manuscript mapping above.

The 3600-second Direct entry point is retained for the separate large-case workflow. The main entry points also use these budgets through explicit per-size configuration, so a new full run needs no separate Direct rerun.

<a id="chinese-guide"></a>

## 中文使用说明

本文件夹对应论文 *Solving Large-Scale High-Speed Rail Corridor Timetabling via Primal–Dual Precedence Learning*，可作为 GitHub 仓库文件使用。

包含 200、300、400、500、600 列车五种规模，每种 10 组，共 50 组算例。每组仅保留 `trains.csv`、`stations.csv`、`segments.csv` 三份输入。另包含 344 列原始参考列车、生成配置、车次编号库，以及 Direct MILP、Vote+LP、IO+LP、TS、ALNS 的运行脚本。

包内不含求解结果、运行图、日志、图片、统计结果表、缓存或已有可行性检查结果。脚本运行后仍会正常生成新的输出文件；“不含输出”指此次交付和上传的文件范围。所有算例输入均与原实验文件逐字节一致。

### 最快开始

解压后，在本文件夹打开终端，使用 Python 3.10 或更高版本：

```text
python -m pip install -r requirements.txt
python -B verify_inputs.py
```

求解主实验和生成算例需要能处理完整模型规模的 Gurobi 许可证。只运行 `verify_inputs.py` 不需要安装任何第三方依赖。

检查主实验配置和 TS/ALNS 输入，不启动实验：

```text
python -B 00.CHECK_CONFIG_AND_DATA.py
python -B RUN_1_TO_6.py --dry-run
python -B RUN_TS80_ALNS_FIXED.py --check
```

由于不携带已有可行性检查结果，首次预览显示 `NOT_CHECKED` 是正常的。

### 运行论文实验

先独立检查可行性，再运行完整主实验：

```text
python trainset_tools/check_feasibility.py --selected-only
python RUN_1_TO_6.py
```

只运行所提方法及 Vote 对照、不运行 Direct 时，第二条改成 `python RUN_2_TO_6.py`。主实验由 `case_plan.csv` 的 `enabled` 列控制选中案例，默认全部启用。K=500/600 的 Direct 时限已接入 3600 秒，其余规模为 1800 秒。主实验输出写入 `outputs/`。

TS/ALNS 单独运行：

```text
python RUN_TS80_ALNS_FIXED.py
```

默认覆盖全部 50 组，每个方法使用 6、7、8、9、10 五个种子，共 500 次搜索，输出写入 `TS_ALNS/`。TS/ALNS 通过脚本顶部 `K_VALUES` 选择规模，不读取 `case_plan.csv` 的启用列。

只想重跑 20 个大规模 Direct，首次运行 `python RUN_500_600_DIRECT_3600.py --new-batch`；以后可加 `--resume latest` 续跑。原脚本默认查找已有批次，因此首次需明确使用 `--new-batch`。完整主实验已经按规模设置时限，不需要再重复运行此独立入口。

### 重新生成算例

已提供全部算例，无需先重新生成。需要从 344 列原始输入重新生成时运行：

```text
python trainset_tools/generate_trainsets.py
```

新生成内容放入独立的 `regenerated_instances/`，不覆盖随包输入。该目录必须为空；可用 `--output-dir` 指定新的目录。只检查生成器输入可加 `--check`，不会求解或写文件。

### 文件说明

- 本 `README.md`：中英文安装与运行说明；[输入字段解释](#input-format)；[文章实验与脚本对应关系](#manuscript-experiment-map)。
- `MANIFEST.csv`：文件来源、SHA-256 校验值及必要的打包调整记录。
- `.gitignore`：避免后续运行结果、日志及缓存被加入 Git。
- `LICENSE`：代码、配置和文档采用 MIT 许可证；`instances/`、`trainset_tools/data/` 内的输入 CSV 及 `train_number_registry.csv` 采用 CC BY 4.0 许可证。

未纳入当前文章的 alpha 敏感性、部分固定和历史试验目录。校验程序会检查原始交付文件的完整性；自行修改配置或脚本后出现哈希不一致属于预期行为。
