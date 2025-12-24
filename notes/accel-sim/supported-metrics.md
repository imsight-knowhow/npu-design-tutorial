# Accel-Sim: Supported Outputs and Metrics

This note summarizes what Accel-Sim can produce *after simulation* (raw run artifacts + the default “supported” metrics that upstream tools extract for CSV/correlation).

Upstream reference: `extern/accel-sim-framework` (shallow clone).

## Available Outputs

### Run artifacts (files)

| Output | Produced by | Typical location | Notes |
|---|---|---|---|
| Job stdout capture (`*.o<jobid>`) | `run_simulations.py` job manager (Torque/Slurm/local) | `sim_run_<cuda>/.../<app+args>/<config>/` | Primary source for most metrics; parsed by `get_stats.py`. |
| Job stderr capture (`*.e<jobid>`) | `run_simulations.py` job manager | `sim_run_<cuda>/.../<app+args>/<config>/` | Useful for crashes, missing files, misconfig. |
| Interactive rerun script (`justrun.sh`) | `run_simulations.py` | `sim_run_<cuda>/.../<app+args>/<config>/justrun.sh` | Replays the exact command used for that run directory. |
| Tee’d simulator log (`gpgpu-sim-out_<timestamp>.txt`) | `justrun.sh` | `sim_run_<cuda>/.../<app+args>/<config>/` | Convenience log if you rerun interactively; still contains the same “printed stats” lines. |
| Resolved config (`gpgpusim.config`) | `run_simulations.py` | `sim_run_<cuda>/.../<app+args>/<config>/gpgpusim.config` | Includes base config + benchmark-specific options + (for SASS mode) `trace.config`. |
| AccelWattch power report (`accelwattch_power_report.log`) | AccelWattch (optional) | `sim_run_<cuda>/.../<app+args>/<config>/` | Only when running an AccelWattch config; per-kernel power estimations. |
| Aggregated CSV (user-generated) | `util/job_launching/get_stats.py` | wherever you redirect stdout | Common: `stats.csv`, `per-kernel-instance.csv`, `per-app-for-correlation.csv`. |
| Correlation plots + summaries (user-generated) | `util/plotting/plot-correlation.py` | `extern/accel-sim-framework/util/plotting/correl-html/` | Interactive HTML plots, CSVs, and text summaries. |
| Bar-chart plots (user-generated) | `util/plotting/plot-get-stats.py` | `extern/accel-sim-framework/util/plotting/htmls/` | Convenience plots from `get_stats.py` CSV output. |

### Default parsed metrics (via `get_stats.py`)

`extern/accel-sim-framework/util/job_launching/get_stats.py` extracts metrics from job stdout (`*.o<jobid>`) using regexes defined in `extern/accel-sim-framework/util/job_launching/stats/example_stats.yml`.

Notes:
- **Granularity** depends on flags:
  - default: “`final_kernel`” (end-of-run snapshot / whole app).
  - `-k`: per **kernel name** (diffing aggregate counters between `kernel_name = ...` boundaries).
  - `-K`: per **kernel instance** (kernel name suffixed with `--<instance_id>`).
- `get_stats.py` also prints `Accel-Sim-build` and `GPGPU-Sim-build` (parsed from the first ~100 lines), even though they are not in the YAML file.

| Metric | Class | Kernel-diffable? | Units | What it represents |
|---|---:|---:|---:|---|
| `Accel-Sim-build` | meta | n/a | string | Accel-Sim build identifier printed in stdout (`Accel-Sim ... [build ...]`). |
| `GPGPU-Sim-build` | meta | n/a | string | GPGPU-Sim build identifier printed in stdout (`GPGPU-Sim ... [build ...]`). |
| `gpu_tot_sim_insn` | aggregate | yes | instructions | Total simulated (dynamic) instructions. |
| `gpgpu_simulation_time` | aggregate | yes | seconds | Wall-clock simulation time (parsed from `(... sec)` in the printed line). |
| `gpu_tot_sim_cycle` | aggregate | yes | cycles | Total simulated cycles. |
| `L2_cache_stats_breakdown[GLOBAL_ACC_R][HIT]` | aggregate | yes | accesses | L2 global-read hit count. |
| `L2_cache_stats_breakdown[GLOBAL_ACC_R][TOTAL_ACCESS]` | aggregate | yes | accesses | L2 global-read total accesses. |
| `L2_cache_stats_breakdown[GLOBAL_ACC_W][HIT]` | aggregate | yes | accesses | L2 global-write hit count. |
| `L2_cache_stats_breakdown[GLOBAL_ACC_W][TOTAL_ACCESS]` | aggregate | yes | accesses | L2 global-write total accesses. |
| `Total_core_cache_stats_breakdown[GLOBAL_ACC_R][TOTAL_ACCESS]` | aggregate | yes | accesses | Core (L1D / “core cache”) global-read total accesses. |
| `Total_core_cache_stats_breakdown[GLOBAL_ACC_R][HIT]` | aggregate | yes | accesses | Core (L1D / “core cache”) global-read hit count. |
| `Total_core_cache_stats_breakdown[GLOBAL_ACC_W][HIT]` | aggregate | yes | accesses | Core (L1D / “core cache”) global-write hit count. |
| `Total_core_cache_stats_breakdown[GLOBAL_ACC_W][TOTAL_ACCESS]` | aggregate | yes | accesses | Core (L1D / “core cache”) global-write total accesses. |
| `Total_core_cache_stats_breakdown[GLOBAL_ACC_R][MSHR_HIT]` | aggregate | yes | accesses | Core-cache global-read MSHR-hit count. |
| `gpgpu_n_tot_w_icount` | aggregate | yes | warp-inst | Total warp instruction count. |
| `total dram reads` | aggregate | yes | transactions | Total DRAM read transactions. |
| `total dram writes` | aggregate | yes | transactions | Total DRAM write transactions. |
| `kernel_launch_uid` | aggregate | yes* | id | Kernel launch UID as printed by the simulator. (*Diffing is mechanically supported but interpret with care.) |
| `gpgpu_n_shmem_bkconflict` | aggregate | yes | conflicts | Shared-memory bank conflict count. |
| `gpgpu_n_l1cache_bkconflict` | aggregate | yes | conflicts | L1 cache bank conflict count. |
| `gpu_ipc` | absolute | no | inst/cycle | Snapshot IPC printed by the simulator (per-kernel reset, not diffed). |
| `gpu_occupancy` | absolute | no | % | Snapshot occupancy percentage. |
| `L2_BW` | absolute | no | GB/s | Snapshot L2 bandwidth. |
| `gpgpu_simulation_rate (inst/sec)` | rate | no | inst/sec | Simulation throughput (instructions per second). |
| `gpgpu_simulation_rate (cycle/sec)` | rate | no | cycle/sec | Simulation throughput (cycles per second). |
| `gpgpu_silicon_slowdown` | rate | no | x | Simulated slowdown vs “silicon” (hardware). |
| `gpu_tot_ipc` | rate | no | inst/cycle | Total IPC (as printed; treated as a snapshot/rate by upstream tooling). |

## Extending the metric set

To extract more outputs from the simulator stdout, add new regex entries to a YAML file in the same format as `extern/accel-sim-framework/util/job_launching/stats/example_stats.yml`, then point `get_stats.py` to it via `-s /path/to/your_stats.yml`.
