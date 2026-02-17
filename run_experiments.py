# run_experiments.py
"""
Master Runner Script for SAPEX-F Simulation Experiments

Implements the nested experiment loop:
  1. Select Comparison Algorithm
  2. Select Stress Scenario
  3. Select Parameter Set
  4. Collect experiment data, store results

Usage:
    python run_experiments.py                              # Run default experiments
    python run_experiments.py --algorithms sapex random     # Specific algorithms
    python run_experiments.py --scenarios thundering_herd   # Specific scenario
    python run_experiments.py --dry-run                     # Preview without running
    python run_experiments.py --list                        # List all options
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from result_logger import ResultLogger


# ============================================================================
# CONFIGURATION
# ============================================================================

# All algorithms to compare
ALGORITHMS = [
    "lowest_latency",
    "lowest_hop_count",
    "random",
    "round_robin",
    "sapex",
]

# Topology files
TOPOLOGIES = {
    "small": "topology.json",
    "sciera_large": "topologies/sciera_large.json",
}

# Stress scenarios (traffic files)
SCENARIOS = {
    "thundering_herd": "scenarios/thundering_herd.json",
    "path_failure_recovery": "scenarios/path_failure_recovery.json",
    "shared_bottleneck": "scenarios/shared_bottleneck.json",
}

# ---- Parameter sets ----
# Network related
NUM_PACKETS_OPTIONS = [1000, 2000, 5000, 10000, 20000, 50000]
PACKET_SIZE_BYTES = 1500  # Fixed per spec

# Algorithm related
T_ROUND_OPTIONS_MS = [1000, 2000, 5000, 10000]    # Allocation epoch (jitter)
COOLDOWN_OPTIONS_MS = [2000, 5000, 10000]          # Expiry times for cooldown
LAMBDA_DIV_OPTIONS = [0.3, 0.5, 0.7, 1.0]         # Weight for diversity reward
POINT_BUDGET_OPTIONS = [100, 250, 500]             # Application point budgets


# ---- Predefined experiment sets ----
EXPERIMENT_SETS = {
    "quick": {
        "description": "Quick smoke test (1 algorithm × 1 scenario × minimal params)",
        "algorithms": ["sapex"],
        "topologies": ["small"],
        "scenarios": ["thundering_herd"],
        "num_packets": [1000],
        "t_round_ms": [2000],
        "cooldown_ms": [5000],
        "lambda_div": [0.5],
        "point_budget": [100],
    },
    "algorithm_comparison": {
        "description": "Compare all algorithms with fixed parameters across all scenarios",
        "algorithms": ALGORITHMS,
        "topologies": ["sciera_large"],
        "scenarios": list(SCENARIOS.keys()),
        "num_packets": [5000],
        "t_round_ms": [2000],
        "cooldown_ms": [5000],
        "lambda_div": [0.5],
        "point_budget": [100],
    },
    "sapex_tuning": {
        "description": "Tune SAPEX parameters (T_round, cooldown, lambda_div, budget)",
        "algorithms": ["sapex"],
        "topologies": ["sciera_large"],
        "scenarios": list(SCENARIOS.keys()),
        "num_packets": [5000],
        "t_round_ms": T_ROUND_OPTIONS_MS,
        "cooldown_ms": COOLDOWN_OPTIONS_MS,
        "lambda_div": LAMBDA_DIV_OPTIONS,
        "point_budget": POINT_BUDGET_OPTIONS,
    },
    "scalability": {
        "description": "Test with increasing packet counts",
        "algorithms": ["sapex", "round_robin", "lowest_latency"],
        "topologies": ["sciera_large"],
        "scenarios": ["thundering_herd"],
        "num_packets": NUM_PACKETS_OPTIONS,
        "t_round_ms": [2000],
        "cooldown_ms": [5000],
        "lambda_div": [0.5],
        "point_budget": [100],
    },
    "full_sweep": {
        "description": "Full parameter sweep (WARNING: very many combinations!)",
        "algorithms": ALGORITHMS,
        "topologies": ["sciera_large"],
        "scenarios": list(SCENARIOS.keys()),
        "num_packets": [1000, 5000, 10000],
        "t_round_ms": T_ROUND_OPTIONS_MS,
        "cooldown_ms": COOLDOWN_OPTIONS_MS,
        "lambda_div": LAMBDA_DIV_OPTIONS,
        "point_budget": [100, 500],
    },
}


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class ExperimentRunner:
    def __init__(self, output_base_dir="results", dry_run=False, verbose=True):
        self.output_base_dir = Path(output_base_dir)
        self.dry_run = dry_run
        self.verbose = verbose
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_results = []
        self.logger = ResultLogger(base_dir=str(self.output_base_dir))

    def log(self, msg):
        if self.verbose:
            print(msg)

    def generate_experiment_configs(
        self,
        algorithms,
        topologies,
        scenarios,
        num_packets_list,
        t_round_list,
        cooldown_list,
        lambda_div_list,
        point_budget_list,
    ):
        """
        Generate all experiment configurations using the nested loop:
          1. Algorithm
          2. Scenario
          3. Parameter set
        """
        configs = []

        # Loop 1: Algorithms
        for algo in algorithms:
            # Loop 2: Topologies
            for topo_name in topologies:
                topo_file = TOPOLOGIES.get(topo_name, topo_name)

                # Loop 3: Stress Scenarios
                for scenario_name in scenarios:
                    traffic_file = SCENARIOS.get(scenario_name, scenario_name)

                    # Loop 4: Parameter combinations
                    # For non-SAPEX algorithms, many params are irrelevant—
                    # but we still run them to have comparable baselines
                    param_combos = list(itertools.product(
                        num_packets_list,
                        t_round_list,
                        cooldown_list,
                        lambda_div_list,
                        point_budget_list,
                    ))

                    for (n_pkts, t_round, cooldown, lam_div, budget) in param_combos:
                        exp_name = (
                            f"{algo}__{scenario_name}__{topo_name}"
                            f"__np{n_pkts}_tr{t_round}_cd{cooldown}"
                            f"_ld{lam_div}_b{budget}"
                        )

                        config = {
                            "topology": topo_file,
                            "traffic": traffic_file,
                            "algorithm": algo,
                            "scenario": scenario_name,
                            "output_dir": str(
                                self.output_base_dir / self.timestamp / scenario_name / algo
                            ),
                            "experiment_name": exp_name,
                            "parameters": {
                                "num_packets": n_pkts,
                                "packet_size_bytes": PACKET_SIZE_BYTES,
                                "t_round_ms": t_round,
                                "cooldown_ms": cooldown,
                                "lambda_div": lam_div,
                                "point_budget": budget,
                            },
                        }
                        configs.append(config)

        return configs

    def write_config_file(self, config):
        """Write a temporary config JSON for main.py --config."""
        config_dir = Path(config["output_dir"])
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"{config['experiment_name']}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return str(config_path)

    def run_single(self, config, index, total):
        """Run a single experiment."""
        name = config["experiment_name"]
        self.log(f"\n[{index}/{total}] {name}")
        self.log(f"  Algorithm:  {config['algorithm']}")
        self.log(f"  Scenario:   {config['scenario']}")
        self.log(f"  Topology:   {config['topology']}")
        self.log(f"  Params:     np={config['parameters']['num_packets']}, "
                 f"t_round={config['parameters']['t_round_ms']}, "
                 f"cd={config['parameters']['cooldown_ms']}, "
                 f"ld={config['parameters']['lambda_div']}, "
                 f"budget={config['parameters']['point_budget']}")

        if self.dry_run:
            self.log(f"  [DRY RUN] skipped")
            return {"experiment": name, "status": "dry_run"}

        config_path = self.write_config_file(config)

        cmd = [
            sys.executable, "main.py",
            "--config", config_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent),
                timeout=600,  # 10 min timeout per experiment
            )

            if result.returncode == 0:
                self.log(f"  ✓ Completed")
                status = "success"
            else:
                self.log(f"  ✗ Failed (exit {result.returncode})")
                if result.stderr:
                    # Print first 5 lines of stderr
                    for line in result.stderr.strip().split('\n')[:5]:
                        self.log(f"    {line}")
                status = "failed"

            return {
                "experiment": name,
                "status": status,
                "returncode": result.returncode,
            }

        except subprocess.TimeoutExpired:
            self.log(f"  ✗ Timeout (>600s)")
            return {"experiment": name, "status": "timeout"}
        except Exception as e:
            self.log(f"  ✗ Error: {e}")
            return {"experiment": name, "status": "error", "error": str(e)}

    def run_experiments(self, configs):
        """Run all experiment configs."""
        total = len(configs)

        self.log(f"\n{'='*70}")
        self.log(f"  SAPEX-F Experiment Runner")
        self.log(f"  Timestamp:    {self.timestamp}")
        self.log(f"  Output:       {self.output_base_dir / self.timestamp}")
        self.log(f"  Total runs:   {total}")
        self.log(f"  Dry run:      {self.dry_run}")
        self.log(f"{'='*70}")

        for i, config in enumerate(configs, 1):
            result = self.run_single(config, i, total)
            self.run_results.append(result)

        self.save_summary()
        self.aggregate_all_stats()
        self.print_summary()

    def save_summary(self):
        """Save experiment run summary."""
        if self.dry_run:
            return

        summary_dir = self.output_base_dir / self.timestamp
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "experiment_summary.json"

        summary = {
            "timestamp": self.timestamp,
            "total_experiments": len(self.run_results),
            "successful": sum(1 for r in self.run_results if r["status"] == "success"),
            "failed": sum(1 for r in self.run_results if r["status"] == "failed"),
            "timeout": sum(1 for r in self.run_results if r["status"] == "timeout"),
            "errors": sum(1 for r in self.run_results if r["status"] == "error"),
            "experiments": self.run_results,
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    def aggregate_all_stats(self):
        """Aggregate all stats.csv files into one master CSV."""
        if self.dry_run:
            return

        run_root = self.output_base_dir / self.timestamp
        all_run_dirs = []

        # Walk through all result directories looking for stats.csv
        for dirpath, dirnames, filenames in os.walk(run_root):
            if "stats.csv" in filenames:
                all_run_dirs.append(dirpath)

        if all_run_dirs:
            aggregate_path = str(run_root / "all_results.csv")
            ResultLogger.aggregate_stats_csvs(all_run_dirs, aggregate_path)
            self.log(f"\n  Aggregated CSV: {aggregate_path}")

    def print_summary(self):
        """Print final summary."""
        success = sum(1 for r in self.run_results if r["status"] == "success")
        failed = sum(1 for r in self.run_results if r["status"] == "failed")
        timeout = sum(1 for r in self.run_results if r["status"] == "timeout")
        errors = sum(1 for r in self.run_results if r["status"] == "error")
        dry = sum(1 for r in self.run_results if r["status"] == "dry_run")

        self.log(f"\n{'='*70}")
        self.log(f"  SUMMARY")
        self.log(f"{'='*70}")
        self.log(f"  Total:      {len(self.run_results)}")
        self.log(f"  Successful: {success}")
        self.log(f"  Failed:     {failed}")
        self.log(f"  Timeout:    {timeout}")
        self.log(f"  Errors:     {errors}")
        if dry:
            self.log(f"  Dry-run:    {dry}")
        self.log(f"{'='*70}")


# ============================================================================
# CLI HELPERS
# ============================================================================

def list_options():
    """Print all available options."""
    print("\n" + "="*70)
    print("  AVAILABLE EXPERIMENT OPTIONS")
    print("="*70)

    print("\n  ALGORITHMS:")
    for algo in ALGORITHMS:
        print(f"    - {algo}")

    print("\n  TOPOLOGIES:")
    for name, path in TOPOLOGIES.items():
        print(f"    - {name}: {path}")

    print("\n  STRESS SCENARIOS:")
    for name, path in SCENARIOS.items():
        print(f"    - {name}: {path}")

    print("\n  PARAMETER RANGES:")
    print(f"    num_packets:   {NUM_PACKETS_OPTIONS}")
    print(f"    packet_size:   {PACKET_SIZE_BYTES} bytes (fixed)")
    print(f"    t_round_ms:    {T_ROUND_OPTIONS_MS}")
    print(f"    cooldown_ms:   {COOLDOWN_OPTIONS_MS}")
    print(f"    lambda_div:    {LAMBDA_DIV_OPTIONS}")
    print(f"    point_budget:  {POINT_BUDGET_OPTIONS}")

    print("\n  PREDEFINED EXPERIMENT SETS:")
    for name, cfg in EXPERIMENT_SETS.items():
        n_combos = (
            len(cfg["algorithms"]) *
            len(cfg["topologies"]) *
            len(cfg["scenarios"]) *
            len(cfg["num_packets"]) *
            len(cfg["t_round_ms"]) *
            len(cfg["cooldown_ms"]) *
            len(cfg["lambda_div"]) *
            len(cfg["point_budget"])
        )
        print(f"    - {name} ({n_combos} runs): {cfg['description']}")

    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Master Runner for SAPEX-F Simulation Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py --preset quick              # Quick smoke test
  python run_experiments.py --preset algorithm_comparison
  python run_experiments.py --preset sapex_tuning --dry-run

  # Custom selection:
  python run_experiments.py \\
      --algorithms sapex lowest_latency round_robin \\
      --scenarios thundering_herd shared_bottleneck \\
      --topologies sciera_large \\
      --num-packets 1000 5000 \\
      --t-round 2000 5000 \\
      --cooldown 5000 \\
      --lambda-div 0.5 \\
      --budget 100
        """,
    )

    # Predefined sets
    parser.add_argument("--preset", type=str, choices=list(EXPERIMENT_SETS.keys()),
                        help="Use a predefined experiment set")

    # Custom selections
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=None)
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="Scenario names or custom traffic JSON paths")
    parser.add_argument("--topologies", nargs="+", default=None,
                        help="Topology names or custom topology JSON paths")

    # Parameter overrides
    parser.add_argument("--num-packets", nargs="+", type=int, default=None)
    parser.add_argument("--t-round", nargs="+", type=int, default=None)
    parser.add_argument("--cooldown", nargs="+", type=int, default=None)
    parser.add_argument("--lambda-div", nargs="+", type=float, default=None)
    parser.add_argument("--budget", nargs="+", type=int, default=None)

    # General options
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Base directory for results (default: results)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview experiments without running them")
    parser.add_argument("--list", action="store_true", dest="list_options",
                        help="List all available options and exit")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")

    args = parser.parse_args()

    if args.list_options:
        list_options()
        return 0

    # Determine experiment parameters
    if args.preset:
        preset = EXPERIMENT_SETS[args.preset]
        algorithms = args.algorithms or preset["algorithms"]
        topologies = args.topologies or preset["topologies"]
        scenarios_ = args.scenarios or preset["scenarios"]
        num_packets = args.num_packets or preset["num_packets"]
        t_round = args.t_round or preset["t_round_ms"]
        cooldown = args.cooldown or preset["cooldown_ms"]
        lambda_div = args.lambda_div or preset["lambda_div"]
        budget = args.budget or preset["point_budget"]
    else:
        # Default to a quick comparison if nothing specified
        algorithms = args.algorithms or ["sapex"]
        topologies = args.topologies or ["small"]
        scenarios_ = args.scenarios or ["thundering_herd"]
        num_packets = args.num_packets or [1000]
        t_round = args.t_round or [2000]
        cooldown = args.cooldown or [5000]
        lambda_div = args.lambda_div or [0.5]
        budget = args.budget or [100]

    runner = ExperimentRunner(
        output_base_dir=args.output_dir,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )

    configs = runner.generate_experiment_configs(
        algorithms=algorithms,
        topologies=topologies,
        scenarios=scenarios_,
        num_packets_list=num_packets,
        t_round_list=t_round,
        cooldown_list=cooldown,
        lambda_div_list=lambda_div,
        point_budget_list=budget,
    )

    if not configs:
        print("No experiment configurations generated. Check your arguments.")
        return 1

    runner.run_experiments(configs)
    return 0


if __name__ == "__main__":
    exit(main())
