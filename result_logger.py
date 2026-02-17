# result_logger.py
"""
Automatic Result Logging for SAPEX-F experiments.

Writes results to structured CSV and JSON files for downstream analysis and plotting.
Output structure:
    results/<run_id>/
        config.json          - Experiment configuration
        stats.json           - Full metrics report (JSON)
        stats.csv            - Summary row for aggregation across runs
        per_flow_stats.csv   - Per-flow detailed metrics
        per_path_stats.csv   - Per-path utilization
        latencies_raw.csv    - Raw latency samples for distribution analysis
        path_switches.csv    - Path switch events timeline
"""

import csv
import json
import os
from datetime import datetime


class ResultLogger:
    """Writes simulation results to structured CSV/JSON files."""

    def __init__(self, base_dir="results"):
        self.base_dir = base_dir

    def create_run_directory(self, run_id=None):
        """Create a unique directory for this run's results."""
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        run_dir = os.path.join(self.base_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def save_all(self, run_dir, config, metrics_collector):
        """
        Save all results for a simulation run.
        
        Args:
            run_dir: Directory to save results in
            config: Experiment configuration dict
            metrics_collector: MetricsCollector instance with gathered data
        """
        self.save_config(run_dir, config)
        
        report = metrics_collector.get_full_report()
        self.save_stats_json(run_dir, report, config)
        self.save_stats_csv(run_dir, report, config)
        self.save_per_flow_csv(run_dir, report)
        self.save_per_path_csv(run_dir, report)
        self.save_raw_latencies_csv(run_dir, metrics_collector)
        self.save_path_switches_csv(run_dir, metrics_collector)

    def save_config(self, run_dir, config):
        """Save experiment configuration."""
        filepath = os.path.join(run_dir, "config.json")
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

    def save_stats_json(self, run_dir, report, config):
        """Save full metrics report as JSON."""
        filepath = os.path.join(run_dir, "stats.json")
        output = {
            "experiment": config.get("experiment_name", "unknown"),
            "algorithm": config.get("algorithm", "unknown"),
            "scenario": config.get("scenario", "unknown"),
            "parameters": config.get("parameters", {}),
            "metrics": report,
        }
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

    def save_stats_csv(self, run_dir, report, config):
        """
        Save a single summary row to stats.csv.
        This CSV is designed to be appended/concatenated across runs
        for comparative analysis.
        """
        filepath = os.path.join(run_dir, "stats.csv")
        
        global_stats = report["global"]
        params = config.get("parameters", {})
        
        row = {
            "experiment_name": config.get("experiment_name", ""),
            "algorithm": config.get("algorithm", ""),
            "scenario": config.get("scenario", ""),
            "topology": config.get("topology", ""),
            # Parameters
            "num_packets": params.get("num_packets", ""),
            "packet_size_bytes": params.get("packet_size_bytes", 1500),
            "t_round_ms": params.get("t_round_ms", ""),
            "cooldown_ms": params.get("cooldown_ms", ""),
            "lambda_div": params.get("lambda_div", ""),
            "point_budget": params.get("point_budget", ""),
            # Global metrics
            "total_packets_sent": global_stats.get("total_packets_sent", 0),
            "total_packets_received": global_stats.get("total_packets_received", 0),
            "total_packets_lost": global_stats.get("total_packets_lost", 0),
            "packet_loss_rate_percent": global_stats.get("packet_loss_rate_percent", 0),
            "latency_avg_ms": global_stats.get("latency_avg_ms", 0),
            "latency_median_ms": global_stats.get("latency_median_ms", 0),
            "latency_p95_ms": global_stats.get("latency_p95_ms", 0),
            "latency_p99_ms": global_stats.get("latency_p99_ms", 0),
            "latency_min_ms": global_stats.get("latency_min_ms", 0),
            "latency_max_ms": global_stats.get("latency_max_ms", 0),
            "total_path_switches": global_stats.get("total_path_switches", 0),
            "wall_clock_seconds": global_stats.get("wall_clock_seconds", ""),
        }

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)

    def save_per_flow_csv(self, run_dir, report):
        """Save per-flow statistics as CSV."""
        filepath = os.path.join(run_dir, "per_flow_stats.csv")
        
        per_flow = report.get("per_flow", {})
        if not per_flow:
            return

        # Get all possible keys from the first flow
        sample_keys = list(next(iter(per_flow.values())).keys())
        fieldnames = ["flow_name"] + sample_keys

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for flow_name, stats in per_flow.items():
                row = {"flow_name": flow_name, **stats}
                writer.writerow(row)

    def save_per_path_csv(self, run_dir, report):
        """Save per-path utilization as CSV."""
        filepath = os.path.join(run_dir, "per_path_stats.csv")
        
        per_path = report.get("per_path", {})
        if not per_path:
            return

        fieldnames = ["path", "packets_forwarded", "bytes_forwarded"]

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for path_key, stats in per_path.items():
                row = {"path": path_key, **stats}
                writer.writerow(row)

    def save_raw_latencies_csv(self, run_dir, metrics_collector):
        """Save raw latency samples for distribution analysis."""
        filepath = os.path.join(run_dir, "latencies_raw.csv")

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["flow_name", "latency_ms"])
            for flow_name, latencies in metrics_collector.flow_latencies.items():
                for lat in latencies:
                    writer.writerow([flow_name, round(lat, 4)])

    def save_path_switches_csv(self, run_dir, metrics_collector):
        """Save path switch timeline for oscillation analysis."""
        filepath = os.path.join(run_dir, "path_switches.csv")

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["flow_name", "sim_time_ms", "new_path"])
            for flow_name, history in metrics_collector.flow_path_history.items():
                for sim_time, path_tuple in history:
                    writer.writerow([flow_name, round(sim_time, 2), " -> ".join(path_tuple)])

    @staticmethod
    def aggregate_stats_csvs(results_dirs, output_filepath):
        """
        Aggregate stats.csv from multiple run directories into a single CSV.
        Useful for cross-experiment comparison and plotting.
        
        Args:
            results_dirs: List of run directories containing stats.csv
            output_filepath: Path for the aggregated output CSV
        """
        all_rows = []
        fieldnames = None

        for run_dir in results_dirs:
            stats_path = os.path.join(run_dir, "stats.csv")
            if not os.path.exists(stats_path):
                continue
            
            with open(stats_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                if fieldnames is None:
                    fieldnames = reader.fieldnames
                for row in reader:
                    all_rows.append(row)

        if not all_rows or not fieldnames:
            return

        with open(output_filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
