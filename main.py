# main.py
import argparse
import json
import os
import sys
from simulation import Simulation
from sapex_algorithm import SapexAlgorithm
from shortest_path_algorithm import ShortestPathAlgorithm
from algorithms.lowest_latency import LowestLatencyAlgorithm
from algorithms.lowest_hop_count import LowestHopCountAlgorithm
from algorithms.round_robin import RoundRobinAlgorithm
from algorithms.random_path import RandomPathAlgorithm
from metrics import MetricsCollector
from result_logger import ResultLogger

# Available algorithms mapping
ALGORITHMS = {
    "sapex": SapexAlgorithm,
    "shortest_path": ShortestPathAlgorithm,
    "lowest_latency": LowestLatencyAlgorithm,
    "lowest_hop_count": LowestHopCountAlgorithm,
    "round_robin": RoundRobinAlgorithm,
    "random": RandomPathAlgorithm,
}


def load_config(config_path):
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def merge_config(args, config):
    """
    Merge CLI arguments with config file.
    CLI arguments take precedence over config file values.
    """
    # Start with config file values as defaults
    merged = {
        "topology": config.get("topology", "topologies/sciera_large.json"),
        "traffic": config.get("traffic", "traffic_sciera_large.json"),
        "algorithm": config.get("algorithm", "sapex"),
        "scenario": config.get("scenario", "default"),
        "output_dir": config.get("output_dir", "results"),
        "experiment_name": config.get("experiment_name", "default"),
        "parameters": config.get("parameters", {}),
    }
    
    # Override with CLI arguments if provided
    if args.topology:
        merged["topology"] = args.topology
    if args.traffic:
        merged["traffic"] = args.traffic
    if args.algorithm:
        merged["algorithm"] = args.algorithm
    if args.output_dir:
        merged["output_dir"] = args.output_dir
    if args.experiment_name:
        merged["experiment_name"] = args.experiment_name
    
    return merged


def main():
    parser = argparse.ArgumentParser(description="SCION Path Selection Simulation Framework")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a configuration file (JSON) that specifies all simulation parameters."
    )
    parser.add_argument(
        "--topology",
        type=str,
        default=None,
        help="Path to the SCION topology file."
    )
    parser.add_argument(
        "--traffic",
        type=str,
        default=None,
        help="Path to the traffic scenario file."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=list(ALGORITHMS.keys()),
        default=None,
        help="Path selection algorithm to use (sapex, shortest_path)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        dest="output_dir",
        help="Directory to save simulation results."
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        dest="experiment_name",
        help="Name of the experiment (used for result file naming)."
    )
    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = load_config(args.config)
        settings = merge_config(args, config)
    else:
        # Use defaults or CLI arguments
        settings = {
            "topology": args.topology or "topology.json",
            "traffic": args.traffic or "traffic.json",
            "algorithm": args.algorithm or "sapex",
            "scenario": "default",
            "output_dir": args.output_dir or "results",
            "experiment_name": args.experiment_name or "default",
            "parameters": {},
        }

    # Get the algorithm class
    algorithm_name = settings["algorithm"]
    if algorithm_name not in ALGORITHMS:
        print(f"Error: Unknown algorithm '{algorithm_name}'. Available: {list(ALGORITHMS.keys())}")
        return 1
    
    algorithm_class = ALGORITHMS[algorithm_name]
    
    print(f"=== Simulation Configuration ===")
    print(f"Topology: {settings['topology']}")
    print(f"Traffic: {settings['traffic']}")
    print(f"Algorithm: {algorithm_name}")
    print(f"Scenario: {settings['scenario']}")
    print(f"Experiment: {settings['experiment_name']}")
    print(f"Output Dir: {settings['output_dir']}")
    if settings.get("parameters"):
        print(f"Parameters: {json.dumps(settings['parameters'], indent=2)}")
    print(f"================================\n")

    # Create output directory if it doesn't exist
    if settings["output_dir"]:
        os.makedirs(settings["output_dir"], exist_ok=True)

    # Initialize metrics collector (topology will be set after simulation is created)
    metrics = MetricsCollector()
    metrics.start_collection()

    # Create and run simulation
    sim = Simulation(
        settings["topology"],
        settings["traffic"],
        algorithm_class
    )

    # Pass additional parameters to simulation if provided
    params = settings.get("parameters", {})
    if params:
        sim.config_parameters = params

        # Apply algorithm-specific parameters
        algo = sim.path_selection_algorithm
        if hasattr(algo, 'T_round') and "t_round_ms" in params:
            algo.T_round = params["t_round_ms"]
        if hasattr(algo, 'cooldown_duration') and "cooldown_ms" in params:
            algo.cooldown_duration = params["cooldown_ms"]
        if hasattr(algo, 'budget') and "point_budget" in params:
            algo.budget = params["point_budget"]
    
    # Attach metrics collector to simulation
    sim.metrics_collector = metrics
    
    sim.env.process(sim.run())
    sim.env.run()

    metrics.stop_collection()

    # Save results to file if output directory is specified
    if settings["output_dir"]:
        logger = ResultLogger(base_dir=settings["output_dir"])
        run_dir = logger.create_run_directory(run_id=settings["experiment_name"])
        logger.save_all(run_dir, settings, metrics)
        print(f"\nResults saved to: {run_dir}/")

    return 0


if __name__ == "__main__":
    exit(main())