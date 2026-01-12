# main.py
import argparse
from simulation import Simulation
import sapex_algorithm
# To use your own algorithm, you would import it here, e.g.:
# from my_sapex_f import SapexAlgorithm

def main():
    parser = argparse.ArgumentParser(description="SCION Path Selection Simulation Framework")
    parser.add_argument(
        "--topology",
        type=str,
        default="topology.json",
        help="Path to the SCION topology file."
    )
    parser.add_argument(
        "--traffic",
        type=str,
        default="traffic.json",
        help="Path to the traffic scenario file."
    )
    args = parser.parse_args()

    # --- To use a different algorithm, change this line ---
    # For example: sim = Simulation(args.topology, args.traffic, SapexAlgorithm)
    sim = Simulation(args.topology, args.traffic, SapexAlgorithm)
    
    sim.env.process(sim.run())
    sim.env.run()


if __name__ == "__main__":
    main()