# simulation.py
import simpy
import json
import os
from topology import Topology
from application import Application
from shortest_path_algorithm import ShortestPathAlgorithm
from events import EventManager
from app_registry import ApplicationRegistry

class Simulation:
    def __init__(self, topology_file, traffic_file, algorithm_class=ShortestPathAlgorithm):
        self.env = simpy.Environment()
             
        # Load traffic scenario (must be after event_manager is created)
        self.traffic_scenario = self.load_traffic_scenario(traffic_file, topology_file)

        self.topology = Topology(self.env, self.traffic_scenario['topology'])
        self.path_selection_algorithm = algorithm_class(self.topology)

        self.results = {"packet_loss": 0, "latencies": []}

        # Metrics collector (will be set by main.py if needed)
        self.metrics_collector = None

        # Application registry for path-app tracking
        self.app_registry = ApplicationRegistry()

        # Event manager for scheduled failures
        self.event_manager = EventManager(
            self.env,
            self.path_selection_algorithm,
            self.app_registry
        )


        # Load events if present
        if 'events' in self.traffic_scenario:
            self.event_manager.load_events(self.traffic_scenario)


    def _resolve_path(self, maybe_path, base_dir):
        """Resolve paths using absolute, CWD-relative, script-relative, then scenario-relative lookup."""
        if not maybe_path:
            return None

        if os.path.isabs(maybe_path):
            return maybe_path if os.path.isfile(maybe_path) else None

        candidates = [
            os.path.normpath(maybe_path),
            os.path.normpath(os.path.join(os.getcwd(), maybe_path)),
            os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), maybe_path)),
            os.path.normpath(os.path.join(base_dir, maybe_path)),
        ]

        for path in candidates:
            if os.path.isfile(path):
                return path

        return None

    def load_traffic_scenario(self, filename, topology_file=None):
        print(f"Loading traffic scenario from {filename}...")
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Traffic scenario file '{filename}' not found.")

        with open(filename, 'r') as f:
            scenario = json.load(f)

        scenario_dir = os.path.dirname(os.path.abspath(filename))
        loaded = dict(scenario)

        # Supported format A (wrapper): {"topology": ..., "traffic": "traffic.json", ...}
        # Supported format B (self-contained): contains flows/events directly.
        traffic_ref = scenario.get('traffic')
        if traffic_ref is not None:
            traffic_path = self._resolve_path(traffic_ref, scenario_dir)
            if not traffic_path or not os.path.isfile(traffic_path):
                raise FileNotFoundError(
                    f"Traffic file '{traffic_ref}' resolved to '{traffic_path}' was not found."
                )

            with open(traffic_path, 'r') as f:
                traffic = json.load(f)

            loaded['flows'] = traffic.get('flows', [])
            loaded['events'] = traffic.get('events', [])
            loaded['duration_ms'] = traffic.get('duration_ms', loaded.get('duration_ms', 1000))
            loaded['drain'] = traffic.get('drain', loaded.get('drain', {}))
        else:
            if 'flows' not in loaded:
                raise ValueError(
                    "Scenario file must define either 'traffic' or inline 'flows'."
                )
            loaded['events'] = loaded.get('events', [])
            loaded['duration_ms'] = loaded.get('duration_ms', 1000)
            loaded['drain'] = loaded.get('drain', {})

        topology_ref = loaded.get('topology') or topology_file
        if not topology_ref:
            raise ValueError("Topology is not specified (missing scenario 'topology' and CLI topology).")

        topology_path = self._resolve_path(topology_ref, scenario_dir)
        if not topology_path or not os.path.isfile(topology_path):
            raise FileNotFoundError(
                f"Topology file '{topology_ref}' resolved to '{topology_path}' was not found."
            )
        loaded['topology'] = topology_path

        return loaded

    def _metrics_snapshot(self):
        """Capture a compact traffic progress snapshot for drain convergence checks."""
        sent = 0
        received = 0
        lost = 0

        if self.metrics_collector:
            sent = self.metrics_collector.total_packets_sent
            received = self.metrics_collector.total_packets_received
            lost = self.metrics_collector.total_packets_lost

        # Estimate queued packets still waiting on links.
        queued_packets = 0
        seen_links = set()
        for node in self.topology.nodes.values():
            for link in node.ports.values():
                link_id = id(link)
                if link_id in seen_links:
                    continue
                seen_links.add(link_id)
                queued_packets += len(link.queue.items)

        return sent, received, lost, queued_packets

    def _run_drain_phase(self):
        """Run optional post-traffic drain to let in-flight packets settle."""
        drain_cfg = self.traffic_scenario.get('drain', {}) or {}
        enabled = drain_cfg.get('enabled', True)
        if not enabled:
            return

        max_drain_ms = int(drain_cfg.get('max_ms', 60000))
        idle_window_ms = int(drain_cfg.get('idle_window_ms', 5000))
        check_interval_ms = int(drain_cfg.get('check_interval_ms', 500))

        if max_drain_ms <= 0 or check_interval_ms <= 0:
            return

        print(
            f"\nStarting drain phase (max={max_drain_ms}ms, "
            f"idle_window={idle_window_ms}ms, check_interval={check_interval_ms}ms)..."
        )

        drain_start = self.env.now
        deadline = drain_start + max_drain_ms
        previous_snapshot = None
        unchanged_ms = 0

        while self.env.now < deadline:
            step_until = min(deadline, self.env.now + check_interval_ms)
            self.env.run(until=step_until)

            snapshot = self._metrics_snapshot()
            if snapshot == previous_snapshot:
                unchanged_ms += check_interval_ms
            else:
                unchanged_ms = 0
                previous_snapshot = snapshot

            _, _, _, queued_packets = snapshot
            if queued_packets == 0 and unchanged_ms >= idle_window_ms:
                print(f"Drain phase converged at t={self.env.now:.2f}ms (queues empty, counters stable).")
                return

        print(f"Drain phase reached max duration at t={self.env.now:.2f}ms.")

    def run(self):
        print("Starting beaconing process...")
        self.topology.initiate_beaconing(self.path_selection_algorithm)
        # Give beaconing some time to propagate
        yield self.env.timeout(2000)

        print("\nAll available paths discovered:")
        for (src, dst), paths in self.path_selection_algorithm.path_store.items():
            print(f"  Paths from {src} to {dst}:")
            for i, path in enumerate(paths):
                 print(f"    {i+1}: {' -> '.join([str(hop) for hop in path])}")

        # Pre-flight check: validate that each traffic flow has at least one AS-level path.
        missing_flow_pairs = []
        for flow in self.traffic_scenario.get('flows', []):
            src_as = str(flow.get('source', '')).split(',')[0]
            dst_as = str(flow.get('destination', '')).split(',')[0]
            candidate_paths = self.path_selection_algorithm.path_store.get((src_as, dst_as), [])
            if not candidate_paths:
                missing_flow_pairs.append((flow.get('name', 'unnamed-flow'), src_as, dst_as))

        if missing_flow_pairs:
            print("\nWARNING: Some traffic flows have no discovered AS path before data-plane start:")
            for flow_name, src_as, dst_as in missing_flow_pairs:
                print(f"  - {flow_name}: {src_as} -> {dst_as} (0 paths)")
            print("These flows will log 'No path found' and send 0 packets unless path discovery is expanded.")

        # Enable probing if the algorithm supports it
        if hasattr(self.path_selection_algorithm, 'enable_probing'):
            # Collect one host per AS for probing
            probe_hosts = {}
            for node_id, node in self.topology.nodes.items():
                if hasattr(node, 'isd_as'):  # It's a host
                    as_id = node.isd_as
                    if as_id not in probe_hosts:
                        probe_hosts[as_id] = node
                        node.path_selector = self.path_selection_algorithm

            # Check if algorithm has probing_interval configured
            if hasattr(self.path_selection_algorithm, 'probing_interval') and \
               self.path_selection_algorithm.probing_interval:
                interval = self.path_selection_algorithm.probing_interval
                self.path_selection_algorithm.enable_probing(interval, self.env, probe_hosts)
                self.env.process(self.path_selection_algorithm.probe_paths())
                print(f"\nPath probing enabled with {interval}ms interval")

        print("\nStarting applications based on traffic scenario...")
        for flow in self.traffic_scenario['flows']:
            source_host = self.topology.get_host(flow['source'])
            destination_host = self.topology.get_host(flow['destination'])

            if source_host and destination_host:
                app = Application(
                    self.env,
                    f"App-{flow['name']}",
                    source_host,
                    destination_host,
                    self.path_selection_algorithm,
                    flow,
                    self.results,
                    self.app_registry,
                    metrics_collector=self.metrics_collector
                )
                self.env.process(app.run())
            else:
                print(f"Warning: Could not find source or destination host for flow {flow['name']}")

        # Schedule event manager process
        self.env.process(self.event_manager.schedule_events())

        # Run the simulation for a specified duration
        simulation_duration = self.traffic_scenario.get("duration_ms", 1000)
        print(f"\nRunning simulation for {simulation_duration}ms...")
        self.env.run(until=simulation_duration)

        # Optional post-duration settling period for in-flight packets.
        self._run_drain_phase()

        print("\nSimulation finished.")
        self.print_results()

    def print_results(self):
        print("\n--- Simulation Results ---")
        total_lost = self.results['packet_loss']
        total_received = len(self.results['latencies'])
        total_sent = total_lost + total_received
        
        loss_rate = (total_lost / total_sent * 100) if total_sent > 0 else 0
        avg_latency = sum(self.results['latencies']) / total_received if total_received > 0 else 0

        print(f"Total Packets Sent: {total_sent}")
        print(f"Total Packets Received: {total_received}")
        print(f"Total Packets Lost: {total_lost}")
        print(f"Packet Loss Rate: {loss_rate:.2f}%")
        print(f"Average Packet Latency: {avg_latency:.2f}ms")
