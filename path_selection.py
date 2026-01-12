# path_selection.py
from abc import ABC, abstractmethod
import networkx as nx
import random

class PathSelectionAlgorithm(ABC):
    def __init__(self, topology):
        self.topology = topology
        self.path_store = {} # (src_as, dst_as) -> [path1, path2, ...]
        self.unavailable_paths = {} # (src_as, dst_as) -> set(tuple(path))
        self.probing_enabled = False
        self.probing_interval = None  # Milliseconds between probes
        self.probe_results = {}  # {tuple(path): [latency1, latency2, ...]}
        self.pending_probes = {}  # {probe_id: (path, send_time)}
        self.probe_counter = 0
        self.env = None  # Will be set by simulation
        self.probe_hosts = {}  # {as_id: host} for sending probes

    @abstractmethod
    def select_path(self, source_as, destination_as):
        """Selects and returns the best path."""
        pass

    def mark_path_down(self, router_path):
        """
        Mark a specific path as unavailable due to failure.

        Args:
            router_path (list): Router sequence like ["br1-110-1", "br1-111-1", "br1-112-1"]

        Returns:
            list: Affected (src_as, dst_as) pairs that used this path
        """
        path_tuple = tuple(router_path)
        affected_pairs = []

        # Find all AS pairs that use this path
        for (src_as, dst_as), paths in self.path_store.items():
            if router_path in paths:
                if (src_as, dst_as) not in self.unavailable_paths:
                    self.unavailable_paths[(src_as, dst_as)] = set()
                self.unavailable_paths[(src_as, dst_as)].add(path_tuple)
                affected_pairs.append((src_as, dst_as))

        print(f"Marked path DOWN: {router_path}")
        print(f"Affected AS pairs: {affected_pairs}")
        return affected_pairs

    def mark_path_up(self, router_path):
        """
        Mark a previously failed path as available again.

        Args:
            router_path (list): Router sequence to restore

        Returns:
            list: Affected (src_as, dst_as) pairs that can now use this path
        """
        path_tuple = tuple(router_path)
        affected_pairs = []

        # Remove from unavailable paths
        for (src_as, dst_as), unavail_set in list(self.unavailable_paths.items()):
            if path_tuple in unavail_set:
                unavail_set.remove(path_tuple)
                affected_pairs.append((src_as, dst_as))
                # Clean up empty sets
                if not unavail_set:
                    del self.unavailable_paths[(src_as, dst_as)]

        print(f"Marked path UP: {router_path}")
        print(f"Affected AS pairs: {affected_pairs}")
        return affected_pairs

    def is_path_available(self, router_path):
        """
        Check if a path is currently available (not marked down).

        Args:
            router_path (list): Router sequence to check

        Returns:
            bool: True if path is available, False otherwise
        """
        path_tuple = tuple(router_path)

        # Check if this path is in any unavailable set
        for unavail_set in self.unavailable_paths.values():
            if path_tuple in unavail_set:
                return False
        return True

    def enable_probing(self, interval_ms, env, probe_hosts):
        """
        Enable periodic path probing.

        Args:
            interval_ms: Milliseconds between probe cycles
            env: SimPy environment for scheduling
            probe_hosts: Dictionary mapping AS IDs to host objects for sending probes
        """
        self.probing_enabled = True
        self.probing_interval = interval_ms
        self.env = env
        self.probe_hosts = probe_hosts

    def probe_paths(self):
        """
        SimPy process that periodically probes all known paths.
        This should be called as: env.process(algorithm.probe_paths())
        """
        from packet import ProbePacket

        if not self.probing_enabled:
            return

        while True:
            # Wait for the next probing interval
            yield self.env.timeout(self.probing_interval)

            # Probe all paths in the path store
            for (src_as, dst_as), paths in self.path_store.items():
                # Check if we have a host in the source AS to send probes
                if src_as not in self.probe_hosts:
                    continue

                source_host = self.probe_hosts[src_as]

                for path in paths:
                    # Skip unavailable paths
                    if not self.is_path_available(path):
                        continue

                    # Create and send probe
                    self.probe_counter += 1
                    probe_id = f"probe_{self.probe_counter}"

                    probe = ProbePacket(
                        source=source_host.node_id,
                        destination=path[-1],  # Last router in path
                        path=path.copy(),
                        probe_id=probe_id,
                        timestamp=self.env.now
                    )

                    # Track pending probe
                    self.pending_probes[probe_id] = (tuple(path), self.env.now)

                    # Send probe
                    source_host.send_packet(probe)

            # Start listening for probe responses
            self.env.process(self._collect_probe_responses())

    def _collect_probe_responses(self):
        """
        Collect probe responses and update latency measurements.
        """
        # Wait a reasonable time for probes to return (e.g., 2x the max expected RTT)
        max_wait = 500  # milliseconds
        yield self.env.timeout(max_wait)

        # Process any probes that should have returned by now
        # In a real implementation, this would check the hosts' queues
        # For now, we rely on the update_probe_result() being called

    def update_probe_result(self, probe_id, rtt):
        """
        Update probe results when a probe returns.

        Args:
            probe_id: Identifier of the probe
            rtt: Round-trip time in milliseconds
        """
        if probe_id not in self.pending_probes:
            return

        path_tuple, send_time = self.pending_probes[probe_id]

        # Store probe result
        if path_tuple not in self.probe_results:
            self.probe_results[path_tuple] = []

        self.probe_results[path_tuple].append(rtt)

        # Keep only recent measurements (e.g., last 10)
        if len(self.probe_results[path_tuple]) > 10:
            self.probe_results[path_tuple].pop(0)

        # Remove from pending
        del self.pending_probes[probe_id]

    def get_path_latency(self, router_path):
        """
        Get the average probed latency for a path.

        Args:
            router_path (list): Router sequence

        Returns:
            float: Average RTT in milliseconds, or None if no probe data
        """
        path_tuple = tuple(router_path)

        if path_tuple not in self.probe_results or not self.probe_results[path_tuple]:
            return None

        return sum(self.probe_results[path_tuple]) / len(self.probe_results[path_tuple])

    def discover_paths(self, use_graph_traversal=False):
        """
        Optional discovery process that finds all simple paths using graph traversal.
        This is a fallback mechanism - normally paths should be populated by beaconing.

        Args:
            use_graph_traversal: If True, use NetworkX to find all paths (bypasses beaconing)
        """
        if not use_graph_traversal:
            print("Path discovery via beaconing is enabled. Paths will be discovered through beacon propagation.")
            return

        # Fallback: graph-based path discovery (for testing/debugging)
        print("WARNING: Using graph traversal for path discovery. This bypasses SCION beaconing.")
        all_ases = {node.split('-')[0] for node in self.topology.graph.nodes() if '-' in node}
        for src_as in all_ases:
            for dst_as in all_ases:
                if src_as != dst_as:
                    # Find routers in source and destination ASes
                    src_routers = [r for r in self.topology.graph.nodes() if r.startswith(src_as + '-')]
                    dst_routers = [r for r in self.topology.graph.nodes() if r.startswith(dst_as + '-')]
                    if not src_routers or not dst_routers:
                        continue

                    # Find all paths between the first router of each AS
                    # In a real scenario, you'd do this for all border routers
                    paths = list(nx.all_simple_paths(self.topology.graph, source=src_routers[0], target=dst_routers[0]))
                    self.path_store[(src_as, dst_as)] = paths





        
                
                        
                        
            
