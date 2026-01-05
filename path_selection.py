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


# --- Example Implementation ---
class ShortestPathAlgorithm(PathSelectionAlgorithm):
    def __init__(self, topology, use_beaconing=True):
        super().__init__(topology)
        self.use_beaconing = use_beaconing
        # Only use graph traversal if beaconing is disabled
        self.discover_paths(use_graph_traversal=not use_beaconing)

    def select_path(self, source_as, destination_as):
        available_paths = self.path_store.get((source_as, destination_as), [])
        if not available_paths:
            return None

        # Filter out unavailable paths
        available_paths = [p for p in available_paths if self.is_path_available(p)]
        if not available_paths:
            return None

        # Select the path with the minimum number of hops
        return min(available_paths, key=len)


#On the initial empty setting, I comment out the new algorithm below, in order to ensure the plain setup properly works

#To integrate path lists, every path may be handled as objects with the following attributes
class PathCandidate:
    """
    Wrapper class to hold state and metrics for a specific path,
    in order to track 'PROBING' vs 'ACTIVE' and store RTT history.
    """
    def __init__(self, router_path):
        self.router_path = router_path      # The actual list of router IDs
        self.state = "PROBING"              # INACTIVE, PROBING, ACTIVE, COOLDOWN
        self.score = 0.0                    # S(p,a)

        # Metrics for filtering
        self.latency_history = []
        self.avg_latency = 1000.0               # Default high RTT until measured
        self.packet_loss_count = 0
        self.packets_sent = 0
        self.cost = 1                       # Cost to budget (simplified to 1 per path)

        # UMCC: Throughput tracking
        self.throughput_history = []        # Bytes/sec over time windows
        self.bytes_sent = 0
        self.bytes_received = 0
        self.last_throughput_time = 0

        # UMCC: Congestion detection state
        self.is_congested = False
        self.congestion_start_time = None
        self.shared_bottleneck_interfaces = set()  # Set of interface IDs in shared bottleneck

    def update_latency(self, latency):
        self.latency_history.append(latency)
        # Keep last 10 measurements
        if len(self.latency_history) > 10:
            self.latency_history.pop(0)
        self.avg_latency = sum(self.latency_history) / len(self.latency_history)

    def update_throughput(self, bytes_transferred, time_window):
        """
        Update throughput measurement.

        Args:
            bytes_transferred: Number of bytes in the time window
            time_window: Duration in milliseconds
        """
        if time_window > 0:
            throughput_mbps = (bytes_transferred * 8) / (time_window * 1000)  # Convert to Mbps
            self.throughput_history.append(throughput_mbps)
            # Keep last 10 measurements
            if len(self.throughput_history) > 10:
                self.throughput_history.pop(0)

    def get_avg_throughput(self):
        """Get average throughput in Mbps"""
        if not self.throughput_history:
            return 0.0
        return sum(self.throughput_history) / len(self.throughput_history)

    def get_loss_rate(self):
        """Calculate current packet loss rate"""
        if self.packets_sent == 0:
            return 0.0
        return self.packet_loss_count / self.packets_sent

    def detect_congestion(self, rtt_threshold_increase=1.5, loss_threshold=0.05, throughput_decrease=0.7):
        """
        Detect if this path is experiencing congestion based on recent metrics.

        Args:
            rtt_threshold_increase: RTT increase ratio to trigger congestion (e.g., 1.5 = 50% increase)
            loss_threshold: Packet loss rate threshold (e.g., 0.05 = 5%)
            throughput_decrease: Throughput decrease ratio (e.g., 0.7 = 30% decrease)

        Returns:
            bool: True if congestion detected
        """
        # Need at least some history to detect changes
        if len(self.latency_history) < 3:
            return False

        # Check for RTT increase
        recent_rtt = sum(self.latency_history[-3:]) / 3
        baseline_rtt = self.latency_history[0] if len(self.latency_history) > 0 else recent_rtt
        rtt_increased = recent_rtt > baseline_rtt * rtt_threshold_increase

        # Check for packet loss
        loss_rate = self.get_loss_rate()
        high_loss = loss_rate > loss_threshold

        # Check for throughput decrease
        throughput_decreased = False
        if len(self.throughput_history) >= 3:
            recent_throughput = sum(self.throughput_history[-3:]) / 3
            baseline_throughput = self.throughput_history[0] if self.throughput_history else recent_throughput
            if baseline_throughput > 0:
                throughput_decreased = recent_throughput < baseline_throughput * throughput_decrease

        # Congestion if any two conditions are met
        congestion_signals = sum([rtt_increased, high_loss, throughput_decreased])
        return congestion_signals >= 2

    def get_interface_ids(self):
        """
        Extract interface IDs from the router path.
        Each interface is identified as ISD-AS-RouterID.

        Returns:
            set: Set of interface IDs (strings) in this path
        """
        interface_ids = set()
        for router_id in self.router_path:
            # Router ID format: "1-ff00:0:110-br1-110-1"
            # Interface ID is the full router ID
            interface_ids.add(router_id)
        return interface_ids


class SapexAlgorithm(PathSelectionAlgorithm):
    # initialization method
    def __init__(self, topology, use_beaconing=True, enable_probing=False, probing_interval=1000, enable_umcc=True):
        super().__init__(topology)
        self.use_beaconing = use_beaconing
        self.discover_paths(use_graph_traversal=not use_beaconing)

        # Dictionary to store our PathCandidate objects
        # Key: tuple(tuple(path_list)), Value: PathCandidate object
        self.candidates_map = {}

        #Assign an initial value to the point budget of the application
        self.budget = 3  # Example

        # Initialize metric constraints for the application
        self.max_latency = 200.0 # ms
        self.max_loss_rate = 0.1 # 10%
        self.min_throughput = 0 # change later

        # Initialize the partition size N for the application
        self.partition_size_N = 2

        # Set probing interval if enabled
        if enable_probing:
            self.probing_interval = probing_interval

        # UMCC: Shared bottleneck detection
        self.enable_umcc = enable_umcc
        self.shared_bottlenecks = []  # List of sets of interface IDs that form bottlenecks
        self.bottleneck_representatives = {}  # Maps bottleneck index to the chosen representative path
    
    #Step 1: Retrieve paths -- Beaconing integration needed

    def _sync_candidates(self, source_as, destination_as):
        """
        Synchronizes the 'stateless' network paths discovered by beaconing with the
        'stateful' PathCandidate objects used by the algorithm.

        This method acts as a bridge:
        1. It retrieves raw paths (lists of router IDs) from the network topology.
        2. It checks if the algorithm has seen these paths before.
        3. It returns PathCandidate objects that preserve historical metrics (RTT, Loss)
           across simulation steps.
        """

        # 1. Retrieve Raw Data:
        # Fetch the latest list of available paths from the global path store.
        # These were simple lists of strings (e.g., ['router A', 'router B']).
        raw_paths = self.path_store.get((source_as, destination_as), [])

        # This list will hold the pathCandidate objects(extended versions of raw paths)
        # we pass to the algorithm
        current_candidates = []

        for p in raw_paths:
            # 2. Data Transformation (List -> Tuple):
            # We convert the mutable list 'p' into an immutable 'tuple'.
            # This is required to use the path as a key in a Python dictionary.
            p_key = tuple(p)

            # 3. We check if we already have a PathCandidate object for this specific path.
            if p_key not in self.candidates_map:
                # CASE: NEW PATH
                # If this path was just discovered by a beacon, we initialize a new
                # PathCandidate object. It starts with default state (PROBING) and empty history.
                self.candidates_map[p_key] = PathCandidate(p)

                # If probing is enabled, use probe data for initial latency
                if self.probing_enabled:
                    probe_latency = self.get_path_latency(p)
                    if probe_latency is not None:
                        self.candidates_map[p_key].avg_latency = probe_latency
                        self.candidates_map[p_key].latency_history = [probe_latency]
            else:
                # CASE: EXISTING PATH
                # Update with latest probe data if available
                if self.probing_enabled:
                    probe_latency = self.get_path_latency(p)
                    if probe_latency is not None:
                        candidate = self.candidates_map[p_key]
                        # Merge probe data with feedback data
                        # Probe data can supplement when no recent feedback exists
                        if len(candidate.latency_history) == 0:
                            candidate.update_latency(probe_latency)

            # 4. Aggregation:
            # We retrieve the specific object (whether new or old) and add it to our current list.
            current_candidates.append(self.candidates_map[p_key])

        # Return the list of stateful objects to be filtered and scored
        return current_candidates
    
    #~Retrieve the Shared Bottleneck List (may be not in the initial setup of simple simulation)
    
    #Step 2: Filter the retrieved path set by some constraints 
        #According to topology config we have in .jsonfile, here latency and bandwidth are potential attributes that 
        #might initally be processed by the draft algorithm. 
            
        #Though, I need to find a way to extract those values along with latency, throughput, packet loss and path length

    # BRIDGE METHOD: Called by Application to report metrics
         
    def update_path_feedback(self, router_path, latency_sample, is_loss, packet_size=1500):
        """
        Ingests real-time performance metrics from the Data Plane (Application)
        into the Control Plane (Algorithm Memory).

        Args:
            router_path (list): The sequence of router IDs used by the packet.
            latency_sample (float): One-way latency in milliseconds (ignored if is_loss=True).
            is_loss (bool): True if the packet was dropped, False if it arrived.
            packet_size (int): Size of the packet in bytes (for throughput calculation).
        """

        # Convert the mutable list of routers into an immutable tuple.
        # This allows us to use the path as a hashable key to look up our memory object.
        p_key = tuple(router_path)

        # Ensuring that we actually have a record (PathCandidate) for this specific path.
        if p_key in self.candidates_map:

            # Changes made to 'candidate' here persist in self.candidates_map.
            candidate = self.candidates_map[p_key]

            candidate.packets_sent += 1

            if is_loss:
                # Case: Packet Loss
                # Increment the loss counter (Numerator for loss rate).
                # We do NOT update latency statistics for lost packets.
                candidate.packet_loss_count += 1
            else:
                # Case: Successful Delivery
                # Pass the latency sample to the candidate object to update
                # its internal sliding window average.
                candidate.update_latency(latency_sample)

                # UMCC: Update throughput tracking
                candidate.bytes_received += packet_size
                if candidate.last_throughput_time > 0:
                    time_window = self.env.now - candidate.last_throughput_time if self.env else 1
                    if time_window >= 100:  # Update every 100ms
                        candidate.update_throughput(candidate.bytes_received, time_window)
                        candidate.bytes_received = 0
                        candidate.last_throughput_time = self.env.now if self.env else 0
                else:
                    candidate.last_throughput_time = self.env.now if self.env else 0

    def detect_shared_bottlenecks(self, active_paths):
        """
        UMCC Algorithm 1: Shared Bottleneck Detection
        Implements the shared bottleneck detection algorithm from the 2024 paper.

        Args:
            active_paths (list): List of PathCandidate objects currently in use

        Returns:
            list: List of sets, where each set contains interface IDs of a shared bottleneck
        """
        if not self.enable_umcc or len(active_paths) < 2:
            return []

        # Step 1: Measure RTT, throughput, and packet loss on each used path
        # (Already done via update_path_feedback)

        # Step 2: Watch for packet loss, RTT, and throughput decreases, or delay increases
        congested_paths = []
        for path_candidate in active_paths:
            if path_candidate.detect_congestion():
                congested_paths.append(path_candidate)
                if not path_candidate.is_congested:
                    path_candidate.is_congested = True
                    path_candidate.congestion_start_time = self.env.now if self.env else 0

        # Step 3: If it affects at least two paths with minimal similarity
        if len(congested_paths) < 2:
            return []

        print(f"[{self.env.now if self.env else 0:.2f}] UMCC: Detected congestion on {len(congested_paths)} paths")

        # Step 4: Assign each SCION Border Router interface a globally unique ID (ISD-AS-Intf)
        # (Already implemented in PathCandidate.get_interface_ids())

        # Step 5: Build the intersection of all included interface IDs of all affected paths
        if not congested_paths:
            return []

        # Start with interfaces from the first congested path
        common_interfaces = congested_paths[0].get_interface_ids()

        # Intersect with all other congested paths
        for path_candidate in congested_paths[1:]:
            common_interfaces = common_interfaces.intersection(path_candidate.get_interface_ids())

        # If no common interfaces, there's no shared bottleneck
        if not common_interfaces:
            return []

        # Step 6: Check all other paths that are not affected by the same event.
        # If an interface is also in another path, remove it from the list.
        non_congested_paths = [p for p in active_paths if p not in congested_paths]

        for path_candidate in non_congested_paths:
            path_interfaces = path_candidate.get_interface_ids()
            # Remove interfaces that appear in non-congested paths
            common_interfaces = common_interfaces - path_interfaces

        # If no interfaces remain after filtering, no shared bottleneck
        if not common_interfaces:
            return []

        print(f"[{self.env.now if self.env else 0:.2f}] UMCC: Identified shared bottleneck interfaces: {common_interfaces}")

        # Mark congested paths with the shared bottleneck interfaces
        for path_candidate in congested_paths:
            path_candidate.shared_bottleneck_interfaces = common_interfaces

        # Step 7: Keep only one path that uses the shared bottleneck
        # Return the detected bottleneck as a set of interface IDs
        return [common_interfaces]

    def apply_bottleneck_constraints(self, filtered_paths):
        """
        Apply UMCC bottleneck constraints to path selection.
        When multiple paths share a bottleneck, only select one representative path.

        Args:
            filtered_paths (list): List of PathCandidate objects after filtering

        Returns:
            list: Filtered list with only one path per shared bottleneck
        """
        if not self.enable_umcc or not filtered_paths:
            return filtered_paths

        # Detect shared bottlenecks among active/candidate paths
        bottlenecks = self.detect_shared_bottlenecks(filtered_paths)

        if not bottlenecks:
            return filtered_paths

        # For each bottleneck, keep only the best path
        result_paths = []
        excluded_paths = set()

        for bottleneck_interfaces in bottlenecks:
            # Find all paths that use this bottleneck
            affected_paths = []
            for path_candidate in filtered_paths:
                path_interfaces = path_candidate.get_interface_ids()
                # Check if this path uses any of the bottleneck interfaces
                if bottleneck_interfaces.intersection(path_interfaces):
                    affected_paths.append(path_candidate)

            if not affected_paths:
                continue

            # Step 7: Keep only one path that uses the shared bottleneck
            # Choose the path with the best metrics (lowest latency, lowest loss rate)
            best_path = min(affected_paths,
                          key=lambda p: (p.avg_latency, p.get_loss_rate()))

            # Add the best path to results
            if best_path not in result_paths:
                result_paths.append(best_path)

            # Mark others as excluded
            for path in affected_paths:
                if path != best_path:
                    excluded_paths.add(tuple(path.router_path))
                    print(f"[{self.env.now if self.env else 0:.2f}] UMCC: Excluding path {path.router_path} due to shared bottleneck")

        # Add paths that don't use any bottleneck
        for path_candidate in filtered_paths:
            if tuple(path_candidate.router_path) not in excluded_paths and path_candidate not in result_paths:
                result_paths.append(path_candidate)

        return result_paths
    
    def calculate_diversity_bonus(self, candidate_path_obj, active_path_objects):
        #weight_diversity * (1 - ((interfaces of active paths of the application.intersection(
            # interfaces of path p)) / interfaces of path p))
        # Find out where to get "interfaces of active paths of the application" and how to
        #fetch this from the application.py or app_registry.py to path_selection.py
        #What if the application have not used any path yet? So there is no paths info
        #of used paths of app a. --> simply assign 1 to diversity bonus, to indicate full bonus
        # or zero to indicate no bonus in the first call
        weight_diversity = 1
        ep_set = candidate_path_obj.get_interface_ids()
        size_ep = len(ep_set)
        
        # Safety check: if path has no interfaces (e.g. source->dest direct?), avoid division by zero
        if size_ep == 0:
            return 0.0
        
        # Extract interfaces already in use by the application (Ea)
        ea_set = set()
        for active_path in active_path_objects:

            if active_path != candidate_path_obj:
                ea_set.update(active_path.get_interface_ids())
        
        if not ea_set:
            return weight_diversity
        
        overlap_count = len(ep_set.intersection(ea_set))
        fraction = overlap_count / len(ea_set)
        diversity_bonus = weight_diversity * (1 - len(fraction))        
        return diversity_bonus  
            
    #When the application requests a path, it must introduce itself to the algorithm 
    # so the algorithm can access that specific path_scoring_randomness variable 
    # --> app_instance argument
    def select_path(self, source_as, destination_as, app_instance=None):

        retrieved_paths = self._sync_candidates(source_as, destination_as)
        if not retrieved_paths:
            return None

        # Filter out paths marked as down
        retrieved_paths = [
            p for p in retrieved_paths
            if self.is_path_available(p.router_path)
        ]

        if not retrieved_paths:
            return None

        #Compare the path metrics with those constraints and remove the insufficient ones
        #from the path set (python list of paths)
        #if else logic can be used
        filtered_paths = []
        for p in retrieved_paths:
            loss_rate = (p.packet_loss_count / p.packets_sent) if p.packets_sent > 0 else 0.0

            if p.avg_latency <= self.max_latency and loss_rate <= self.max_loss_rate:
                filtered_paths.append(p)
            else:
                p.state = "INACTIVE"
        #Fallback if strict filtering kills all paths(least-worst)
        if not filtered_paths:
            filtered_paths = retrieved_paths

        # UMCC: Apply shared bottleneck detection and filtering
        # Step 8: Repeat in case more congestion is located
        if self.enable_umcc:
            filtered_paths = self.apply_bottleneck_constraints(filtered_paths)
        

            
    #Initialize the partition size N for the application    


    #Step 3: Compute the composite scores S(p,a) for each path
        # Detect bottlenecks across ALL filtered paths first
        detected_bottlenecks_list = self.detect_shared_bottlenecks(filtered_paths)
        weight_sb = 0.1 
        new_active_set = []
        #multipliers depending on the state of the path
        alpha_probe = 0.7 
        alpha_inactive = 0.4
        alpha_state = 0 
        #random multiplier per each app will be created stored in application.py
        
        #Step 3.1: Calculate the base scores
        #for loop on path dictionary/list
        for p in filtered_paths:
            #initalize weights for each path at the beginning of the iteration
            weight_throughput = 0.1
            weight_packet_loss = -0.1
            weight_rtt = -0.1
            
            #Priority to probing data
            rtt_val = self.get_path_latency(p.router_path)
            #if there is no probing data yet, then use the current value of the path
            if rtt_val is None:
                rtt_val = p.avg_latency

            #p.base_score = (weight_throughput * throughput) + (weight_packet_loss * packet_loss_rate) + (weight_rtt * rtt)
            base_score = (p.weight_throughput * p.get_avg_throughput()) + (p.weight_packet_loss * p.get_loss_rate()) \
                + (p.weight_rtt * rtt_val)
                
            #Step 3.2: Calculate the SB_penaltys for each path
            #use detect_shared_bottlenecks(self, active_paths) to get sbl interfaces
            # weight_sb * ( 
            # len(common(umcc_shared_bottleneck_interfaces, interfaces_of_path_p)) / len(interfaces_of_path_p))
            Ep = p.get_interface_ids()
            shared_interface_count = len(SBL.intersection(Ep))
            interface_count_of_path = len(Ep)
            if interface_count_of_path > 0:
                sb_penalty = (weight_sb * shared_interface_count) / interface_count_of_path
            else:
                sb_penalty = 0
                
            #Step 3.3: Calculate the diversity bonuses for each path
            # a helper method, to decouple 
            div_bonus = self.calculate_diversity_bonus(p, current_active, weight_diversity)

            #Step 4.4: integrate state logic into the scoring
            if p.state == 'ACTIVE':
                alpha_state = 1
            elif p.state == 'PROBING':
                alpha_state = alpha_probe
            elif p.state == 'INACTIVE':
                alpha_state = alpha_inactive    
            
            #Step 4.5: final scoring
            #get the app-specific randomness value for the path selection
            lambda_rand = app_instance.path_selection_randomness if app_instance else 1
            final_score = lambda_rand * ((alpha_state * base_score) - sb_penalty + div_bonus)
            
            
            


            

    #Step 4: Sort paths by descending composite scores (S(p,a))
            #May be merge sort as the default --> O(nlogn) time complexity 
            #Different sorting algorithms could be tested for a comparison
        filtered_paths.sort(key=lambda x: x.score, reverse=True)
        #reverse=True (descending order), take scores of path sobjects as sorting argument

    #Step 5: Group paths into N sized Groups
    
    #Initialize the allocation epoch T_round 
        
                
        #Step 6: 
        #create an emtpy list (will serve as the path set)
        #a for loop to to look into each partition in the sorted path list
            #an inner for loop to look into the paths inside the partitions

                #find the 'best' paths among the partitions('best' will be corrected
                #with the relevant details) 
                #subtract the cost of each best path from the balance of the app
                #populate the path list with those 'best' paths 
                
                #return the path list 
                
        current_budget = self.budget
        active_set = []
        
        for candidate in filtered_paths: #simple iteration for now
            if current_budget >= candidate.cost:
                current_budget -= candidate.cost
                candidate.state = "ACTIVE"
                active_set.append(candidate)
            else:
                candidate.state = "INACTIVE"
    
    # Step 7-8-9 need to continuosuly watch for failures and react, 
    # so may be while loops or threads might be an option
    
    #Step 7: Probing
    #...
    
    #Step 8: UMCC
    #...
    
    #Step 9: Failure handling and maintenance
    #if event_type='path_down' occurs then remove that path from the ACTIVE list.
    #if event_type='path_up' occurs then set that paths' state from ... 
    #...

    
    #Step 10: Jitter
    # Change the delay randomly (assign a random number) whenever 
    # ACTIVE path list changes
        # 0 < delay <= T_round/2
        # Randomly select one of the ACTIVE paths to distribute load
        if active_set:
            return random.choice(active_set).router_path        
        
    #apply the randomized delay so wait: sleep(delay) can be used
    
    
        # SAVE STATE: This ensures next time we call select_path, 
        # these paths are treated as "already in use" for the app.
        self.active_set = new_active_set
        # change their state attributes to 'ACTIVE' 
        # ...
        
                
                        
                        
            
