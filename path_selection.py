# path_selection.py
from abc import ABC, abstractmethod
import networkx as nx
import random

class PathSelectionAlgorithm(ABC):
    def __init__(self, topology):
        self.topology = topology
        self.path_store = {} # (src_as, dst_as) -> [path1, path2, ...]
        self.unavailable_paths = {} # (src_as, dst_as) -> set(tuple(path))

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

    def update_latency(self, latency):
        self.latency_history.append(latency)
        # Keep last 10 measurements
        if len(self.latency_history) > 10:
            self.latency_history.pop(0)
        self.avg_latency = sum(self.latency_history) / len(self.latency_history)


class SapexAlgorithm(PathSelectionAlgorithm):
    # initialization method
    def __init__(self, topology, use_beaconing=True):
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
            
            # CASE: EXISTING PATH
            # If the path exists in the map, we do nothing. The existing object
            # (which holds valuable RTT history and scores) stays in memory.
            
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
         
    def update_path_feedback(self, router_path, latency_sample, is_loss):
        """
        Ingests real-time performance metrics from the Data Plane (Application)
        into the Control Plane (Algorithm Memory).
        
        Args:
            router_path (list): The sequence of router IDs used by the packet.
            latency_sample (float): One-way latency in milliseconds (ignored if is_loss=True).
            is_loss (bool): True if the packet was dropped, False if it arrived.
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
        
        
    def select_path(self, source_as, destination_as):

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
        

            
    #Initialize the partition size N for the application    


    #Step 3: Compute the composite scores S(p,a) for each path
            #for loop on path dictionary
        for p in filtered_paths:
            p.score = ... #apply the detailed formula later
        
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
    #...

    
    #Step 10: Jitter
    # Change the delay randomly (assign a random number) whenever 
    # ACTIVE path list changes
        # 0 < delay <= T_round/2
        # Randomly select one of the ACTIVE paths to distribute load
        if active_set:
            return random.choice(active_set).router_path        
        
    #apply the randomized delay so wait: sleep(delay) can be used
    
    
    
                
                        
                        
            
