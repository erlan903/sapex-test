# beaconing.py
from packet import BeaconPacket

class BeaconingProcess:
    def __init__(self, env, start_router, path_selection_algorithm, interval=1000, topology=None):
        self.env = env
        self.start_router = start_router
        self.path_selection_algorithm = path_selection_algorithm
        self.interval = interval
        self.topology = topology
        self.origin_as = self.extract_as_from_router_id(start_router.node_id)

    def extract_as_from_router_id(self, router_id):
        """Extract AS ID from router ID. E.g., '71-20965-br-fra-1' -> '71-20965'"""
        # Format: ISD-AS-router_name
        # Example: "71-20965-br-fra-1" -> "71-20965"
        # The AS part is everything before the last occurrence of a router name pattern
        # We can identify AS by looking for the pattern before "-br"
        if '-br' in router_id:
            as_id = router_id.split('-br')[0]
            return as_id
        return router_id

    def start(self):
        """Periodically send beacons from this core AS"""
        while True:
            # Create beacon with proper AS information
            beacon = BeaconPacket(
                origin_router_id=self.start_router.node_id,
                origin_as_id=self.origin_as
            )
            #print(f"[{self.env.now:.2f}] Core AS {self.origin_as} (router {self.start_router.node_id}) sending beacon.")

            # Send beacon directly to all neighbors (don't process it locally)
            # The origin router's AS is already in the beacon's hop list
            for neighbor_id, link in self.start_router.ports.items():
                beacon_copy = beacon.clone()
                link.enqueue(beacon_copy)
            #print(f" waiting for next beacon interval {self.interval}")
            yield self.env.timeout(self.interval)

    def register_path(self, beacon, receiving_router_id):
        """
        Register a path discovered via beaconing into the path store.
        Called when a beacon reaches a border router.

        Args:
            beacon: BeaconPacket containing the path information
            receiving_router_id: ID of the router that received this beacon
        """
        receiving_as = self.extract_as_from_router_id(receiving_router_id)
        origin_as = beacon.origin_as_id

        # Don't register paths to ourselves
        if receiving_as == origin_as:
            return

        # Extract router-level path from beacon for data packet forwarding
        router_path = beacon.get_router_path()

        # Add the receiving router to complete the path
        if receiving_router_id not in router_path:
            router_path.append(receiving_router_id)

        # Register the path in the path store
        # Format: path_store[(src_as, dst_as)] = [path1, path2, ...]
        path_key = (origin_as, receiving_as)

        if path_key not in self.path_selection_algorithm.path_store:
            self.path_selection_algorithm.path_store[path_key] = []

        # Check if this path already exists (avoid duplicates)
        if router_path not in self.path_selection_algorithm.path_store[path_key]:
            self.path_selection_algorithm.path_store[path_key].append(router_path)

            # Also register the reverse path (for up-segments)
            reverse_path = list(reversed(router_path))
            reverse_key = (receiving_as, origin_as)

            if reverse_key not in self.path_selection_algorithm.path_store:
                self.path_selection_algorithm.path_store[reverse_key] = []

            if reverse_path not in self.path_selection_algorithm.path_store[reverse_key]:
                self.path_selection_algorithm.path_store[reverse_key].append(reverse_path)

            print(f"[{self.env.now:.2f}] AS {receiving_as} registered path from {origin_as}: {router_path}")
            print(f"[{self.env.now:.2f}] AS {receiving_as} registered reverse path to {origin_as}: {reverse_path}")

            # Also create combined paths to other non-core ASes through the core
            # This simulates SCION path combination for inter-leaf communication
            if origin_as != receiving_as:
                self._create_combined_paths(origin_as, receiving_as, router_path, reverse_path)

            # Expand transitive combinations so multi-core leaf-to-leaf paths are discovered.
            self._expand_transitive_combinations()

    def _register_composed_path(self, src_as, dst_as, router_path):
        """Register a composed path if it is new and loop-free at AS-level."""
        if src_as == dst_as:
            return False

        if self._has_as_loop(router_path):
            return False

        key = (src_as, dst_as)
        if key not in self.path_selection_algorithm.path_store:
            self.path_selection_algorithm.path_store[key] = []

        if router_path in self.path_selection_algorithm.path_store[key]:
            return False

        self.path_selection_algorithm.path_store[key].append(router_path)
        return True

    def _has_as_loop(self, router_path):
        """
        Reject paths that revisit a previously left AS (avoids composition cycles).
        Consecutive routers in the same AS are valid for intra-AS stitching.
        """
        seen = set()
        previous_as = None
        for router_id in router_path:
            as_id = self.extract_as_from_router_id(router_id)
            if as_id != previous_as:
                if as_id in seen:
                    return True
                seen.add(as_id)
            previous_as = as_id
        return False

    def _bridge_within_as(self, as_id, from_router, to_router):
        """Find all router paths inside a single AS between two border routers."""
        if from_router == to_router:
            return [[from_router]]

        if not self.topology:
            return []

        try:
            candidate_paths = self.topology.all_router_paths(from_router, to_router)
        except Exception:
            return []

        if not candidate_paths:
            return []

        valid_paths = []
        for path in candidate_paths:
            # Ensure bridge does not leave the intended AS.
            if all(self.extract_as_from_router_id(router_id) == as_id for router_id in path):
                valid_paths.append(path)
        return valid_paths

    def _stitch_paths(self, left_path, right_path, middle_as):
        """Compose src->middle and middle->dst router paths into one or more src->dst paths."""
        if not left_path or not right_path:
            return []

        left_tail = left_path[-1]
        right_head = right_path[0]
        tail_as = self.extract_as_from_router_id(left_tail)
        head_as = self.extract_as_from_router_id(right_head)

        if tail_as != middle_as or head_as != middle_as:
            return []

        if left_tail == right_head:
            return [left_path + right_path[1:]]

        bridges = self._bridge_within_as(middle_as, left_tail, right_head)
        if not bridges:
            return []

        stitched = []
        for bridge in bridges:
            stitched.append(left_path + bridge[1:] + right_path[1:])
        return stitched

    def _expand_transitive_combinations(self):
        """
        Build transitive path combinations:
          if we know A->B and B->C, compose A->C.
        """
        changed = True
        while changed:
            changed = False
            all_items = list(self.path_selection_algorithm.path_store.items())

            for (src_as, mid_as), left_paths in all_items:
                for (mid_as_2, dst_as), right_paths in all_items:
                    if mid_as != mid_as_2:
                        continue
                    if src_as == dst_as:
                        continue

                    for left_path in left_paths:
                        for right_path in right_paths:
                            combined_paths = self._stitch_paths(left_path, right_path, mid_as)
                            for combined in combined_paths:
                                if self._register_composed_path(src_as, dst_as, combined):
                                    changed = True
                                    print(
                                        f"[{self.env.now:.2f}] Created transit path {src_as} -> {dst_as}: {combined}"
                                    )

    def _create_combined_paths(self, core_as, leaf_as, down_path, up_path):
        """Create combined paths between leaf ASes through the core"""
        # Find all paths in the path store that go through this core AS
        for (src_as, dst_as), paths in list(self.path_selection_algorithm.path_store.items()):
            # If there's a path from another leaf AS to the core, combine it with our path
            if dst_as == core_as and src_as != leaf_as and src_as != core_as:
                # We have:
                # - up_path: leaf_as -> core_as (our reverse path)
                # - paths: src_as -> core_as (existing up-segment)
                # We need to create: src_as -> core_as -> leaf_as
                for existing_path in paths:
                    # Combine using router-level stitching at the core AS boundary.
                    combined_paths = self._stitch_paths(existing_path, down_path, core_as)
                    for combined_path in combined_paths:
                        if self._register_composed_path(src_as, leaf_as, combined_path):
                            print(f"[{self.env.now:.2f}] Created combined path {src_as} -> {leaf_as}: {combined_path}")