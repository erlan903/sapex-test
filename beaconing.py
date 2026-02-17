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
                    # Combine: src_as -> core -> leaf_as
                    # The existing_path goes from src_as to core
                    # The down_path goes from core to leaf_as
                    # We need to connect them at the core router
                    combined_path = existing_path[:-1] + down_path
                    combined_key = (src_as, leaf_as)

                    if combined_key not in self.path_selection_algorithm.path_store:
                        self.path_selection_algorithm.path_store[combined_key] = []

                    if combined_path not in self.path_selection_algorithm.path_store[combined_key]:
                        self.path_selection_algorithm.path_store[combined_key].append(combined_path)
                        print(f"[{self.env.now:.2f}] Created combined path {src_as} -> {leaf_as}: {combined_path}")