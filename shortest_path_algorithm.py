from path_selection import PathSelectionAlgorithm

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