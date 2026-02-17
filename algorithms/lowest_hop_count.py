# algorithms/lowest_hop_count.py
"""
Lowest Hop Count Path Selection Algorithm.

Selects the path with the fewest number of hops (routers).
Equivalent to the existing ShortestPathAlgorithm but explicitly named.
"""

from path_selection import PathSelectionAlgorithm


class LowestHopCountAlgorithm(PathSelectionAlgorithm):
    """Selects the path with the fewest hops (shortest in router count)."""

    def __init__(self, topology, use_beaconing=True):
        super().__init__(topology)
        self.use_beaconing = use_beaconing
        self.discover_paths(use_graph_traversal=not use_beaconing)

    def select_path(self, source_as, destination_as, app_instance=None):
        available_paths = self.path_store.get((source_as, destination_as), [])
        if not available_paths:
            return None

        # Filter out unavailable paths
        available_paths = [p for p in available_paths if self.is_path_available(p)]
        if not available_paths:
            return None

        # Select the path with fewest hops
        return min(available_paths, key=len)
