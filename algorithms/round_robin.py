# algorithms/round_robin.py
"""
Round-Robin / ECMP Path Selection Algorithm.

Cycles through all available paths in order, distributing traffic equally.

"""

from path_selection import PathSelectionAlgorithm


class RoundRobinAlgorithm(PathSelectionAlgorithm):
    """
    Round-Robin (ECMP) path selection.
    Distributes packets equally across all available paths.
    """

    def __init__(self, topology, use_beaconing=True):
        super().__init__(topology)
        self.use_beaconing = use_beaconing
        self.discover_paths(use_graph_traversal=not use_beaconing)
        # Track the round-robin index per (src, dst) pair
        self._rr_index = {}

    def select_path(self, source_as, destination_as, app_instance=None):
        available_paths = self.path_store.get((source_as, destination_as), [])
        if not available_paths:
            return None

        # Filter out unavailable paths
        available_paths = [p for p in available_paths if self.is_path_available(p)]
        if not available_paths:
            return None

        pair_key = (source_as, destination_as)
        if pair_key not in self._rr_index:
            self._rr_index[pair_key] = 0

        # Select the next path in round-robin order
        index = self._rr_index[pair_key] % len(available_paths)
        self._rr_index[pair_key] += 1

        return available_paths[index]
