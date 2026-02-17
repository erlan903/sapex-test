# algorithms/random_path.py
"""
Random Path Selection Algorithm.

Selects a random available path for each packet.
"""

import random
from path_selection import PathSelectionAlgorithm


class RandomPathAlgorithm(PathSelectionAlgorithm):
    """Selects a random available path each time."""

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

        return random.choice(available_paths)
