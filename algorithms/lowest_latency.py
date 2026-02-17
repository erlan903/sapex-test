# algorithms/lowest_latency.py
"""
Lowest Latency Path Selection Algorithm.

Selects the path with the minimum estimated end-to-end latency from all
available paths between a source and destination AS.

Latency source:
  - Probed RTT (preferred) — running average of measured round-trip times
    from SCMP-style echo probes sent along each candidate path.
  - Topology estimate (fallback) — end-to-end latency derived from the
    configured link weights along the path, used only before probes return.

The algorithm compares *all* discovered paths between the same (src, dst)
pair and picks the one with the lowest end-to-end latency, regardless of
hop count.
"""

from path_selection import PathSelectionAlgorithm


class LowestLatencyAlgorithm(PathSelectionAlgorithm):
    """Selects the path with the lowest end-to-end latency."""

    def __init__(self, topology, use_beaconing=True):
        super().__init__(topology)
        self.use_beaconing = use_beaconing
        self.discover_paths(use_graph_traversal=not use_beaconing)

    # ------------------------------------------------------------------
    # End-to-end latency helpers
    # ------------------------------------------------------------------

    def _get_end_to_end_latency(self, path):
        """
        Return the best available end-to-end latency estimate for *path*.

        Prefers the probed RTT (measured). Falls back to the topology-
        configured latency (sum of edge weights along the path).
        """
        probed = self.get_path_latency(path)
        if probed is not None:
            return probed
        return self._topology_latency(path)

    def _topology_latency(self, path):
        """
        Derive end-to-end latency from topology edge weights.

        This is only used as a cold-start estimate before any probes have
        completed for this path.
        """
        total = 0.0
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            if self.topology.graph.has_edge(src, dst):
                total += self.topology.graph[src][dst].get('latency', 0)
            else:
                total += 10_000  # penalise missing edges
        return total

    # ------------------------------------------------------------------
    # Path selection
    # ------------------------------------------------------------------

    def select_path(self, source_as, destination_as, app_instance=None):
        candidates = self.path_store.get((source_as, destination_as), [])
        if not candidates:
            return None

        # Keep only paths whose links are currently up
        candidates = [p for p in candidates if self.is_path_available(p)]
        if not candidates:
            return None

        # Compare every candidate path on its end-to-end latency
        return min(candidates, key=self._get_end_to_end_latency)
