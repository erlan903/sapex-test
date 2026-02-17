# metrics.py
"""

Tracks:
    - Path Switching Frequency (oscillation count)
    - Transfer Time (time to deliver all packets per flow)
    - Per-flow latency statistics (min, max, avg, median, p95, p99)
    - Per-flow packet loss rate
    - Per-flow throughput
    - Per-path utilization
"""

import time as wall_time
from collections import defaultdict
from collections.abc import Iterable


class MetricsCollector:
    """
    Collects and aggregates simulation metrics across all flows and paths.
    Attach to a Simulation instance to automatically gather data.
    """

    def __init__(self):
        # --- Per-flow tracking ---
        # flow_name -> list of latencies (ms)
        self.flow_latencies = defaultdict(list)
        # flow_name -> count of lost packets
        self.flow_packet_loss = defaultdict(int)
        # flow_name -> count of sent packets
        self.flow_packets_sent = defaultdict(int)
        # flow_name -> count of received packets
        self.flow_packets_received = defaultdict(int)
        # flow_name -> bytes sent
        self.flow_bytes_sent = defaultdict(int)
        # flow_name -> first packet send time (sim ms)
        self.flow_start_time = {}
        # flow_name -> last packet receive time (sim ms)
        self.flow_end_time = {}
        # flow_name -> list of (sim_time, path_tuple) to track switches
        self.flow_path_history = defaultdict(list)

        # --- Per-path tracking ---
        # path_tuple -> total packets forwarded
        self.path_packet_count = defaultdict(int)
        # path_tuple -> total bytes forwarded
        self.path_bytes = defaultdict(int)

        # --- Global tracking ---
        self.all_latencies = []
        self.total_packets_sent = 0
        self.total_packets_received = 0
        self.total_packets_lost = 0

        # Wall-clock timing
        self._wall_start = None
        self._wall_end = None

    def start_collection(self):
        """Mark the start of metrics collection (wall-clock)."""
        self._wall_start = wall_time.time()

    def stop_collection(self):
        """Mark the end of metrics collection (wall-clock)."""
        self._wall_end = wall_time.time()

    # ---- Recording methods (called during simulation) ----

    def record_packet_sent(self, flow_name, sim_time, path, packet_size=1500):
        """Record a packet being sent."""
        if isinstance(sim_time, str) and not self._is_path_like(path):
            # Legacy call signature: (flow_name, app_name, sim_time)
            sim_time, path = path, []

        if not isinstance(sim_time, (int, float)):
            sim_time = 0

        if not self._is_path_like(path):
            path = []

        self.flow_packets_sent[flow_name] += 1
        self.flow_bytes_sent[flow_name] += packet_size
        self.total_packets_sent += 1

        path_tuple = tuple(path)
        self.path_packet_count[path_tuple] += 1
        self.path_bytes[path_tuple] += packet_size

        if flow_name not in self.flow_start_time:
            self.flow_start_time[flow_name] = sim_time

    def record_packet_received(self, flow_name, sim_time, latency, packet_size=None):
        """Record a packet being received."""
        if isinstance(sim_time, str) and packet_size is not None:
            # Legacy call signature: (flow_name, app_name, sim_time, latency)
            sim_time, latency = latency, packet_size

        if not isinstance(sim_time, (int, float)):
            sim_time = 0
        if not isinstance(latency, (int, float)):
            latency = 0

        self.flow_latencies[flow_name].append(latency)
        self.flow_packets_received[flow_name] += 1
        self.flow_end_time[flow_name] = sim_time
        self.total_packets_received += 1
        self.all_latencies.append(latency)

    def record_packet_loss(self, flow_name):
        """Record a packet loss."""
        self.flow_packet_loss[flow_name] += 1
        self.total_packets_lost += 1

    def record_path_switch(self, flow_name, sim_time, new_path):
        """Record a path switch event."""
        path_tuple = tuple(new_path)
        history = self.flow_path_history[flow_name]

        # Only record if the path actually changed
        if not history or history[-1][1] != path_tuple:
            history.append((sim_time, path_tuple))

    # ---- Aggregation methods (called after simulation) ----

    def get_flow_stats(self, flow_name):
        """Get comprehensive statistics for a single flow."""
        latencies = self.flow_latencies.get(flow_name, [])
        sent = self.flow_packets_sent.get(flow_name, 0)
        received = self.flow_packets_received.get(flow_name, 0)
        lost = self.flow_packet_loss.get(flow_name, 0)
        bytes_sent = self.flow_bytes_sent.get(flow_name, 0)
        start = self.flow_start_time.get(flow_name, 0)
        end = self.flow_end_time.get(flow_name, 0)

        transfer_time = end - start if end > start else 0
        loss_rate = (lost / sent * 100) if sent > 0 else 0
        throughput_mbps = (bytes_sent * 8 / (transfer_time * 1000)) if transfer_time > 0 else 0

        stats = {
            "packets_sent": sent,
            "packets_received": received,
            "packets_lost": lost,
            "loss_rate_percent": round(loss_rate, 4),
            "bytes_sent": bytes_sent,
            "transfer_time_ms": round(transfer_time, 2),
            "throughput_mbps": round(throughput_mbps, 4),
            "path_switches": self.get_path_switch_count(flow_name),
        }

        if latencies:
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)
            stats.update({
                "latency_min_ms": round(min(latencies), 4),
                "latency_max_ms": round(max(latencies), 4),
                "latency_avg_ms": round(sum(latencies) / n, 4),
                "latency_median_ms": round(sorted_lat[n // 2], 4),
                "latency_p95_ms": round(sorted_lat[int(n * 0.95)], 4),
                "latency_p99_ms": round(sorted_lat[int(n * 0.99)], 4),
                "latency_stddev_ms": round(self._stddev(latencies), 4),
            })
        else:
            stats.update({
                "latency_min_ms": 0,
                "latency_max_ms": 0,
                "latency_avg_ms": 0,
                "latency_median_ms": 0,
                "latency_p95_ms": 0,
                "latency_p99_ms": 0,
                "latency_stddev_ms": 0,
            })

        return stats

    def get_path_switch_count(self, flow_name):
        """Get the number of path switches for a flow (oscillation metric)."""
        history = self.flow_path_history.get(flow_name, [])
        # First entry is initial path selection, so switches = len - 1
        return max(0, len(history) - 1)

    def get_total_path_switches(self):
        """Get total path switches across all flows."""
        total = 0
        for flow_name in self.flow_path_history:
            total += self.get_path_switch_count(flow_name)
        return total

    def get_global_stats(self):
        """Get global simulation statistics."""
        total_sent = self.total_packets_sent
        total_received = self.total_packets_received
        total_lost = self.total_packets_lost
        loss_rate = (total_lost / total_sent * 100) if total_sent > 0 else 0

        stats = {
            "total_packets_sent": total_sent,
            "total_packets_received": total_received,
            "total_packets_lost": total_lost,
            "packet_loss_rate_percent": round(loss_rate, 4),
            "total_path_switches": self.get_total_path_switches(),
        }

        if self.all_latencies:
            sorted_lat = sorted(self.all_latencies)
            n = len(sorted_lat)
            stats.update({
                "latency_min_ms": round(min(self.all_latencies), 4),
                "latency_max_ms": round(max(self.all_latencies), 4),
                "latency_avg_ms": round(sum(self.all_latencies) / n, 4),
                "latency_median_ms": round(sorted_lat[n // 2], 4),
                "latency_p95_ms": round(sorted_lat[int(n * 0.95)], 4),
                "latency_p99_ms": round(sorted_lat[int(n * 0.99)], 4),
            })
        else:
            stats.update({
                "latency_min_ms": 0,
                "latency_max_ms": 0,
                "latency_avg_ms": 0,
                "latency_median_ms": 0,
                "latency_p95_ms": 0,
                "latency_p99_ms": 0,
            })

        if self._wall_start and self._wall_end:
            stats["wall_clock_seconds"] = round(self._wall_end - self._wall_start, 3)

        return stats

    def get_per_path_stats(self):
        """Get utilization stats per path."""
        path_stats = {}
        for path_tuple, count in self.path_packet_count.items():
            path_key = " -> ".join(path_tuple)
            path_stats[path_key] = {
                "packets_forwarded": count,
                "bytes_forwarded": self.path_bytes.get(path_tuple, 0),
            }
        return path_stats

    def get_full_report(self):
        """Get the complete metrics report as a dictionary."""
        report = {
            "global": self.get_global_stats(),
            "per_flow": {},
            "per_path": self.get_per_path_stats(),
        }

        for flow_name in self.flow_packets_sent:
            report["per_flow"][flow_name] = self.get_flow_stats(flow_name)

        return report

    # ---- Helpers ----

    @staticmethod
    def _stddev(values):
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    @staticmethod
    def _is_path_like(value):
        return isinstance(value, Iterable) and not isinstance(value, (str, bytes))
