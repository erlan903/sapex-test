# packet.py
import copy
import time

class HopInfo:
    """Represents AS-level hop information in a Path Construction Beacon (PCB)"""
    def __init__(self, as_id, router_id, ingress_if=None, egress_if=None, link_metrics=None):
        self.as_id = as_id  # ISD-AS identifier (e.g., "71-20965")
        self.router_id = router_id  # Border router ID (e.g., "br1-110-1")
        self.ingress_if = ingress_if  # Interface where PCB entered this AS
        self.egress_if = egress_if  # Interface where PCB exits this AS
        self.link_metrics = link_metrics or {}  # Dict with latency, bandwidth, etc.

    def __repr__(self):
        return f"HopInfo(AS={self.as_id}, Router={self.router_id})"

class Packet:
    def __init__(self, source, destination, path, payload="", size=1500, is_beacon=False):
        self.source = source
        self.destination = destination
        self.path = path
        self.payload = payload
        self.size = size # in bytes
        self.is_beacon = is_beacon
        self.creation_time = 0

    def clone(self):
        return copy.deepcopy(self)

class BeaconPacket(Packet):
    """Path Construction Beacon (PCB) for SCION path discovery"""
    def __init__(self, origin_router_id, origin_as_id, timestamp=None):
        super().__init__(source=origin_router_id, destination=None,
                         path=[origin_router_id], is_beacon=True, size=100)
        self.origin_as_id = origin_as_id  # Origin AS (ISD-AS)
        self.timestamp = timestamp or time.time()  # Creation time
        self.hops = []  # List of HopInfo objects (AS-level path)
        self.segment_type = "down"  # down, core, or up segment

        # Initialize with origin AS as first hop
        origin_hop = HopInfo(as_id=origin_as_id, router_id=origin_router_id)
        self.hops.append(origin_hop)

    def add_hop(self, as_id, router_id, ingress_if, egress_if=None, link_metrics=None):
        """Add a new hop to the PCB path"""
        hop = HopInfo(as_id, router_id, ingress_if, egress_if, link_metrics)
        self.hops.append(hop)

    def get_as_path(self):
        """Get the list of AS IDs in this beacon's path"""
        return [hop.as_id for hop in self.hops]

    def get_router_path(self):
        """Get the list of router IDs in this beacon's path (for data forwarding)"""
        return [hop.router_id for hop in self.hops]

    def get_total_latency(self):
        """Get total latency across hops based on link metrics."""
        total = 0
        for hop in self.hops:
            if hop.link_metrics and "latency" in hop.link_metrics:
                latency = hop.link_metrics.get("latency")
                if latency is not None:
                    total += latency
        return total

    def clone(self):
        """Create a deep copy of this beacon"""
        return copy.deepcopy(self)

class ProbePacket(Packet):
    """Probe packet for measuring path latency"""
    def __init__(self, source, destination, path, probe_id=None, timestamp=None):
        super().__init__(source=source, destination=destination, path=path,
                         payload="", size=64, is_beacon=False)
        self.probe_id = probe_id  # Unique identifier for this probe
        self.timestamp = timestamp or time.time()  # When probe was sent
        self.is_probe = True  # Flag to identify probe packets
        self.rtt = None  # Round-trip time (set when probe returns)