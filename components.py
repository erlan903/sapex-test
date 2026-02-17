# components.py
import simpy
from collections import deque

class Link:
    def __init__(self, env, latency, bandwidth_mbps, destination_node, loss_notifier):
        self.env = env
        self.latency = latency
        # Bandwidth in bytes per millisecond
        self.bandwidth = (bandwidth_mbps * 1_000_000) / 8 / 1000
        self.queue = simpy.Store(env)
        self.destination = destination_node
        self.loss_notifier = loss_notifier
        self.env.process(self.run())

    def run(self):
        while True:
            packet = yield self.queue.get()
            # Propagation delay
            yield self.env.timeout(self.latency)
            # Transmission delay
            transmission_delay = (packet.size * 8) / (self.bandwidth * 1000)
            yield self.env.timeout(transmission_delay)
            self.destination.receive_packet(packet)
    
    def enqueue(self, packet):
        self.queue.put(packet)


class Node:
    def __init__(self, env, node_id):
        self.env = env
        self.node_id = node_id
        self.ports = {}

    def add_link(self, to_node_id, latency, bandwidth, destination_node):
        link = Link(self.env, latency, bandwidth, destination_node, self.notify_loss)
        self.ports[to_node_id] = link
        return link

    def notify_loss(self, packet):
        # This can be extended to notify the sending application
        pass

    def receive_packet(self, packet):
        raise NotImplementedError

class Router(Node):
    def __init__(self, env, node_id):
        super().__init__(env, node_id)
        self.forwarding_table = {} # simple next-hop forwarding
        self.beaconing_process = None  # Will be set by topology when beaconing is initiated
        self.as_id = self.extract_as_from_router_id(node_id)

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

    def set_beaconing_process(self, beaconing_process):
        """Set the beaconing process for path registration"""
        self.beaconing_process = beaconing_process

    def receive_packet(self, packet):
        # Beacon packets are handled by the beaconing process logic
        if packet.is_beacon:
            # Get AS-level path to check for loops
            as_path = packet.get_as_path()
            current_as = self.as_id

            # AS-level loop detection
            if current_as in as_path:
                # Already visited this AS, drop the beacon
                return

            # Add hop information to the beacon
            # Find which neighbor this beacon came from to determine ingress interface
            ingress_if = None
            if len(packet.path) > 0:
                previous_router = packet.path[-1]
                ingress_if = previous_router  # Simplified: use router ID as interface ID

            # Get link metrics for this hop
            link_metrics = {}
            if ingress_if and ingress_if in self.ports:
                link = self.ports[ingress_if]
                link_metrics = {
                    'latency': link.latency,
                    'bandwidth': link.bandwidth
                }

            # Add this hop to the beacon's AS-level path
            packet.add_hop(
                as_id=current_as,
                router_id=self.node_id,
                ingress_if=ingress_if,
                link_metrics=link_metrics
            )

            # Update router-level path (for backward compatibility)
            packet.path.append(self.node_id)

            # Register this path if we have a beaconing process
            if self.beaconing_process:
                self.beaconing_process.register_path(packet, self.node_id)

            # Forward beacon to neighbors (not in AS path)
            for neighbor_id, link in self.ports.items():
                # Avoid router-level loops (additional safety check)
                if neighbor_id not in packet.path:
                    # Clone beacon and forward
                    beacon_copy = packet.clone()
                    link.enqueue(beacon_copy)

        else: # Data packet or probe packet
            # Check if this is a probe packet that reached its destination
            if hasattr(packet, 'is_probe') and packet.is_probe:
                try:
                    current_hop_index = packet.path.index(self.node_id)
                    # If this is the last hop (destination router), reflect the probe back
                    if current_hop_index == len(packet.path) - 1:
                        # Reverse the path and send probe back
                        packet.path = list(reversed(packet.path))
                        packet.source, packet.destination = packet.destination, packet.source
                        # Forward back along reversed path
                        next_hop = packet.path[1]
                        if next_hop in self.ports:
                            self.ports[next_hop].enqueue(packet)
                        return
                    else:
                        # Forward probe to next hop
                        next_hop = packet.path[current_hop_index + 1]
                        if next_hop in self.ports:
                            self.ports[next_hop].enqueue(packet)
                        return
                except (ValueError, IndexError):
                    print(f"[{self.env.now:.2f}] Router {self.node_id}: Invalid probe path. Dropping.")
                    return

            if packet.destination == self.node_id:
                # Packet for a host in this AS, find the host link
                # This part needs a more complex routing logic in a full implementation
                pass
            else:
                # Forward along the path
                try:
                    current_hop_index = packet.path.index(self.node_id)
                    next_hop = packet.path[current_hop_index + 1]
                    if next_hop in self.ports:
                        self.ports[next_hop].enqueue(packet)
                    else:
                        print(f"[{self.env.now:.2f}] Router {self.node_id}: Dead end for packet to {packet.destination}. Dropping.")
                except (ValueError, IndexError):
                    print(f"[{self.env.now:.2f}] Router {self.node_id}: Invalid path on packet. Dropping.")

class Host(Node):
    def __init__(self, env, node_id, topology):
        super().__init__(env, node_id)
        self.isd_as = node_id.split(',')[0]
        self.topology = topology
        self.in_queue = simpy.Store(env)
        self.application = None # To be set by the application instance
        self.path_selector = None  # To be set if this host is used for probing

    def send_packet(self, packet):
        # Send to the connected router
        router_id = list(self.topology.graph.neighbors(self.node_id))[0]
        if router_id in self.ports:
            self.ports[router_id].enqueue(packet)
        else:
            print(f"Error: Host {self.node_id} has no link to router {router_id}")

    def receive_packet(self, packet):
        # Check if this is a returning probe packet
        if hasattr(packet, 'is_probe') and packet.is_probe:
            # Calculate RTT
            rtt = self.env.now - packet.timestamp
            packet.rtt = rtt

            # Update path selector with probe result if available
            if self.path_selector and hasattr(self.path_selector, 'update_probe_result'):
                self.path_selector.update_probe_result(packet.probe_id, rtt)
            return

        # Regular packet - put in application queue
        self.in_queue.put(packet)