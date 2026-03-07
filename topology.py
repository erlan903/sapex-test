# topology.py
import json
import networkx as nx
from components import Host, Router
from beaconing import BeaconingProcess

class Topology:
    def __init__(self, env, filename):
        self.env = env
        self.graph = nx.Graph()
        self.nodes = {}
        self.hosts = {}
        self.core_ases = set()
        self.as_border_routers = {}
        self._load_from_json(filename)

    def _load_from_json(self, filename):
        with open(filename, 'r') as f:
            topo_data = json.load(f)

        for isd_as, as_details in topo_data.items():
            if as_details.get("core", False):
                self.core_ases.add(isd_as)

            self.as_border_routers[isd_as] = []

            # Add routers (border and internal)
            for router_id, router_details in as_details.get("border_routers", {}).items():
                router = Router(self.env, f"{isd_as}-{router_id}")
                self.nodes[router.node_id] = router
                self.graph.add_node(router.node_id, instance=router)
                self.as_border_routers[isd_as].append(router.node_id)

            # Add hosts
            for host_id, host_details in as_details.get("hosts", {}).items():
                full_host_id = f"{isd_as},{host_details['addr']}"
                host = Host(self.env, full_host_id, self)
                self.hosts[full_host_id] = host
                self.nodes[full_host_id] = host
                self.graph.add_node(full_host_id, instance=host)
                # Connect host to a router in the AS (simplified: connect to first border router)
                if as_details.get("border_routers"):
                    first_router_id = list(as_details["border_routers"].keys())[0]
                    self._add_link(host.node_id, f"{isd_as}-{first_router_id}", 1, 1000) # 1ms latency, 1Gbps


        # Add links between routers
        for isd_as, as_details in topo_data.items():
            for router_id, router_details in as_details.get("border_routers", {}).items():
                local_router_id = f"{isd_as}-{router_id}"
                for link in router_details.get("interfaces", []):
                    remote_isd_as = link["isd_as"]
                    remote_router_id_full = f"{remote_isd_as}-{link['neighbor_router']}"
                    
                    # Avoid duplicate link creation
                    if not self.graph.has_edge(local_router_id, remote_router_id_full):
                        self._add_link(local_router_id, remote_router_id_full, link['latency_ms'], link['bandwidth_mbps'])

        # Add synthetic intra-AS links between border routers so combined paths can
        # traverse inside an AS between different edge routers.
        intra_as_latency_ms = 1
        intra_as_bandwidth_mbps = 10000
        for isd_as, routers in self.as_border_routers.items():
            if len(routers) < 2:
                continue
            for i in range(len(routers)):
                for j in range(i + 1, len(routers)):
                    r1 = routers[i]
                    r2 = routers[j]
                    if not self.graph.has_edge(r1, r2):
                        self._add_link(r1, r2, intra_as_latency_ms, intra_as_bandwidth_mbps)

    def _add_link(self, from_node, to_node, latency, bandwidth):
        self.graph.add_edge(from_node, to_node, latency=latency, bandwidth=bandwidth)
        # Create directional links for both directions.
        # Reusing a single link object for both directions causes incorrect
        # forwarding because each Link has exactly one destination endpoint.
        self.nodes[from_node].add_link(to_node, latency, bandwidth, self.nodes[to_node])
        self.nodes[to_node].add_link(from_node, latency, bandwidth, self.nodes[from_node])


    def get_host(self, host_id):
        return self.hosts.get(host_id)
        
    def get_node(self, node_id):
        return self.nodes.get(node_id)

    def all_router_paths(self, from_router_id, to_router_id):
        """
        Return all simple router-to-router paths.
        This is used by beacon/path stitching to provide feasible compositions.
        Path scoring/ranking is intentionally handled by path-selection algorithms.
        """
        if from_router_id not in self.graph or to_router_id not in self.graph:
            return []

        router_nodes = [
            node_id for node_id, node in self.nodes.items()
            if isinstance(node, Router)
        ]
        router_subgraph = self.graph.subgraph(router_nodes)

        try:
            return list(
                nx.all_simple_paths(
                    router_subgraph,
                    source=from_router_id,
                    target=to_router_id,
                    cutoff=max(1, len(router_nodes) - 1),
                )
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def initiate_beaconing(self, path_selection_algorithm):
        """Start beaconing processes from core ASes and set up path registration for all routers"""
        beaconing_processes = {}

        # Create beaconing processes for all core-AS border routers.
        # This is required in topologies where different core routers connect to
        # different peers/leaf ASes.
        for isd_as in self.core_ases:
            for router_id in self.as_border_routers.get(isd_as, []):
                start_router = self.get_node(router_id)
                if not start_router:
                    continue

                beaconing_process = BeaconingProcess(
                    self.env,
                    start_router,
                    path_selection_algorithm,
                    interval=1000,
                    topology=self
                )
                beaconing_processes[f"{isd_as}:{router_id}"] = beaconing_process
                self.env.process(beaconing_process.start())

        # Set beaconing process reference on all routers for path registration
        # All routers need access to a beaconing process to register discovered paths
        # For simplicity, we'll use the first core AS's beaconing process for all routers
        if beaconing_processes:
            primary_beaconing_process = list(beaconing_processes.values())[0]
            for node_id, node in self.nodes.items():
                if hasattr(node, 'set_beaconing_process'):
                    node.set_beaconing_process(primary_beaconing_process)