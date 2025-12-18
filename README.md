# SCION Path Selection Simulation Framework

A discrete event simulation framework for evaluating SCION path selection algorithms with realistic beaconing-based path discovery.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Implementing Custom Algorithms](#implementing-custom-algorithms)
- [Output and Metrics](#output-and-metrics)
- [Examples](#examples)
- [Technical Details](#technical-details)

## Overview

This simulation framework models the SCION (Scalability, Control, and Isolation On Next-generation networks) architecture, focusing on:

- **Path Construction Beaconing (PCB)**: Realistic SCION beaconing protocol for path discovery
- **AS-level Topology**: Inter-domain routing with core and non-core ASes
- **Path Selection**: Pluggable algorithms for selecting paths between ASes
- **Traffic Simulation**: Configurable traffic flows with realistic network delays
- **Performance Metrics**: Packet loss, latency, and throughput measurements

The framework uses [SimPy](https://simpy.readthedocs.io/) for discrete event simulation, providing accurate timing and resource modeling.

## Architecture

### Core Components

```

Simulation Engine                         
    (simulation.py)                          

   Topology       Applications 
 (topology.py) (application.py)

Nodes  Beaconing 
        Process  

       Path Selection 
         Algorithm    
```

### File Structure

- **main.py**: Entry point for running simulations
- **simulation.py**: Orchestrates the simulation lifecycle
- **topology.py**: Loads and manages network topology from JSON
- **components.py**: Defines network components (Routers, Hosts, Links)
- **packet.py**: Packet structures including BeaconPacket with HopInfo
- **beaconing.py**: SCION beaconing protocol implementation
- **path_selection.py**: Abstract interface and example algorithms
- **application.py**: Traffic generation and measurement

## Installation

### Requirements

- Python 3.8+
- SimPy
- NetworkX

### Setup

```bash
# Clone or navigate to the repository
cd sapexf-simulation

# Install dependencies
pip install -r requirements.txt
```

The `requirements.txt` contains:
```
simpy
networkx
```

## Quick Start

Run the simulation with default configuration:

```bash
python main.py
```

Run with custom topology and traffic files:

```bash
python main.py --topology my_topology.json --traffic my_traffic.json
```

### Expected Output

```
Path discovery via beaconing is enabled. Paths will be discovered through beacon propagation.
Starting beaconing process...
[0.00] Core AS 1-ff00:0:110 (router 1-ff00:0:110-br1-110-1) sending beacon.
[10.00] AS 1-ff00:0:111 registered path from 1-ff00:0:110: ['1-ff00:0:110-br1-110-1', '1-ff00:0:111-br1-111-1']
[20.00] AS 1-ff00:0:112 registered path from 1-ff00:0:110: ['1-ff00:0:110-br1-110-1', '1-ff00:0:112-br1-112-1']
[20.00] Created combined path 1-ff00:0:111 -> 1-ff00:0:112: ['1-ff00:0:111-br1-111-1', '1-ff00:0:110-br1-110-1', '1-ff00:0:112-br1-112-1']

All available paths discovered:
  Paths from 1-ff00:0:110 to 1-ff00:0:111:
    1: 1-ff00:0:110-br1-110-1 -> 1-ff00:0:111-br1-111-1
  ...

--- Simulation Results ---
Total Packets Sent: 3414
Total Packets Received: 3414
Total Packets Lost: 0
Packet Loss Rate: 0.00%
Average Packet Latency: 30.12ms
```

## How It Works

### 1. SCION Beaconing Protocol

The simulation implements the SCION path discovery mechanism:

#### Beacon Origination (Core ASes)

Core ASes periodically send Path Construction Beacons (PCBs) to their neighbors:

```python
# From beaconing.py
beacon = BeaconPacket(
    origin_router_id=self.start_router.node_id,
    origin_as_id=self.origin_as
)
# Send to all neighbors every 1000ms (configurable)
```

#### Beacon Propagation

When a router receives a beacon:

1. **AS-level Loop Detection**: Check if the current AS is already in the beacon's path
2. **Hop Information Addition**: Add AS ID, router ID, interfaces, and link metrics
3. **Path Registration**: Register the discovered path in the local path store
4. **Forwarding**: Forward the beacon to neighbors (excluding loops)

```python
# From components.py - Router.receive_packet()
if packet.is_beacon:
    as_path = packet.get_as_path()
    if current_as in as_path:
        return  # Drop beacon (loop detected)

    # Add this hop to the beacon
    packet.add_hop(as_id, router_id, ingress_if, link_metrics)

    # Register the path
    beaconing_process.register_path(packet, self.node_id)

    # Forward to neighbors
    for neighbor_id, link in self.ports.items():
        if neighbor_id not in packet.path:
            link.enqueue(packet.clone())
```

#### Path Combination

For inter-leaf communication, the framework automatically combines:
- **Up-segments**: Non-core AS � Core AS
- **Down-segments**: Core AS � Non-core AS

Example: To reach AS 112 from AS 111:
```
AS 111 � AS 110 (core) � AS 112
[up-segment]  [down-segment]
```

### 2. Path Selection

Applications query the path selection algorithm for paths between source and destination ASes:

```python
# From application.py
path = self.path_selector.select_path(self.source.isd_as, self.destination.isd_as)
```

The `ShortestPathAlgorithm` selects the path with minimum hop count, but you can implement custom algorithms.

### 3. Packet Forwarding

Data packets follow the selected path hop-by-hop:

```python
# From components.py - Router.receive_packet()
current_hop_index = packet.path.index(self.node_id)
next_hop = packet.path[current_hop_index + 1]
self.ports[next_hop].enqueue(packet)
```

### 4. Network Delays

The simulation models realistic network delays:

- **Propagation Delay**: Based on link latency (e.g., 10ms)
- **Transmission Delay**: Based on packet size and bandwidth
- **Processing Delay**: Implicit in event scheduling

```python
# From components.py - Link.run()
yield self.env.timeout(self.latency)  # Propagation delay
transmission_delay = (packet.size * 8) / (self.bandwidth * 1000)
yield self.env.timeout(transmission_delay)  # Transmission delay
```

## Configuration

### Topology Configuration (topology.json)

Define the network structure with ASes, routers, hosts, and links:

```json
{
  "1-ff00:0:110": {
    "core": true,
    "border_routers": {
      "br1-110-1": {
        "interfaces": [
          {
            "isd_as": "1-ff00:0:111",
            "neighbor_router": "br1-111-1",
            "latency_ms": 10,
            "bandwidth_mbps": 100
          }
        ]
      }
    },
    "hosts": {
      "server1": {
        "addr": "192.168.0.1"
      }
    }
  }
}
```

#### Topology Format

- **AS Identifier**: `ISD-ASff00:ASN` format (e.g., `1-ff00:0:110`)
- **Core Flag**: `"core": true` for core ASes, `false` for non-core
- **Border Routers**: Dictionary of router names to interface configurations
- **Interfaces**: Define links to neighboring ASes with latency and bandwidth
- **Hosts**: End hosts within the AS with IP addresses

### Traffic Configuration (traffic.json)

Define traffic flows between hosts:

```json
{
  "duration_ms": 10000,
  "flows": [
    {
      "name": "WebServerTraffic",
      "source": "1-ff00:0:111,10.0.0.5",
      "destination": "1-ff00:0:112,172.16.5.5",
      "start_time_ms": 1000,
      "data_size_kb": 5000
    }
  ]
}
```

#### Traffic Format

- **duration_ms**: Total simulation time in milliseconds
- **flows**: List of traffic flows
  - **name**: Flow identifier for logging
  - **source**: Source host in format `AS,IP`
  - **destination**: Destination host in format `AS,IP`
  - **start_time_ms**: When the flow starts (ms)
  - **data_size_kb**: Total data to transfer (KB)

## Implementing Custom Algorithms

Create your own path selection algorithm by extending `PathSelectionAlgorithm`:

### Step 1: Create Your Algorithm Class

```python
# my_algorithm.py
from path_selection import PathSelectionAlgorithm

class MyCustomAlgorithm(PathSelectionAlgorithm):
    def __init__(self, topology, use_beaconing=True):
        super().__init__(topology)
        self.use_beaconing = use_beaconing
        self.discover_paths(use_graph_traversal=not use_beaconing)
        # Add custom initialization here

    def select_path(self, source_as, destination_as):
        """
        Select the best path from source to destination AS.

        Args:
            source_as: Source AS identifier (e.g., "1-ff00:0:111")
            destination_as: Destination AS identifier

        Returns:
            List of router IDs representing the path, or None if no path exists
        """
        available_paths = self.path_store.get((source_as, destination_as), [])
        if not available_paths:
            return None

        # Example: Select path with lowest total latency
        best_path = None
        min_latency = float('inf')

        for path in available_paths:
            total_latency = self._calculate_path_latency(path)
            if total_latency < min_latency:
                min_latency = total_latency
                best_path = path

        return best_path

    def _calculate_path_latency(self, path):
        """Calculate total latency for a path"""
        total = 0
        for i in range(len(path) - 1):
            current_router = path[i]
            next_router = path[i + 1]
            # Get link latency from topology
            if self.topology.graph.has_edge(current_router, next_router):
                latency = self.topology.graph[current_router][next_router].get('latency', 0)
                total += latency
        return total
```

### Step 2: Use Your Algorithm

Modify `main.py`:

```python
from my_algorithm import MyCustomAlgorithm

# Replace ShortestPathAlgorithm with your algorithm
sim = Simulation(args.topology, args.traffic, MyCustomAlgorithm)
```

### Available Information

Your algorithm has access to:

- **self.path_store**: Dictionary of discovered paths
  - Key: `(source_as, destination_as)` tuple
  - Value: List of paths (each path is a list of router IDs)

- **self.topology**: The network topology with:
  - `topology.graph`: NetworkX graph with routers and links
  - `topology.nodes`: Dictionary of all network nodes
  - Link attributes: `latency`, `bandwidth`

### Algorithm Guidelines

1. **Use Beaconing**: Set `use_beaconing=True` to rely on realistic path discovery
2. **Handle Missing Paths**: Return `None` if no path exists
3. **Return Router-level Path**: Path must be a list of router IDs in order
4. **Consider Metrics**: Access link metrics from beacon `HopInfo` objects
5. **Stateful Algorithms**: Store state in instance variables for adaptive selection

## Output and Metrics

### Console Output

The simulation provides detailed logging:

1. **Beaconing Phase** (0-2000ms by default):
   - Beacon transmission events
   - Path registration messages
   - Combined path creation

2. **Path Discovery Summary**:
   - All discovered paths between AS pairs
   - Router-level paths for data forwarding

3. **Traffic Phase**:
   - Application start events
   - Path selection results
   - Packet transmission/reception (if verbose)

4. **Results Summary**:
   - Total packets sent/received/lost
   - Packet loss rate (%)
   - Average end-to-end latency (ms)

### Accessing Detailed Results

Modify `simulation.py` to capture additional metrics:

```python
# In Simulation class
def print_results(self):
    print("\n--- Simulation Results ---")
    # Access detailed latency data
    latencies = self.results['latencies']
    if latencies:
        print(f"Min Latency: {min(latencies):.2f}ms")
        print(f"Max Latency: {max(latencies):.2f}ms")
        print(f"Median Latency: {sorted(latencies)[len(latencies)//2]:.2f}ms")
```

## Examples

### Example 1: Simple Star Topology

The default `topology.json` creates a star topology:

```
        AS 111 (client1)
             |
             | 10ms, 100Mbps
             |
        AS 110 (core)
             |
             | 20ms, 50Mbps
             |
        AS 112 (client2)
```

Traffic flows from AS 111 to AS 112 through the core AS 110.

### Example 2: Adding a New AS

Add a new AS to `topology.json`:

```json
"1-ff00:0:113": {
  "core": false,
  "border_routers": {
    "br1-113-1": {
      "interfaces": [
        {
          "isd_as": "1-ff00:0:110",
          "neighbor_router": "br1-110-1",
          "latency_ms": 15,
          "bandwidth_mbps": 75
        }
      ]
    }
  },
  "hosts": {
    "client3": {
      "addr": "10.0.0.10"
    }
  }
}
```

Update the core AS to include the new link:

```json
"1-ff00:0:110": {
  "core": true,
  "border_routers": {
    "br1-110-1": {
      "interfaces": [
        // ... existing interfaces ...
        {
          "isd_as": "1-ff00:0:113",
          "neighbor_router": "br1-113-1",
          "latency_ms": 15,
          "bandwidth_mbps": 75
        }
      ]
    }
  }
}
```

### Example 3: Latency-Based Path Selection

Create a latency-optimized algorithm:

```python
class LatencyOptimizedAlgorithm(PathSelectionAlgorithm):
    def select_path(self, source_as, destination_as):
        available_paths = self.path_store.get((source_as, destination_as), [])
        if not available_paths:
            return None

        # Calculate total latency for each path
        path_latencies = []
        for path in available_paths:
            total_latency = 0
            for i in range(len(path) - 1):
                if self.topology.graph.has_edge(path[i], path[i+1]):
                    total_latency += self.topology.graph[path[i]][path[i+1]]['latency']
            path_latencies.append((path, total_latency))

        # Return path with minimum latency
        return min(path_latencies, key=lambda x: x[1])[0]
```

### Example 4: Multiple Traffic Flows

Configure concurrent flows in `traffic.json`:

```json
{
  "duration_ms": 10000,
  "flows": [
    {
      "name": "Flow1",
      "source": "1-ff00:0:111,10.0.0.5",
      "destination": "1-ff00:0:112,172.16.5.5",
      "start_time_ms": 1000,
      "data_size_kb": 5000
    },
    {
      "name": "Flow2",
      "source": "1-ff00:0:112,172.16.5.5",
      "destination": "1-ff00:0:111,10.0.0.5",
      "start_time_ms": 2000,
      "data_size_kb": 3000
    }
  ]
}
```

## Technical Details

### Packet Structure

#### BeaconPacket

```python
class BeaconPacket:
    origin_as_id: str          # Originating AS
    timestamp: float           # Creation time
    hops: List[HopInfo]        # AS-level path information
    segment_type: str          # "down", "core", or "up"
    path: List[str]            # Router-level path (for forwarding)
```

#### HopInfo

```python
class HopInfo:
    as_id: str                 # AS identifier at this hop
    router_id: str             # Border router identifier
    ingress_if: str            # Ingress interface
    egress_if: str             # Egress interface (optional)
    link_metrics: dict         # {"latency": ms, "bandwidth": bytes/ms}
```

### Beaconing Parameters

Modify beaconing behavior in `topology.py`:

```python
BeaconingProcess(
    env,
    start_router,
    path_selection_algorithm,
    interval=1000,  # Beacon interval in ms (default: 1000)
    topology=self
)
```

### Simulation Timing

Default timing in `simulation.py`:

- **Beaconing Phase**: 0-2000ms (configurable in `run()` method)
- **Traffic Phase**: Starts after beaconing, duration from `traffic.json`
- **Beacon Interval**: 1000ms (1 second)

Adjust in `simulation.py`:

```python
def run(self):
    self.topology.initiate_beaconing(self.path_selection_algorithm)
    yield self.env.timeout(5000)  # Wait 5 seconds for path discovery
    # ...
```

### Path Store Format

The `path_store` dictionary structure:

```python
{
    ("1-ff00:0:111", "1-ff00:0:112"): [
        ["1-ff00:0:111-br1-111-1", "1-ff00:0:110-br1-110-1", "1-ff00:0:112-br1-112-1"],
        # Additional paths if available
    ],
    # More AS pairs...
}
```

### Disabling Beaconing (for testing)

To use graph-based path discovery instead of beaconing:

```python
# In main.py or when creating the algorithm
algorithm = ShortestPathAlgorithm(topology, use_beaconing=False)
```

This bypasses beaconing and uses NetworkX to find all paths, useful for:
- Comparing beaconing vs. omniscient path knowledge
- Debugging path selection algorithms
- Baseline performance measurements

## Performance Considerations

### Simulation Speed

- **Beaconing Overhead**: Beacon propagation is realistic but adds events
- **Large Topologies**: O(N�) path combinations in dense networks
- **Packet-Level Simulation**: Each packet is an event

### Optimization Tips

1. **Reduce Beacon Frequency**: Increase interval for faster simulation
2. **Limit Path Discovery Window**: Reduce timeout in `simulation.py`
3. **Use Graph Traversal**: Set `use_beaconing=False` for small tests
4. **Batch Analysis**: Run multiple simulations in parallel

## Troubleshooting

### No Paths Found

**Symptom**: `App: No path found. Stopping.`

**Causes**:
- Beaconing hasn't completed (increase timeout)
- Topology connectivity issue (check link definitions)
- Missing reverse links in topology

**Solution**:
```python
# In simulation.py, increase beaconing time
yield self.env.timeout(5000)  # From 2000 to 5000
```

### Path Registration Not Working

**Symptom**: Empty path store after beaconing

**Debug**:
1. Check beacon sending: Verify core AS has `"core": true`
2. Verify router connections: Ensure bidirectional links
3. Enable debug logging in `beaconing.py` and `components.py`

### Packets Not Forwarding

**Symptom**: Packets sent but not received

**Causes**:
- Path format mismatch (wrong router IDs)
- Host not connected to router
- Missing forwarding logic for destination host

## Dynamic Path Failure Simulation

The framework supports **dynamic path failure events** that can be scheduled at specific simulation times. This enables realistic testing of path selection algorithms under failure conditions.

### Features

- **Configuration-based Events**: Define path down/up events in traffic.json
- **Application Notifications**: Applications are notified when their paths fail
- **Automatic Failover**: Applications can re-select alternative paths
- **Path Recovery**: Paths can be restored at scheduled times
- **Path-level Granularity**: Mark specific router sequences as unavailable

### Quick Example

Add events to your `traffic.json`:

```json
{
  "duration_ms": 10000,
  "flows": [
    {
      "name": "WebServerTraffic",
      "source": "1-ff00:0:111,10.0.0.5",
      "destination": "1-ff00:0:112,172.16.5.5",
      "start_time_ms": 1000,
      "data_size_kb": 5000
    }
  ],
  "events": [
    {
      "type": "path_down",
      "time_ms": 3000,
      "path": ["1-ff00:0:111-br1-111-1", "1-ff00:0:110-br1-110-1", "1-ff00:0:112-br1-112-1"],
      "description": "Primary path failure - simulating link congestion"
    },
    {
      "type": "path_up",
      "time_ms": 7000,
      "path": ["1-ff00:0:111-br1-111-1", "1-ff00:0:110-br1-110-1", "1-ff00:0:112-br1-112-1"],
      "description": "Primary path recovery"
    }
  ]
}
```

### Event Configuration

#### Event Types

- **`path_down`**: Marks a path as unavailable. Path selection algorithms will filter out this path.
- **`path_up`**: Restores a previously failed path, making it available for selection again.

#### Event Fields

- **`type`** (required): Event type - either `"path_down"` or `"path_up"`
- **`time_ms`** (required): Absolute simulation time in milliseconds when the event occurs
- **`path`** (required): Array of router IDs representing the path (must match beaconing format)
- **`description`** (optional): Human-readable description for logging

#### Path Format

Paths must be specified as complete router sequences in the format `"ISD-AS-router"`:

```json
["1-ff00:0:111-br1-111-1", "1-ff00:0:110-br1-110-1", "1-ff00:0:112-br1-112-1"]
```

To find valid paths, run the simulation first and check the "All available paths discovered" output.

### How It Works

#### 1. Event Loading

Events are loaded from `traffic.json` during simulation initialization:

```python
# From simulation.py
self.event_manager = EventManager(
    self.env,
    self.path_selection_algorithm,
    self.app_registry
)
self.event_manager.load_events(scenario)
```

#### 2. Event Scheduling

Events are scheduled as SimPy processes and execute at their designated times:

```python
# From events.py
def schedule_events(self):
    for event in self.events:
        yield self.env.timeout(event['time_ms'] - self.env.now)
        if event['type'] == 'path_down':
            self._execute_path_down(event)
        elif event['type'] == 'path_up':
            self._execute_path_up(event)
```

#### 3. Path Marking

When a `path_down` event occurs:

1. The path is marked unavailable in the path selection algorithm
2. The path selector identifies affected AS pairs
3. Applications using the path are notified via callbacks

```python
# From path_selection.py
def mark_path_down(self, router_path):
    """Mark a specific path as unavailable due to failure."""
    # Store in unavailable_paths dictionary
    # Return list of affected (src_as, dst_as) pairs
```

#### 4. Application Notification

Applications receive a callback when their current path goes down:

```python
# From application.py
def on_path_down(self, router_path):
    """Callback invoked when the current path goes down."""
    print(f"Path down notification received")
    self.is_path_down = True
    self._attempt_path_reselection()
```

#### 5. Path Re-selection

Applications automatically attempt to find alternative paths:

```python
def _attempt_path_reselection(self):
    """Attempt to select a new path after failure."""
    new_path = self.path_selector.select_path(
        self.source.isd_as,
        self.destination.isd_as
    )

    if new_path:
        print(f"Switched to new path: {' -> '.join(new_path)}")
        self.current_path = new_path
        self.is_path_down = False
    else:
        print(f"No alternative path available")
```

### Architecture Components

The path failure system consists of three main components:

#### ApplicationRegistry (app_registry.py)

Tracks which applications are using which paths:

```python
class ApplicationRegistry:
    def register_path_usage(self, application, router_path)
    def notify_path_down(self, router_path, affected_as_pairs)
    def notify_path_up(self, router_path, affected_as_pairs)
```

#### EventManager (events.py)

Manages scheduled path failure events:

```python
class EventManager:
    def load_events(self, config_dict)
    def schedule_events(self)  # SimPy generator
    def _execute_path_down(self, event)
    def _execute_path_up(self, event)
```

#### Path Availability Methods (path_selection.py)

Added to the `PathSelectionAlgorithm` base class:

```python
def mark_path_down(self, router_path)      # Mark path unavailable
def mark_path_up(self, router_path)        # Restore path
def is_path_available(self, router_path)   # Check availability
```

### Example Output

When a path failure event occurs, you'll see output like:

```
[3000.00] EVENT: Path down - ['1-ff00:0:111-br1-111-1', '1-ff00:0:110-br1-110-1', '1-ff00:0:112-br1-112-1']
[3000.00]   Description: Primary path failure - simulating link congestion
Marked path DOWN: ['1-ff00:0:111-br1-111-1', '1-ff00:0:110-br1-110-1', '1-ff00:0:112-br1-112-1']
Affected AS pairs: [('1-ff00:0:111', '1-ff00:0:112')]
[3000.00]   Affected AS pairs: [('1-ff00:0:111', '1-ff00:0:112')]
  Notifying 1 application(s) using this path
[3000.00] App App-WebServerTraffic: Path down notification received
[3000.00] App App-WebServerTraffic: Switched to new path: 1-ff00:0:111-br1-111-1 -> 1-ff00:0:112-br1-112-1
```

If a path recovery event occurs:

```
[7000.00] EVENT: Path up - ['1-ff00:0:111-br1-111-1', '1-ff00:0:110-br1-110-1', '1-ff00:0:112-br1-112-1']
[7000.00]   Description: Primary path recovery
Marked path UP: ['1-ff00:0:111-br1-111-1', '1-ff00:0:110-br1-110-1', '1-ff00:0:112-br1-112-1']
Affected AS pairs: [('1-ff00:0:111', '1-ff00:0:112')]
```

### Testing Scenarios

#### Scenario 1: Single Path Failure with Alternative

```json
{
  "events": [
    {
      "type": "path_down",
      "time_ms": 3000,
      "path": ["AS111-router1", "AS110-core", "AS112-router1"],
      "description": "Primary path fails"
    }
  ]
}
```

Application switches to alternative path if available.

#### Scenario 2: Cascading Failures

```json
{
  "events": [
    {
      "type": "path_down",
      "time_ms": 3000,
      "path": ["AS111-router1", "AS110-core1", "AS112-router1"]
    },
    {
      "type": "path_down",
      "time_ms": 5000,
      "path": ["AS111-router1", "AS110-core2", "AS112-router1"]
    }
  ]
}
```

Tests behavior when all paths fail.

#### Scenario 3: Failure and Recovery Cycle

```json
{
  "events": [
    {
      "type": "path_down",
      "time_ms": 3000,
      "path": ["AS111-router1", "AS110-core", "AS112-router1"]
    },
    {
      "type": "path_up",
      "time_ms": 7000,
      "path": ["AS111-router1", "AS110-core", "AS112-router1"]
    }
  ]
}
```

Path becomes available again after recovery.

#### Scenario 4: Multiple Concurrent Flows

```json
{
  "flows": [
    {"name": "Flow1", "source": "AS111,host1", "destination": "AS112,host2", "start_time_ms": 1000},
    {"name": "Flow2", "source": "AS111,host1", "destination": "AS113,host3", "start_time_ms": 1000}
  ],
  "events": [
    {
      "type": "path_down",
      "time_ms": 4000,
      "path": ["AS111-router", "AS110-core", "AS112-router"],
      "description": "Affects only Flow1"
    }
  ]
}
```

Only affected applications are notified.

### Edge Cases

#### No Alternative Paths

When all paths fail and no alternatives exist:

```
[3000.00] App App-WebServerTraffic: Path down notification received
[3000.00] App App-WebServerTraffic: No alternative path available
```

The application continues to retry every 10ms until a path becomes available or the simulation ends.

#### Event Before Application Start

If an event occurs before traffic starts:
- The path is marked down in the path selector
- Applications started later will not select the failed path
- No notifications are sent (no apps running yet)

#### Path Recovery Without Failure

If a `path_up` event targets a path that wasn't down:
- The event is processed normally
- No state changes occur
- No errors are raised

### Custom Algorithm Integration

Path selection algorithms automatically respect path availability:

```python
class MyAlgorithm(PathSelectionAlgorithm):
    def select_path(self, source_as, destination_as):
        paths = self.path_store.get((source_as, destination_as), [])

        # Filter unavailable paths automatically
        available = [p for p in paths if self.is_path_available(p)]

        # Your selection logic here
        return my_selection_logic(available)
```

Both `ShortestPathAlgorithm` and `SapexAlgorithm` already filter unavailable paths.

### Performance Metrics with Failures

Track how failures affect performance:

```python
# In simulation.py print_results()
def print_results(self):
    print("\n--- Simulation Results ---")
    print(f"Total Packets Sent: {total_sent}")
    print(f"Total Packets Lost: {total_lost}")
    print(f"Packet Loss Rate: {loss_rate:.2f}%")
    print(f"Average Latency: {avg_latency:.2f}ms")

    # Analyze impact of failures
    if total_lost > 0:
        print(f"\nPackets lost during path failures")
```

### Best Practices

1. **Event Timing**: Schedule events after beaconing completes (after 2000ms)
2. **Path Validation**: Run simulation once to see discovered paths before adding events
3. **Recovery Timing**: Allow sufficient time between failure and recovery
4. **Multiple Events**: Space events apart to observe individual effects
5. **Application Start Time**: Ensure apps start before failure events if testing failover

## Contributing

To extend the framework:

1. **New Metrics**: Modify `application.py` to track additional statistics
2. **Complex Topologies**: Add multi-core or hierarchical AS structures
3. **Advanced Beaconing**: Implement beacon filtering, expiration, updates
4. **QoS Features**: Add bandwidth reservation or priority queuing
5. **Failure Simulation**: Extend event types (link-level, router-level, AS-level failures)
6. **Automatic Failure Detection**: Trigger path_down based on loss/latency thresholds

## References

- [SCION Architecture](https://www.scion-architecture.net/)
- [SimPy Documentation](https://simpy.readthedocs.io/)
- [NetworkX Documentation](https://networkx.org/)

## License

This simulation framework is provided as-is for research and educational purposes.
