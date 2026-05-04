import json
import random
import numpy as np
import networkx as nx
from PIL import Image

NSEMPTY = -1


# ----------------------------
# Single-road NaSch
# ----------------------------

def step_road(road, vmax=5, p=0.3, circular=True):
    n = len(road)
    speeds = road[:]

    for i, v in enumerate(road):
        if v == NSEMPTY:
            continue

        v = min(v + 1, vmax)

        gap = 0
        for d in range(1, vmax + 1):
            j = i + d
            if circular:
                j %= n
            elif j >= n:
                break

            if road[j] == NSEMPTY:
                gap += 1
            else:
                break

        v = min(v, gap)

        if random.random() < p:
            v = max(v - 1, 0)

        speeds[i] = v

    new = [NSEMPTY] * n

    for i, v in enumerate(speeds):
        if v == NSEMPTY:
            continue

        j = i + v

        if circular:
            j %= n
            new[j] = v
        else:
            if j < n:
                new[j] = v

    return new


# ----------------------------
# Graph-road representation
# ----------------------------

class GraphRoadNetwork:
    def __init__(self, nodes, edges, edge_lengths, in_nodes=None, out_nodes=None):
        self.nodes = nodes
        self.edges = edges                  # [(u, v), ...]
        self.edge_lengths = edge_lengths    # number of cells per edge

        self.node_to_idx = {n: i for i, n in enumerate(nodes)}
        self.edge_to_idx = {e: i for i, e in enumerate(edges)}

        self.in_nodes = set(in_nodes or [])
        self.out_nodes = set(out_nodes or [])

        self.out_edges = {n: [] for n in nodes}
        self.in_edges = {n: [] for n in nodes}

        for ei, (u, v) in enumerate(edges):
            self.out_edges[u].append(ei)
            self.in_edges[v].append(ei)

        self.roads = [
            [NSEMPTY] * L for L in edge_lengths
        ]

    @classmethod
    def from_networkx(cls, G, inout=None, cell_length_m=7.0):
        nodes = list(G.nodes())
        edges = []
        lengths = []

        for u, v, data in G.edges(data=True):
            edges.append((u, v))

            if "length" in data:
                length_m = float(data["length"])
            else:
                x1 = float(G.nodes[u]["x"])
                y1 = float(G.nodes[u]["y"])
                x2 = float(G.nodes[v]["x"])
                y2 = float(G.nodes[v]["y"])
                length_m = haversine_m(y1, x1, y2, x2)

            cells = max(1, int(round(length_m / cell_length_m)))
            lengths.append(cells)

        in_nodes = inout.get("in", []) if inout else []
        out_nodes = inout.get("out", []) if inout else []

        return cls(nodes, edges, lengths, in_nodes, out_nodes)

    def insert_car(self, edge_idx, speed=0):
        road = self.roads[edge_idx]
        if road[0] == NSEMPTY:
            road[0] = speed
            return True
        return False

    def random_inject(self, rate=0.2):
        for node in self.in_nodes:
            for edge_idx in self.out_edges.get(node, []):
                if random.random() < rate:
                    self.insert_car(edge_idx, speed=0)

    def choose_next_edge(self, node):
        choices = self.out_edges.get(node, [])
        if not choices:
            return None
        return random.choice(choices)

    def step(self, vmax=5, p=0.3, injection_rate=0.0):
        new_roads = [
            [NSEMPTY] * len(road) for road in self.roads
        ]

        # First update speeds edge-by-edge.
        speed_roads = [
            update_edge_speeds(road, vmax, p)
            for road in self.roads
        ]

        # Then move cars.
        for ei, road in enumerate(speed_roads):
            u, v_node = self.edges[ei]
            L = len(road)

            for i, speed in enumerate(road):
                if speed == NSEMPTY:
                    continue

                j = i + speed

                if j < L:
                    if new_roads[ei][j] == NSEMPTY:
                        new_roads[ei][j] = speed
                    continue

                # Car reaches junction.
                overflow = j - L
                next_edge = self.choose_next_edge(v_node)

                # Leaving the system.
                if v_node in self.out_nodes or next_edge is None:
                    continue

                next_road = new_roads[next_edge]

                if overflow < len(next_road) and next_road[overflow] == NSEMPTY:
                    next_road[overflow] = speed
                else:
                    # Junction blocked: keep car at end of current edge.
                    new_roads[ei][L - 1] = 0

        self.roads = new_roads

        if injection_rate > 0:
            self.random_inject(injection_rate)


def update_edge_speeds(road, vmax=5, p=0.3):
    L = len(road)
    speeds = road[:]

    for i, v in enumerate(road):
        if v == NSEMPTY:
            continue

        v = min(v + 1, vmax)

        gap = 0
        for d in range(1, vmax + 1):
            j = i + d
            if j >= L:
                gap += 1
                continue

            if road[j] == NSEMPTY:
                gap += 1
            else:
                break

        v = min(v, gap)

        if random.random() < p:
            v = max(v - 1, 0)

        speeds[i] = v

    return speeds


# ----------------------------
# Geometry helper
# ----------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * R * np.arcsin(np.sqrt(a))


# ----------------------------
# Visualization for single road
# ----------------------------

def color(v):
    if v == NSEMPTY:
        return (255, 255, 255)
    if v == 0:
        return (255, 0, 0)
    if v in (1, 2):
        return (255, 128, 0)
    if v in (3, 4):
        return (255, 255, 0)
    if v >= 5:
        return (0, 255, 0)
    return (0, 0, 0)


def simulate_road(highway, iters=500, vmax=5, p=0.3, output="nasch.png"):
    road = highway[:]
    img = Image.new("RGB", (len(road), iters), (255, 255, 255))

    for t in range(iters):
        for x, v in enumerate(road):
            img.putpixel((x, t), color(v))

        road = step_road(road, vmax=vmax, p=p, circular=True)

    img.save(output)
    print(f"Saved {output}")


# ----------------------------
# Graph loading
# ----------------------------

def load_graph_network(gexf_path="erd.gexf", inout_path="inout.json"):
    with open(inout_path) as f:
        inout = json.load(f)

    G = nx.read_gexf(gexf_path)

    network = GraphRoadNetwork.from_networkx(
        G,
        inout=inout,
        cell_length_m=7.0,
    )

    return network


def simulate_graph(
    gexf_path="erd.gexf",
    inout_path="inout.json",
    iters=500,
    vmax=5,
    p=0.3,
    injection_rate=0.2,
):
    network = load_graph_network(gexf_path, inout_path)

    for _ in range(iters):
        network.step(
            vmax=vmax,
            p=p,
            injection_rate=injection_rate,
        )

    return network

def populate_random(network, density=0.2, vmax=5, speed_mode="random"):
    """
    density: probability that each cell starts with a car
    speed_mode:
        "zero"   -> all cars start stopped
        "random" -> random speed from 0..vmax
        "vmax"   -> all cars start at vmax
    """
    for road in network.roads:
        for i in range(len(road)):
            if random.random() < density:
                if speed_mode == "zero":
                    road[i] = 0
                elif speed_mode == "vmax":
                    road[i] = vmax
                else:
                    road[i] = random.randint(0, vmax)
            else:
                road[i] = NSEMPTY
                
# ----------------------------
# Main examples
# ----------------------------

if __name__ == "__main__":
    mode = "graph"  # "road" or "graph"

    if mode == "road":
        highway = [
            -1, -1, -1, -1, -1, 2, 3, -1, -1, 4,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, 2, 3, -1, -1, 4, -1, -1, -1, -1, -1, -1,
        ] * 8

        simulate_road(
            highway,
            iters=500,
            vmax=5,
            p=0.3,
            output="nasch.png",
        )

    elif mode == "graph":
        network = simulate_graph(
            gexf_path="erd.gexf",
            inout_path="inout.json",
            iters=500,
            vmax=5,
            p=0.3,
            injection_rate=0.2,
        )

        print("Simulation finished.")
        print("Number of edges:", len(network.edges))
        print("Number of road segments:", len(network.roads))