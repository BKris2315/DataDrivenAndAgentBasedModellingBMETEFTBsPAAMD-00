import os
import json
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import numpy as np
from PIL import Image

NSEMPTY = -1


# ============================================================
# Single-road NaSch model
# ============================================================

def step_road(road, vmax=5, p=0.3, circular=True):
    """
    Original one-dimensional NaSch road.
    road[cell] = -1 for empty, otherwise speed 0..vmax.
    """
    n = len(road)
    speeds = road[:]

    for i, v in enumerate(road):
        if v == NSEMPTY:
            continue

        # 1. Acceleration
        v = min(v + 1, vmax)

        # 2. Slowing down due to gap
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

        # 3. Random braking
        if random.random() < p:
            v = max(v - 1, 0)

        speeds[i] = v

    # 4. Movement
    new = [NSEMPTY] * n

    for i, v in enumerate(speeds):
        if v == NSEMPTY:
            continue

        j = i + v

        if circular:
            j %= n
            new[j] = v
        elif j < n:
            new[j] = v

    return new


# ============================================================
# Car representation for graph mode
# ============================================================

@dataclass
class Car:
    car_id: int
    speed: int
    origin: str
    destination: str
    path: list
    path_pos: int = 0


@dataclass
class JunctionRequest:
    car_id: int
    from_edge: int
    to_edge: int | None
    arrival_node: str
    target_cell: int
    old_cell: int


# ============================================================
# Graph-road NaSch model
# ============================================================

class GraphRoadNetwork:
    """
    Graph-based NaSch traffic model.

    Important representation change:
        roads[edge_idx][cell_idx] = -1      empty cell
        roads[edge_idx][cell_idx] = car_id  occupied cell

    Speeds and destinations live in self.cars[car_id].
    """

    def __init__(
        self,
        nodes,
        edges,
        edge_lengths,
        G=None,
        route_graph=None,
        pos=None,
        edge_length_m=None,
        in_nodes=None,
        out_nodes=None,
    ):
        self.nodes = list(nodes)
        self.edges = list(edges)
        self.edge_lengths = list(edge_lengths)
        self.G = G
        self.route_graph = route_graph
        self.pos = pos or {}
        self.edge_length_m = edge_length_m or [float(L) for L in edge_lengths]

        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.edge_to_idx = {e: i for i, e in enumerate(self.edges)}

        self.in_nodes = set(in_nodes or [])
        self.out_nodes = set(out_nodes or [])

        self.out_edges = {n: [] for n in self.nodes}
        self.in_edges = {n: [] for n in self.nodes}

        for ei, (u, v) in enumerate(self.edges):
            self.out_edges.setdefault(u, []).append(ei)
            self.in_edges.setdefault(v, []).append(ei)

        self.roads = [[NSEMPTY] * L for L in self.edge_lengths]
        self.cars = {}
        self.next_car_id = 0
        self.finished_cars = 0
        self.failed_spawns = 0
        self.spawned_cars = 0
        self.od_pairs = []
        self.od_by_origin = {}
        self.build_od_cache()

    @classmethod
    def from_networkx(
        cls,
        G,
        inout=None,
        cell_length_m=7.0,
        min_cells=None,
        bidirectional_if_undirected=True,
    ):
        """
        Build the internal edge-list representation from a NetworkX graph.

        If G is undirected, roads are made bidirectional by default:
            u -> v and v -> u both become separate road segments.

        This is important because cars follow directed edge pairs internally.
        Without reverse edges, many shortest paths can become unusable.
        """
        if min_cells is None:
            min_cells = 1

        nodes = list(G.nodes())
        edges = []
        lengths_cells = []
        lengths_m = []
        pos = {}

        for n, d in G.nodes(data=True):
            lat = float(d["x"])
            lon = float(d["y"])
            pos[n] = (lat, lon)

        def add_edge(u, v, length_m):
            edges.append((u, v))
            cells = max(min_cells, int(round(length_m / cell_length_m)))
            cells = max(1, cells)
            lengths_m.append(length_m)
            lengths_cells.append(cells)

        for u, v, data in G.edges(data=True):
            if "length" in data:
                length_m = float(data["length"])
            else:
                lat1 = float(G.nodes[u]["x"])
                lon1 = float(G.nodes[u]["y"])
                lat2 = float(G.nodes[v]["x"])
                lon2 = float(G.nodes[v]["y"])
                length_m = haversine_m(lat1, lon1, lat2, lon2)

            add_edge(u, v, length_m)

            if (not G.is_directed()) and bidirectional_if_undirected:
                add_edge(v, u, length_m)

        route_graph = nx.DiGraph()
        route_graph.add_nodes_from(nodes)
        for (u, v), length_m in zip(edges, lengths_m):
            route_graph.add_edge(u, v, length=length_m)

        in_nodes = inout.get("in", []) if inout else []
        out_nodes = inout.get("out", []) if inout else []

        return cls(
            nodes=nodes,
            edges=edges,
            edge_lengths=lengths_cells,
            G=G,
            route_graph=route_graph,
            pos=pos,
            edge_length_m=lengths_m,
            in_nodes=in_nodes,
            out_nodes=out_nodes,
        )

    # --------------------------------------------------------
    # Spawning and routing
    # --------------------------------------------------------

    def build_od_cache(self):
        """
        Precompute reachable origin-destination pairs.

        This avoids repeatedly trying impossible OD pairs during injection.
        """
        self.od_pairs = []
        self.od_by_origin = {origin: [] for origin in self.in_nodes}

        if not self.in_nodes or not self.out_nodes or self.route_graph is None:
            return

        for origin in self.in_nodes:
            for destination in self.out_nodes:
                if origin == destination:
                    continue
                path = self.shortest_path(origin, destination)
                if path is None or len(path) < 2:
                    continue
                self.od_pairs.append((origin, destination))
                self.od_by_origin.setdefault(origin, []).append(destination)

    def random_reachable_od(self):
        if not self.od_pairs:
            return None, None
        return random.choice(self.od_pairs)

    def random_node_with_outgoing_edge(self):
        candidates = [n for n in self.nodes if self.out_edges.get(n)]
        if not candidates:
            return None
        return random.choice(candidates)

    def random_reachable_destination_from(self, origin, allowed_destinations=None, max_tries=30):
        if allowed_destinations is None:
            allowed_destinations = self.nodes

        allowed_destinations = list(allowed_destinations)
        if not allowed_destinations:
            return None

        for _ in range(max_tries):
            destination = random.choice(allowed_destinations)
            if destination == origin:
                continue
            path = self.shortest_path(origin, destination)
            if path is not None and len(path) >= 2:
                return destination

        return None

    def choose_spawn_od(
        self,
        boundary_probability=0.7,
        boundary_sources="inout",
        boundary_destinations="inout",
    ):
        """
        Choose an OD pair.

        With probability boundary_probability:
            use boundary nodes, e.g. in/out nodes.

        Otherwise:
            use a random internal graph trip.
        """
        boundary_sources_set = set()
        if boundary_sources in ("in", "inout"):
            boundary_sources_set |= self.in_nodes
        if boundary_sources in ("out", "inout"):
            boundary_sources_set |= self.out_nodes

        boundary_destinations_set = set()
        if boundary_destinations in ("in", "inout"):
            boundary_destinations_set |= self.in_nodes
        if boundary_destinations in ("out", "inout"):
            boundary_destinations_set |= self.out_nodes

        use_boundary = random.random() < boundary_probability

        if use_boundary and boundary_sources_set and boundary_destinations_set:
            origins = [n for n in boundary_sources_set if self.out_edges.get(n)]
            random.shuffle(origins)

            for origin in origins:
                destination = self.random_reachable_destination_from(
                    origin,
                    allowed_destinations=boundary_destinations_set,
                )
                if destination is not None:
                    return origin, destination

        origin = self.random_node_with_outgoing_edge()
        if origin is None:
            return None, None

        destination = self.random_reachable_destination_from(origin)
        return origin, destination

    def shortest_path(self, origin, destination):
        """
        Route only over edges that actually exist in the simulation.
        """
        if self.route_graph is None:
            return None
        try:
            return nx.shortest_path(self.route_graph, origin, destination, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def spawn_car(self, origin, destination, speed=0):
        """
        Spawn a car at origin with a fixed destination.
        The car follows a shortest path and despawns at destination.
        """
        if origin == destination:
            return False

        path = self.shortest_path(origin, destination)
        if path is None or len(path) < 2:
            self.failed_spawns += 1
            return False

        first_edge = self.edge_to_idx.get((path[0], path[1]))
        if first_edge is None:
            self.failed_spawns += 1
            return False

        road = self.roads[first_edge]
        if road[0] != NSEMPTY:
            self.failed_spawns += 1
            return False

        car_id = self.next_car_id
        self.next_car_id += 1

        self.cars[car_id] = Car(
            car_id=car_id,
            speed=speed,
            origin=origin,
            destination=destination,
            path=path,
            path_pos=0,
        )

        road[0] = car_id
        self.spawned_cars += 1
        return True

    def random_inject(
        self,
        rate=0.2,
        max_new_cars=None,
        boundary_probability=0.7,
        boundary_sources="inout",
        boundary_destinations="inout",
    ):
        """
        Inject new cars with mixed demand.

        Default behavior:
            70% boundary-to-boundary trips
            30% random internal graph trips

        boundary_sources / boundary_destinations can be:
            "in"    -> only in_nodes
            "out"   -> only out_nodes
            "inout" -> in_nodes union out_nodes
        """
        if max_new_cars is None:
            boundary_count = max(1, len(self.in_nodes | self.out_nodes))
            max_new_cars = max(1, int(round(rate * boundary_count)))

        attempts = max_new_cars * 3
        inserted = 0

        for _ in range(attempts):
            origin, destination = self.choose_spawn_od(
                boundary_probability=boundary_probability,
                boundary_sources=boundary_sources,
                boundary_destinations=boundary_destinations,
            )

            if origin is None or destination is None:
                self.failed_spawns += 1
                continue

            if self.spawn_car(origin, destination, speed=0):
                inserted += 1

            if inserted >= max_new_cars:
                break

        return inserted, attempts

    def populate_random_od(self, density=0.05, vmax=5):
        """
        Random initial distribution with valid origin/destination cars.

        This places cars throughout the graph, but only on edges that can still
        reach at least one output node from their downstream endpoint.
        """
        if not self.out_nodes:
            raise ValueError("populate_random_od needs out_nodes in inout.json")

        destinations = list(self.out_nodes)

        for ei, road in enumerate(self.roads):
            u, v = self.edges[ei]

            reachable_destinations = []
            for destination in destinations:
                path_tail = self.shortest_path(v, destination)
                if path_tail is not None and len(path_tail) >= 1:
                    reachable_destinations.append((destination, path_tail))

            if not reachable_destinations:
                continue

            for cell_idx in range(len(road)):
                if road[cell_idx] != NSEMPTY:
                    continue

                if random.random() >= density:
                    continue

                selected_destination, path_tail = random.choice(reachable_destinations)
                selected_path = [u] + path_tail

                if len(selected_path) < 2:
                    continue

                car_id = self.next_car_id
                self.next_car_id += 1

                self.cars[car_id] = Car(
                    car_id=car_id,
                    speed=random.randint(0, vmax),
                    origin=u,
                    destination=selected_destination,
                    path=selected_path,
                    path_pos=0,
                )

                road[cell_idx] = car_id
                self.spawned_cars += 1

    def reverse_edge_idx(self, edge_idx):
        u, v = self.edges[edge_idx]
        return self.edge_to_idx.get((v, u))
    
    def get_next_edge_for_car(self, car_id, current_edge):
        car = self.cars[car_id]

        if car.path_pos + 2 >= len(car.path):
            return None

        u = car.path[car.path_pos + 1]
        v = car.path[car.path_pos + 2]
        next_edge = self.edge_to_idx.get((u, v))

        if next_edge == self.reverse_edge_idx(current_edge):
            return None

    # --------------------------------------------------------
    # NaSch update on graph
    # --------------------------------------------------------

    def edge_gap(self, edge_idx, cell_idx, vmax):
        """
        Gap ahead on the current edge only.

        Near a junction, this allows cars to request movement through the
        junction. Blocking is then handled by the junction resolver.
        """
        road = self.roads[edge_idx]
        L = len(road)
        gap = 0

        for d in range(1, vmax + 1):
            j = cell_idx + d

            if j >= L:
                gap += 1
                continue

            if road[j] == NSEMPTY:
                gap += 1
            else:
                break

        return gap

    def update_speeds(self, vmax=5, p=0.3):
        """
        Apply NaSch acceleration, slowing, and randomization.
        """
        for ei, road in enumerate(self.roads):
            for i, car_id in enumerate(road):
                if car_id == NSEMPTY:
                    continue

                car = self.cars[car_id]

                speed = min(car.speed + 1, vmax)
                gap = self.edge_gap(ei, i, vmax)
                speed = min(speed, gap)

                if random.random() < p:
                    speed = max(speed - 1, 0)

                car.speed = speed

    def collect_movements(self):
        """
        Build normal movements and junction requests from current speeds.
        """
        new_roads = [[NSEMPTY] * len(road) for road in self.roads]
        requests = []

        # Process cars from front to back on each edge.
        for ei, road in enumerate(self.roads):
            u, arrival_node = self.edges[ei]
            L = len(road)

            for i in range(L - 1, -1, -1):
                car_id = road[i]
                if car_id == NSEMPTY:
                    continue

                car = self.cars[car_id]
                target = i + car.speed

                if target < L:
                    if new_roads[ei][target] == NSEMPTY:
                        new_roads[ei][target] = car_id
                    else:
                        # Safety fallback: if target is occupied in the new road,
                        # place it as far forward as possible behind that cell.
                        placed = False
                        for k in range(target - 1, i - 1, -1):
                            if new_roads[ei][k] == NSEMPTY:
                                new_roads[ei][k] = car_id
                                car.speed = max(0, k - i)
                                placed = True
                                break
                        if not placed:
                            new_roads[ei][i] = car_id
                            car.speed = 0
                    continue

                overflow = target - L

                # Destination reached: despawn.
                if arrival_node == car.destination:
                    self.finished_cars += 1
                    del self.cars[car_id]
                    continue

                to_edge = self.get_next_edge_for_car(car_id, ei)

                requests.append(
                    JunctionRequest(
                        car_id=car_id,
                        from_edge=ei,
                        to_edge=to_edge,
                        arrival_node=arrival_node,
                        target_cell=overflow,
                        old_cell=i,
                    )
                )

        return new_roads, requests

    def resolve_junction_requests(self, requests):
        """
        Resolve competing junction movements.

        A simplified right-hand rule is used when several cars request the same
        junction at the same timestep. One winner per junction is allowed.
        """
        by_node = {}
        for r in requests:
            by_node.setdefault(r.arrival_node, []).append(r)

        accepted = []
        rejected = []

        for node, node_requests in by_node.items():
            valid = [r for r in node_requests if r.to_edge is not None]
            invalid = [r for r in node_requests if r.to_edge is None]

            rejected.extend(invalid)

            if not valid:
                continue

            if len(valid) == 1:
                accepted.append(valid[0])
            else:
                winner = self.right_hand_rule(node, valid)
                accepted.append(winner)
                rejected.extend([r for r in valid if r is not winner])

        return accepted, rejected

    def right_hand_rule(self, node, requests):
        """
        Simplified yield-to-the-right rule.

        For each request, count how many other incoming cars are on its right.
        The car with the fewest blockers wins. Ties are random.
        """
        if len(requests) == 1:
            return requests[0]

        scored = []

        for r in requests:
            incoming_angle = self.edge_angle_toward_node(r.from_edge)
            blockers_on_right = 0

            for other in requests:
                if other is r:
                    continue

                other_angle = self.edge_angle_toward_node(other.from_edge)
                diff = normalize_angle(other_angle - incoming_angle)

                # Right side approximation: other approach is clockwise from us.
                if 0 < diff < np.pi:
                    blockers_on_right += 1

            scored.append((blockers_on_right, r))

        best_score = min(score for score, _ in scored)
        candidates = [r for score, r in scored if score == best_score]
        return random.choice(candidates)

    def apply_junction_results(self, new_roads, accepted, rejected):
        """
        Apply accepted junction movements and keep rejected cars at edge ends.
        """
        for r in accepted:
            car = self.cars.get(r.car_id)
            if car is None:
                continue

            if r.to_edge is None:
                self.block_at_edge_end(new_roads, r)
                continue

            next_road = new_roads[r.to_edge]
            target = min(r.target_cell, len(next_road) - 1)

            if target >= 0 and next_road[target] == NSEMPTY:
                next_road[target] = r.car_id
                car.path_pos += 1
            else:
                self.block_at_edge_end(new_roads, r)

        for r in rejected:
            if r.car_id in self.cars:
                self.block_at_edge_end(new_roads, r)

    def block_at_edge_end(self, new_roads, request):
        """
        Keep a car at the end of its current edge if it cannot enter junction.
        """
        road = new_roads[request.from_edge]
        car = self.cars.get(request.car_id)
        if car is None:
            return

        for pos in range(len(road) - 1, -1, -1):
            if road[pos] == NSEMPTY:
                road[pos] = request.car_id
                car.speed = 0
                return

        # Extremely rare fallback: the edge is fully occupied in new_roads.
        # Keep the car alive but do not place it; this should not happen if the
        # update is collision-free. We place it at the old cell if possible.
        old = min(request.old_cell, len(road) - 1)
        if road[old] == NSEMPTY:
            road[old] = request.car_id
            car.speed = 0

    def step(
        self,
        vmax=5,
        p=0.3,
        injection_rate=0.0,
        max_new_cars=None,
        boundary_probability=0.7,
        boundary_sources="inout",
        boundary_destinations="inout",
    ):
        self.update_speeds(vmax=vmax, p=p)
        new_roads, requests = self.collect_movements()
        accepted, rejected = self.resolve_junction_requests(requests)
        self.apply_junction_results(new_roads, accepted, rejected)
        self.roads = new_roads

        if injection_rate > 0:
            self.random_inject(
                rate=injection_rate,
                max_new_cars=max_new_cars,
                boundary_probability=boundary_probability,
                boundary_sources=boundary_sources,
                boundary_destinations=boundary_destinations,
            )

    # --------------------------------------------------------
    # Geometry for right-hand rule and plotting
    # --------------------------------------------------------

    def edge_angle_toward_node(self, edge_idx):
        """
        Angle of an incoming edge in the direction of travel, toward its target.
        """
        u, v = self.edges[edge_idx]
        x1, y1 = self.pos[u]
        x2, y2 = self.pos[v]
        return np.arctan2(y2 - y1, x2 - x1)

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    def total_cars_on_network(self):
        return sum(1 for road in self.roads for cell in road if cell != NSEMPTY)

    def edge_car_counts(self):
        return [sum(1 for cell in road if cell != NSEMPTY) for road in self.roads]

    def edge_densities(self):
        return [
            sum(1 for cell in road if cell != NSEMPTY) / len(road)
            for road in self.roads
        ]

    def routing_diagnostics(self, samples=200):
        """
        Quick sanity check for OD reachability.
        """
        if not self.in_nodes or not self.out_nodes:
            print("No in_nodes or out_nodes defined.")
            return

        origins = list(self.in_nodes)
        destinations = list(self.out_nodes)
        ok = 0
        failed = 0

        for _ in range(samples):
            o = random.choice(origins)
            d = random.choice(destinations)
            if o == d:
                continue
            path = self.shortest_path(o, d)
            if path is None:
                failed += 1
            else:
                ok += 1

        print("Routing diagnostics")
        print("  simulated directed edges:", len(self.edges))
        print("  reachable OD samples:", ok)
        print("  unreachable OD samples:", failed)


# ============================================================
# Geometry helpers
# ============================================================

def normalize_angle(angle):
    return angle % (2 * np.pi)


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


# ============================================================
# Visualization
# ============================================================

def color_speed(v):
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


def color_cell_graph(network, cell):
    if cell == NSEMPTY:
        return color_speed(NSEMPTY)
    car = network.cars.get(cell)
    if car is None:
        return (0, 0, 0)
    return color_speed(car.speed)

def reverse_edge_idx(self, edge_idx):
    u, v = self.edges[edge_idx]
    return self.edge_to_idx.get((v, u))

def simulate_road(highway, iters=500, vmax=5, p=0.3, output="nasch.png"):
    road = highway[:]
    img = Image.new("RGB", (len(road), iters), (255, 255, 255))

    for t in range(iters):
        for x, v in enumerate(road):
            img.putpixel((x, t), color_speed(v))

        road = step_road(road, vmax=vmax, p=p, circular=True)

    img.save(output)
    print(f"Saved {output}")


def flatten_network(network):
    flat = []
    for road in network.roads:
        flat.extend(road)
    return flat


def simulate_graph_with_image(
    G,
    network,
    iters=500,
    save_image_every=10,
    vmax=5,
    p=0.3,
    injection_rate=0.2,
    output="figures",
    max_new_cars=None,
    boundary_probability=0.7,
    boundary_sources="inout",
    boundary_destinations="inout",
):
    width = sum(len(road) for road in network.roads)
    img = Image.new("RGB", (width, iters), (255, 255, 255))

    for t in range(iters):
        x = 0
        for road in network.roads:
            for cell in road:
                img.putpixel((x, t), color_cell_graph(network, cell))
                x += 1

        network.step(
            vmax=vmax,
            p=p,
            injection_rate=injection_rate,
            max_new_cars=max_new_cars,
            boundary_probability=boundary_probability,
            boundary_sources=boundary_sources,
            boundary_destinations=boundary_destinations,
        )
        if (t + 1) % save_image_every == 0:
            plot_network_density(G, network, output=os.path.join(output, f'network_density_{t}.png'), use_density=True)
        

    img.save(os.path.join(output, 'nasch_graph.png'))
    print(f"Saved {output}")


def plot_network(G, output=None):
    pos = {}
    for n, d in G.nodes(data=True):
        lat = float(d["x"])
        lon = float(d["y"])
        pos[n] = (lat, lon)

    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw(
        G,
        pos,
        node_size=10,
        node_color="red",
        edge_color="gray",
        with_labels=False,
        arrows=False,
        ax=ax,
    )
    ax.set_title("Road Network")
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=200)
        plt.close(fig)
        print(f"Saved {output}")
    else:
        plt.show()


def plot_network_density(
    G,
    network,
    output="figures/network_density.png",
    use_density=True,
):
    pos = {}
    for n, d in G.nodes(data=True):
        lat = float(d["x"])
        lon = float(d["y"])
        pos[n] = (lat, lon)

    if use_density:
        values = network.edge_densities()
        label = "Car density on street"
    else:
        values = network.edge_car_counts()
        label = "Number of cars on street"

    vmax_value = max(values) if values else 1
    if vmax_value == 0:
        vmax_value = 1

    norm = mpl.colors.Normalize(vmin=0, vmax=vmax_value)
    cmap = plt.cm.viridis
    edge_colors = [cmap(norm(v)) for v in values]

    fig, ax = plt.subplots(figsize=(10, 10))

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=8,
        node_color="black",
        alpha=0.6,
        ax=ax,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        width=2.0,
        arrows=False,
        ax=ax,
    )

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(label)

    ax.set_title("Road Network Colored by Traffic")
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {output}")


# ============================================================
# Loading and simulation helpers
# ============================================================

def load_graph_network(
    gexf_path="data/erd.gexf",
    inout_path="data/inout.json",
    cell_length_m=7.0,
    min_cells=1,
):
    with open(inout_path) as f:
        inout = json.load(f)

    G = nx.read_gexf(gexf_path)

    network = GraphRoadNetwork.from_networkx(
        G,
        inout=inout,
        cell_length_m=cell_length_m,
        min_cells=min_cells,
    )

    return G, network


def simulate_graph(
    gexf_path="data/erd.gexf",
    inout_path="data/inout.json",
    iters=500,
    vmax=5,
    p=0.3,
    injection_rate=0.2,
    initial_density=0.0,
    cell_length_m=7.0,
    min_cells=1,
    output="figures",
):
    G, network = load_graph_network(
        gexf_path=gexf_path,
        inout_path=inout_path,
        cell_length_m=cell_length_m,
        min_cells=min_cells,
    )

    if initial_density > 0:
        network.populate_random_od(density=initial_density, vmax=vmax)

    for _ in range(iters):
        network.step(vmax=vmax, p=p, injection_rate=injection_rate, max_new_cars=None)
        plot_network_density(G, network, output=f"{output}/network_density_{_}.png", use_density=True)

    return G, network


# ============================================================
# Main example
# ============================================================

if __name__ == "__main__":
    G, network = load_graph_network(
        gexf_path="data/erd.gexf",
        inout_path="data/inout.json",
        cell_length_m=7.0,
        min_cells=3,
    )

    plot_network(G, output="figures/network.png")

    # Optional initial OD-based distribution.
    # Each initialized car gets a real destination and path.
    network.populate_random_od(
        density=0.05,
        vmax=5,
    )

    simulate_graph_with_image(
        network,
        iters=500,
        vmax=5,
        p=0.3,
        injection_rate=0.8,
        output="figures/",
    )

    plot_network_density(
        G,
        network,
        output="figures/network_density.png",
        use_density=True,
    )

    print("Cars currently on network:", network.total_cars_on_network())
    print("Cars spawned:", network.spawned_cars)
    print("Cars finished:", network.finished_cars)
    print("Failed spawns:", network.failed_spawns)
