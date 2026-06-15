import os
import json
import random
from dataclasses import dataclass

# The managed workspace may not allow Matplotlib to write into the user home.
os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, PillowWriter
import networkx as nx
import igraph as ig
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
        elif j < n:
            new[j] = v

    return new


# ============================================================
# Data structures
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
class MovePlan:
    car_id: int
    from_edge: int
    from_cell: int
    to_edge: int
    to_cell: int
    crosses_junction: bool
    junction: str | None = None
    turn_type: str | None = None
    priority: int = 0


# ============================================================
# Graph-road traffic simulator
# ============================================================

class GraphRoadNetwork:
    """
    Graph-based NaSch/cellular traffic model.

    Representation:
        roads[edge_idx][cell_idx] = -1      empty cell
        roads[edge_idx][cell_idx] = car_id  occupied cell

    Each graph edge is represented as a one-way cell road.
    If the input graph is undirected, each undirected street becomes two one-way roads.
    """

    def __init__(
        self,
        nodes,
        edges,
        edge_lengths,
        G=None,
        route_graph=None,
        ig_graph=None,
        node_to_ig=None,
        ig_to_node=None,
        pos=None,
        edge_length_m=None,
        in_nodes=None,
        out_nodes=None,
        k_paths=3,
        congestion_weight=4.0,
        reroute_at_junctions=True,
        reroute_improvement_threshold=0.05,
    ):
        self.nodes = list(nodes)
        self.edges = list(edges)
        self.edge_lengths = list(edge_lengths)
        self.G = G
        self.route_graph = route_graph
        self.ig_graph = ig_graph
        self.node_to_ig = node_to_ig or {}
        self.ig_to_node = ig_to_node or {}
        self.path_cache = {}
        self.pos = pos or {}
        self.edge_length_m = edge_length_m or [float(L) for L in edge_lengths]
        self.k_paths = k_paths
        self.congestion_weight = congestion_weight
        self.reroute_at_junctions = reroute_at_junctions
        self.reroute_improvement_threshold = reroute_improvement_threshold

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
        self.blocked_junction_moves = 0
        self.accepted_junction_moves = 0
        self.internal_moves = 0
        self.junction_blocked_counts = {n: 0 for n in self.nodes}
        self.junction_accepted_counts = {n: 0 for n in self.nodes}
        self.od_flow_counts = {}

        self.od_pairs = []
        self.od_by_origin = {}
        self.snapshots = []

        self.build_od_cache()

    @classmethod
    def from_networkx(
        cls,
        G,
        inout=None,
        cell_length_m=7.0,
        min_cells=3,
        bidirectional_if_undirected=True,
        k_paths=3,
        congestion_weight=4.0,
        reroute_at_junctions=True,
        reroute_improvement_threshold=0.05,
    ):
        if min_cells is None:
            min_cells = 1

        nodes = list(G.nodes())
        edges = []
        lengths_cells = []
        lengths_m = []
        pos = {}

        for n, d in G.nodes(data=True):
            lon = float(d["x"])
            lat = float(d["y"])
            pos[n] = (lon, lat)

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
                lon1 = float(G.nodes[u]["x"])
                lat1 = float(G.nodes[u]["y"])
                lon2 = float(G.nodes[v]["x"])
                lat2 = float(G.nodes[v]["y"])
                length_m = haversine_m(lat1, lon1, lat2, lon2)

            add_edge(u, v, length_m)

            if (not G.is_directed()) and bidirectional_if_undirected:
                add_edge(v, u, length_m)

        route_graph = nx.DiGraph()
        route_graph.add_nodes_from(nodes)
        for (u, v), length_m in zip(edges, lengths_m):
            route_graph.add_edge(u, v, length=length_m)

        node_to_ig = {node: i for i, node in enumerate(nodes)}
        ig_to_node = {i: node for node, i in node_to_ig.items()}
        ig_edges = [(node_to_ig[u], node_to_ig[v]) for u, v in edges]
        ig_graph = ig.Graph(n=len(nodes), edges=ig_edges, directed=True)
        ig_graph.es["length"] = lengths_m

        in_nodes = inout.get("in", []) if inout else []
        out_nodes = inout.get("out", []) if inout else []

        return cls(
            nodes=nodes,
            edges=edges,
            edge_lengths=lengths_cells,
            G=G,
            route_graph=route_graph,
            ig_graph=ig_graph,
            node_to_ig=node_to_ig,
            ig_to_node=ig_to_node,
            pos=pos,
            edge_length_m=lengths_m,
            in_nodes=in_nodes,
            out_nodes=out_nodes,
            k_paths=k_paths,
            congestion_weight=congestion_weight,
            reroute_at_junctions=reroute_at_junctions,
            reroute_improvement_threshold=reroute_improvement_threshold,
        )

    # --------------------------------------------------------
    # Routing and spawning
    # --------------------------------------------------------

    def build_od_cache(self):
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

    def shortest_path(self, origin, destination):
        """
        Fast shortest path using igraph.
        Returns node names, not igraph vertex IDs.
        """
        if self.ig_graph is None:
            return None

        if origin not in self.node_to_ig or destination not in self.node_to_ig:
            return None

        key = (origin, destination, "shortest")
        if key in self.path_cache:
            return self.path_cache[key]

        source = self.node_to_ig[origin]
        target = self.node_to_ig[destination]

        path_ids = self.ig_graph.get_shortest_paths(
            source,
            to=target,
            weights="length",
            output="vpath",
        )[0]

        if not path_ids or len(path_ids) < 2:
            self.path_cache[key] = None
            return None

        path = [self.ig_to_node[i] for i in path_ids]
        self.path_cache[key] = path
        return path

    def candidate_paths(self, origin, destination):
        """Return cached short route alternatives for an origin-destination pair."""
        if self.ig_graph is None or origin == destination:
            return []

        key = (origin, destination, "candidates", self.k_paths)
        if key in self.path_cache:
            return self.path_cache[key]

        if origin not in self.node_to_ig or destination not in self.node_to_ig:
            self.path_cache[key] = []
            return []

        source = self.node_to_ig[origin]
        target = self.node_to_ig[destination]
        paths = []

        path_id_lists = self.ig_graph.get_k_shortest_paths(
            source,
            to=target,
            k=max(1, self.k_paths),
            weights="length",
            output="vpath",
        )

        for path_ids in path_id_lists:
            if path_ids and len(path_ids) >= 2:
                path = [self.ig_to_node[i] for i in path_ids]
                if path not in paths:
                    paths.append(path)

        # If k-shortest produced few alternatives, add some mildly perturbed
        # shortest paths. This helps on networks with many near-equivalent roads.
        base_lengths = np.array(self.ig_graph.es["length"], dtype=float)
        tries = max(0, self.k_paths - len(paths)) * 2

        for attempt in range(tries):
            noise = np.random.uniform(0.9, 1.3, size=len(base_lengths))
            weights = base_lengths * noise

            path_ids = self.ig_graph.get_shortest_paths(
                source,
                to=target,
                weights=weights.tolist(),
                output="vpath",
            )[0]

            if path_ids and len(path_ids) >= 2:
                path = [self.ig_to_node[i] for i in path_ids]
                if path not in paths:
                    paths.append(path)

        self.path_cache[key] = paths
        return paths

    def path_edge_indices(self, path):
        edge_indices = []
        for u, v in zip(path[:-1], path[1:]):
            edge_idx = self.edge_to_idx.get((u, v))
            if edge_idx is None:
                return None
            edge_indices.append(edge_idx)
        return edge_indices

    def route_score(self, path, avoid_blocked_start=False):
        """
        Score a route using physical length and current traffic density.

        A longer but emptier route can beat a shorter congested one. The density
        term is squared so almost-empty roads stay close to pure shortest-path
        routing, while very full roads become unattractive quickly.
        """
        edge_indices = self.path_edge_indices(path)
        if edge_indices is None:
            return float("inf")

        score = 0.0
        for position, edge_idx in enumerate(edge_indices):
            road = self.roads[edge_idx]
            length_m = self.edge_length_m[edge_idx]
            density = sum(1 for cell in road if cell != NSEMPTY) / len(road)
            score += length_m * (1.0 + self.congestion_weight * density * density)

            if avoid_blocked_start and position == 0 and road[0] != NSEMPTY:
                score += length_m * (10.0 + self.congestion_weight)

        return score

    def random_path(
        self,
        origin,
        destination,
        avoid_blocked_start=False,
        forbidden_first_edge=None,
    ):
        """
        Congestion-aware route choice.

        The method keeps the old name because the rest of the simulator calls it
        as the route-choice hook. It now checks several candidate routes and
        chooses one with the best current density-adjusted score.
        """
        paths = self.candidate_paths(origin, destination)
        if not paths:
            return None

        if forbidden_first_edge is not None:
            allowed_paths = []
            for path in paths:
                edge_indices = self.path_edge_indices(path)
                if edge_indices and edge_indices[0] != forbidden_first_edge:
                    allowed_paths.append(path)
            paths = allowed_paths
            if not paths:
                return None

        scored_paths = [
            (self.route_score(path, avoid_blocked_start=avoid_blocked_start), path)
            for path in paths
        ]
        scored_paths = [(score, path) for score, path in scored_paths if np.isfinite(score)]
        if not scored_paths:
            return None

        best_score = min(score for score, _ in scored_paths)
        tolerance = max(1.0, best_score * 0.02)
        best_paths = [
            path for score, path in scored_paths
            if score <= best_score + tolerance
        ]
        return random.choice(best_paths)

    def nodes_for_role(self, role):
        if role == "in":
            return set(self.in_nodes)
        if role == "out":
            return set(self.out_nodes)
        if role == "inout":
            return set(self.in_nodes | self.out_nodes)
        if role == "all":
            return set(self.nodes)
        return set()

    def internal_nodes(self):
        boundary = self.in_nodes | self.out_nodes
        return [n for n in self.nodes if n not in boundary]

    def random_node_with_outgoing_edge(self, candidates=None):
        if candidates is None:
            candidates = self.nodes
        candidates = [n for n in candidates if self.out_edges.get(n)]
        if not candidates:
            return None
        return random.choice(candidates)

    def random_reachable_destination_and_path_from(
        self,
        origin,
        allowed_destinations=None,
        max_tries=50,
        avoid_blocked_start=False,
    ):
        if allowed_destinations is None:
            allowed_destinations = self.nodes

        allowed_destinations = list(allowed_destinations)
        if not allowed_destinations:
            return None, None

        for _ in range(max_tries):
            destination = random.choice(allowed_destinations)
            if destination == origin:
                continue
            path = self.random_path(
                origin,
                destination,
                avoid_blocked_start=avoid_blocked_start,
            )
            if path is not None and len(path) >= 2:
                return destination, path

        return None, None

    def random_reachable_destination_from(self, origin, allowed_destinations=None, max_tries=50):
        destination, _ = self.random_reachable_destination_and_path_from(
            origin,
            allowed_destinations=allowed_destinations,
            max_tries=max_tries,
        )
        return destination

    def choose_destination_from_pools(
        self,
        origin,
        preferred_destinations,
        fallback_destinations=None,
        avoid_blocked_start=False,
    ):
        pools = [
            list(preferred_destinations or []),
            list(fallback_destinations or []),
            list(self.nodes),
        ]

        tried = set()
        for pool in pools:
            key = tuple(sorted(pool))
            if not pool or key in tried:
                continue
            tried.add(key)
            destination, path = self.random_reachable_destination_and_path_from(
                origin,
                allowed_destinations=pool,
                avoid_blocked_start=avoid_blocked_start,
            )
            if destination is not None:
                return destination, path

        return None, None

    def choose_spawn_od(
        self,
        boundary_probability=0.7,
        boundary_sources="in",
        boundary_destinations="out",
        boundary_to_boundary_probability=0.85,
        city_to_city_probability=0.8,
    ):
        boundary_sources_set = self.nodes_for_role(boundary_sources)
        boundary_destinations_set = self.nodes_for_role(boundary_destinations)
        internal_nodes = set(self.internal_nodes())

        boundary_origins = [n for n in boundary_sources_set if self.out_edges.get(n)]
        city_origins = [n for n in internal_nodes if self.out_edges.get(n)]

        use_boundary = random.random() < boundary_probability or not city_origins

        if use_boundary and boundary_origins:
            random.shuffle(boundary_origins)
            for origin in boundary_origins:
                prefer_boundary_destination = random.random() < boundary_to_boundary_probability
                if prefer_boundary_destination:
                    preferred = boundary_destinations_set - {origin}
                    fallback = internal_nodes
                else:
                    preferred = internal_nodes
                    fallback = boundary_destinations_set - {origin}

                destination, _ = self.choose_destination_from_pools(
                    origin,
                    preferred,
                    fallback_destinations=fallback,
                    avoid_blocked_start=True,
                )
                if destination is not None:
                    return origin, destination

        fallback_origins = city_origins or boundary_origins
        random.shuffle(fallback_origins)

        for origin in fallback_origins:
            prefer_city_destination = random.random() < city_to_city_probability
            if prefer_city_destination:
                preferred = internal_nodes - {origin}
                fallback = boundary_destinations_set
            else:
                preferred = boundary_destinations_set - {origin}
                fallback = internal_nodes

            destination, _ = self.choose_destination_from_pools(
                origin,
                preferred,
                fallback_destinations=fallback,
                avoid_blocked_start=True,
            )
            if destination is not None:
                return origin, destination

        origin = self.random_node_with_outgoing_edge()
        if origin is None:
            return None, None

        destination = self.random_reachable_destination_from(origin)
        return origin, destination

    def record_od_flow(self, origin, destination):
        key = (origin, destination)
        self.od_flow_counts[key] = self.od_flow_counts.get(key, 0) + 1

    def record_junction_result(self, junction, accepted):
        if junction is None:
            return

        if accepted:
            self.junction_accepted_counts[junction] = (
                self.junction_accepted_counts.get(junction, 0) + 1
            )
        else:
            self.junction_blocked_counts[junction] = (
                self.junction_blocked_counts.get(junction, 0) + 1
            )

    def spawn_car(self, origin, destination, speed=0):
        if origin == destination:
            self.failed_spawns += 1
            return False

        path = self.random_path(origin, destination, avoid_blocked_start=True)
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
        self.record_od_flow(origin, destination)
        return True

    def random_inject(
        self,
        rate=0.2,
        max_new_cars=None,
        boundary_probability=0.7,
        boundary_sources="in",
        boundary_destinations="out",
        boundary_to_boundary_probability=0.85,
        city_to_city_probability=0.8,
    ):
        if max_new_cars is None:
            boundary_count = max(1, len(self.in_nodes | self.out_nodes))
            max_new_cars = max(1, int(round(rate * boundary_count)))

        attempts = max_new_cars * 4
        inserted = 0

        for _ in range(attempts):
            origin, destination = self.choose_spawn_od(
                boundary_probability=boundary_probability,
                boundary_sources=boundary_sources,
                boundary_destinations=boundary_destinations,
                boundary_to_boundary_probability=boundary_to_boundary_probability,
                city_to_city_probability=city_to_city_probability,
            )

            if origin is None or destination is None:
                self.failed_spawns += 1
                continue

            if self.spawn_car(origin, destination, speed=0):
                inserted += 1

            if inserted >= max_new_cars:
                break

        return inserted, attempts

    def populate_random_od(
        self,
        density=0.05,
        vmax=5,
        destinations_mode="mixed",
        city_to_city_probability=0.8,
    ):
        """
        Initial cars throughout the graph.

        destinations_mode:
            "out"   -> destinations from out_nodes
            "inout" -> destinations from in_nodes union out_nodes
            "all"   -> destinations can be anywhere
            "mixed" -> mostly internal destinations, sometimes boundary exits
        """
        internal_destinations = set(self.internal_nodes())
        boundary_destinations = set(self.in_nodes | self.out_nodes)
        exit_destinations = set(self.out_nodes) or boundary_destinations

        def destination_pools(origin):
            if destinations_mode == "out":
                return set(self.out_nodes) - {origin}, internal_destinations
            if destinations_mode == "inout":
                return boundary_destinations - {origin}, internal_destinations
            if destinations_mode == "all":
                return set(self.nodes) - {origin}, []

            prefer_internal = random.random() < city_to_city_probability
            if prefer_internal:
                return internal_destinations - {origin}, exit_destinations
            return exit_destinations - {origin}, internal_destinations

        for ei, road in enumerate(self.roads):
            u, v = self.edges[ei]

            for cell_idx in range(len(road)):
                if road[cell_idx] != NSEMPTY:
                    continue

                if random.random() >= density:
                    continue

                preferred, fallback = destination_pools(v)
                destination, path_tail = self.choose_destination_from_pools(
                    v,
                    preferred,
                    fallback_destinations=fallback,
                )
                if destination is None or path_tail is None:
                    continue

                selected_path = [u] + path_tail

                if len(selected_path) < 2:
                    continue

                car_id = self.next_car_id
                self.next_car_id += 1

                self.cars[car_id] = Car(
                    car_id=car_id,
                    speed=random.randint(0, vmax),
                    origin=u,
                    destination=destination,
                    path=selected_path,
                    path_pos=0,
                )

                road[cell_idx] = car_id
                self.spawned_cars += 1
                self.record_od_flow(u, destination)

    # --------------------------------------------------------
    # Edge and route helpers
    # --------------------------------------------------------

    def reverse_edge_idx(self, edge_idx):
        u, v = self.edges[edge_idx]
        return self.edge_to_idx.get((v, u))

    def edge_density_value(self, edge_idx):
        road = self.roads[edge_idx]
        return sum(1 for cell in road if cell != NSEMPTY) / len(road)

    def maybe_reroute_car_at_junction(self, car_id, current_edge, allow_u_turn=False):
        if not self.reroute_at_junctions:
            return

        car = self.cars[car_id]
        arrival_node = self.edges[current_edge][1]
        if arrival_node == car.destination:
            return

        forbidden_first_edge = None
        if not allow_u_turn:
            forbidden_first_edge = self.reverse_edge_idx(current_edge)

        new_tail = self.random_path(
            arrival_node,
            car.destination,
            forbidden_first_edge=forbidden_first_edge,
        )
        if new_tail is None or len(new_tail) < 2:
            return

        current_tail = car.path[car.path_pos + 1:]
        if current_tail == new_tail:
            return

        current_score = self.route_score(current_tail)
        new_score = self.route_score(new_tail)
        if not np.isfinite(new_score):
            return

        current_next_edge = self.get_next_edge_from_path(
            current_tail,
            current_edge=current_edge,
            allow_u_turn=allow_u_turn,
        )
        current_next_is_crowded = (
            current_next_edge is None
            or self.edge_density_value(current_next_edge) >= 0.75
        )
        improvement = (
            not np.isfinite(current_score)
            or new_score < current_score * (1.0 - self.reroute_improvement_threshold)
        )

        if improvement or current_next_is_crowded:
            car.path = car.path[:car.path_pos + 1] + new_tail

    def get_next_edge_from_path(self, path, current_edge, allow_u_turn=False):
        if len(path) < 2:
            return None

        next_edge = self.edge_to_idx.get((path[0], path[1]))
        if next_edge is None:
            return None

        if not allow_u_turn and next_edge == self.reverse_edge_idx(current_edge):
            return None

        return next_edge

    def get_next_edge_for_car(self, car_id, current_edge, allow_u_turn=False):
        car = self.cars[car_id]

        if car.path_pos + 1 >= len(car.path):
            return None

        self.maybe_reroute_car_at_junction(
            car_id,
            current_edge,
            allow_u_turn=allow_u_turn,
        )

        if car.path_pos + 2 >= len(car.path):
            return None

        next_path = car.path[car.path_pos + 1:car.path_pos + 3]
        return self.get_next_edge_from_path(
            next_path,
            current_edge=current_edge,
            allow_u_turn=allow_u_turn,
        )

    def find_car_position(self, car_id):
        for ei, road in enumerate(self.roads):
            for ci, cell in enumerate(road):
                if cell == car_id:
                    return ei, ci
        return None, None

    # --------------------------------------------------------
    # NaSch speed update and movement planning
    # --------------------------------------------------------

    def edge_gap(self, edge_idx, cell_idx, vmax, car_id=None, allow_u_turn=False):
        road = self.roads[edge_idx]
        L = len(road)
        gap = 0
        car = self.cars.get(car_id) if car_id is not None else None
        arrival_node = self.edges[edge_idx][1]
        next_edge = None
        next_edge_known = False

        for d in range(1, vmax + 1):
            j = cell_idx + d

            if j >= L:
                if car_id is None:
                    gap += 1
                    continue

                if car is not None and arrival_node == car.destination:
                    gap += 1
                    continue

                if not next_edge_known:
                    next_edge = self.get_next_edge_for_car(
                        car_id,
                        edge_idx,
                        allow_u_turn=allow_u_turn,
                    )
                    next_edge_known = True

                if next_edge is None:
                    break

                next_cell = j - L
                next_road = self.roads[next_edge]
                if next_cell >= len(next_road):
                    gap += 1
                    continue

                if next_road[next_cell] == NSEMPTY:
                    gap += 1
                else:
                    break

                continue

            if road[j] == NSEMPTY:
                gap += 1
            else:
                break

        return gap

    def update_speeds(self, vmax=5, p=0.3, allow_u_turn=False):
        for ei, road in enumerate(self.roads):
            for i, car_id in enumerate(road):
                if car_id == NSEMPTY:
                    continue

                car = self.cars[car_id]

                speed = min(car.speed + 1, vmax)
                gap = self.edge_gap(
                    ei,
                    i,
                    vmax,
                    car_id=car_id,
                    allow_u_turn=allow_u_turn,
                )
                speed = min(speed, gap)

                if random.random() < p:
                    speed = max(speed - 1, 0)

                car.speed = speed

    def plan_movements(self, allow_u_turn=False):
        """
        Plan all movements before applying any of them.

        Internal moves stay on the same edge.
        Junction moves are resolved later per node.
        """
        internal_plans = []
        junction_plans = []

        for ei, road in enumerate(self.roads):
            u, arrival_node = self.edges[ei]
            L = len(road)

            for i in range(L - 1, -1, -1):
                car_id = road[i]
                if car_id == NSEMPTY or car_id not in self.cars:
                    continue

                car = self.cars[car_id]
                target = i + car.speed

                if target < L:
                    internal_plans.append(
                        MovePlan(
                            car_id=car_id,
                            from_edge=ei,
                            from_cell=i,
                            to_edge=ei,
                            to_cell=target,
                            crosses_junction=False,
                        )
                    )
                    continue

                # Destination reached at this node.
                if arrival_node == car.destination:
                    junction_plans.append(
                        MovePlan(
                            car_id=car_id,
                            from_edge=ei,
                            from_cell=i,
                            to_edge=-1,
                            to_cell=-1,
                            crosses_junction=True,
                            junction=arrival_node,
                            turn_type="exit",
                            priority=100,
                        )
                    )
                    continue

                next_edge = self.get_next_edge_for_car(car_id, ei, allow_u_turn=allow_u_turn)
                if next_edge is None:
                    # No valid route continuation. It will be blocked at the edge end.
                    junction_plans.append(
                        MovePlan(
                            car_id=car_id,
                            from_edge=ei,
                            from_cell=i,
                            to_edge=ei,
                            to_cell=L - 1,
                            crosses_junction=True,
                            junction=arrival_node,
                            turn_type="blocked",
                            priority=-1,
                        )
                    )
                    continue

                overflow = target - L
                to_cell = min(overflow, len(self.roads[next_edge]) - 1)
                turn_type = self.turn_type(ei, next_edge)

                junction_plans.append(
                    MovePlan(
                        car_id=car_id,
                        from_edge=ei,
                        from_cell=i,
                        to_edge=next_edge,
                        to_cell=to_cell,
                        crosses_junction=True,
                        junction=arrival_node,
                        turn_type=turn_type,
                        priority=self.turn_priority(turn_type),
                    )
                )

        return internal_plans, junction_plans

    # --------------------------------------------------------
    # Junction resolution
    # --------------------------------------------------------

    def resolve_junctions(self, junction_plans):
        """
        Allow multiple compatible junction movements.

        Per junction:
            1. exits are accepted immediately
            2. invalid/no-route cars are rejected
            3. valid requests are sorted by priority
            4. accept a request if it does not conflict with already accepted ones
            5. otherwise reject only that request
        """
        by_node = {}
        for plan in junction_plans:
            by_node.setdefault(plan.junction, []).append(plan)

        accepted = []
        rejected = []

        for node, plans in by_node.items():
            exits = [p for p in plans if p.turn_type == "exit"]
            blocked = [p for p in plans if p.turn_type == "blocked" or p.to_edge is None]
            valid = [p for p in plans if p.turn_type not in ("exit", "blocked") and p.to_edge >= 0]

            accepted.extend(exits)
            rejected.extend(blocked)

            # Shuffle before sorting to avoid deterministic bias among exact ties.
            random.shuffle(valid)
            valid.sort(key=lambda p: (-p.priority, self.right_hand_score(p, valid)))

            local_accepted = []
            for plan in valid:
                if all(not self.movements_conflict(plan, other) for other in local_accepted):
                    local_accepted.append(plan)
                else:
                    rejected.append(plan)

            accepted.extend(local_accepted)

        return accepted, rejected

    def movements_conflict(self, a, b):
        """
        Approximate junction conflict model.

        This is intentionally simple but much less restrictive than one-car-per-junction.
        """
        if a.car_id == b.car_id:
            return False

        # Same outgoing road and same target cell definitely conflicts.
        if a.to_edge == b.to_edge and a.to_cell == b.to_cell:
            return True

        # Same outgoing road with very close entry positions conflicts.
        if a.to_edge == b.to_edge and abs(a.to_cell - b.to_cell) <= 1:
            return True

        # Left turns conflict with straight/right movements from other approaches.
        if a.turn_type == "left" and b.turn_type in ("straight", "right"):
            return True
        if b.turn_type == "left" and a.turn_type in ("straight", "right"):
            return True

        # Opposing left turns are allowed.
        # Parallel straight movements are allowed.
        return False

    def right_hand_score(self, plan, competing_plans):
        """
        Lower score wins. More cars on your right means worse priority.
        """
        score = 0
        a_angle = self.edge_angle_toward_node(plan.from_edge)

        for other in competing_plans:
            if other is plan:
                continue
            b_angle = self.edge_angle_toward_node(other.from_edge)
            diff = normalize_angle(b_angle - a_angle)
            if 0 < diff < np.pi:
                score += 1

        return score

    def turn_priority(self, turn_type):
        # Higher is better.
        if turn_type == "right":
            return 3
        if turn_type == "straight":
            return 2
        if turn_type == "left":
            return 1
        if turn_type == "uturn":
            return 0
        return 0

    def turn_type(self, from_edge, to_edge):
        if to_edge == self.reverse_edge_idx(from_edge):
            return "uturn"

        a = self.edge_angle_toward_node(from_edge)
        b = self.edge_angle_away_from_node(to_edge)
        diff = signed_angle_diff(a, b)

        # Coordinates are geographic-ish; this is an approximation.
        if abs(diff) < np.pi / 4:
            return "straight"
        if diff < 0:
            return "right"
        return "left"

    # --------------------------------------------------------
    # Apply movements
    # --------------------------------------------------------

    def apply_plans(self, internal_plans, accepted_junction, rejected_junction):
        new_roads = [[NSEMPTY] * len(road) for road in self.roads]

        # Apply internal moves first, front-to-back by target cell.
        internal_plans = sorted(internal_plans, key=lambda p: (p.to_edge, -p.to_cell))
        for plan in internal_plans:
            if plan.car_id not in self.cars:
                continue
            self.place_or_block_internal(new_roads, plan)

        # Apply accepted junction movements.
        for plan in accepted_junction:
            if plan.car_id not in self.cars:
                continue

            if plan.turn_type == "exit":
                self.record_junction_result(plan.junction, accepted=True)
                del self.cars[plan.car_id]
                self.finished_cars += 1
                continue

            if self.place_junction_move(new_roads, plan):
                self.accepted_junction_moves += 1
                self.record_junction_result(plan.junction, accepted=True)
            else:
                self.block_at_edge_end(new_roads, plan)
                self.blocked_junction_moves += 1
                self.record_junction_result(plan.junction, accepted=False)

        # Rejected junction movements stay at the end of their current edge.
        for plan in rejected_junction:
            if plan.car_id in self.cars:
                self.block_at_edge_end(new_roads, plan)
                self.blocked_junction_moves += 1
                self.record_junction_result(plan.junction, accepted=False)

        self.roads = new_roads

    def place_junction_move(self, new_roads, plan):
        target_road = new_roads[plan.to_edge]
        target_cell = min(max(plan.to_cell, 0), len(target_road) - 1)
        from_road_len = len(new_roads[plan.from_edge])

        for cell in range(target_cell, -1, -1):
            if target_road[cell] != NSEMPTY:
                continue

            target_road[cell] = plan.car_id
            car = self.cars[plan.car_id]
            car.path_pos += 1
            car.speed = max(0, from_road_len - plan.from_cell + cell)
            return True

        return False

    def place_or_block_internal(self, new_roads, plan):
        road = new_roads[plan.to_edge]
        car = self.cars[plan.car_id]

        if road[plan.to_cell] == NSEMPTY:
            road[plan.to_cell] = plan.car_id
            self.internal_moves += 1
            return

        # Place as far forward as possible between old and target.
        for k in range(plan.to_cell - 1, plan.from_cell - 1, -1):
            if road[k] == NSEMPTY:
                road[k] = plan.car_id
                car.speed = max(0, k - plan.from_cell)
                return

        # Last fallback.
        if road[plan.from_cell] == NSEMPTY:
            road[plan.from_cell] = plan.car_id
            car.speed = 0

    def block_at_edge_end(self, new_roads, plan):
        road = new_roads[plan.from_edge]
        car = self.cars.get(plan.car_id)
        if car is None:
            return

        for pos in range(len(road) - 1, -1, -1):
            if road[pos] == NSEMPTY:
                road[pos] = plan.car_id
                car.speed = 0
                return

        old = min(plan.from_cell, len(road) - 1)
        if road[old] == NSEMPTY:
            road[old] = plan.car_id
            car.speed = 0

    def step(
        self,
        vmax=5,
        p=0.3,
        injection_rate=0.0,
        max_new_cars=None,
        boundary_probability=0.7,
        boundary_sources="in",
        boundary_destinations="out",
        boundary_to_boundary_probability=0.85,
        city_to_city_probability=0.8,
        allow_u_turn=False,
        record_snapshot=False,
        t=None,
    ):
        self.update_speeds(vmax=vmax, p=p, allow_u_turn=allow_u_turn)
        internal_plans, junction_plans = self.plan_movements(allow_u_turn=allow_u_turn)
        accepted, rejected = self.resolve_junctions(junction_plans)
        self.apply_plans(internal_plans, accepted, rejected)

        if injection_rate > 0:
            self.random_inject(
                rate=injection_rate,
                max_new_cars=max_new_cars,
                boundary_probability=boundary_probability,
                boundary_sources=boundary_sources,
                boundary_destinations=boundary_destinations,
                boundary_to_boundary_probability=boundary_to_boundary_probability,
                city_to_city_probability=city_to_city_probability,
            )

        if record_snapshot:
            self.record_snapshot(t=t)

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    def edge_angle_toward_node(self, edge_idx):
        u, v = self.edges[edge_idx]
        x1, y1 = self.pos[u]
        x2, y2 = self.pos[v]
        return np.arctan2(y2 - y1, x2 - x1)

    def edge_angle_away_from_node(self, edge_idx):
        u, v = self.edges[edge_idx]
        x1, y1 = self.pos[u]
        x2, y2 = self.pos[v]
        return np.arctan2(y2 - y1, x2 - x1)

    # --------------------------------------------------------
    # Snapshots and diagnostics
    # --------------------------------------------------------

    def record_snapshot(self, t=None):
        cars_state = {}
        edge_counts = [0] * len(self.roads)
        edge_speed_sum = [0.0] * len(self.roads)

        for ei, road in enumerate(self.roads):
            for ci, car_id in enumerate(road):
                if car_id == NSEMPTY or car_id not in self.cars:
                    continue

                car = self.cars[car_id]
                cars_state[car_id] = {
                    "edge": ei,
                    "cell": ci,
                    "speed": car.speed,
                    "origin": car.origin,
                    "destination": car.destination,
                }
                edge_counts[ei] += 1
                edge_speed_sum[ei] += car.speed

        edge_density = [edge_counts[i] / len(self.roads[i]) for i in range(len(self.roads))]
        edge_mean_speed = [
            edge_speed_sum[i] / edge_counts[i] if edge_counts[i] else 0.0
            for i in range(len(self.roads))
        ]
        total_cars = sum(edge_counts)
        mean_speed = sum(edge_speed_sum) / total_cars if total_cars else 0.0

        self.snapshots.append(
            {
                "t": t,
                "cars": cars_state,
                "edge_counts": edge_counts,
                "edge_density": edge_density,
                "edge_mean_speed": edge_mean_speed,
                "total_cars": total_cars,
                "finished_cars": self.finished_cars,
                "spawned_cars": self.spawned_cars,
                "failed_spawns": self.failed_spawns,
                "accepted_junction_moves": self.accepted_junction_moves,
                "blocked_junction_moves": self.blocked_junction_moves,
                "mean_speed": mean_speed,
                "junction_accepted_counts": dict(self.junction_accepted_counts),
                "junction_blocked_counts": dict(self.junction_blocked_counts),
            }
        )

    def total_cars_on_network(self):
        return sum(1 for road in self.roads for cell in road if cell != NSEMPTY)

    def edge_car_counts(self):
        return [sum(1 for cell in road if cell != NSEMPTY) for road in self.roads]

    def edge_densities(self):
        return [sum(1 for cell in road if cell != NSEMPTY) / len(road) for road in self.roads]

    def edge_mean_speeds(self):
        values = []
        for road in self.roads:
            speeds = [self.cars[cell].speed for cell in road if cell != NSEMPTY and cell in self.cars]
            values.append(sum(speeds) / len(speeds) if speeds else 0.0)
        return values

    def mean_speed(self):
        speeds = [
            self.cars[cell].speed
            for road in self.roads
            for cell in road
            if cell != NSEMPTY and cell in self.cars
        ]
        return sum(speeds) / len(speeds) if speeds else 0.0

    def routing_diagnostics(self, samples=200):
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
            path = self.random_path(o, d)
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


def signed_angle_diff(a, b):
    """Return signed angle difference b-a in [-pi, pi]."""
    return (b - a + np.pi) % (2 * np.pi) - np.pi


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


def simulate_road(highway, iters=500, vmax=5, p=0.3, output="nasch.png"):
    road = highway[:]
    img = Image.new("RGB", (len(road), iters), (255, 255, 255))

    for t in range(iters):
        for x, v in enumerate(road):
            img.putpixel((x, t), color_speed(v))
        road = step_road(road, vmax=vmax, p=p, circular=True)

    img.save(output)
    print(f"Saved {output}")


def simulate_graph_with_image(
    G,
    network,
    iters=500,
    save_image_every=50,
    vmax=5,
    p=0.3,
    injection_rate=0.05,
    output="figures",
    max_new_cars=2,
    boundary_probability=0.7,
    boundary_sources="in",
    boundary_destinations="out",
    boundary_to_boundary_probability=0.85,
    city_to_city_probability=0.8,
    allow_u_turn=False,
    record_snapshots=True,
):
    os.makedirs(output, exist_ok=True)

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
            boundary_to_boundary_probability=boundary_to_boundary_probability,
            city_to_city_probability=city_to_city_probability,
            allow_u_turn=allow_u_turn,
            record_snapshot=record_snapshots,
            t=t,
        )

        if save_image_every and (t + 1) % save_image_every == 0:
            plot_network_density(
                G,
                network,
                output=os.path.join(output, f"network_density_{t + 1}.png"),
                use_density=True,
            )
            plot_network_speed(
                G,
                network,
                output=os.path.join(output, f"network_speed_{t + 1}.png"),
            )
            print(
                t + 1,
                "cars:", network.total_cars_on_network(),
                "finished:", network.finished_cars,
                "mean speed:", round(network.mean_speed(), 2),
            )

    img.save(os.path.join(output, "nasch_graph.png"))
    print(f"Saved {os.path.join(output, 'nasch_graph.png')}")


def plot_network(G, output=None):
    pos = {n: (float(d["x"]), float(d["y"])) for n, d in G.nodes(data=True)}

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
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        plt.savefig(output, dpi=200)
        plt.close(fig)
        print(f"Saved {output}")
    else:
        plt.show()


def plot_network_density(G, network, output="figures/network_density.png", use_density=True):
    if use_density:
        values = network.edge_densities()
        label = "Car density on street"
    else:
        values = network.edge_car_counts()
        label = "Number of cars on street"

    plot_network_edge_values(G, network, values, label, output, title="Road Network Colored by Traffic")


def plot_network_speed(G, network, output="figures/network_speed.png"):
    values = network.edge_mean_speeds()
    plot_network_edge_values(G, network, values, "Mean car speed on street", output, title="Road Network Colored by Speed")


def plot_network_edge_values(G, network, values, label, output, title):
    pos = {n: (float(d["x"]), float(d["y"])) for n, d in G.nodes(data=True)}

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

    # For undirected G, NetworkX draws fewer edges than network.edges if we doubled directions.
    # Therefore draw directly from network.edges so values align with simulated roads.
    edge_collection = mpl.collections.LineCollection(
        [[pos[u], pos[v]] for u, v in network.edges],
        colors=edge_colors,
        linewidths=1.7,
        alpha=0.95,
    )
    ax.add_collection(edge_collection)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(label)

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.autoscale()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {output}")


def snapshot_frame_indices(snapshots, every=1, max_frames=None):
    if not snapshots:
        raise ValueError("No snapshots available. Run with record_snapshots=True first.")

    every = max(1, int(every))
    indices = list(range(0, len(snapshots), every))

    if max_frames is not None and len(indices) > max_frames:
        selected = np.linspace(0, len(indices) - 1, int(max_frames), dtype=int)
        indices = [indices[i] for i in selected]

    return indices


def draw_boundary_nodes(ax, network, pos):
    in_points = [pos[n] for n in network.in_nodes if n in pos]
    out_points = [pos[n] for n in network.out_nodes if n in pos]

    if in_points:
        xs, ys = zip(*in_points)
        ax.scatter(xs, ys, s=45, color="#2563eb", edgecolor="white", linewidth=0.6, label="entry")

    if out_points:
        xs, ys = zip(*out_points)
        ax.scatter(xs, ys, s=45, color="#dc2626", edgecolor="white", linewidth=0.6, label="exit")


def animate_network_density(
    G,
    network,
    output="figures/traffic_density.gif",
    every=5,
    fps=8,
    max_frames=200,
    use_density=True,
):
    """
    Animate edge density or car count over the road network.

    Run the simulation with record_snapshots=True before calling this function.
    GIF output is used so no external ffmpeg installation is required.
    """
    frames = snapshot_frame_indices(network.snapshots, every=every, max_frames=max_frames)
    value_key = "edge_density" if use_density else "edge_counts"
    label = "Road density" if use_density else "Cars on road"

    pos = {n: (float(d["x"]), float(d["y"])) for n, d in G.nodes(data=True)}
    segments = [[pos[u], pos[v]] for u, v in network.edges]

    if use_density:
        vmax_value = 1.0
    else:
        vmax_value = max(max(snapshot[value_key]) for snapshot in network.snapshots)
        vmax_value = max(1, vmax_value)

    norm = mpl.colors.Normalize(vmin=0, vmax=vmax_value)
    cmap = plt.cm.magma_r

    fig, ax = plt.subplots(figsize=(10, 10))
    edge_collection = mpl.collections.LineCollection(
        segments,
        linewidths=2.0,
        alpha=0.95,
    )
    ax.add_collection(edge_collection)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=5,
        node_color="black",
        alpha=0.25,
        ax=ax,
    )
    draw_boundary_nodes(ax, network, pos)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.01)
    cbar.set_label(label)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.autoscale()
    if network.in_nodes or network.out_nodes:
        ax.legend(loc="upper right", frameon=True)

    def update(frame_idx):
        snapshot = network.snapshots[frame_idx]
        values = snapshot[value_key]
        edge_collection.set_color([cmap(norm(value)) for value in values])
        ax.set_title(
            f"{label} at t={snapshot['t']} | cars={snapshot['total_cars']} | "
            f"finished={snapshot['finished_cars']}"
        )
        return [edge_collection]

    update(frames[0])
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    anim.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved {output}")


def car_xy_from_snapshot(network, car_state):
    edge_idx = int(car_state["edge"])
    cell_idx = int(car_state["cell"])
    u, v = network.edges[edge_idx]
    x1, y1 = network.pos[u]
    x2, y2 = network.pos[v]

    road_len = len(network.roads[edge_idx])
    alpha = (cell_idx + 0.5) / road_len
    x = x1 + alpha * (x2 - x1)
    y = y1 + alpha * (y2 - y1)
    return x, y


def animate_moving_cars(
    G,
    network,
    output="figures/moving_cars.gif",
    every=5,
    fps=8,
    max_frames=200,
    max_cars=None,
    car_size=12,
):
    """
    Animate individual cars as moving dots on the map.

    This is most readable for lower-density runs. For crowded simulations, use
    max_cars to cap the number of plotted vehicles or prefer density animation.
    """
    frames = snapshot_frame_indices(network.snapshots, every=every, max_frames=max_frames)
    pos = {n: (float(d["x"]), float(d["y"])) for n, d in G.nodes(data=True)}
    segments = [[pos[u], pos[v]] for u, v in network.edges]

    max_speed = 1
    for snapshot in network.snapshots:
        for car_state in snapshot["cars"].values():
            max_speed = max(max_speed, int(car_state["speed"]))

    norm = mpl.colors.Normalize(vmin=0, vmax=max_speed)
    cmap = plt.cm.turbo

    fig, ax = plt.subplots(figsize=(10, 10))
    road_collection = mpl.collections.LineCollection(
        segments,
        colors="#b8b8b8",
        linewidths=0.8,
        alpha=0.55,
    )
    ax.add_collection(road_collection)

    draw_boundary_nodes(ax, network, pos)
    cars_artist = ax.scatter(
        [],
        [],
        s=car_size,
        c=[],
        cmap=cmap,
        norm=norm,
        alpha=0.9,
        linewidth=0,
    )

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.01)
    cbar.set_label("Car speed")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.autoscale()
    if network.in_nodes or network.out_nodes:
        ax.legend(loc="upper right", frameon=True)

    def update(frame_idx):
        snapshot = network.snapshots[frame_idx]
        car_items = list(snapshot["cars"].items())
        if max_cars is not None:
            car_items = car_items[:max_cars]

        xy = []
        speeds = []
        for _, car_state in car_items:
            xy.append(car_xy_from_snapshot(network, car_state))
            speeds.append(int(car_state["speed"]))

        if xy:
            cars_artist.set_offsets(np.array(xy))
            cars_artist.set_array(np.array(speeds))
        else:
            cars_artist.set_offsets(np.empty((0, 2)))
            cars_artist.set_array(np.array([]))

        ax.set_title(
            f"Moving cars at t={snapshot['t']} | cars={snapshot['total_cars']} | "
            f"finished={snapshot['finished_cars']}"
        )
        return [cars_artist]

    update(frames[0])
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    anim.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved {output}")


def require_snapshots(network):
    if not network.snapshots:
        raise ValueError("No snapshots available. Run with record_snapshots=True first.")
    return network.snapshots


def snapshot_times(snapshots):
    return [snapshot["t"] if snapshot.get("t") is not None else i for i, snapshot in enumerate(snapshots)]


def snapshot_mean_speed(snapshot):
    if "mean_speed" in snapshot:
        return snapshot["mean_speed"]

    edge_counts = snapshot.get("edge_counts", [])
    edge_speeds = snapshot.get("edge_mean_speed", [])
    total_cars = sum(edge_counts)
    if not total_cars:
        return 0.0

    weighted_speed = sum(count * speed for count, speed in zip(edge_counts, edge_speeds))
    return weighted_speed / total_cars


def plot_traffic_timeseries(network, output="figures/analytics/traffic_timeseries.png"):
    snapshots = require_snapshots(network)
    t = snapshot_times(snapshots)

    active = [snapshot["total_cars"] for snapshot in snapshots]
    spawned = [snapshot["spawned_cars"] for snapshot in snapshots]
    finished = [snapshot["finished_cars"] for snapshot in snapshots]
    mean_speed = [snapshot_mean_speed(snapshot) for snapshot in snapshots]
    failed_spawns = [snapshot.get("failed_spawns", np.nan) for snapshot in snapshots]
    blocked = [snapshot.get("blocked_junction_moves", np.nan) for snapshot in snapshots]
    accepted = [snapshot.get("accepted_junction_moves", np.nan) for snapshot in snapshots]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    ax = axes[0, 0]
    ax.plot(t, active, label="active cars", color="#2563eb")
    ax.plot(t, spawned, label="spawned cars", color="#16a34a")
    ax.plot(t, finished, label="finished cars", color="#dc2626")
    ax.set_ylabel("Cars")
    ax.set_title("Traffic Throughput")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(t, mean_speed, color="#7c3aed")
    ax.set_ylabel("Cells per step")
    ax.set_title("Mean Speed")

    ax = axes[1, 0]
    ax.plot(t, failed_spawns, color="#ea580c")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Failed spawns")
    ax.set_title("Entry Pressure")

    ax = axes[1, 1]
    ax.plot(t, blocked, label="blocked junction moves", color="#be123c")
    ax.plot(t, accepted, label="accepted junction moves", color="#0891b2")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Cumulative moves")
    ax.set_title("Junction Flow")
    ax.legend()

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {output}")


def plot_fundamental_diagram(
    network,
    output="figures/analytics/fundamental_diagram.png",
    max_points=60000,
):
    snapshots = require_snapshots(network)
    density = []
    speed = []

    for snapshot in snapshots:
        density.extend(snapshot["edge_density"])
        speed.extend(snapshot["edge_mean_speed"])

    density = np.array(density, dtype=float)
    speed = np.array(speed, dtype=float)

    if len(density) > max_points:
        idx = np.linspace(0, len(density) - 1, max_points, dtype=int)
        density = density[idx]
        speed = speed[idx]

    flow = density * speed

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        density,
        speed,
        c=flow,
        s=5,
        alpha=0.25,
        cmap="viridis",
        linewidth=0,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Estimated flow = density * speed")
    ax.set_xlabel("Road density")
    ax.set_ylabel("Mean speed")
    ax.set_title("Density-Speed Fundamental Diagram")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {output}")


def edge_bottleneck_statistics(network, congestion_threshold=0.6):
    snapshots = require_snapshots(network)
    density = np.array([snapshot["edge_density"] for snapshot in snapshots], dtype=float)
    speed = np.array([snapshot["edge_mean_speed"] for snapshot in snapshots], dtype=float)

    return {
        "mean_density": density.mean(axis=0),
        "max_density": density.max(axis=0),
        "mean_speed": speed.mean(axis=0),
        "time_congested": (density >= congestion_threshold).mean(axis=0),
    }


def plot_network_metric_map(G, network, values, label, output, title, cmap_name="magma_r"):
    pos = {n: (float(d["x"]), float(d["y"])) for n, d in G.nodes(data=True)}
    vmax_value = max(values) if len(values) else 1
    if vmax_value == 0:
        vmax_value = 1

    norm = mpl.colors.Normalize(vmin=0, vmax=vmax_value)
    cmap = plt.get_cmap(cmap_name)
    edge_colors = [cmap(norm(value)) for value in values]

    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=5,
        node_color="black",
        alpha=0.25,
        ax=ax,
    )
    edge_collection = mpl.collections.LineCollection(
        [[pos[u], pos[v]] for u, v in network.edges],
        colors=edge_colors,
        linewidths=1.9,
        alpha=0.95,
    )
    ax.add_collection(edge_collection)
    draw_boundary_nodes(ax, network, pos)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(label)

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.autoscale()
    if network.in_nodes or network.out_nodes:
        ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {output}")


def plot_bottleneck_map(
    G,
    network,
    output="figures/analytics/bottleneck_map.png",
    metric="time_congested",
    congestion_threshold=0.6,
):
    stats = edge_bottleneck_statistics(network, congestion_threshold=congestion_threshold)
    labels = {
        "mean_density": "Mean density",
        "max_density": "Maximum density",
        "mean_speed": "Mean speed",
        "time_congested": f"Share of time density >= {congestion_threshold}",
    }
    values = stats[metric]
    plot_network_metric_map(
        G,
        network,
        values,
        labels.get(metric, metric),
        output,
        title="Bottleneck Map",
    )


def plot_bottleneck_ranking(
    network,
    output="figures/analytics/bottleneck_ranking.png",
    top_n=15,
    congestion_threshold=0.6,
):
    stats = edge_bottleneck_statistics(network, congestion_threshold=congestion_threshold)
    score = stats["time_congested"]
    top_indices = np.argsort(score)[::-1][:top_n]

    labels = []
    values = []
    for edge_idx in top_indices:
        u, v = network.edges[edge_idx]
        labels.append(f"{edge_idx}: {u}->{v}")
        values.append(score[edge_idx])

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(values)), values, color="#be123c")
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"Share of snapshots with density >= {congestion_threshold}")
    ax.set_title("Top Bottleneck Roads")
    ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {output}")


def plot_junction_pressure_map(
    G,
    network,
    output="figures/analytics/junction_pressure_map.png",
    metric="blocked",
):
    pos = {n: (float(d["x"]), float(d["y"])) for n, d in G.nodes(data=True)}
    blocked = np.array([network.junction_blocked_counts.get(n, 0) for n in G.nodes()], dtype=float)
    accepted = np.array([network.junction_accepted_counts.get(n, 0) for n in G.nodes()], dtype=float)
    total = blocked + accepted

    if metric == "ratio":
        values = np.divide(blocked, total, out=np.zeros_like(blocked), where=total > 0)
        label = "Blocked / all junction requests"
    elif metric == "accepted":
        values = accepted
        label = "Accepted junction moves"
    else:
        values = blocked
        label = "Blocked junction moves"

    sizes = 12 + 80 * np.divide(total, total.max(), out=np.zeros_like(total), where=total.max() > 0)
    vmax_value = values.max() if len(values) else 1
    if vmax_value == 0:
        vmax_value = 1

    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_edges(G, pos, edge_color="#b8b8b8", width=0.8, alpha=0.5, ax=ax)
    sc = ax.scatter(
        [pos[n][0] for n in G.nodes()],
        [pos[n][1] for n in G.nodes()],
        s=sizes,
        c=values,
        cmap="inferno",
        vmin=0,
        vmax=vmax_value,
        alpha=0.9,
        linewidth=0,
    )
    draw_boundary_nodes(ax, network, pos)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(label)
    ax.set_title("Junction Pressure Map")
    ax.set_aspect("equal")
    ax.axis("off")
    if network.in_nodes or network.out_nodes:
        ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {output}")


def node_flow_group(network, node, axis):
    if axis == "origin" and node in network.in_nodes:
        return node
    if axis == "destination" and node in network.out_nodes:
        return node
    return "internal"


def plot_od_flow_matrix(network, output="figures/analytics/od_flow_matrix.png"):
    row_labels = sorted(network.in_nodes) + ["internal"]
    col_labels = sorted(network.out_nodes) + ["internal"]
    matrix = np.zeros((len(row_labels), len(col_labels)), dtype=int)
    row_index = {label: i for i, label in enumerate(row_labels)}
    col_index = {label: i for i, label in enumerate(col_labels)}

    for (origin, destination), count in network.od_flow_counts.items():
        row = node_flow_group(network, origin, axis="origin")
        col = node_flow_group(network, destination, axis="destination")
        matrix[row_index[row], col_index[col]] += count

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, cmap="Blues")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Spawned trips")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Destination")
    ax.set_ylabel("Origin")
    ax.set_title("Origin-Destination Flow Matrix")

    if matrix.size <= 100:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {output}")


def load_summary_record(summary):
    if isinstance(summary, dict):
        return summary

    with open(summary) as f:
        return json.load(f)


def plot_scenario_comparison(
    summaries,
    output="figures/analytics/scenario_comparison.png",
    labels=None,
):
    records = [load_summary_record(summary) for summary in summaries]
    if labels is None:
        labels = [f"scenario {i + 1}" for i in range(len(records))]

    metrics = [
        ("cars_on_network", "Cars left"),
        ("finished_cars", "Finished trips"),
        ("failed_spawns", "Failed spawns"),
        ("mean_speed", "Mean speed"),
        ("blocked_junction_moves", "Blocked junction moves"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for ax, (key, title) in zip(axes, metrics):
        values = [record.get(key, 0) for record in records]
        ax.bar(labels, values, color="#2563eb")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.25)

    axes[-1].axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {output}")


def plot_additional_result_figures(
    G,
    network,
    output_dir="figures/analytics",
    scenario_summaries=None,
    scenario_labels=None,
    congestion_threshold=0.6,
):
    os.makedirs(output_dir, exist_ok=True)

    outputs = {
        "traffic_timeseries": os.path.join(output_dir, "traffic_timeseries.png"),
        "fundamental_diagram": os.path.join(output_dir, "fundamental_diagram.png"),
        "bottleneck_map": os.path.join(output_dir, "bottleneck_map.png"),
        "bottleneck_ranking": os.path.join(output_dir, "bottleneck_ranking.png"),
        "junction_pressure_map": os.path.join(output_dir, "junction_pressure_map.png"),
        "od_flow_matrix": os.path.join(output_dir, "od_flow_matrix.png"),
    }

    plot_traffic_timeseries(network, output=outputs["traffic_timeseries"])
    plot_fundamental_diagram(network, output=outputs["fundamental_diagram"])
    plot_bottleneck_map(
        G,
        network,
        output=outputs["bottleneck_map"],
        congestion_threshold=congestion_threshold,
    )
    plot_bottleneck_ranking(
        network,
        output=outputs["bottleneck_ranking"],
        congestion_threshold=congestion_threshold,
    )
    plot_junction_pressure_map(G, network, output=outputs["junction_pressure_map"])
    plot_od_flow_matrix(network, output=outputs["od_flow_matrix"])

    if scenario_summaries:
        outputs["scenario_comparison"] = os.path.join(output_dir, "scenario_comparison.png")
        plot_scenario_comparison(
            scenario_summaries,
            output=outputs["scenario_comparison"],
            labels=scenario_labels,
        )

    return outputs


def save_snapshots_json(network, output="figures/snapshots.json"):
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(network.snapshots, f)
    print(f"Saved {output}")


# ============================================================
# Loading and simulation helpers
# ============================================================

def load_graph_network(
    gexf_path="data/erd.gexf",
    inout_path="data/inout.json",
    cell_length_m=7.0,
    min_cells=3,
    k_paths=3,
    congestion_weight=4.0,
    reroute_at_junctions=True,
    reroute_improvement_threshold=0.05,
):
    with open(inout_path) as f:
        inout = json.load(f)

    G = nx.read_gexf(gexf_path)

    network = GraphRoadNetwork.from_networkx(
        G,
        inout=inout,
        cell_length_m=cell_length_m,
        min_cells=min_cells,
        k_paths=k_paths,
        congestion_weight=congestion_weight,
        reroute_at_junctions=reroute_at_junctions,
        reroute_improvement_threshold=reroute_improvement_threshold,
    )

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
        k_paths=4,
    )

    network.routing_diagnostics(samples=500)

    plot_network(G, output="figures/network.png")

    network.populate_random_od(
        density=0.02,
        vmax=5,
        destinations_mode="mixed",
        city_to_city_probability=0.8,
    )

    plot_network_density(
        G,
        network,
        output="figures/network_density_init.png",
        use_density=True,
    )

    simulate_graph_with_image(
        G,
        network,
        iters=1000,
        save_image_every=100,
        vmax=5,
        p=0.3,
        injection_rate=0.05,
        max_new_cars=2,
        boundary_probability=0.7,
        boundary_sources="in",
        boundary_destinations="out",
        boundary_to_boundary_probability=0.85,
        city_to_city_probability=0.8,
        allow_u_turn=False,
        record_snapshots=True,
        output="figures",
    )

    plot_network_density(
        G,
        network,
        output="figures/network_density_final.png",
        use_density=True,
    )

    plot_network_speed(
        G,
        network,
        output="figures/network_speed_final.png",
    )

    save_snapshots_json(network, output="figures/snapshots.json")

    print("Cars currently on network:", network.total_cars_on_network())
    print("Cars spawned:", network.spawned_cars)
    print("Cars finished:", network.finished_cars)
    print("Failed spawns:", network.failed_spawns)
    print("Accepted junction moves:", network.accepted_junction_moves)
    print("Blocked junction moves:", network.blocked_junction_moves)
    print("Mean speed:", round(network.mean_speed(), 2))
