import os
import json
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib as mpl
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
    ):
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

    def random_path(self, origin, destination):
        """
        Fast route choice using igraph.

        For speed, this defaults to shortest path. If k_paths > 1, it creates
        variation by adding small random penalties to edge weights and caching
        a few alternatives per OD pair. This is much faster than
        networkx.shortest_simple_paths for repeated simulation use.
        """
        if self.ig_graph is None or origin == destination:
            return None

        key = (origin, destination, self.k_paths)
        if key in self.path_cache:
            paths = self.path_cache[key]
            return random.choice(paths) if paths else None

        if origin not in self.node_to_ig or destination not in self.node_to_ig:
            self.path_cache[key] = []
            return None

        source = self.node_to_ig[origin]
        target = self.node_to_ig[destination]
        paths = []

        base_lengths = np.array(self.ig_graph.es["length"], dtype=float)
        tries = max(1, self.k_paths)

        for attempt in range(tries):
            if attempt == 0 or self.k_paths <= 1:
                weights = base_lengths
            else:
                noise = np.random.uniform(0.95, 1.25, size=len(base_lengths))
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
        return random.choice(paths) if paths else None

    def random_node_with_outgoing_edge(self):
        candidates = [n for n in self.nodes if self.out_edges.get(n)]
        if not candidates:
            return None
        return random.choice(candidates)

    def random_reachable_destination_from(self, origin, allowed_destinations=None, max_tries=50):
        if allowed_destinations is None:
            allowed_destinations = self.nodes

        allowed_destinations = list(allowed_destinations)
        if not allowed_destinations:
            return None

        for _ in range(max_tries):
            destination = random.choice(allowed_destinations)
            if destination == origin:
                continue
            path = self.random_path(origin, destination)
            if path is not None and len(path) >= 2:
                return destination

        return None

    def choose_spawn_od(
        self,
        boundary_probability=0.7,
        boundary_sources="inout",
        boundary_destinations="inout",
    ):
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

    def spawn_car(self, origin, destination, speed=0):
        if origin == destination:
            self.failed_spawns += 1
            return False

        path = self.random_path(origin, destination)
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
            )

            if origin is None or destination is None:
                self.failed_spawns += 1
                continue

            if self.spawn_car(origin, destination, speed=0):
                inserted += 1

            if inserted >= max_new_cars:
                break

        return inserted, attempts

    def populate_random_od(self, density=0.05, vmax=5, destinations_mode="out"):
        """
        Initial cars throughout the graph.

        destinations_mode:
            "out"   -> destinations from out_nodes
            "inout" -> destinations from in_nodes union out_nodes
            "all"   -> destinations can be anywhere
        """
        if destinations_mode == "out":
            destinations = list(self.out_nodes)
        elif destinations_mode == "inout":
            destinations = list(self.in_nodes | self.out_nodes)
        else:
            destinations = list(self.nodes)

        if not destinations:
            destinations = list(self.nodes)

        for ei, road in enumerate(self.roads):
            u, v = self.edges[ei]

            reachable = []
            for destination in destinations:
                if destination == v:
                    continue
                path_tail = self.random_path(v, destination)
                if path_tail is not None and len(path_tail) >= 1:
                    reachable.append((destination, path_tail))

            if not reachable:
                continue

            for cell_idx in range(len(road)):
                if road[cell_idx] != NSEMPTY:
                    continue

                if random.random() >= density:
                    continue

                destination, path_tail = random.choice(reachable)
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

    # --------------------------------------------------------
    # Edge and route helpers
    # --------------------------------------------------------

    def reverse_edge_idx(self, edge_idx):
        u, v = self.edges[edge_idx]
        return self.edge_to_idx.get((v, u))

    def get_next_edge_for_car(self, car_id, current_edge, allow_u_turn=False):
        car = self.cars[car_id]

        if car.path_pos + 2 >= len(car.path):
            return None

        u = car.path[car.path_pos + 1]
        v = car.path[car.path_pos + 2]
        next_edge = self.edge_to_idx.get((u, v))

        if next_edge is None:
            return None

        if not allow_u_turn and next_edge == self.reverse_edge_idx(current_edge):
            return None

        return next_edge

    def find_car_position(self, car_id):
        for ei, road in enumerate(self.roads):
            for ci, cell in enumerate(road):
                if cell == car_id:
                    return ei, ci
        return None, None

    # --------------------------------------------------------
    # NaSch speed update and movement planning
    # --------------------------------------------------------

    def edge_gap(self, edge_idx, cell_idx, vmax):
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
                del self.cars[plan.car_id]
                self.finished_cars += 1
                continue

            target_road = new_roads[plan.to_edge]
            target_cell = min(max(plan.to_cell, 0), len(target_road) - 1)

            if target_road[target_cell] == NSEMPTY:
                target_road[target_cell] = plan.car_id
                car = self.cars[plan.car_id]
                car.path_pos += 1
                self.accepted_junction_moves += 1
            else:
                self.block_at_edge_end(new_roads, plan)
                self.blocked_junction_moves += 1

        # Rejected junction movements stay at the end of their current edge.
        for plan in rejected_junction:
            if plan.car_id in self.cars:
                self.block_at_edge_end(new_roads, plan)
                self.blocked_junction_moves += 1

        self.roads = new_roads

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
        boundary_sources="inout",
        boundary_destinations="inout",
        allow_u_turn=False,
        record_snapshot=False,
        t=None,
    ):
        self.update_speeds(vmax=vmax, p=p)
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

        self.snapshots.append(
            {
                "t": t,
                "cars": cars_state,
                "edge_counts": edge_counts,
                "edge_density": edge_density,
                "edge_mean_speed": edge_mean_speed,
                "total_cars": sum(edge_counts),
                "finished_cars": self.finished_cars,
                "spawned_cars": self.spawned_cars,
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
    boundary_sources="inout",
    boundary_destinations="inout",
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
        destinations_mode="inout",
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
        boundary_sources="inout",
        boundary_destinations="inout",
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
