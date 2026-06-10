---
marp: true
theme: default
paginate: true
size: 16:9
---

# City Traffic Simulation with a Nagel-Schreckenberg Model

**Agent-based modelling project**

Data-driven and Agent-based Modelling  
BMETEFTBsPAAMD-00

---

## Research Question

How can a microscopic traffic model simulate vehicle flow through a city road
network with:

- multiple entry points,
- multiple exit points,
- internal destinations,
- cars generated inside the city,
- congestion-aware route choice,
- and junction interactions?

The goal is not to build a full traffic engineering model. The goal is to build
a clear agent-based simulation where local vehicle rules create recognizable
network-level traffic patterns.

---

## Project Idea

This project extends the classical one-dimensional Nagel-Schreckenberg model to
a road graph.

Instead of cars moving on one circular highway, cars move on directed road
segments connected by junctions.

Each car has:

- an origin,
- a destination,
- a route through the network,
- a current speed,
- and a current position on one road segment.

---

## Input Data

The model uses two project data files:

- `data/erd.gexf`: the city road network,
- `data/inout.json`: manually selected entry and exit nodes.

The `inout.json` file separates boundary points into:

- `in`: nodes where external traffic enters the city,
- `out`: nodes where traffic can leave the city.

This makes it possible to model through-traffic as well as local city traffic.

---

## Road Network Representation

Each graph edge is converted into a cellular road.

If the input graph is undirected, every street becomes two directed roads:

- one road from `u` to `v`,
- one road from `v` to `u`.

Each cell can contain:

- `-1`: empty cell,
- `car_id`: occupied by one vehicle.

This preserves the simple NaSch idea while allowing movement across a real road
network.

---

## Main Visual: Road Network

![Road network](figures/network.png)

This figure shows the underlying road graph. It is useful as the first visual
because every later traffic result is mapped back onto this same geography.

---

## Vehicle Types

The simulator creates two broad kinds of trips.

**Boundary-origin trips**

- start from an `in` node,
- usually go to an `out` node,
- sometimes stop at an internal city destination.

**City-origin trips**

- start from an internal node,
- usually go to another internal destination,
- sometimes leave the city through an `out` node.

---

## Important Demand Parameters

The notebook exposes the main traffic assumptions through `RUN_CONFIG`.

| Parameter | Meaning |
|---|---|
| `boundary_probability` | Probability that a new car starts at a city entry |
| `boundary_to_boundary_probability` | Probability that an entry car leaves through an exit |
| `city_to_city_probability` | Probability that an internal car stays inside the city |
| `injection_rate` | Rate of new car insertion |
| `max_new_cars` | Maximum cars inserted per step |

These parameters make the simulation easy to turn into scenario experiments.

---

## NaSch Update Rules

At every time step, each car follows the standard NaSch logic:

1. **Acceleration**  
   Increase speed by one, up to `vmax`.

2. **Collision avoidance**  
   Reduce speed so the car does not pass through the next occupied cell.

3. **Random slowdown**  
   With probability `p`, reduce speed by one.

4. **Movement**  
   Move forward by the updated speed.

The graph extension adds route following and junction resolution.

---

## Junction Handling

When a car reaches the end of a road segment, it may need to cross a junction.

The simulator first plans all movements, then resolves junction conflicts.

The current model allows multiple compatible movements at a junction:

- exits are accepted immediately,
- blocked cars stay at the end of their road,
- valid turns are prioritized,
- conflicting movements are rejected for that time step.

This is more realistic than allowing only one car through every junction.

---

## Route Choice

For each origin-destination pair, the model caches several short candidate
routes.

Routes are scored by:

- physical route length,
- current density on the route,
- blocked first cells when spawning.

The route score is:

```text
score = sum(edge_length * (1 + congestion_weight * density^2))
```

A longer but emptier route can therefore be selected over the shortest route
when the shortest route is crowded.

---

## Dynamic Rerouting

Cars do not only choose a route at spawn time.

When a car reaches a junction, it can reconsider the remaining route to its
destination.

This matters because congestion is dynamic:

- a route that was good at departure may become crowded,
- a different road may become available,
- rerouting can spread traffic across the network.

This is the main mechanism that gives the model adaptive behavior.

---

## Initial Traffic State

The simulation can populate the road network before the time loop starts.

Initial cars are assigned real destinations and paths. In the default mixed
mode:

- most initial cars choose internal city destinations,
- some choose an exit node and leave the city.

This avoids starting from an empty network and allows congestion patterns to
develop more quickly.

---

## Main Simulation Loop

At each time step:

1. Update speeds using NaSch rules.
2. Plan internal and junction-crossing moves.
3. Resolve junction conflicts.
4. Apply accepted moves and blocked moves.
5. Remove cars that reached their destination.
6. Inject new cars.
7. Optionally record a snapshot.
8. Optionally save density and speed figures.

This separates decision making from movement application, which avoids update
order artifacts.

---

## Output Structure

The notebook is designed for reproducible experiment runs.

Change this single line:

```python
SAVE_DIR = Path("results/baseline")
```

The notebook then writes:

- `config.json`: all run parameters,
- `summary.json`: final metrics,
- `snapshots.json`: time-dependent simulation state,
- `figures/`: generated plots and images.

This makes scenario comparisons much cleaner.

---

## Density Visualization

![Initial density](figures/network_density_init.png)

The density plot colors each directed road by the fraction of occupied cells.

This helps identify:

- initially busy areas,
- roads where cars accumulate,
- possible bottlenecks,
- and spatial imbalance in the traffic demand.

---

## Final Density

![Final density](figures/network_density_final.png)

Comparing initial and final density shows whether the system:

- clears traffic efficiently,
- accumulates congestion,
- concentrates congestion near exits,
- or spreads traffic through alternative routes.

---

## Speed Visualization

![Final speed](DataDrivenAndAgentBasedModellingBMETEFTBsPAAMD-00/project1/results/baseline/figures/network_speed_final.png)

Mean speed by road segment is often easier to interpret than density alone.

Low speed can indicate:

- queues,
- frequent junction blocking,
- high density,
- or repeated random slowdowns.

Density and speed should be interpreted together.

---

## Space-Time Image

![NaSch graph image](figures/nasch_graph.png)

The `nasch_graph.png` image is a compact space-time representation.

Each row is one simulation time step.

Each column is one road cell in the flattened network.

Cell colors represent vehicle speed:

- white: empty,
- red: stopped,
- orange/yellow: slow to medium,
- green: fast.

---

## Core Summary Metrics

Useful metrics for the final slide or report:

| Metric | Interpretation |
|---|---|
| `cars_on_network` | Remaining active traffic |
| `spawned_cars` | Total inserted cars |
| `finished_cars` | Completed trips |
| `failed_spawns` | Demand that could not enter |
| `mean_speed` | Overall mobility level |
| `blocked_junction_moves` | Junction pressure |
| `accepted_junction_moves` | Successful junction flow |

These are written into `summary.json` by the notebook.

---

## Suggested Experiment 1: Demand Sweep

Run the same model with different `injection_rate` values:

```text
0.005, 0.01, 0.02, 0.04, 0.08
```

For each run, compare:

- final number of cars on the network,
- finished cars,
- failed spawns,
- mean speed,
- blocked junction moves.

Expected result: after a critical demand level, congestion grows faster than
throughput.

---

## Suggested Experiment 2: Rerouting On vs Off

Compare two scenarios:

```python
reroute_at_junctions = True
reroute_at_junctions = False
```

Keep all other parameters fixed.

Interesting questions:

- Does adaptive routing reduce congestion?
- Does it increase completed trips?
- Does it shift congestion to previously unused roads?
- Does it create new bottlenecks near popular exits?

This is a clean way to show why route choice matters.

---

## Suggested Experiment 3: Congestion Sensitivity

Vary `congestion_weight`:

```text
0.0, 1.0, 2.0, 4.0, 8.0
```

Interpretation:

- `0.0`: pure shortest-path routing,
- low values: weak congestion avoidance,
- high values: strong preference for empty roads.

This experiment can show whether route choice smooths traffic or creates
excessive detours.

---

## Suggested Experiment 4: Local vs Through Traffic

Vary:

```python
boundary_to_boundary_probability
city_to_city_probability
```

This changes the structure of demand:

- mostly through-traffic,
- mostly local city traffic,
- mixed traffic.

The resulting congestion patterns should differ because through-traffic tends
to connect boundary nodes, while local traffic spreads destinations across the
city.

---

## Strong Additional Figures

The current project already has network, density, speed, and space-time plots.

The most useful extra figures would be:

1. cumulative throughput over time,
2. active cars over time,
3. mean speed over time,
4. failed spawns over time,
5. blocked junction moves over time,
6. density-speed scatter plot,
7. edge bottleneck ranking,
8. scenario comparison bar charts.

These would make the results easier to present quantitatively.

---

## Figure Idea: Throughput Curves

Plot time series from `snapshots.json`:

- active cars,
- spawned cars,
- finished cars,
- mean speed.

This answers:

- Is the system stabilizing?
- Is congestion accumulating?
- Does throughput saturate?
- When does the network start to break down?

This should be one of the first extra result figures.

---

## Figure Idea: Fundamental Diagram

Create a scatter plot with points for edge-time observations:

- x-axis: edge density,
- y-axis: mean speed or estimated flow.

Expected pattern:

- low density: high speed,
- medium density: highest flow,
- high density: speed collapse.

This connects the simulation back to classical traffic-flow theory.

---

## Figure Idea: Bottleneck Map

Aggregate each edge over all snapshots:

- mean density,
- maximum density,
- mean speed,
- time spent above a congestion threshold.

Then color the network by these aggregate values.

This produces a strong "where did congestion happen?" figure.

It is often more presentation-friendly than showing only the final time step.

---

## Figure Idea: Junction Pressure

Track where junction moves are rejected.

A useful map could color nodes by:

- number of blocked incoming moves,
- number of accepted crossing moves,
- blocked-to-accepted ratio.

This would show whether congestion is caused mainly by:

- road capacity,
- entry pressure,
- or junction conflicts.

This requires adding per-junction counters to the simulator.

---

## Figure Idea: OD Flow Matrix

Build a matrix or heatmap:

- rows: origin category or entry node,
- columns: destination category or exit node,
- values: number of spawned trips.

This would make demand assumptions visible.

For a presentation, this is helpful because it explains where cars are trying to
go before showing where congestion appears.

---

## Animation Idea: Animated Density Map

Use the saved snapshots to animate network density over time.

For each frame:

- compute density per edge,
- draw the road graph,
- color edges by density,
- save frames as GIF or MP4.

This is probably the clearest animation for a presentation because it keeps the
map stable and shows congestion spreading or clearing.

---

## Animation Idea: Moving Cars on the Map

Animate individual cars as points moving along road segments.

For each car:

- use current edge,
- use cell index as progress along the edge,
- interpolate between the edge's two node coordinates,
- color by speed or destination type.

This looks intuitive and lively, but it can get visually crowded with many cars.
It is best for short clips or lower-density scenarios.

---

## Animation Idea: Queue Growth at Entries

Show entry nodes and nearby roads over time.

Possible visual encodings:

- entry node size = failed spawn attempts,
- incoming edge color = queue density,
- label = cumulative inserted cars.

This would be useful if you discuss network capacity or demand pressure.

---

## Animation Idea: Scenario Comparison

Make a side-by-side animation:

- left: shortest-path only,
- right: congestion-aware rerouting.

Use the same random seed and demand parameters.

This is one of the best ways to communicate the value of the adaptive route
choice algorithm.

The audience can immediately see whether traffic spreads out or jams.

---

## Data to Log for Better Results

The current snapshots are enough for many figures.

For even stronger analysis, add logs for:

- car birth time,
- car finish time,
- route length,
- number of reroutes,
- final trip duration,
- rejected junction node,
- selected route score at spawn time.

These would enable travel-time distributions and route-choice analysis.

---

## Limitations

The model is intentionally simplified.

Important limitations:

- no traffic lights,
- no lanes,
- no overtaking,
- no acceleration differences between drivers,
- simplified junction conflict rules,
- no calibrated real-world demand,
- no explicit travel-time objective,
- no public transport or pedestrians.

These limitations are acceptable for a first agent-based model, but they should
be stated clearly.

---

## Strengths of the Project

The project already includes several strong modeling ideas:

- microscopic vehicle agents,
- graph-based roads,
- boundary and internal demand,
- real route planning,
- congestion-aware alternatives,
- dynamic rerouting at junctions,
- structured notebook outputs,
- reproducible experiment configuration.

This is a solid foundation for scenario analysis.

---

## Main Takeaway

A simple NaSch model can be extended from a one-road toy system into a city-scale
agent-based traffic simulation.

The interesting behavior comes from combining:

- local vehicle rules,
- finite road capacity,
- entry and exit demand,
- junction conflicts,
- and adaptive route choice.

The next step is to strengthen the result section with time series, bottleneck
maps, and animations of congestion evolution.

---

## Appendix: Recommended Presentation Figure Set

For a clean final presentation, use this order:

1. road network map,
2. entry and exit node map,
3. initial density map,
4. animated density map or selected frames,
5. final density map,
6. final speed map,
7. active cars and mean speed over time,
8. throughput curve,
9. bottleneck map,
10. scenario comparison: rerouting on vs off.

This gives the audience both the spatial story and the quantitative story.

---

## Appendix: Best Animation Set

If time is limited, prioritize these:

1. **Animated density map**  
   Best balance of clarity and information.

2. **Side-by-side rerouting comparison**  
   Best for explaining the main algorithmic contribution.

3. **Moving vehicle dots**  
   Best for intuition, but can become busy.

4. **Space-time movie for a selected corridor**  
   Best for connecting back to the original NaSch model.

The first two would give the strongest presentation impact.
