# Project 1: City NaSch Traffic Model

This project implements a Nagel-Schreckenberg-style cellular traffic model on a
city road graph. Each road segment is converted into a one-way cell road; for an
undirected input graph, both directions are simulated separately.

The model uses:

- `data/erd.gexf` as the road network.
- `data/inout.json` to mark city entry nodes (`in`) and exit nodes (`out`).
- `src/nasch.py` for loading, simulation, routing, diagnostics, and plotting.

## Traffic behavior

At each time step, cars update their speed with the usual NaSch acceleration,
gap braking, random slowdown, and movement rules. Cars can cross junctions when
their movement reaches the end of a road segment; compatible junction movements
can happen in the same time step.

New cars are injected using two trip types:

- Boundary-origin trips start at an `in` node. They usually choose an `out` node
  as destination, but sometimes choose an internal city destination.
- City-origin trips start from an internal node. They usually choose another
  internal destination, but sometimes leave the city through an `out` node.

The most important probability parameters are:

- `boundary_probability`: chance that a new injected car starts at a boundary
  entry instead of inside the city.
- `boundary_to_boundary_probability`: chance that a boundary-origin car leaves
  through an exit instead of stopping inside the city.
- `city_to_city_probability`: chance that a city-origin car chooses an internal
  city destination instead of leaving the city.

## Routing

For every origin-destination pair, the simulator caches several short candidate
routes. Route choice is congestion-aware: each route is scored by its physical
length and current road density, so a longer but emptier route can be selected
when the shortest route is crowded. Cars also reconsider the remaining route
when they reach a junction.

## Running

From this directory:

```bash
python src/nasch.py
```

The script writes network and traffic plots into `figures/`, plus a snapshot
JSON file for further analysis.

The notebook `notebooks/NagelSchreckenberg.ipynb` is set up for organized
experiment runs. Change the `SAVE_DIR` value in the first code cell to put all
figures, `config.json`, `summary.json`, and `snapshots.json` into a new output
folder.

For a ready-to-use project explanation and slide-style outline, see
`PRESENTATION.md`.
