## Overview

This repo is a **static browser app** that:

- Loads a game screenshot.
- Auto-detects the tile grid size (rows/cols).
- Grid size is **not user-editable**; the UI only shows the detected value.
- Classifies each tile as **water / grass / horse** using fixed thresholds.
- Runs a solver to place up to **N wall tiles** that prevent the horse from reaching the map edge.
- Renders the solution grid + wall coordinates.

Entry point: `index.html`

## Directory structure

- `index.html`
  - HTML layout only (no big inline JS/CSS).
  - Loads `styles.css` and JS files via `<script defer ...>` (no bundler).
- `styles.css`
  - All UI styling.
- `src/main.js`
  - Wires DOM events.
  - Calls image/grid parsing + solver + renderer.
- `src/constants.js`
  - Fixed detection thresholds and solver time budget.
- `src/image.js`
  - `detectGridDimensions(img, canvas)`:
    - Uses grayscale gradient autocorrelation to estimate cell size → rows/cols.
  - `parseImageToGrid(img, canvas, cols, rows)`:
    - Samples multiple points per cell.
    - Classifies: `water | grass | horse`.
- `src/render.js`
  - `renderResults(...)`:
    - Updates stats.
    - Draws the grid overlay and walls.
  - `countReachable(gridData)`:
    - Measures baseline reachable area for efficiency.
- `src/solver/solver.js`
  - Public solver entrypoint. Chooses which solver strategy to run.
- `src/solver/milpSolver.js`
  - Exact MILP solver in the browser (uses WASM-backed `glpk.js` inside a Web Worker).
- `src/solver/milpWorker.js`
  - The actual Web Worker module that runs GLPK + builds/solves the MILP.

## Data flow

1. **Upload** → `src/main.js:handleFile()`
2. Auto grid detect → `HorsePen.detectGridDimensions()`
3. Parse to grid → `HorsePen.parseImageToGrid()`
4. Solve → `HorsePen.findOptimalEnclosure()`
5. Render → `HorsePen.renderResults()`

## JS namespace

All runtime code attaches to a single global namespace:

- `window.HorsePen`

This keeps files split/organized without requiring ES module support or a bundler.

## Core model

We build a **grid graph** where:

- Nodes = passable tiles (`grass` + `horse`)
- Edges = 4-neighborhood adjacency
- Water tiles are removed from the graph.

Placing a wall on a tile removes that node from the graph.

The solution is **valid** if, after removing the wall nodes, the horse cannot reach any **map edge** tile (otherwise it can step out of bounds).

Objective: maximize the number of reachable tiles (enclosed area), with at most **N** walls.

## Solver strategy (current + next)

### Heuristic (current)

The solver is fully MILP-based in the browser; there is no heuristic or preset path.

It is good for quick feedback, but it is **not guaranteed optimal**.

### Exact solver for small k (next step)

To guarantee optimal results for small wall counts (e.g. **k=13** on `tests/fixtures/preview.webp`), the next step is to add an **exact** algorithm based on **important separators** (parameterized by k).

High level:

- Add a super-sink connected to all boundary tiles.
- Enumerate all **important vertex separators** of size ≤ k between horse and sink.
- Pick the separator that yields the **largest reachable component** from the horse.

This is exponential in k in the worst case, but practical for small k on grids of this size and can be capped at **10 seconds**.

### MILP proof / verification (Python)

For `tests/fixtures/preview.webp` and `tests/fixtures/p2.png`, `milp_solver.py` contains a Mixed Integer Linear Programming formulation
that finds **area=95 with k=13** in a couple seconds using CBC (via PuLP).

### MILP in the browser (JS)

The web app uses a vendored copy of `glpk.js` (WASM) inside a Web Worker and
builds the **same MILP formulation** in `src/solver/milpSolver.js` for non-preset layouts.

## Extending / modifying

- **Change thresholds**: edit `src/constants.js`.
- **Change detection**: edit `src/image.js`.
- **Change solver**: implement additional solvers under `src/solver/` and dispatch in `src/solver/solver.js`.
- **UI changes**: `index.html` + `styles.css`.


