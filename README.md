# Horse Pen Optimizer

Upload a game screenshot and compute an **optimal** set of wall placements to maximize a score:

- **Score** = enclosed tiles \(+\) **3 × enclosed cherries**
- **Hard constraint**: you can place at most **K** walls (user input)

This project runs entirely in the browser (static HTML/CSS/JS). The optimization is solved exactly with a **MILP** in a **Web Worker** using a vendored **WASM GLPK** build.

## How to run

Because the app uses ES module Web Workers and WASM, you should serve it over HTTP (opening the HTML via `file://` may break in many browsers).

From the repo root:

```bash
python3 -m http.server 8000
```

Then open:

- `http://localhost:8000/horse_pen_optimizer.html`

## How to use

- Drop/upload a screenshot (PNG/WebP/JPG)
- The app will:
  - auto-detect the grid size
  - classify tiles (water/grass/horse/cherries)
  - solve for the best wall placement under your wall budget
- Click **Analyze & Optimize**

## What counts as “enclosed”

After walls are placed, the horse’s reachable region must **not** touch the grid boundary (otherwise the horse can escape). The solver maximizes the score of the reachable enclosed region.

## Project structure

- `horse_pen_optimizer.html`: UI shell
- `styles.css`: styling
- `src/constants.js`: thresholds + scoring constants
- `src/image.js`: grid auto-detection + tile classification (water/horse/cherries)
- `src/render.js`: minimap rendering + score display
- `src/main.js`: wiring (upload → parse → solve → render)
- `src/solver/solver.js`: solver entrypoint
- `src/solver/milpSolver.js`: worker orchestration + result decoding
- `src/solver/milpWorker.js`: builds + solves the MILP using GLPK (WASM)
- `vendor/glpk/`: vendored `glpk.js` + `glpk.wasm`

See `ARCHITECTURE.md` for a deeper walkthrough.

## Notes

- **Water & horse thresholds are fixed** (tuned for the game UI).
- Cherry detection is designed to be robust for pixel-art sprites by combining per-cell sampling with a pixel-level scan.


