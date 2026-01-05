(() => {
  const HP = window.HorsePen;

  /**
   * Exact MILP solver (non-blocking).
   *
   * Uses `glpk.js` (WASM) in a Web Worker to solve the same MILP formulation
   * as `milp_solver.py`, without freezing the UI.
   *
   * Returns a promise resolving to:
   *   { area, walls: [{x,y}], enclosed: Set("x,y"), debug: {...} }
   */
  HP.solveEnclosureMILPAsync = function solveEnclosureMILPAsync(gridData, maxWalls, opts = {}) {
    const timeBudgetMs = Math.max(1000, Math.min(HP.TIME_BUDGET_MS, opts.timeBudgetMs ?? HP.TIME_BUDGET_MS));
    const { promise } = startWorkerSolve(gridData, maxWalls, timeBudgetMs);
    return promise;
  };

  /**
   * Start a worker solve and return { promise, cancel }.
   */
  HP.startMILPWorkerSolve = function startMILPWorkerSolve(gridData, maxWalls, timeBudgetMs = HP.TIME_BUDGET_MS) {
    return startWorkerSolve(gridData, maxWalls, timeBudgetMs);
  };

  function startWorkerSolve(gridData, maxWalls, timeBudgetMs) {
    const { grid, width, height, horsePos } = gridData || {};
    if (!grid || !horsePos) {
      return {
        promise: Promise.reject(new Error("Missing grid/horse for MILP")),
        cancel: () => {},
      };
    }

    // Pack water mask for transfer
    const waterMask = new Uint8Array(width * height);
    const cherryMask = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        waterMask[idx] = grid[y][x] === "water" ? 1 : 0;
        cherryMask[idx] = grid[y][x] === "cherry" ? 1 : 0;
      }
    }
    // Ensure horse cell is treated as passable
    waterMask[horsePos.y * width + horsePos.x] = 0;
    cherryMask[horsePos.y * width + horsePos.x] = 0;

    const cherryBonus = HP.SCORING?.CHERRY_BONUS ?? 3;

    const worker = createMILPWorker();
    const solveId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;

    const cancel = () => {
      try {
        worker.terminate();
      } catch {}
    };

    const promise = new Promise((resolve, reject) => {
      // GLPK loads/compiles a WASM module on first use; allow extra startup headroom.
      const hardTimeoutMs = timeBudgetMs + 12000;
      const timeout = setTimeout(() => {
        cancel();
        reject(new Error(`MILP timed out (solve budget ${timeBudgetMs}ms)`));
      }, hardTimeoutMs);

      worker.onmessage = (ev) => {
        const msg = ev.data;
        if (!msg || msg.solveId !== solveId) return;
        clearTimeout(timeout);
        cancel();

        if (!msg.ok) {
          reject(new Error(msg.error || "MILP failed"));
          return;
        }

        const walls = msg.walls || [];
        const wallSet = new Set(walls.map((w) => `${w.x},${w.y}`));
        const { enclosed, escapes } = floodFillEnclosed(gridData, wallSet);
        if (escapes) {
          reject(new Error("MILP returned escaping wall set (bug)"));
          return;
        }

        resolve({
          area: enclosed.size,
          walls,
          enclosed,
          debug: { solver: "milp_worker", ms: msg.ms ?? null, status: msg.status ?? null },
        });
      };

      worker.onerror = (err) => {
        clearTimeout(timeout);
        cancel();
        // Some browsers hide worker error details; include whatever is available.
        const msg =
          err?.message ||
          (typeof err === "object" && err
            ? `${err.filename || "worker"}:${err.lineno || "?"}:${err.colno || "?"}`
            : "MILP worker error");
        reject(new Error(msg));
      };

      // transfer water mask buffer for performance
      worker.postMessage(
        {
          solveId,
          width,
          height,
          horseX: horsePos.x,
          horseY: horsePos.y,
          waterMask,
          cherryMask,
          cherryBonus,
          maxWalls,
          timeBudgetMs,
        },
        [waterMask.buffer, cherryMask.buffer]
      );
    });

    return { promise, cancel };
  }

  function floodFillEnclosed(data, wallSet) {
    const { grid, width, height, horsePos } = data;
    const inBounds = (x, y) => x >= 0 && x < width && y >= 0 && y < height;
    const isPassable = (x, y) => inBounds(x, y) && grid[y][x] !== "water";
    const key = (x, y) => `${x},${y}`;

    const startKey = key(horsePos.x, horsePos.y);
    if (wallSet.has(startKey)) return { enclosed: new Set(), escapes: true };

    const enclosed = new Set([startKey]);
    const q = [{ x: horsePos.x, y: horsePos.y }];
    let escapes = false;

    while (q.length) {
      const { x, y } = q.shift();
      if (x === 0 || x === width - 1 || y === 0 || y === height - 1) {
        escapes = true;
        break;
      }
      const neigh = [
        { x: x - 1, y },
        { x: x + 1, y },
        { x, y: y - 1 },
        { x, y: y + 1 },
      ];
      for (const n of neigh) {
        if (!isPassable(n.x, n.y)) continue;
        const nk = key(n.x, n.y);
        if (wallSet.has(nk) || enclosed.has(nk)) continue;
        enclosed.add(nk);
        q.push(n);
      }
    }

    return { enclosed, escapes };
  }

  function createMILPWorker() {
    const url = new URL("./src/solver/milpWorker.js", window.location.href).toString();
    return new Worker(url, { type: "module" });
  }
})();


