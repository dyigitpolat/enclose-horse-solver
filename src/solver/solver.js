(() => {
  const HP = window.HorsePen;

  /**
   * Async solve entrypoint (non-blocking).
   * Uses MILP in a worker (exact). Presets are disabled.
   */
  HP.solveEnclosure = async function solveEnclosure(data, maxWalls, opts = {}) {
    const milpSolve = async (gridData) => {
      const out = await HP.solveEnclosureMILPAsync?.(gridData, maxWalls, opts);
      if (!out) throw new Error("MILP returned null");
      return out;
    };

    const isBudgetExhausted = (sol) => {
      const d = sol?.debug || {};
      const ms = d.ms;
      const budgetMs = d.timeBudgetMs;
      const isOptimal = d.isOptimal;
      if (ms == null || budgetMs == null) return isOptimal === false;
      const used = ms >= Math.max(0, budgetMs - 75);
      return used && isOptimal !== true;
    };

    const floodFillEnclosedWithWalls = (gridData, wallSet) => {
      const { grid, width, height, horsePos } = gridData;
      const inBounds = (x, y) => x >= 0 && x < width && y >= 0 && y < height;
      const key = (x, y) => `${x},${y}`;
      const isPassable = (x, y) => {
        if (!inBounds(x, y)) return false;
        if (grid[y][x] === "water") return false;
        if (wallSet.has(key(x, y))) return false;
        return true;
      };

      const startKey = key(horsePos.x, horsePos.y);
      if (!isPassable(horsePos.x, horsePos.y)) return { enclosed: new Set(), escapes: true };

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
          if (enclosed.has(nk)) continue;
          enclosed.add(nk);
          q.push(n);
        }
      }

      return { enclosed, escapes };
    };

    const scoreOnFullGrid = (fullGridData, enclosed) => {
      const { grid, width, height } = fullGridData;
      const cherryBonus = HP.SCORING?.CHERRY_BONUS ?? 3;
      let cherriesInPen = 0;
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          if (grid[y][x] !== "cherry") continue;
          if (enclosed.has(`${x},${y}`)) cherriesInPen++;
        }
      }
      const area = enclosed.size;
      const score = area + cherryBonus * cherriesInPen;
      return { area, cherries: cherriesInPen, score };
    };

    const computeCropBounds = (width, height, horsePos, frac) => {
      const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
      const winW = clamp(Math.round(width * frac), 5, width);
      const winH = clamp(Math.round(height * frac), 5, height);

      let x0 = Math.floor(horsePos.x - winW / 2);
      x0 = clamp(x0, 0, width - winW);
      let y0 = Math.floor(horsePos.y - winH / 2);
      y0 = clamp(y0, 0, height - winH);

      return { x0, y0, x1: x0 + winW, y1: y0 + winH, frac };
    };

    const cropGridData = (fullGridData, bounds) => {
      const { grid, horsePos } = fullGridData;
      const { x0, y0, x1, y1 } = bounds;
      const subGrid = [];
      for (let y = y0; y < y1; y++) {
        subGrid.push(grid[y].slice(x0, x1));
      }
      return {
        grid: subGrid,
        width: x1 - x0,
        height: y1 - y0,
        horsePos: { x: horsePos.x - x0, y: horsePos.y - y0 },
        debug: { croppedFrom: { x0, y0, x1, y1 } },
      };
    };

    // 1) First attempt: full grid
    let fullAttempt = null;
    let fullAttemptErr = null;
    try {
      fullAttempt = await milpSolve(data);
    } catch (e) {
      fullAttemptErr = e;
    }

    if (fullAttempt && !isBudgetExhausted(fullAttempt)) return fullAttempt;

    // 2) If the budget was exhausted (or the solve timed out), retry on progressively smaller crops around the horse.
    if (!data?.horsePos) {
      if (fullAttemptErr) throw fullAttemptErr;
      throw new Error("Missing horse position");
    }

    const maxCropAttempts = Math.max(1, Math.min(10, Number(opts.cropMaxAttempts ?? 8)));
    const shrink = Number(opts.cropShrinkFactor ?? 0.8);

    // Keep best valid solution found on the FULL grid (for safety), but stop once we get a non-exhausted crop solve.
    let best = fullAttempt;
    if (best && best?.debug?.solver === "milp_worker") {
      best.debug = { ...best.debug, cropRetry: { used: false } };
    }

    let frac = shrink;
    for (let attempt = 1; attempt <= maxCropAttempts; attempt++) {
      const bounds = computeCropBounds(data.width, data.height, data.horsePos, frac);
      const croppedData = cropGridData(data, bounds);

      let cropSol = null;
      try {
        cropSol = await milpSolve(croppedData);
      } catch {
        // If even the cropped solve fails, shrink again.
        frac *= shrink;
        continue;
      }

      // Map walls back to full-grid coords and re-evaluate enclosure on the full grid.
      const wallsFull = (cropSol.walls || []).map((w) => ({ x: w.x + bounds.x0, y: w.y + bounds.y0 }));
      const wallSetFull = new Set(wallsFull.map((w) => `${w.x},${w.y}`));
      const { enclosed, escapes } = floodFillEnclosedWithWalls(data, wallSetFull);
      if (escapes) {
        frac *= shrink;
        continue;
      }

      const scored = scoreOnFullGrid(data, enclosed);
      const mapped = {
        area: scored.area,
        score: scored.score,
        cherries: scored.cherries,
        walls: wallsFull,
        enclosed,
        debug: {
          ...(cropSol.debug || {}),
          solver: "milp_worker",
          cropRetry: {
            used: true,
            attempt,
            frac,
            bounds,
            exhausted: isBudgetExhausted(cropSol),
          },
        },
      };

      // Track best by score (full-grid evaluation)
      if (!best || (mapped.score ?? 0) > (best.score ?? 0)) {
        best = mapped;
      }

      // Stop once we get a crop solve that did NOT exhaust its budget.
      if (!isBudgetExhausted(cropSol)) {
        return mapped;
      }

      frac *= shrink;
    }

    // If all crop attempts exhausted or failed, return the best we have (likely the full-grid incumbent).
    if (best) return best;
    if (fullAttemptErr) throw fullAttemptErr;
    throw new Error("MILP failed");
  };
})();


