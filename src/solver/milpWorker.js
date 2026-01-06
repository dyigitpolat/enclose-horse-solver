// Web Worker (ES module) for the exact MILP solve using glpk.js (WASM).
// This keeps the UI responsive and avoids browser "long task" freezes.

import glpkInit from "../../vendor/glpk/glpk.js";

let glpPromise = null;

async function getGlp() {
  if (!glpPromise) glpPromise = glpkInit();
  return await glpPromise;
}

async function solveMILP(payload) {
  const t0 = Date.now();
  const { width, height, horseX, horseY, waterMask, cherryMask, cherryBonus = 3, maxWalls, timeBudgetMs } = payload;

  const glp = await getGlp();

  const total = width * height;
  // idOf: cellIndex -> nodeId (passable), -1 otherwise
  const idOf = new Int32Array(total);
  for (let i = 0; i < total; i++) idOf[i] = -1;
  const coordX = [];
  const coordY = [];
  const nodeCherry = [];
  for (let i = 0; i < total; i++) {
    if (waterMask[i] === 1) continue;
    idOf[i] = coordX.length;
    coordX.push(i % width);
    coordY.push((i / width) | 0);
    nodeCherry.push(cherryMask ? cherryMask[i] : 0);
  }

  const n = coordX.length;
  if (n === 0) throw new Error("empty passable grid");
  const horseIndex = horseY * width + horseX;
  const horseId = idOf[horseIndex];
  if (horseId < 0) throw new Error("horse not on passable cell");
  if (n > 1200) {
    throw new Error(
      `Grid too large for browser MILP solver: ${width}×${height} = ${n} passable cells (limit: 1200). ` +
      `Try a smaller image or use the Python solver for very large grids.`
    );
  }

  const edges = [];
  const und = new Set();
  const addUnd = (a, b) => {
    const u = a < b ? a : b;
    const v = a < b ? b : a;
    und.add(u + "," + v);
  };

  // Build directed edges + undirected set
  for (let id = 0; id < n; id++) {
    const x = coordX[id],
      y = coordY[id];
    const neigh = [
      [x - 1, y],
      [x + 1, y],
      [x, y - 1],
      [x, y + 1],
    ];
    for (const [nx, ny] of neigh) {
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      const nid = idOf[ny * width + nx];
      if (nid < 0) continue;
      edges.push([id, nid]);
      addUnd(id, nid);
    }
  }

  const boundary = [];
  for (let id = 0; id < n; id++) {
    const x = coordX[id],
      y = coordY[id];
    if (x === 0 || x === width - 1 || y === 0 || y === height - 1) boundary.push(id);
  }

  const M = n;

  // --- Build GLPK JSON model (same formulation as milp_solver.py) ---
  const constraints = {};
  const ensureConstraint = (name, bnds) => {
    if (!constraints[name]) constraints[name] = { name, bnds, vars: {} };
  };
  const addCoef = (cName, vName, coef) => {
    const c = constraints[cName];
    c.vars[vName] = (c.vars[vName] || 0) + coef;
  };

  // Helper bounds
  const FX = (v) => ({ type: glp.GLP_FX, lb: v, ub: v });
  const UP = (ub) => ({ type: glp.GLP_UP, lb: 0, ub });

  // Budget
  ensureConstraint("budget", UP(maxWalls));

  // Flow constraints (encode inflow - outflow)
  for (let v = 0; v < n; v++) {
    ensureConstraint("flow_" + v, v === horseId ? FX(1) : FX(0));
  }

  // Binary vars and simple constraints
  const binaries = [];
  const bounds = [];
  const objVars = [];

  for (let i = 0; i < n; i++) {
    const rName = "r_" + i;
    const wName = "w_" + i;

    binaries.push(rName, wName);
    bounds.push({ name: rName, type: glp.GLP_DB, lb: 0, ub: 1 });
    bounds.push({ name: wName, type: glp.GLP_DB, lb: 0, ub: 1 });

    objVars.push({ name: rName, coef: 1 + (nodeCherry[i] ? cherryBonus : 0) });

    // budget: sum w <= k
    addCoef("budget", wName, 1);

    // r + w <= 1
    const rw = "rw_" + i;
    ensureConstraint(rw, UP(1));
    addCoef(rw, rName, 1);
    addCoef(rw, wName, 1);
  }

  // Horse fixed
  ensureConstraint("horse_r", FX(1));
  addCoef("horse_r", "r_" + horseId, 1);
  ensureConstraint("horse_w", FX(0));
  addCoef("horse_w", "w_" + horseId, 1);

  // Cherries cannot have walls placed on them
  for (let i = 0; i < n; i++) {
    if (nodeCherry[i] === 1) {
      const c = "cherry_w_" + i;
      ensureConstraint(c, FX(0));
      addCoef(c, "w_" + i, 1);
    }
  }

  // Boundary cannot be reachable
  for (const b of boundary) {
    if (b === horseId) continue;
    const c = "boundary_" + b;
    ensureConstraint(c, FX(0));
    addCoef(c, "r_" + b, 1);
  }

  // Closure constraints
  for (const uv of und) {
    const parts = uv.split(",");
    const u = +parts[0],
      v = +parts[1];

    // r_u - w_v - r_v <= 0
    const c1 = "cl1_" + u + "_" + v;
    ensureConstraint(c1, UP(0));
    addCoef(c1, "r_" + u, 1);
    addCoef(c1, "w_" + v, -1);
    addCoef(c1, "r_" + v, -1);

    // r_v - w_u - r_u <= 0
    const c2 = "cl2_" + u + "_" + v;
    ensureConstraint(c2, UP(0));
    addCoef(c2, "r_" + v, 1);
    addCoef(c2, "w_" + u, -1);
    addCoef(c2, "r_" + u, -1);
  }

  // Horse flow equation: inflow - outflow + sum(r) = 1
  for (let i = 0; i < n; i++) {
    addCoef("flow_" + horseId, "r_" + i, 1);
  }

  // Flow vars and caps + conservation
  for (let ei = 0; ei < edges.length; ei++) {
    const u = edges[ei][0],
      v = edges[ei][1];
    const fName = "f_" + u + "_" + v;

    // 0 <= f <= M helps the solver a lot
    bounds.push({ name: fName, type: glp.GLP_DB, lb: 0, ub: M });

    // cap: f - M*r_u <= 0
    const cap1 = "cap1_" + u + "_" + v;
    ensureConstraint(cap1, UP(0));
    addCoef(cap1, fName, 1);
    addCoef(cap1, "r_" + u, -M);

    // cap: f - M*r_v <= 0
    const cap2 = "cap2_" + u + "_" + v;
    ensureConstraint(cap2, UP(0));
    addCoef(cap2, fName, 1);
    addCoef(cap2, "r_" + v, -M);

    // inflow/outflow coefficients
    addCoef("flow_" + v, fName, 1); // inflow to v
    addCoef("flow_" + u, fName, -1); // outflow from u
  }

  // For v != horse: inflow - outflow = r_v  => inflow - outflow - r_v = 0
  for (let v = 0; v < n; v++) {
    if (v === horseId) continue;
    addCoef("flow_" + v, "r_" + v, -1);
  }

  const subjectTo = Object.values(constraints).map((c) => ({
    name: c.name,
    bnds: c.bnds,
    vars: Object.entries(c.vars).map(([name, coef]) => ({ name, coef })),
  }));

  const model = {
    name: "horse_pen",
    objective: { direction: glp.GLP_MAX, name: "obj", vars: objVars },
    subjectTo,
    bounds,
    binaries,
  };

  const opts = {
    msglev: glp.GLP_MSG_OFF,
    presol: true,
    tmlim: Math.max(1, timeBudgetMs / 1000),
    mipgap: 0,
  };

  const res = glp.solve(model, opts);
  const status = res?.result?.status ?? glp.GLP_UNDEF;
  const isOptimal = status === glp.GLP_OPT;
  const isFeasible = status === glp.GLP_FEAS || isOptimal;
  if (!isFeasible) {
    throw new Error("GLPK returned status " + status);
  }

  const vars = res.result.vars || {};
  const walls = [];
  for (let i = 0; i < n; i++) {
    const val = vars["w_" + i] || 0;
    if (val > 0.5) walls.push({ x: coordX[i], y: coordY[i] });
  }

  return { walls, ms: Date.now() - t0, status, isOptimal };
}

self.onmessage = (ev) => {
  const payload = ev.data;
  const solveId = payload && payload.solveId;
  (async () => {
    try {
      const out = await solveMILP(payload);
      self.postMessage({
        ok: true,
        solveId,
        walls: out.walls,
        ms: out.ms,
        status: out.status,
        isOptimal: out.isOptimal,
      });
    } catch (e) {
      self.postMessage({ ok: false, solveId, error: e?.message ? e.message : String(e) });
    }
  })();
};

self.addEventListener("unhandledrejection", (ev) => {
  try {
    self.postMessage({ ok: false, solveId: null, error: String(ev?.reason || "Unhandled promise rejection") });
  } catch {}
});


