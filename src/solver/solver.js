(() => {
  const HP = window.HorsePen;

  /**
   * Async solve entrypoint (non-blocking).
   * Uses MILP in a worker (exact). Presets are disabled.
   */
  HP.solveEnclosure = async function solveEnclosure(data, maxWalls, opts = {}) {
    const milp = await HP.solveEnclosureMILPAsync?.(data, maxWalls, opts);
    if (!milp) throw new Error("MILP returned null");
    return milp;
  };
})();


