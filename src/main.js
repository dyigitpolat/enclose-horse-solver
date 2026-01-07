// ============ STATE ============
let imageLoaded = false;
let gridData = null;
let solution = null;
let detectedGrid = null; // { cols, rows }
let cancelSolve = null;
let resizeRaf = null;
let loadToken = 0;

// ============ DOM ============
const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const analyzeBtn = document.getElementById("analyzeBtn");
const status = document.getElementById("status");
const loading = document.getElementById("loading");
const loadingText = loading?.querySelector(".loading-text");
const resultsContent = document.getElementById("resultsContent");
const placeholder = document.getElementById("placeholder");
const gridCanvas = document.getElementById("gridCanvas");
const detectionCanvas = document.getElementById("detectionCanvas");

const autoDetectInfo = document.getElementById("autoDetectInfo");
const detectedDims = document.getElementById("detectedDims");
const parseSummary = document.getElementById("parseSummary");

const wallCountInput = document.getElementById("wallCount");
const timeBudgetSecInput = document.getElementById("timeBudgetSec");

const areaValue = document.getElementById("areaValue");
const wallsUsed = document.getElementById("wallsUsed");
const wallsLeft = document.getElementById("wallsLeft");
const efficiency = document.getElementById("efficiency");
const wallList = document.getElementById("wallList");

// ============ EVENTS ============
uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("dragover");
});
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", (e) => {
  if (e.target.files.length) handleFile(e.target.files[0]);
});
analyzeBtn.addEventListener("click", analyze);

document.getElementById("debugToggle").addEventListener("click", () => {
  const info = document.getElementById("debugInfo");
  const toggle = document.getElementById("debugToggle");
  info.classList.toggle("visible");
  toggle.textContent = info.classList.contains("visible")
    ? "▾ Hide Debug Info"
    : "▸ Show Debug Info";
});

// ============ UI HELPERS ============
function showStatus(message, type) {
  status.textContent = message;
  status.className = `status ${type}`;
}

function setLoadingText(text) {
  if (loadingText) loadingText.textContent = text;
}

function clearCanvas(canvas) {
  const ctx = canvas?.getContext?.("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// ============ FILE HANDLING ============
function handleFile(file) {
  const token = (loadToken += 1);

  if (!file.type.startsWith("image/")) {
    showStatus("Please upload an image file", "error");
    return;
  }

  // Reset UI immediately (so old minimap/solution doesn't linger)
  imageLoaded = false;
  setLoadingText("Loading image...");
  loading.classList.add("active");
  placeholder.style.display = "none";
  resultsContent.style.display = "flex";
  clearCanvas(gridCanvas);
  areaValue.textContent = "0";
  wallsUsed.textContent = "0";
  wallsLeft.textContent = "0";
  efficiency.textContent = "0";
  wallList.innerHTML = "—";

  // Reset per-image state
  detectedGrid = null;
  autoDetectInfo.style.display = "none";
  parseSummary.textContent = "";
  gridData = null;
  solution = null;
  if (typeof cancelSolve === "function") cancelSolve();
  cancelSolve = null;

  // Update thumbnail immediately (avoid blocking on FileReader base64 conversion).
  // Revoke any previous object URL to avoid leaking blobs.
  const prevUrl = preview.dataset?.blobUrl;
  if (prevUrl) {
    URL.revokeObjectURL(prevUrl);
    delete preview.dataset.blobUrl;
  }

  preview.onload = null;
  preview.onerror = null;

  const blobUrl = URL.createObjectURL(file);
  preview.dataset.blobUrl = blobUrl;
  preview.src = blobUrl;
  preview.style.display = "block";
  uploadZone.classList.add("has-image");
  uploadZone.querySelector(".upload-icon").style.display = "none";
  uploadZone.querySelector(".upload-text").style.display = "none";

  preview.onload = () => {
    // Ignore stale loads if the user picks another file quickly.
    if (token !== loadToken) return;

    // The thumbnail is ready; free the blob URL resource.
    const url = preview.dataset?.blobUrl;
    if (url) {
      URL.revokeObjectURL(url);
      delete preview.dataset.blobUrl;
    }

    imageLoaded = true;
    analyzeBtn.disabled = true;

    // Yield so the browser can paint the new thumbnail + minimap loading overlay
    // before we run CPU-heavy parsing on the main thread.
    setTimeout(() => {
      if (token !== loadToken) return;

      try {
        const dims = window.HorsePen.detectGridDimensions(preview, detectionCanvas);
        detectedGrid = dims;
        detectedDims.textContent = `${dims.cols} × ${dims.rows}`;
        autoDetectInfo.style.display = "block";

        // Parse immediately to verify detection is sane and the horse is visible.
        const parsed = window.HorsePen.parseImageToGrid(preview, detectionCanvas, dims.cols, dims.rows);
        const validation = validateParsedGrid(parsed);
        if (!validation.ok) {
          parseSummary.textContent = "";
          loading.classList.remove("active");
          setLoadingText("Calculating optimal enclosure...");
          resultsContent.style.display = "none";
          placeholder.style.display = "flex";
          showStatus(`Image processing failed: ${validation.reason}`, "error");
          analyzeBtn.disabled = true;
          return;
        }

        gridData = parsed;
        parseSummary.textContent = validation.summary;
        analyzeBtn.disabled = false;
        showStatus("Image loaded and processed successfully.", "success");

        const wallCount = parseInt(wallCountInput.value, 10);
        // Render on the next frame so layout (panel sizing) is finalized; otherwise the canvas can be tiny/appear blank.
        requestAnimationFrame(() => {
          try {
            window.HorsePen.renderPreview({
              gridData,
              maxWalls: Number.isFinite(wallCount) ? wallCount : 0,
              gridCanvas,
              areaValueEl: areaValue,
              wallsUsedEl: wallsUsed,
              wallsLeftEl: wallsLeft,
              efficiencyEl: efficiency,
              wallListEl: wallList,
            });
          } catch (e) {
            // eslint-disable-next-line no-console
            console.error("renderPreview failed", e);
          } finally {
            loading.classList.remove("active");
            setLoadingText("Calculating optimal enclosure...");
          }
        });
      } catch (err) {
        loading.classList.remove("active");
        setLoadingText("Calculating optimal enclosure...");
        resultsContent.style.display = "none";
        placeholder.style.display = "flex";
        const msg = String(err?.message || err);
        showStatus(`Image processing failed: ${msg}`, "error");
        analyzeBtn.disabled = true;
      }
    }, 0);
  };

  preview.onerror = () => {
    if (token !== loadToken) return;
    loading.classList.remove("active");
    setLoadingText("Calculating optimal enclosure...");
    resultsContent.style.display = "none";
    placeholder.style.display = "flex";
    showStatus("Failed to load image.", "error");
    analyzeBtn.disabled = true;
  };
}

// ============ ANALYSIS ============
function analyze() {
  if (!imageLoaded) return;

  // Prevent repeated clicks from spawning multiple concurrent solves.
  analyzeBtn.disabled = true;

  setLoadingText("Calculating optimal enclosure...");
  loading.classList.add("active");
  placeholder.style.display = "none";
  // Use flex layout so the canvas can grow to fill available space (desktop + mobile).
  resultsContent.style.display = "flex";

  setTimeout(async () => {
    try {
      // Grid data is parsed+validated on upload. If missing, do it now.
      if (!gridData) {
        if (!detectedGrid) {
          detectedGrid = window.HorsePen.detectGridDimensions(preview, detectionCanvas);
          detectedDims.textContent = `${detectedGrid.cols} × ${detectedGrid.rows}`;
          autoDetectInfo.style.display = "block";
        }
        gridData = window.HorsePen.parseImageToGrid(
          preview,
          detectionCanvas,
          detectedGrid.cols,
          detectedGrid.rows
        );
        const validation = validateParsedGrid(gridData);
        if (!validation.ok) throw new Error(validation.reason);
        parseSummary.textContent = validation.summary;
      }

      const debugText = `Detected: ${gridData.debug.waterCount} water, ${gridData.debug.grassCount} grass, horse at ${
        gridData.horsePos ? `(${gridData.horsePos.x}, ${gridData.horsePos.y})` : "NOT FOUND"
      } | grid ${gridData.width}×${gridData.height} | cherries ${gridData.debug.cherryCount ?? 0} | horse method ${
        gridData.debug.horseMethod
      }`;
      document.getElementById("debugInfo").textContent = debugText;

      const wallCount = parseInt(wallCountInput.value, 10);
      const budgetSecRaw = parseFloat(timeBudgetSecInput?.value ?? "10");
      const budgetSec = Number.isFinite(budgetSecRaw) ? Math.max(1, Math.min(120, budgetSecRaw)) : 10;
      const scaledBudgetSec = Math.round(budgetSec * 0.3);
      const timeBudgetMs = Math.round(scaledBudgetSec * 1000);

      // Cancel any in-flight solve
      if (typeof cancelSolve === "function") cancelSolve();
      cancelSolve = null;

      showStatus(`Solving (MILP, up to ${budgetSec}s)…`, "success");

      // Prefer async MILP (worker) to avoid UI freezes; solver.js handles fallback.
      const solvePromise = window.HorsePen.solveEnclosure(gridData, wallCount, { timeBudgetMs });
      solution = await solvePromise;

      window.HorsePen.renderResults({
        gridData,
        solution,
        maxWalls: wallCount,
        gridCanvas,
        areaValueEl: areaValue,
        wallsUsedEl: wallsUsed,
        wallsLeftEl: wallsLeft,
        efficiencyEl: efficiency,
        wallListEl: wallList,
      });

      loading.classList.remove("active");
      analyzeBtn.disabled = false;
      if (solution?.debug?.solver === "milp_worker") {
        const usedMs = solution.debug.ms ?? null;
        const isOptimal = solution.debug.isOptimal;
        const budgetMs = solution.debug.timeBudgetMs ?? timeBudgetMs;
        const budgetUsed = usedMs != null && budgetMs != null ? usedMs >= Math.max(0, budgetMs - 50) : false;
        const cropRetry = solution?.debug?.cropRetry;
        const usedCrop = !!cropRetry?.used;

        if (usedCrop) {
          const b = cropRetry?.bounds;
          const boundsText = b ? ` (view x:${b.x0}..${b.x1 - 1}, y:${b.y0}..${b.y1 - 1})` : "";
          const exhaustedText =
            cropRetry?.exhausted === true
              ? ` Time budget was exhausted again; returned best found.`
              : ` Cropped retry finished within budget.`;
          showStatus(
            `Solved with cropped retry around the horse${boundsText}.${exhaustedText} Solution may be suboptimal for the full map.`,
            "warning"
          );
        } else if (isOptimal === false || (isOptimal !== true && budgetUsed)) {
          showStatus(
            `Time budget exhausted (${Math.round(budgetMs / 1000)}s). Returned best found: score ${solution.score} (area ${solution.area}, ${solution.walls.length} walls). Increase time budget and re-run to improve / prove optimality.`,
            "warning"
          );
        } else {
          showStatus(
            `MILP solved in ${usedMs ?? "?"}ms: score ${solution.score} (area ${solution.area}, ${solution.walls.length} walls)`,
            "success"
          );
        }
      } else {
        showStatus(`Found enclosure of ${solution.area} tiles using ${solution.walls.length} walls`, "success");
      }
    } catch (err) {
      loading.classList.remove("active");
      analyzeBtn.disabled = false;
      // Keep the minimap visible (preview) even if optimization fails.
      if (gridData) {
        placeholder.style.display = "none";
        resultsContent.style.display = "flex";
        window.HorsePen.renderPreview({
          gridData,
          maxWalls: parseInt(wallCountInput.value, 10),
          gridCanvas,
          areaValueEl: areaValue,
          wallsUsedEl: wallsUsed,
          wallsLeftEl: wallsLeft,
          efficiencyEl: efficiency,
          wallListEl: wallList,
        });
      } else {
        resultsContent.style.display = "none";
        placeholder.style.display = "flex";
      }
      const msg = String(err?.message || err);
      if (msg.includes("timed out")) {
        showStatus(`Error: ${msg}. Try increasing the time budget and re-run.`, "warning");
      } else {
        showStatus(`Error: ${msg}`, "error");
      }
      // eslint-disable-next-line no-console
      console.error(err);
    }
  }, 50);
}

// Keep the map fitting the viewport while resizing.
window.addEventListener("resize", () => {
  if (!gridData) return;
  if (resizeRaf) cancelAnimationFrame(resizeRaf);
  resizeRaf = requestAnimationFrame(() => {
    resizeRaf = null;
    try {
      if (solution) {
        window.HorsePen.renderResults({
          gridData,
          solution,
          maxWalls: parseInt(wallCountInput.value, 10),
          gridCanvas,
          areaValueEl: areaValue,
          wallsUsedEl: wallsUsed,
          wallsLeftEl: wallsLeft,
          efficiencyEl: efficiency,
          wallListEl: wallList,
        });
      } else {
        window.HorsePen.renderPreview({
          gridData,
          maxWalls: parseInt(wallCountInput.value, 10),
          gridCanvas,
          areaValueEl: areaValue,
          wallsUsedEl: wallsUsed,
          wallsLeftEl: wallsLeft,
          efficiencyEl: efficiency,
          wallListEl: wallList,
        });
      }
    } catch {
      // ignore resize redraw errors
    }
  });
});

function validateParsedGrid(parsed) {
  if (!parsed || !parsed.grid || !parsed.horsePos) {
    return { ok: false, reason: "horse not found", summary: "" };
  }

  const total = parsed.width * parsed.height;
  const water = parsed.debug?.waterCount ?? 0;
  const grass = parsed.debug?.grassCount ?? 0;

  const method = parsed.debug?.horseMethod ?? "unknown";

  // Sanity checks: extremely unlikely to be correct if these fail.
  if (total < 25) return { ok: false, reason: "grid too small", summary: "" };
  if (water === 0) return { ok: false, reason: "no water detected (bad grid parse)", summary: "" };
  if (water >= total - 1) return { ok: false, reason: "almost all water (bad grid parse)", summary: "" };

  // Horse detection: brightest_square is the only method now (always reliable)

  const { x, y } = parsed.horsePos;
  const cherries = parsed.debug?.cherryCount ?? 0;
  const summary = `Parsed: horse (${x}, ${y}) • water ${water}/${total} • cherries ${cherries} • method ${method}`;
  return { ok: true, reason: "", summary };
}


