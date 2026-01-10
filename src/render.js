(() => {
  const HP = window.HorsePen;

  HP.renderPreview = function renderPreview({
    gridData,
    maxWalls,
    gridCanvas,
    areaValueEl,
    wallsUsedEl,
    wallsLeftEl,
    efficiencyEl,
    wallListEl,
  }) {
    if (!gridData) return;
    const { grid, width, height, horsePos } = gridData;

    // Stats: not optimized yet
    if (areaValueEl) areaValueEl.textContent = "0";
    if (wallsUsedEl) wallsUsedEl.textContent = "0";
    if (wallsLeftEl) wallsLeftEl.textContent = String(Number.isFinite(maxWalls) ? maxWalls : 0);
    if (efficiencyEl) efficiencyEl.textContent = "0";
    if (wallListEl) wallListEl.innerHTML = '<em>Not optimized yet — click "Analyze &amp; Optimize".</em>';

    // Draw base grid (no walls/enclosure)
    const canvas = gridCanvas;
    const ctx = canvas.getContext("2d");

    const cellSize = computeCellSizeToFitViewport(canvas, width, height);
    const dpr = window.devicePixelRatio || 1;
    canvas.style.width = `${width * cellSize}px`;
    canvas.style.height = `${height * cellSize}px`;
    canvas.width = Math.round(width * cellSize * dpr);
    canvas.height = Math.round(height * cellSize * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = false;

    const rootStyle = getComputedStyle(document.documentElement);
    const cssVar = (name, fallback) => (rootStyle.getPropertyValue(name).trim() || fallback);
    const colors = {
      grass: cssVar("--grass", "#2d5a3d"),
      grassDark: cssVar("--grass-light", "#1f4029"),
      water: cssVar("--water", "#2f79b8"),
      waterDark: cssVar("--water-dark", "#1f5688"),
      cherry: cssVar("--cherry", "#ff3b3b"),
      cherryLeaf: cssVar("--cherry-leaf", "#2ecc71"),
      horse: cssVar("--horse", "#ffffff"),
      horseBorder: "#5dff88",
    };

    // Build portal cell -> hue color map from portal pairs
    const portalColorMap = new Map();
    const portalPairs = gridData.debug?.portalPairs || gridData.portalPairs || [];
    for (const pair of portalPairs) {
      const hue = pair.h ?? pair.hueKey ?? 270; // fallback to purple
      const colorRgb = hslToRgb(hue, 0.85, 0.65);
      const color = `rgb(${colorRgb[0]},${colorRgb[1]},${colorRgb[2]})`;
      portalColorMap.set(`${pair.a.x},${pair.a.y}`, color);
      portalColorMap.set(`${pair.b.x},${pair.b.y}`, color);
    }

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const cellType = grid[y][x];
        let color;
        if (cellType === "water") {
          color = (x + y) % 2 === 0 ? colors.water : colors.waterDark;
        } else {
          color = (x + y) % 2 === 0 ? colors.grass : colors.grassDark;
        }

        ctx.fillStyle = color;
        ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);

        // Draw cherry marker on top of the tile.
        if (cellType === "cherry") {
          const cx = x * cellSize + cellSize * 0.5;
          const cy = y * cellSize + cellSize * 0.6;
          const r = Math.max(2, cellSize * 0.12);
          ctx.fillStyle = colors.cherry;
          ctx.beginPath();
          ctx.arc(cx - r * 0.55, cy, r, 0, Math.PI * 2);
          ctx.arc(cx + r * 0.55, cy, r, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = colors.cherryLeaf;
          ctx.fillRect(cx - 1, cy - r - 3, 2, 4);
        }

        // Draw portal marker (swirling oval gateway) - color-coded by pair
        if (cellType === "portal") {
          const portalKey = `${x},${y}`;
          const portalColor = portalColorMap.get(portalKey) || "rgb(148,0,211)"; // fallback purple
          
          const cx = x * cellSize + cellSize * 0.5;
          const cy = y * cellSize + cellSize * 0.5;
          const r = Math.max(3, cellSize * 0.25);
          
          // Parse RGB for lighter/darker variants
          const match = portalColor.match(/rgb\((\d+),(\d+),(\d+)\)/);
          const [_, rVal, gVal, bVal] = match ? match.map(Number) : [0, 148, 0, 211];
          
          // Outer glow (lighter, more transparent)
          ctx.fillStyle = `rgba(${Math.min(255, rVal + 40)},${Math.min(255, gVal + 40)},${Math.min(255, bVal + 40)},0.4)`;
          ctx.beginPath();
          ctx.ellipse(cx, cy, r * 1.1, r * 0.7, 0, 0, Math.PI * 2);
          ctx.fill();
          
          // Portal oval (main color)
          ctx.fillStyle = `rgba(${rVal},${gVal},${bVal},0.85)`;
          ctx.beginPath();
          ctx.ellipse(cx, cy, r * 0.9, r * 0.55, 0, 0, Math.PI * 2);
          ctx.fill();
          
          // Inner swirl/highlight (lighter)
          ctx.fillStyle = `rgba(${Math.min(255, rVal + 70)},${Math.min(255, gVal + 70)},${Math.min(255, bVal + 70)},0.9)`;
          ctx.beginPath();
          ctx.ellipse(cx - r * 0.2, cy - r * 0.1, r * 0.4, r * 0.25, Math.PI * 0.3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // Draw horse
    if (horsePos) {
      const hx = horsePos.x * cellSize + cellSize / 2;
      const hy = horsePos.y * cellSize + cellSize / 2;
      const hr = cellSize * 0.35;

      ctx.fillStyle = colors.horse;
      ctx.beginPath();
      ctx.arc(hx, hy, hr, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = colors.horseBorder;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  };

  HP.renderResults = function renderResults({
  gridData,
  solution,
  maxWalls,
  gridCanvas,
  areaValueEl,
  wallsUsedEl,
  wallsLeftEl,
  efficiencyEl,
  wallListEl,
}) {
  const { grid, width, height, horsePos } = gridData;
  const { walls, area, enclosed } = solution;

  const wallSet = new Set(walls.map((w) => `${w.x},${w.y}`));

  const view = solution?.debug?.cropRetry?.used ? solution.debug.cropRetry.bounds : null;
  const xStart = view ? view.x0 : 0;
  const yStart = view ? view.y0 : 0;
  const xEnd = view ? view.x1 : width;
  const yEnd = view ? view.y1 : height;
  const viewW = xEnd - xStart;
  const viewH = yEnd - yStart;

  // Stats
  areaValueEl.textContent = String(area);
  wallsUsedEl.textContent = String(walls.length);
  wallsLeftEl.textContent = String(maxWalls - walls.length);

  const cherryBonus = HP.SCORING?.CHERRY_BONUS ?? 3;
  let cherriesInPen = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (grid[y][x] !== "cherry") continue;
      if (enclosed.has(`${x},${y}`)) cherriesInPen++;
    }
  }
  const score = area + cherryBonus * cherriesInPen;
  efficiencyEl.textContent = String(score);

  // Draw
  const canvas = gridCanvas;
  const ctx = canvas.getContext("2d");

  const cellSize = computeCellSizeToFitViewport(canvas, viewW, viewH);
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = `${viewW * cellSize}px`;
  canvas.style.height = `${viewH * cellSize}px`;
  canvas.width = Math.round(viewW * cellSize * dpr);
  canvas.height = Math.round(viewH * cellSize * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = false;

  const rootStyle = getComputedStyle(document.documentElement);
  const cssVar = (name, fallback) => (rootStyle.getPropertyValue(name).trim() || fallback);
  const colors = {
    grass: cssVar("--grass", "#2d5a3d"),
    grassDark: cssVar("--grass-light", "#1f4029"),
    water: cssVar("--water", "#2f79b8"),
    waterDark: cssVar("--water-dark", "#1f5688"),
    wall: cssVar("--wall", "#e8b86d"),
    enclosed: cssVar("--enclosed", "rgba(255, 140, 0, 0.33)"),
    enclosedBorder: cssVar("--enclosed-border", "#ffb000"),
    cherry: cssVar("--cherry", "#ff3b3b"),
    cherryLeaf: cssVar("--cherry-leaf", "#2ecc71"),
    horse: cssVar("--horse", "#ffffff"),
    horseBorder: "#5dff88",
  };

  // Build portal cell -> hue color map from portal pairs
  const portalColorMap = new Map();
  const portalPairs = gridData.debug?.portalPairs || gridData.portalPairs || [];
  for (const pair of portalPairs) {
    const hue = pair.h ?? pair.hueKey ?? 270; // fallback to purple
    const colorRgb = hslToRgb(hue, 0.85, 0.65);
    const color = `rgb(${colorRgb[0]},${colorRgb[1]},${colorRgb[2]})`;
    portalColorMap.set(`${pair.a.x},${pair.a.y}`, color);
    portalColorMap.set(`${pair.b.x},${pair.b.y}`, color);
  }

  for (let y = yStart; y < yEnd; y++) {
    for (let x = xStart; x < xEnd; x++) {
      const cellType = grid[y][x];
      const cellKey = `${x},${y}`;
      const isWall = wallSet.has(cellKey);
      const isEnclosed = enclosed.has(cellKey) && !isWall;

      const dx = x - xStart;
      const dy = y - yStart;

      let color;
      if (isWall) {
        color = colors.wall;
      } else if (cellType === "water") {
        color = (x + y) % 2 === 0 ? colors.water : colors.waterDark;
      } else if (isEnclosed) {
        color = colors.enclosed;
      } else {
        color = (x + y) % 2 === 0 ? colors.grass : colors.grassDark;
      }

      ctx.fillStyle = color;
      ctx.fillRect(dx * cellSize, dy * cellSize, cellSize, cellSize);

      // Draw cherry marker on top of the tile (even if enclosed).
      if (!isWall && cellType === "cherry") {
        const cx = dx * cellSize + cellSize * 0.5;
        const cy = dy * cellSize + cellSize * 0.6;
        const r = Math.max(2, cellSize * 0.12);
        ctx.fillStyle = colors.cherry;
        ctx.beginPath();
        ctx.arc(cx - r * 0.55, cy, r, 0, Math.PI * 2);
        ctx.arc(cx + r * 0.55, cy, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = colors.cherryLeaf;
        ctx.fillRect(cx - 1, cy - r - 3, 2, 4);
      }

      // Draw portal marker (even if enclosed, never on walls) - color-coded by pair
      if (!isWall && cellType === "portal") {
        const portalKey = `${x},${y}`;
        const portalColor = portalColorMap.get(portalKey) || "rgb(148,0,211)"; // fallback purple
        
        const cx = dx * cellSize + cellSize * 0.5;
        const cy = dy * cellSize + cellSize * 0.5;
        const r = Math.max(3, cellSize * 0.25);
        
        // Parse RGB for lighter/darker variants
        const match = portalColor.match(/rgb\((\d+),(\d+),(\d+)\)/);
        const [_, rVal, gVal, bVal] = match ? match.map(Number) : [0, 148, 0, 211];
        
        // Outer glow (lighter, more transparent)
        ctx.fillStyle = `rgba(${Math.min(255, rVal + 40)},${Math.min(255, gVal + 40)},${Math.min(255, bVal + 40)},0.4)`;
        ctx.beginPath();
        ctx.ellipse(cx, cy, r * 1.1, r * 0.7, 0, 0, Math.PI * 2);
        ctx.fill();
        
        // Portal oval (main color)
        ctx.fillStyle = `rgba(${rVal},${gVal},${bVal},0.85)`;
        ctx.beginPath();
        ctx.ellipse(cx, cy, r * 0.9, r * 0.55, 0, 0, Math.PI * 2);
        ctx.fill();
        
        // Inner swirl/highlight (lighter)
        ctx.fillStyle = `rgba(${Math.min(255, rVal + 70)},${Math.min(255, gVal + 70)},${Math.min(255, bVal + 70)},0.9)`;
        ctx.beginPath();
        ctx.ellipse(cx - r * 0.2, cy - r * 0.1, r * 0.4, r * 0.25, Math.PI * 0.3, 0, Math.PI * 2);
        ctx.fill();
      }

      if (isEnclosed && !isWall) {
        ctx.strokeStyle = colors.enclosedBorder;
        ctx.lineWidth = 2;
        ctx.strokeRect(dx * cellSize + 1, dy * cellSize + 1, cellSize - 2, cellSize - 2);
      }

      if (isWall) {
        ctx.fillStyle = "rgba(0,0,0,0.3)";
        ctx.fillRect(dx * cellSize + 2, dy * cellSize + cellSize - 6, cellSize - 4, 4);

        ctx.strokeStyle = "rgba(139, 90, 43, 0.8)";
        ctx.lineWidth = 2;
        const pad = cellSize * 0.25;
        ctx.beginPath();
        ctx.moveTo(dx * cellSize + pad, dy * cellSize + pad);
        ctx.lineTo(dx * cellSize + cellSize - pad, dy * cellSize + cellSize - pad);
        ctx.moveTo(dx * cellSize + cellSize - pad, dy * cellSize + pad);
        ctx.lineTo(dx * cellSize + pad, dy * cellSize + cellSize - pad);
        ctx.stroke();
      }
    }
  }

  // Draw horse
  if (horsePos) {
    // Only draw if horse is inside the current view bounds.
    if (horsePos.x >= xStart && horsePos.x < xEnd && horsePos.y >= yStart && horsePos.y < yEnd) {
      const hx = (horsePos.x - xStart) * cellSize + cellSize / 2;
      const hy = (horsePos.y - yStart) * cellSize + cellSize / 2;
    const hr = cellSize * 0.35;

    ctx.fillStyle = colors.horse;
    ctx.beginPath();
    ctx.arc(hx, hy, hr, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = colors.horseBorder;
    ctx.lineWidth = 2;
    ctx.stroke();
    }
  }

  // Wall list
  if (walls.length > 0) {
    wallListEl.innerHTML = walls
      .map((w) => `<span class="wall-coord-item">(${w.x}, ${w.y})</span>`)
      .join("");
  } else {
    wallListEl.innerHTML = "<em>No walls needed - horse is naturally enclosed by water!</em>";
  }
};

  function computeCellSizeToFitViewport(canvas, gridW, gridH) {
    const container = canvas.parentElement;
    const style = container ? getComputedStyle(container) : null;
    const padX = (style ? parseFloat(style.paddingLeft) + parseFloat(style.paddingRight) : 0) || 0;
    const padY = (style ? parseFloat(style.paddingTop) + parseFloat(style.paddingBottom) : 0) || 0;

    let availableW = container ? container.clientWidth - padX : 0;
    let availableH = container ? container.clientHeight - padY : 0;

    // Fallback if layout hasn't resolved yet.
    if (!Number.isFinite(availableW) || availableW <= 0 || !Number.isFinite(availableH) || availableH <= 0) {
      const rect = container?.getBoundingClientRect?.() ?? { top: 0, width: 900 };
      const margin = 16;
      availableW = Math.max(80, rect.width - padX);
      availableH = Math.max(120, window.innerHeight - rect.top - margin - padY);
    }

    let cell = Math.floor(Math.min(availableW / gridW, availableH / gridH));
    cell = Math.max(2, Math.min(cell, 60));
    return cell;
  }

  HP.countReachable = function countReachable(gridData) {
  if (!gridData?.horsePos) return 0;
  const { grid, width, height, horsePos } = gridData;

  const visited = new Set();
  const queue = [horsePos];
  visited.add(`${horsePos.x},${horsePos.y}`);

  while (queue.length > 0) {
    const pos = queue.shift();
    const neighbors = [
      { x: pos.x - 1, y: pos.y },
      { x: pos.x + 1, y: pos.y },
      { x: pos.x, y: pos.y - 1 },
      { x: pos.x, y: pos.y + 1 },
    ];

    for (const n of neighbors) {
      const nKey = `${n.x},${n.y}`;
      if (
        n.x >= 0 &&
        n.x < width &&
        n.y >= 0 &&
        n.y < height &&
        !visited.has(nKey) &&
        grid[n.y][n.x] !== "water"
      ) {
        visited.add(nKey);
        queue.push(n);
      }
    }
  }

  return visited.size;
};

  // Helper: Convert HSL to RGB (for portal color-coding)
  // h: hue in degrees [0, 360), s: saturation [0, 1], l: lightness [0, 1]
  // Returns [r, g, b] with values [0, 255]
  function hslToRgb(h, s, l) {
    h = ((h % 360) + 360) % 360;
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
    const m = l - c / 2;
    
    let r1, g1, b1;
    if (h < 60) {
      r1 = c; g1 = x; b1 = 0;
    } else if (h < 120) {
      r1 = x; g1 = c; b1 = 0;
    } else if (h < 180) {
      r1 = 0; g1 = c; b1 = x;
    } else if (h < 240) {
      r1 = 0; g1 = x; b1 = c;
    } else if (h < 300) {
      r1 = x; g1 = 0; b1 = c;
    } else {
      r1 = c; g1 = 0; b1 = x;
    }
    
    return [
      Math.round((r1 + m) * 255),
      Math.round((g1 + m) * 255),
      Math.round((b1 + m) * 255)
    ];
  }
})();


