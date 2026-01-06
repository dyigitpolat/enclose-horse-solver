(() => {
  const HP = window.HorsePen;
  const THRESHOLDS = HP.THRESHOLDS;

  HP.detectGridDimensions = function detectGridDimensions(img, detectionCanvas) {
  const canvas = detectionCanvas;
  const ctx = canvas.getContext("2d");

  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  ctx.drawImage(img, 0, 0);

  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const pixels = imgData.data;
  const width = canvas.width;
  const height = canvas.height;

  // Convert to grayscale and calculate gradients
  const gray = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const idx = i * 4;
    gray[i] = (pixels[idx] + pixels[idx + 1] + pixels[idx + 2]) / 3;
  }

  // Horizontal gradient profile
  const colProfile = new Float32Array(width - 1);
  for (let x = 0; x < width - 1; x++) {
    let sum = 0;
    for (let y = 0; y < height; y++) {
      sum += Math.abs(gray[y * width + x + 1] - gray[y * width + x]);
    }
    colProfile[x] = sum;
  }

  // Vertical gradient profile
  const rowProfile = new Float32Array(height - 1);
  for (let y = 0; y < height - 1; y++) {
    let sum = 0;
    for (let x = 0; x < width; x++) {
      sum += Math.abs(gray[(y + 1) * width + x] - gray[y * width + x]);
    }
    rowProfile[y] = sum;
  }

  function scorePeriods(signal, minPeriod, maxPeriod) {
    const n = signal.length;

    // Normalize (matches Python)
    let mean = 0;
    let std = 0;
    for (let i = 0; i < n; i++) mean += signal[i];
    mean /= n;
    for (let i = 0; i < n; i++) {
      const d = signal[i] - mean;
      std += d * d;
    }
    std = Math.sqrt(std / n);
    if (std === 0) return [];

    const normalized = new Float32Array(n);
    for (let i = 0; i < n; i++) normalized[i] = (signal[i] - mean) / std;

    const upper = Math.min(maxPeriod, Math.floor(n / 3));
    const out = [];
    for (let period = minPeriod; period <= upper; period++) {
      let score = 0;
      for (let i = 0; i < n - period; i++) score += normalized[i] * normalized[i + period];
      out.push({ period, score });
    }
    out.sort((a, b) => b.score - a.score);
    return out;
  }

  // Search a wider range than before; still fast for these image sizes.
  const minPeriod = 10;
  const maxPeriod = 140;

  const colPeriods = scorePeriods(colProfile, minPeriod, maxPeriod);
  const rowPeriods = scorePeriods(rowProfile, minPeriod, maxPeriod);

  // Fallback: if scoring fails, default to common layout
  if (colPeriods.length === 0 || rowPeriods.length === 0) {
    return { cols: 19, rows: 23 };
  }

  const topK = 10;
  const colTop = colPeriods.slice(0, topK);
  const rowTop = rowPeriods.slice(0, topK);

  let best = { score: -Infinity, cols: 19, rows: 23 };

  for (const cw of colTop) {
    for (const rh of rowTop) {
      let cols = Math.round(width / cw.period);
      let rows = Math.round(height / rh.period);

      cols = Math.max(5, Math.min(50, cols));
      rows = Math.max(5, Math.min(50, rows));

      const cellW2 = width / cols;
      const cellH2 = height / rows;
      const aspect = cellW2 / cellH2;

      // Prefer near-square cells; reject extremely skewed.
      if (aspect < 0.8 || aspect > 1.25) continue;

      const errW = Math.abs(cellW2 - cw.period) / cw.period;
      const errH = Math.abs(cellH2 - rh.period) / rh.period;
      const aspectPenalty = Math.abs(aspect - 1);

      const score = cw.score + rh.score - 1000 * aspectPenalty - 500 * (errW + errH);

      if (score > best.score) {
        best = { score, cols, rows };
      }
    }
  }

  // If the cross-scoring found nothing, fall back to independent best periods
  if (!Number.isFinite(best.score)) {
    const cols = Math.max(5, Math.min(50, Math.round(width / colPeriods[0].period)));
    const rows = Math.max(5, Math.min(50, Math.round(height / rowPeriods[0].period)));
    return { cols, rows };
  }

  return { cols: best.cols, rows: best.rows };
  };

  HP.parseImageToGrid = function parseImageToGrid(img, detectionCanvas, cols, rows) {
  const canvas = detectionCanvas;
  const ctx = canvas.getContext("2d");

  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  ctx.drawImage(img, 0, 0);

  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const pixels = imgData.data;

  const cellW = canvas.width / cols;
  const cellH = canvas.height / rows;

  const grid = [];
  let horsePos = null;
  let waterCount = 0;
  let grassCount = 0;
  let cherryCount = 0;

  const cherryCandidates = new Uint8Array(cols * rows);

  for (let row = 0; row < rows; row++) {
    grid[row] = [];
    for (let col = 0; col < cols; col++) {
      const samples = [];
      const samplePoints = [
        [0.5, 0.5],
        [0.3, 0.3],
        [0.7, 0.3],
        [0.3, 0.7],
        [0.7, 0.7],
      ];

      for (const [fx, fy] of samplePoints) {
        const px = Math.floor((col + fx) * cellW);
        const py = Math.floor((row + fy) * cellH);
        if (px < canvas.width && py < canvas.height) {
          const idx = (py * canvas.width + px) * 4;
          samples.push({
            r: pixels[idx],
            g: pixels[idx + 1],
            b: pixels[idx + 2],
          });
        }
      }

      if (samples.length === 0) {
        grid[row][col] = "grass";
        grassCount++;
        continue;
      }

      const avg = {
        r: samples.reduce((s, c) => s + c.r, 0) / samples.length,
        g: samples.reduce((s, c) => s + c.g, 0) / samples.length,
        b: samples.reduce((s, c) => s + c.b, 0) / samples.length,
      };

      let cherrySampleHit = false;
      for (const s of samples) {
        if (isCherryPixel(s.r, s.g, s.b)) {
          cherrySampleHit = true;
          break;
        }
      }

      const isWater =
        avg.b > avg.g &&
        avg.b > avg.r &&
        avg.b > THRESHOLDS.WATER_BLUE &&
        avg.r < 80 &&
        avg.g < 110;

      if (isWater) {
        grid[row][col] = "water";
        waterCount++;
      } else {
        grid[row][col] = "grass";
        grassCount++;
      }

      if (!isWater && cherrySampleHit) {
        cherryCandidates[row * cols + col] = 1;
      }
    }
  }

  // Find horse using brightest square algorithm (most reliable)
  let horseMethod = "brightest_square";
  let horseStats = findHorseByBrightestSquare(pixels, canvas.width, canvas.height, cellW, cellH, rows, cols);

  if (horseStats) {
    horsePos = { x: horseStats.x, y: horseStats.y };
    // Adjust counts if we overwrite water/grass
    const prev = grid[horsePos.y][horsePos.x];
    if (prev === "water") waterCount = Math.max(0, waterCount - 1);
    if (prev === "grass") grassCount = Math.max(0, grassCount - 1);
    grid[horsePos.y][horsePos.x] = "horse";
  }

  // Detect cherries (priority: reliability). Cherries are passable tiles (like grass).
  const cherryByPixels = detectCherryCellsByPixels(
    pixels,
    canvas.width,
    canvas.height,
    cellW,
    cellH,
    rows,
    cols,
    grid
  );
  const cherryCells = [];
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const idx = y * cols + x;
      if (grid[y][x] !== "grass") continue;
      if (!cherryCandidates[idx] && !cherryByPixels[idx]) continue;
      grid[y][x] = "cherry";
      grassCount = Math.max(0, grassCount - 1);
      cherryCount++;
      cherryCells.push({ x, y });
    }
  }

  const debug = {
    waterCount,
    grassCount,
    horsePos,
    horseMethod,
    horseBrightness: horseStats?.brightness ?? null,
    horseWhiteness: horseStats?.whiteness ?? null,
    cherryCount,
    cherryCells,
  };

  return { grid, width: cols, height: rows, horsePos, debug };
  };

  function findHorseByBrightestSquare(pixels, width, height, cellW, cellH, rows, cols) {
    // Find the cellSize×cellSize square region with highest mean brightness.
    // The horse sprite is white, so its cell will have the highest mean brightness.
    const cellSize = Math.round((cellW + cellH) / 2);
    
    // Compute grayscale brightness for the entire image
    const gray = new Float32Array(width * height);
    for (let i = 0; i < width * height; i++) {
      const idx = i * 4;
      gray[i] = (pixels[idx] + pixels[idx + 1] + pixels[idx + 2]) / 3;
    }
    
    // Sliding window to find the square with highest mean brightness
    // Step by 1/4 cell to ensure we don't miss the horse
    const step = Math.max(1, Math.floor(cellSize / 4));
    let bestBrightness = 0;
    let bestX = 0;
    let bestY = 0;
    
    for (let y = 0; y <= height - cellSize; y += step) {
      for (let x = 0; x <= width - cellSize; x += step) {
        // Compute mean brightness of this square
        let sum = 0;
        let count = 0;
        for (let dy = 0; dy < cellSize; dy++) {
          for (let dx = 0; dx < cellSize; dx++) {
            const px = x + dx;
            const py = y + dy;
            if (py < height && px < width) {
              sum += gray[py * width + px];
              count++;
            }
          }
        }
        const meanBrightness = count > 0 ? sum / count : 0;
        if (meanBrightness > bestBrightness) {
          bestBrightness = meanBrightness;
          bestX = x;
          bestY = y;
        }
      }
    }
    
    if (bestBrightness < 50) return null; // Sanity check
    
    // Map center of square to grid coordinates
    const centerX = bestX + Math.floor(cellSize / 2);
    const centerY = bestY + Math.floor(cellSize / 2);
    
    const col = Math.min(cols - 1, Math.max(0, Math.floor(centerX / cellW)));
    const row = Math.min(rows - 1, Math.max(0, Math.floor(centerY / cellH)));
    
    // Compute whiteness at center
    const idx = (centerY * width + centerX) * 4;
    const r = pixels[idx];
    const g = pixels[idx + 1];
    const b = pixels[idx + 2];
    const whiteness = Math.abs(r - g) + Math.abs(g - b);
    
    return { x: col, y: row, brightness: bestBrightness, whiteness };
  }

  function isCherryPixel(r, g, b) {
    // Cherries are strong red pixels (sprite-like), not the brown shore pixels.
    // Tuned on p3.png where cherry pixels are around (171, 47, 46).
    return r >= 150 && r - g >= 60 && r - b >= 60 && g <= 170 && b <= 170;
  }

  function detectCherryCellsByPixels(pixels, width, height, cellW, cellH, rows, cols, grid) {
    const counts = new Uint16Array(rows * cols);
    const step = 2; // fast enough and reliable for pixel-art sprites

    for (let y = 0; y < height; y += step) {
      for (let x = 0; x < width; x += step) {
        const idx = (y * width + x) * 4;
        const r = pixels[idx];
        const g = pixels[idx + 1];
        const b = pixels[idx + 2];
        if (!isCherryPixel(r, g, b)) continue;

        const col = Math.min(cols - 1, Math.max(0, Math.floor(x / cellW)));
        const row = Math.min(rows - 1, Math.max(0, Math.floor(y / cellH)));
        if (grid[row]?.[col] === "water") continue;
        counts[row * cols + col]++;
      }
    }

    let max = 0;
    for (let i = 0; i < counts.length; i++) if (counts[i] > max) max = counts[i];

    const out = new Uint8Array(rows * cols);
    if (max === 0) return out;

    const threshold = Math.max(6, Math.floor(max * 0.2));
    for (let i = 0; i < counts.length; i++) if (counts[i] >= threshold) out[i] = 1;

    return out;
  }
})();


