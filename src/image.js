(() => {
  const HP = window.HorsePen;
  const THRESHOLDS = HP.THRESHOLDS;

  // Gaussian blur helper function
  function applyGaussianBlur(pixels, width, height, radius) {
    if (radius <= 0) return pixels;
    
    const result = new Uint8ClampedArray(pixels.length);
    const kernel = createGaussianKernel(radius);
    const kernelSize = kernel.length;
    const halfKernel = Math.floor(kernelSize / 2);
    
    // Apply separable Gaussian blur (horizontal then vertical)
    const temp = new Uint8ClampedArray(pixels.length);
    
    // Horizontal pass
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        for (let c = 0; c < 4; c++) {
          let sum = 0;
          let weightSum = 0;
          for (let kx = -halfKernel; kx <= halfKernel; kx++) {
            const px = Math.max(0, Math.min(width - 1, x + kx));
            const weight = kernel[kx + halfKernel];
            sum += pixels[(y * width + px) * 4 + c] * weight;
            weightSum += weight;
          }
          temp[(y * width + x) * 4 + c] = sum / weightSum;
        }
      }
    }
    
    // Vertical pass
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        for (let c = 0; c < 4; c++) {
          let sum = 0;
          let weightSum = 0;
          for (let ky = -halfKernel; ky <= halfKernel; ky++) {
            const py = Math.max(0, Math.min(height - 1, y + ky));
            const weight = kernel[ky + halfKernel];
            sum += temp[(py * width + x) * 4 + c] * weight;
            weightSum += weight;
          }
          result[(y * width + x) * 4 + c] = sum / weightSum;
        }
      }
    }
    
    return result;
  }
  
  function createGaussianKernel(radius) {
    const sigma = radius / 2;
    const size = Math.ceil(radius * 2) * 2 + 1;
    const kernel = new Float32Array(size);
    const center = Math.floor(size / 2);
    
    for (let i = 0; i < size; i++) {
      const x = i - center;
      kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
    }
    
    return kernel;
  }

  function analyzeColorClusters(pixels, width, height, x0, y0, cellW, cellH, rows, cols, samplePoints) {
    // Sample colors from all cells to determine adaptive thresholds
    const allSamples = [];
    
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        for (const [fx, fy] of samplePoints) {
          const px = Math.floor(x0 + (c + fx) * cellW);
          const py = Math.floor(y0 + (r + fy) * cellH);
          if (px >= 0 && py >= 0 && px < width && py < height) {
            const idx = (py * width + px) * 4;
            allSamples.push({
              r: pixels[idx],
              g: pixels[idx + 1],
              b: pixels[idx + 2]
            });
          }
        }
      }
    }
    
    if (allSamples.length === 0) {
      return {
        waterThreshold: { bMin: 50, rMax: 80, gMax: 110 },
        method: 'fallback'
      };
    }
    
    // Identify blue-dominant pixels (potential water)
    // Use lenient criterion: B is highest OR B is close to highest
    const blueSamples = allSamples.filter(s => 
      (s.b > s.g && s.b > s.r) || 
      (s.b >= s.g - 15 && s.b >= s.r - 15 && s.b > 40)
    );
    
    if (blueSamples.length > 10) {
      // Compute adaptive thresholds
      const bValues = blueSamples.map(s => s.b).sort((a, b) => a - b);
      const rValues = blueSamples.map(s => s.r).sort((a, b) => a - b);
      const gValues = blueSamples.map(s => s.g).sort((a, b) => a - b);
      
      const percentile = (arr, p) => arr[Math.floor(arr.length * p)];
      
      const bP25 = percentile(bValues, 0.25);
      const bStd = Math.sqrt(bValues.reduce((sum, v) => sum + Math.pow(v - bValues.reduce((a,b)=>a+b,0)/bValues.length, 2), 0) / bValues.length);
      
      const rP90 = percentile(rValues, 0.90);
      const gP90 = percentile(gValues, 0.90);
      
      let bMin = Math.max(35, bP25 - 0.5 * bStd);
      let rMax = Math.max(80, rP90);
      let gMax = Math.max(100, gP90);
      
      // Clamp to reasonable ranges
      bMin = Math.max(30, Math.min(80, bMin));
      rMax = Math.max(70, Math.min(180, rMax));
      gMax = Math.max(90, Math.min(200, gMax));
      
      return {
        waterThreshold: { bMin, rMax, gMax },
        method: 'adaptive',
        nBlueSamples: blueSamples.length
      };
    } else {
      // Try to find any blue-ish pixels
      const blueIsh = allSamples.filter(s => s.b > (s.r + s.g) / 2.5);
      
      if (blueIsh.length > 5) {
        const bValues = blueIsh.map(s => s.b).sort((a, b) => a - b);
        const rValues = blueIsh.map(s => s.r).sort((a, b) => a - b);
        const gValues = blueIsh.map(s => s.g).sort((a, b) => a - b);
        
        const percentile = (arr, p) => arr[Math.floor(arr.length * p)];
        
        const bMin = Math.max(30, percentile(bValues, 0.15));
        const rMax = Math.max(80, percentile(rValues, 0.75));
        const gMax = Math.max(110, percentile(gValues, 0.75));
        
        return {
          waterThreshold: { bMin, rMax, gMax },
          method: 'blueish_fallback'
        };
      } else {
        // Absolute fallback
        return {
          waterThreshold: { bMin: 50, rMax: 80, gMax: 110 },
          method: 'fallback'
        };
      }
    }
  }

  HP.detectGridDimensionsFromPixels = function detectGridDimensionsFromPixels(pixels, width, height, useMultiscale = true, trimEdges = 8) {
    // Trim edges to reduce border artifacts
    const trim = Math.min(trimEdges, Math.floor(Math.min(width, height) / 4));
    const trimLeft = trim;
    const trimRight = trim;
    const trimTop = trim;
    const trimBottom = trim;

    if (useMultiscale) {
      // Multi-scale detection with weighted voting
      const scales = [0.0, 0.3, 0.7, 1.2, 1.8];
      const weights = [2.0, 1.5, 1.0, 0.8, 0.5];
      const scaleResults = [];

      for (let i = 0; i < scales.length; i++) {
        const blurRadius = scales[i];
        const weight = weights[i];

        let pixelsToUse = pixels;
        if (blurRadius > 0) {
          pixelsToUse = applyGaussianBlur(pixels, width, height, blurRadius);
        }

        const result = detectGridSingleScale(pixelsToUse, width, height, trimLeft, trimRight, trimTop, trimBottom);
        if (result) {
          scaleResults.push({ cols: result.cols, rows: result.rows, score: result.score, weight });
        }
      }

      if (scaleResults.length === 0) {
        return { cols: 19, rows: 23 };
      }

      // Weighted voting
      const votes = {};
      for (const result of scaleResults) {
        const key = `${result.cols},${result.rows}`;
        votes[key] = (votes[key] || 0) + result.score * result.weight;
      }

      // Pick best based on weighted votes
      let bestKey = null;
      let bestVote = -Infinity;

      for (const [key, vote] of Object.entries(votes)) {
        if (vote > bestVote) {
          bestVote = vote;
          bestKey = key;
        }
      }

      if (!bestKey) {
        console.warn('Multi-scale detection failed, falling back to 19x23');
        return { cols: 19, rows: 23 };
      }

      const [cols, rows] = bestKey.split(',').map(Number);
      return { cols, rows };
    } else {
      // Single-scale detection
      const result = detectGridSingleScale(pixels, width, height, trimLeft, trimRight, trimTop, trimBottom);
      if (result) {
        return { cols: result.cols, rows: result.rows };
      }
      console.warn('Single-scale detection failed, falling back to 19x23');
      return { cols: 19, rows: 23 };
    }
  };

  HP.detectGridDimensions = function detectGridDimensions(img, detectionCanvas, useMultiscale = true, trimEdges = 8) {
    const canvas = detectionCanvas;
    const ctx = canvas.getContext("2d");

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);

    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return HP.detectGridDimensionsFromPixels(imgData.data, canvas.width, canvas.height, useMultiscale, trimEdges);
  };

  function detectGridSingleScale(pixels, width, height, trimLeft, trimRight, trimTop, trimBottom) {
    const widthTrim = width - trimLeft - trimRight;
    const heightTrim = height - trimTop - trimBottom;
    
    // Convert to grayscale (trimmed region)
    const gray = new Float32Array(widthTrim * heightTrim);
    for (let y = 0; y < heightTrim; y++) {
      for (let x = 0; x < widthTrim; x++) {
        const srcIdx = ((y + trimTop) * width + (x + trimLeft)) * 4;
        gray[y * widthTrim + x] = (pixels[srcIdx] + pixels[srcIdx + 1] + pixels[srcIdx + 2]) / 3;
      }
    }

    // Horizontal gradient profile
    const colProfile = new Float32Array(widthTrim - 1);
    for (let x = 0; x < widthTrim - 1; x++) {
      let sum = 0;
      for (let y = 0; y < heightTrim; y++) {
        sum += Math.abs(gray[y * widthTrim + x + 1] - gray[y * widthTrim + x]);
      }
      colProfile[x] = sum;
    }

    // Vertical gradient profile
    const rowProfile = new Float32Array(heightTrim - 1);
    for (let y = 0; y < heightTrim - 1; y++) {
      let sum = 0;
      for (let x = 0; x < widthTrim; x++) {
        sum += Math.abs(gray[(y + 1) * widthTrim + x] - gray[y * widthTrim + x]);
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

    // Fallback: if scoring fails, return null
    if (colPeriods.length === 0 || rowPeriods.length === 0) {
      return null;
    }

    const topK = 10;
    const colTop = colPeriods.slice(0, topK);
    const rowTop = rowPeriods.slice(0, topK);

    let best = { score: -Infinity, cols: 19, rows: 23 };

    for (const cw of colTop) {
      for (const rh of rowTop) {
        let cols = Math.round(widthTrim / cw.period);
        let rows = Math.round(heightTrim / rh.period);

        // Allow up to 35x35 for detection (proper clamping happens later)
        cols = Math.max(5, Math.min(35, cols));
        rows = Math.max(5, Math.min(35, rows));

        const cellW2 = widthTrim / cols;
        const cellH2 = heightTrim / rows;
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
      const cols = Math.max(5, Math.min(35, Math.round(widthTrim / colPeriods[0].period)));
      const rows = Math.max(5, Math.min(35, Math.round(heightTrim / rowPeriods[0].period)));
      
      // Scale back to original dimensions
      const cellWTrim = widthTrim / cols;
      const cellHTrim = heightTrim / rows;
      const colsOrig = Math.max(5, Math.min(35, Math.round(width / cellWTrim)));
      const rowsOrig = Math.max(5, Math.min(35, Math.round(height / cellHTrim)));
      return { cols: colsOrig, rows: rowsOrig, score: -Infinity };
    }

    // Scale back to original dimensions
    const cellWTrim = widthTrim / best.cols;
    const cellHTrim = heightTrim / best.rows;
    const colsOrig = Math.max(5, Math.min(35, Math.round(width / cellWTrim)));
    const rowsOrig = Math.max(5, Math.min(35, Math.round(height / cellHTrim)));
    
    return { cols: colsOrig, rows: rowsOrig, score: best.score };
  }

  function smooth1d(x, windowSize = 3) {
    const n = x.length;
    if (windowSize <= 1 || n === 0) return x;
    let w = Math.max(1, Math.floor(windowSize));
    if (w % 2 === 0) w += 1;
    if (n < w) return x;

    const half = Math.floor(w / 2);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      let cnt = 0;
      for (let j = i - half; j <= i + half; j++) {
        if (j < 0 || j >= n) continue;
        sum += x[j];
        cnt++;
      }
      out[i] = cnt > 0 ? sum / cnt : x[i];
    }
    return out;
  }

  function percentileFloat(arr, p) {
    const n = arr.length;
    if (n === 0) return 0;
    const copy = Array.from(arr);
    copy.sort((a, b) => a - b);
    const idx = Math.max(0, Math.min(n - 1, Math.floor(n * p)));
    return copy[idx];
  }

  function fitGridAxis(profile, nCells, targetCell) {
    const prof = profile;
    const profS = smooth1d(prof, 3);
    const n = profS.length;
    const target = targetCell;
    const pMin = Math.max(10, Math.floor(target * 0.7));
    const pMax = Math.min(200, Math.ceil(target * 1.3));

    if (n <= 0) {
      return { start: 0, period: target, debug: { fallback: true, reason: 1 } };
    }

    const cap = percentileFloat(profS, 0.9);
    const minLines = Math.max(6, nCells - 2);

    let bestScore = -Infinity;
    let bestP = null;
    let bestS = null;

    for (let p = pMin; p <= pMax; p++) {
      for (let s = 0; s < p; s++) {
        let sum = 0;
        let count = 0;
        for (let k = 0; k <= nCells; k++) {
          const idx = s + p * k;
          if (idx > n - 1) break;
          let v = profS[idx];
          if (cap > 0 && v > cap) v = cap;
          sum += v;
          count++;
        }
        if (count < minLines) continue;
        const score = sum / count;
        if (score > bestScore || (score === bestScore && bestS !== null && s < bestS)) {
          bestScore = score;
          bestP = p;
          bestS = s;
        }
      }
    }

    if (bestP == null || bestS == null) {
      return { start: 0, period: target, debug: { fallback: true, reason: 2 } };
    }

    const p0 = bestP;
    const s0 = bestS;
    const win = Math.max(2, Math.round(p0 * 0.35));

    const obs = [];
    const kk = [];
    for (let k = 0; k <= nCells; k++) {
      const t = s0 + p0 * k;
      if (t < -win || t > (n - 1) + win) continue;
      const lo = Math.max(0, Math.floor(t - win));
      const hi = Math.min(n - 1, Math.ceil(t + win));
      if (hi <= lo) continue;

      let bestJ = lo;
      let bestV = -Infinity;
      for (let j = lo; j <= hi; j++) {
        const v = prof[j];
        if (v > bestV) {
          bestV = v;
          bestJ = j;
        }
      }
      obs.push(bestJ);
      kk.push(k);
    }

    let start = s0;
    let period = p0;
    if (obs.length >= 2) {
      const N = obs.length;
      let sumK = 0, sumY = 0, sumKK = 0, sumKY = 0;
      for (let i = 0; i < N; i++) {
        const k = kk[i];
        const y = obs[i];
        sumK += k;
        sumY += y;
        sumKK += k * k;
        sumKY += k * y;
      }
      const denom = N * sumKK - sumK * sumK;
      if (denom !== 0) {
        period = (N * sumKY - sumK * sumY) / denom;
        start = (sumY - period * sumK) / N;
      }
    }

    if (!(period >= p0 * 0.7 && period <= p0 * 1.3)) {
      start = s0;
      period = p0;
    }

    // Canonicalize phase: allow negative start; choose shift with most cell centers inside image.
    const pixelSize = n + 1;
    let bestStart = start;
    let bestIn = -1;
    let bestAbs = Infinity;
    for (const m of [-2, -1, 0, 1, 2]) {
      const sCand = start + m * period;
      let inCount = 0;
      for (let c = 0; c < nCells; c++) {
        const center = sCand + (c + 0.5) * period;
        if (center >= 0 && center < pixelSize) inCount++;
      }
      const absS = Math.abs(sCand);
      if (inCount > bestIn || (inCount === bestIn && absS < bestAbs)) {
        bestIn = inCount;
        bestAbs = absS;
        bestStart = sCand;
      }
    }

    return {
      start: bestStart,
      period,
      debug: {
        coarsePeriod: p0,
        coarseStart: s0,
        refinedPeriod: period,
        refinedStart: bestStart,
        score: bestScore,
      },
    };
  }

  function computeGridGeometry(pixels, width, height, cols, rows) {
    // Compute grayscale
    const gray = new Float32Array(width * height);
    for (let i = 0; i < width * height; i++) {
      const idx = i * 4;
      gray[i] = (pixels[idx] + pixels[idx + 1] + pixels[idx + 2]) / 3;
    }

    // Gradient profiles
    const colProfile = new Float32Array(width - 1);
    for (let x = 0; x < width - 1; x++) {
      let sum = 0;
      for (let y = 0; y < height; y++) {
        sum += Math.abs(gray[y * width + x + 1] - gray[y * width + x]);
      }
      colProfile[x] = sum;
    }

    const rowProfile = new Float32Array(height - 1);
    for (let y = 0; y < height - 1; y++) {
      let sum = 0;
      for (let x = 0; x < width; x++) {
        sum += Math.abs(gray[(y + 1) * width + x] - gray[y * width + x]);
      }
      rowProfile[y] = sum;
    }

    const fx = fitGridAxis(colProfile, cols, width / cols);
    const fy = fitGridAxis(rowProfile, rows, height / rows);

    return {
      x0: fx.start,
      y0: fy.start,
      cellW: fx.period,
      cellH: fy.period,
      axisX: fx.debug,
      axisY: fy.debug,
    };
  }

  HP.parsePixelsToGrid = function parsePixelsToGrid(pixelsOriginal, width, height, cols, rows, blurRadius = 0.0) {
    // Compute grid geometry (offset + cell sizes) on ORIGINAL pixels
    const geom = computeGridGeometry(pixelsOriginal, width, height, cols, rows);
    const x0 = geom.x0;
    const y0 = geom.y0;
    const cellW = geom.cellW;
    const cellH = geom.cellH;

    let pixels = pixelsOriginal;
    if (blurRadius > 0) {
      pixels = applyGaussianBlur(pixelsOriginal, width, height, blurRadius);
    }

    // Use 25 sample points for maximum robustness (5x5 grid)
    // This helps handle small pixel shifts from crop/extend operations
    const samplePoints = [];
    for (const fy of [0.2, 0.35, 0.5, 0.65, 0.8]) {
      for (const fx of [0.2, 0.35, 0.5, 0.65, 0.8]) {
        samplePoints.push([fx, fy]);
      }
    }

    // Analyze color clusters adaptively
    const colorClusters = analyzeColorClusters(pixels, width, height, x0, y0, cellW, cellH, rows, cols, samplePoints);
    const waterThresh = colorClusters.waterThreshold;
    console.log(`[JS] Grid ${cols}x${rows}, geom:`, { x0, y0, cellW, cellH }, 'Color clusters:', colorClusters.method, 'Water thresh:', waterThresh);

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

        for (const [fx, fy] of samplePoints) {
          const px = Math.floor(x0 + (col + fx) * cellW);
          const py = Math.floor(y0 + (row + fy) * cellH);
          if (px >= 0 && py >= 0 && px < width && py < height) {
            const idx = (py * width + px) * 4;
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

        // Use MAJORITY VOTING for robust classification
        let waterVotes = 0;
        let grassVotes = 0;
        let cherrySampleHit = false;

        for (const s of samples) {
          // Check for cherry first
          if (!cherrySampleHit && isCherryPixel(s.r, s.g, s.b)) {
            cherrySampleHit = true;
          }

          // Vote: is this sample point water or grass?
          const isWaterSample =
            s.b > waterThresh.bMin &&
            s.r <= waterThresh.rMax &&
            s.g <= waterThresh.gMax &&
            s.b >= (s.g - 15) &&
            s.b >= (s.r - 15);

          if (isWaterSample) {
            waterVotes++;
          } else {
            grassVotes++;
          }
        }

        // Majority wins
        const isWater = waterVotes > grassVotes;

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
    let horseStats = findHorseByBrightestSquare(pixelsOriginal, width, height, { x0, y0, cellW, cellH }, rows, cols);

    if (horseStats) {
      horsePos = { x: horseStats.x, y: horseStats.y };
      // Adjust counts if we overwrite water/grass
      const prev = grid[horsePos.y][horsePos.x];
      if (prev === "water") waterCount = Math.max(0, waterCount - 1);
      if (prev === "grass") grassCount = Math.max(0, grassCount - 1);
      grid[horsePos.y][horsePos.x] = "horse";
    }

    // Detect cherries (priority: reliability). Cherries are passable tiles (like grass).
    const cherryByPixels = detectCherryCellsByPixels(pixelsOriginal, width, height, { x0, y0, cellW, cellH }, rows, cols, grid);
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
      colorClusters,
      blurRadius,
      gridGeom: { x0, y0, cellW, cellH, axisX: geom.axisX, axisY: geom.axisY },
    };

    return { grid, width: cols, height: rows, horsePos, debug };
  };

  HP.parseImageToGrid = function parseImageToGrid(img, detectionCanvas, cols, rows, blurRadius = 0.0) {
    const canvas = detectionCanvas;
    const ctx = canvas.getContext("2d");

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);

    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return HP.parsePixelsToGrid(imgData.data, canvas.width, canvas.height, cols, rows, blurRadius);
  };

  function findHorseByBrightestSquare(pixels, width, height, geom, rows, cols) {
    // Find the cellSize×cellSize square region with highest mean brightness.
    // The horse sprite is white, so its cell will have the highest mean brightness.
    const cellSize = Math.round((geom.cellW + geom.cellH) / 2);
    
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
    
    const xMin = Math.max(0, Math.floor(geom.x0));
    const yMin = Math.max(0, Math.floor(geom.y0));
    const xMax = Math.min(width, Math.ceil(geom.x0 + cols * geom.cellW));
    const yMax = Math.min(height, Math.ceil(geom.y0 + rows * geom.cellH));

    const yStop = Math.max(yMin, yMax - cellSize);
    const xStop = Math.max(xMin, xMax - cellSize);

    for (let y = yMin; y <= yStop; y += step) {
      for (let x = xMin; x <= xStop; x += step) {
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
    
    const col = Math.min(cols - 1, Math.max(0, Math.floor((centerX - geom.x0) / geom.cellW)));
    const row = Math.min(rows - 1, Math.max(0, Math.floor((centerY - geom.y0) / geom.cellH)));
    
    // Compute whiteness at center
    const idx = (centerY * width + centerX) * 4;
    const r = pixels[idx];
    const g = pixels[idx + 1];
    const b = pixels[idx + 2];
    const whiteness = Math.abs(r - g) + Math.abs(g - b);
    
    return { x: col, y: row, brightness: bestBrightness, whiteness };
  }

  function isCherryPixel(r, g, b) {
    // Cherries are sprite-like strong red pixels. Robust to saturation reduction.
    if (r < 130) return false;
    if ((r - g) < 50 || (r - b) < 50) return false;
    if (r < 1.7 * (g + 1) || r < 1.7 * (b + 1)) return false;
    return true;
  }

  function detectCherryCellsByPixels(pixels, width, height, geom, rows, cols, grid) {
    const counts = new Uint16Array(rows * cols);
    const step = 2; // fast enough and reliable for pixel-art sprites

    const gridLeft = geom.x0;
    const gridTop = geom.y0;
    const gridRight = geom.x0 + cols * geom.cellW;
    const gridBottom = geom.y0 + rows * geom.cellH;

    for (let y = 0; y < height; y += step) {
      for (let x = 0; x < width; x += step) {
        const idx = (y * width + x) * 4;
        const r = pixels[idx];
        const g = pixels[idx + 1];
        const b = pixels[idx + 2];
        if (!isCherryPixel(r, g, b)) continue;

        if (x < gridLeft || x >= gridRight || y < gridTop || y >= gridBottom) continue;
        const col = Math.min(cols - 1, Math.max(0, Math.floor((x - gridLeft) / geom.cellW)));
        const row = Math.min(rows - 1, Math.max(0, Math.floor((y - gridTop) / geom.cellH)));
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


