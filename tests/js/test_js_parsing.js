/**
 * Strict Node.js test for JS image parsing robustness.
 *
 * Requirements:
 * - All p4_* variants must parse IDENTICALLY to p4.png
 * - All p5_* variants must parse IDENTICALLY to p5.png
 *
 * Run:
 *   npm install
 *   node tests/js/test_js_parsing.js
 */

const fs = require("fs");
const path = require("path");

let PNG;
try {
  PNG = require("pngjs").PNG;
} catch (e) {
  console.error('Missing dependency: "pngjs"');
  console.error("Install it with: npm install");
  process.exit(1);
}

// Load the browser parsing code in Node
global.window = { HorsePen: {} };
const HP = global.window.HorsePen;

HP.THRESHOLDS = {
  WATER_BLUE: 50,
  HORSE_BRIGHTNESS: 160,
};

const ROOT_DIR = path.resolve(__dirname, "../..");
const FIXTURES_DIR = path.join(ROOT_DIR, "tests", "fixtures");

require(path.join(ROOT_DIR, "src", "image.js"));

// Fixture policy (as per user feedback):
// - Only test downscales in [0.5, 1.0]
// - Only test JPEG qualities >= 20
// Other non-scale/non-JPEG variants (hue shift, crop/extend, etc.) are still tested.
const RE_SCALE = /_scale_([0-9]+(?:\.[0-9]+)?)/;
const RE_JPGQ = /_jpg_q(\d+)/;
function isAllowedVariant(fileName) {
  const mS = fileName.match(RE_SCALE);
  if (mS) {
    const s = Number.parseFloat(mS[1]);
    if (!Number.isFinite(s) || s < 0.5 || s > 1.0) return false;
  }
  const mQ = fileName.match(RE_JPGQ);
  if (mQ) {
    const q = Number.parseInt(mQ[1], 10);
    if (!Number.isFinite(q) || q < 20) return false;
  }
  return true;
}

function loadPng(filePath) {
  const buf = fs.readFileSync(filePath);
  const png = PNG.sync.read(buf);
  return { width: png.width, height: png.height, pixels: png.data };
}

function parseWithJs(filePath) {
  const { width, height, pixels } = loadPng(filePath);
  const dims = HP.detectGridDimensionsFromPixels(pixels, width, height, true, 8);
  const out = HP.parsePixelsToGrid(pixels, width, height, dims.cols, dims.rows, 0.0);
  return { dims, out };
}

function gridDiff(a, b) {
  const diffs = [];
  if (!Array.isArray(a) || !Array.isArray(b) || a.length === 0 || b.length === 0) return diffs;
  if (a.length !== b.length || a[0].length !== b[0].length) return [["GRID_SIZE_MISMATCH", a[0]?.length, a.length, b[0]?.length, b.length]];
  for (let y = 0; y < a.length; y++) {
    for (let x = 0; x < a[0].length; x++) {
      if (a[y][x] !== b[y][x]) diffs.push([x, y, a[y][x], b[y][x]]);
    }
  }
  return diffs;
}

function sortedCherryCells(debug) {
  const cells = debug?.cherryCells || [];
  return cells
    .map((c) => `${c.x},${c.y}`)
    .sort();
}

function assertFamily(prefix, baseFile, expectedBase) {
  const basePath = path.join(FIXTURES_DIR, baseFile);
  const base = parseWithJs(basePath);
  const baseDims = `${base.dims.cols}x${base.dims.rows}`;
  const baseHorse = base.out.horsePos;
  const baseCounts = {
    water: base.out.debug.waterCount,
    grass: base.out.debug.grassCount,
    cherries: base.out.debug.cherryCount,
  };
  const baseCherries = sortedCherryCells(base.out.debug);

  console.log(`\n=== ${prefix} family ===`);
  console.log(`base=${baseFile} grid=${baseDims} horse=(${baseHorse.x},${baseHorse.y}) counts=${JSON.stringify(baseCounts)}`);

  // Sanity check vs Python reference for base images
  if (expectedBase) {
    const ok =
      baseDims === expectedBase.grid &&
      baseHorse.x === expectedBase.horse.x &&
      baseHorse.y === expectedBase.horse.y &&
      baseCounts.water === expectedBase.water &&
      baseCounts.grass === expectedBase.grass &&
      baseCounts.cherries === expectedBase.cherries;
    if (!ok) {
      throw new Error(
        `${baseFile} mismatch vs expected: got grid=${baseDims} horse=(${baseHorse.x},${baseHorse.y}) counts=${JSON.stringify(
          baseCounts
        )}, expected=${JSON.stringify(expectedBase)}`
      );
    }
  }

  const files = fs
    .readdirSync(FIXTURES_DIR)
    .filter((f) => f.startsWith(prefix + "_") && f.endsWith(".png"))
    .filter(isAllowedVariant)
    .sort();
  const ignored = fs
    .readdirSync(FIXTURES_DIR)
    .filter((f) => f.startsWith(prefix + "_") && f.endsWith(".png") && !isAllowedVariant(f)).length;
  if (ignored) {
    console.log(`[INFO] ignored ${ignored} out-of-policy fixture(s) (scale>=0.5, jpg_q>=20)`);
  }

  let failed = 0;
  for (const file of files) {
    const res = parseWithJs(path.join(FIXTURES_DIR, file));

    let ok = true;
    const dims = `${res.dims.cols}x${res.dims.rows}`;
    const horse = res.out.horsePos;
    const counts = {
      water: res.out.debug.waterCount,
      grass: res.out.debug.grassCount,
      cherries: res.out.debug.cherryCount,
    };
    const cherries = sortedCherryCells(res.out.debug);

    if (dims !== baseDims) ok = false;
    if (horse.x !== baseHorse.x || horse.y !== baseHorse.y) ok = false;
    if (counts.water !== baseCounts.water || counts.grass !== baseCounts.grass || counts.cherries !== baseCounts.cherries) ok = false;
    if (cherries.join("|") !== baseCherries.join("|")) ok = false;

    const diffs = gridDiff(base.out.grid, res.out.grid);
    if (diffs.length) ok = false;

    if (ok) {
      console.log(`[OK]   ${file}`);
    } else {
      failed++;
      console.log(`[FAIL] ${file}`);
      console.log(`  dims:   ${dims} (base ${baseDims})`);
      console.log(`  horse:  (${horse.x},${horse.y}) (base ${baseHorse.x},${baseHorse.y})`);
      console.log(`  counts: ${JSON.stringify(counts)} (base ${JSON.stringify(baseCounts)})`);
      if (cherries.length !== baseCherries.length) {
        console.log(`  cherries: ${cherries.length} (base ${baseCherries.length})`);
      }
      if (diffs.length) {
        console.log(`  cell diffs: ${diffs.length} (showing up to 30)`);
        for (const d of diffs.slice(0, 30)) console.log("   ", d);
      }
    }
  }

  if (failed) {
    throw new Error(`${prefix} family failed: ${failed}/${files.length} variants mismatched`);
  }
}

function main() {
  // Base expectations from Python (see verify_parsing_robustness.py)
  const expected = {
    "p4.png": { grid: "30x30", horse: { x: 7, y: 23 }, water: 413, grass: 486, cherries: 0 },
    "p5.png": { grid: "12x14", horse: { x: 6, y: 7 }, water: 53, grass: 104, cherries: 10 },
  };

  // p2/p3: no hardcoded expectations, but all variants must match their base exactly.
  assertFamily("p2", "p2.png");
  assertFamily("p3", "p3.png");
  assertFamily("p4", "p4.png", expected["p4.png"]);
  assertFamily("p5", "p5.png", expected["p5.png"]);

  console.log("\n✅ All JS parsing robustness checks passed.");
}

main();

