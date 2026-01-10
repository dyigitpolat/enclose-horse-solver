/**
 * Portal detection + pairing test.
 *
 * Run:
 *   npm install
 *   node tests/js/test_js_portals.js
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

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

function main() {
  const p = path.join(FIXTURES_DIR, "portal2.png");
  const res = parseWithJs(p);

  console.log(`portal2.png grid=${res.dims.cols}x${res.dims.rows} horse=(${res.out.horsePos.x},${res.out.horsePos.y})`);

  const pairs = res.out.portalPairs || [];
  console.log(`detected portal pairs: ${pairs.length}`);

  // Fixture invariant (per user request): every pair is 2 cells away.
  assert(pairs.length === 8, `expected 8 portal pairs in portal2.png, got ${pairs.length}`);

  const used = new Set();
  for (const pp of pairs) {
    const ax = pp?.a?.x;
    const ay = pp?.a?.y;
    const bx = pp?.b?.x;
    const by = pp?.b?.y;
    assert([ax, ay, bx, by].every(Number.isFinite), `invalid pair: ${JSON.stringify(pp)}`);
    const d = Math.abs(ax - bx) + Math.abs(ay - by);
    assert(d === 2, `expected manhattan distance 2, got ${d} for ${JSON.stringify(pp)}`);
    const ak = `${ax},${ay}`;
    const bk = `${bx},${by}`;
    assert(!used.has(ak), `portal cell reused across pairs: ${ak}`);
    assert(!used.has(bk), `portal cell reused across pairs: ${bk}`);
    used.add(ak);
    used.add(bk);
  }

  console.log("✅ portal2.png portal pairing looks correct (all pairs distance=2).");

  // Regression: cherries should not be misclassified as portals.
  const p2 = path.join(FIXTURES_DIR, "portal_cherries.png");
  const res2 = parseWithJs(p2);
  const cherries = res2.out.debug?.cherryCells || [];
  const portals = res2.out.debug?.portalCells || [];
  console.log(`portal_cherries.png cherries=${cherries.length} portals=${portals.length}`);
  assert(cherries.length === 2, `expected 2 cherries in portal_cherries.png, got ${cherries.length}`);
  const cherryKeys = new Set(cherries.map((c) => `${c.x},${c.y}`));
  assert(cherryKeys.has("9,7") && cherryKeys.has("5,10"), `unexpected cherry coords: ${Array.from(cherryKeys).join(", ")}`);
}

main();


