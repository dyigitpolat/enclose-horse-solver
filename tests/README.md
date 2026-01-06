# Tests

## Layout

- `tests/fixtures/`: **test input images** (p2/p3/p4*/p5*/preview)
- `tests/python/`: **Python test + verification scripts**
- `tests/js/`: **Node.js test scripts** (runs the same parsing logic as the browser)

## Quick commands

### Python (parsing robustness)

```bash
cd /Users/dogukanyigitpolat/repos/horsepen
source venv/bin/activate
python tests/python/verify_parsing_robustness.py
```

### JavaScript (Node)

```bash
cd /Users/dogukanyigitpolat/repos/horsepen
npm install
npm run test:js
```

## Notes

- The **fixtures** are intentionally committed so we can reproduce parsing/solver regressions.
- The Node test uses `pngjs` (pure JS) to avoid native `canvas` dependencies.


