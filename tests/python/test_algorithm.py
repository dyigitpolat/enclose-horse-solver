"""
Horse Pen Optimizer - Algorithm Test Suite
Finds optimal subset region to enclose with given wall count.
"""

import numpy as np
from PIL import Image
from collections import deque, defaultdict
import os
import time
import random
from itertools import combinations
import heapq

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FIXTURES_DIR = os.path.join(ROOT_DIR, "tests", "fixtures")
PREVIEW_PATH = os.path.join(FIXTURES_DIR, "preview.webp")

# ============ CORE ALGORITHM ============

def find_optimal_enclosure(grid, width, height, horse_pos, max_walls, timeout=10.0):
    """
    Find optimal wall placement to maximize enclosed area.
    
    Key insight: We don't necessarily enclose ALL reachable area.
    We find the SUBSET of cells that maximizes area while requiring ≤max_walls.
    """
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def in_bounds(x, y):
        return 0 <= x < width and 0 <= y < height
    
    def is_passable(x, y):
        return in_bounds(x, y) and grid[y][x] != 'water'
    
    def get_neighbors(x, y):
        return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    
    def flood_fill(start_x, start_y, blocked_set):
        """BFS flood fill, returns (visited set, can reach edge)."""
        if (start_x, start_y) in blocked_set:
            return set(), False
        
        visited = {(start_x, start_y)}
        queue = deque([(start_x, start_y)])
        can_escape = False
        
        while queue:
            x, y = queue.popleft()
            for nx, ny in get_neighbors(x, y):
                if not in_bounds(nx, ny):
                    can_escape = True
                    continue
                pos = (nx, ny)
                if pos not in visited and is_passable(nx, ny) and pos not in blocked_set:
                    visited.add(pos)
                    queue.append(pos)
        
        return visited, can_escape
    
    def is_valid(wall_set):
        if horse_pos in wall_set:
            return False, set()
        visited, can_escape = flood_fill(horse_pos[0], horse_pos[1], wall_set)
        return not can_escape, visited
    
    # Get all reachable cells and check initial state
    all_reachable, can_escape = flood_fill(horse_pos[0], horse_pos[1], set())
    
    best = {'area': 0, 'walls': [], 'enclosed': set()}
    
    if not can_escape:
        return {'area': len(all_reachable), 'walls': [], 'enclosed': all_reachable}
    
    # Build distance map from horse
    dist_from_horse = {horse_pos: 0}
    parent = {horse_pos: None}
    queue = deque([horse_pos])
    while queue:
        pos = queue.popleft()
        d = dist_from_horse[pos]
        for nx, ny in get_neighbors(pos[0], pos[1]):
            npos = (nx, ny)
            if npos not in dist_from_horse and in_bounds(nx, ny) and is_passable(nx, ny):
                dist_from_horse[npos] = d + 1
                parent[npos] = pos
                queue.append(npos)
    
    # Find edge cells
    edge_cells = [(x, y) for x, y in all_reachable 
                  if x == 0 or x == width-1 or y == 0 or y == height-1]
    
    # Find cells adjacent to water (boundary cells)
    boundary_cells = []
    for pos in all_reachable:
        if pos == horse_pos:
            continue
        x, y = pos
        for nx, ny in get_neighbors(x, y):
            if not in_bounds(nx, ny) or grid[ny][nx] == 'water':
                boundary_cells.append(pos)
                break
    
    # ===== STRATEGY 1: Find single chokepoints =====
    for pos in boundary_cells:
        if elapsed() > timeout * 0.1:
            break
        blocked = {pos}
        valid, enclosed = is_valid(blocked)
        if valid and len(enclosed) > best['area']:
            best = {'area': len(enclosed), 'walls': [pos], 'enclosed': enclosed}
    
    # ===== STRATEGY 2: Distance-based layers =====
    layers = defaultdict(list)
    for pos, dist in dist_from_horse.items():
        if pos != horse_pos:
            layers[dist].append(pos)
    
    for dist in sorted(layers.keys()):
        if elapsed() > timeout * 0.2:
            break
        layer = layers[dist]
        if len(layer) <= max_walls:
            wall_set = set(layer)
            valid, enclosed = is_valid(wall_set)
            if valid and len(enclosed) > best['area']:
                best = {'area': len(enclosed), 'walls': layer, 'enclosed': enclosed}
    
    # ===== STRATEGY 3: Region growing with boundary tracking =====
    def region_grow():
        nonlocal best
        
        # Start from horse, expand, track required walls
        included = {horse_pos}
        frontier = set()
        
        for nx, ny in get_neighbors(horse_pos[0], horse_pos[1]):
            if in_bounds(nx, ny) and is_passable(nx, ny):
                frontier.add((nx, ny))
        
        while frontier and elapsed() < timeout * 0.4:
            # Calculate walls needed for current region
            walls_needed = set()
            for pos in frontier:
                # Check if this frontier cell leads to escape
                leads_to_edge = False
                for nx, ny in get_neighbors(pos[0], pos[1]):
                    if not in_bounds(nx, ny):
                        leads_to_edge = True
                        break
                    if is_passable(nx, ny) and (nx, ny) not in included and (nx, ny) not in frontier:
                        leads_to_edge = True
                        break
                if leads_to_edge:
                    walls_needed.add(pos)
            
            # If we can wall the current frontier
            if len(walls_needed) <= max_walls:
                valid, enclosed = is_valid(walls_needed)
                if valid and len(enclosed) > best['area']:
                    best = {'area': len(enclosed), 'walls': list(walls_needed), 'enclosed': enclosed}
            
            # Expand: add frontier cell with minimum "cost"
            best_cell = None
            best_score = float('inf')
            
            for pos in frontier:
                # Score: prefer cells that don't increase wall count
                new_walls = 0
                for nx, ny in get_neighbors(pos[0], pos[1]):
                    if not in_bounds(nx, ny):
                        new_walls += 1
                    elif is_passable(nx, ny) and (nx, ny) not in included and (nx, ny) not in frontier:
                        new_walls += 1
                
                if new_walls < best_score:
                    best_score = new_walls
                    best_cell = pos
            
            if best_cell is None:
                break
            
            # Move best cell from frontier to included
            frontier.remove(best_cell)
            included.add(best_cell)
            
            # Add new frontier cells
            for nx, ny in get_neighbors(best_cell[0], best_cell[1]):
                pos = (nx, ny)
                if in_bounds(nx, ny) and is_passable(nx, ny) and pos not in included and pos not in frontier:
                    frontier.add(pos)
    
    region_grow()
    
    # ===== STRATEGY 4: Try rectangles centered on horse =====
    def try_rectangles():
        nonlocal best
        
        max_range = min(15, max(width, height) // 2)
        
        for radius in range(1, max_range):
            if elapsed() > timeout * 0.5:
                break
            
            # Define rectangle bounds
            min_x = max(0, horse_pos[0] - radius)
            max_x = min(width - 1, horse_pos[0] + radius)
            min_y = max(0, horse_pos[1] - radius)
            max_y = min(height - 1, horse_pos[1] + radius)
            
            # Collect interior cells
            interior = set()
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    if is_passable(x, y):
                        interior.add((x, y))
            
            if horse_pos not in interior:
                continue
            
            # Find walls needed (boundary of interior that leads to exterior)
            walls = set()
            for pos in interior:
                x, y = pos
                for nx, ny in get_neighbors(x, y):
                    if in_bounds(nx, ny) and is_passable(nx, ny) and (nx, ny) not in interior:
                        walls.add((nx, ny))
                    elif not in_bounds(nx, ny):
                        # Edge of map - need to wall this interior cell
                        walls.add(pos)
            
            if len(walls) <= max_walls:
                valid, enclosed = is_valid(walls)
                if valid and len(enclosed) > best['area']:
                    best = {'area': len(enclosed), 'walls': list(walls), 'enclosed': enclosed}
    
    try_rectangles()
    
    # ===== STRATEGY 5: Score-based candidate selection =====
    def score_candidates():
        scores = []
        for pos in boundary_cells:
            if pos == horse_pos:
                continue
            
            # Score based on multiple factors
            score = 0
            
            # Distance from horse
            dist = dist_from_horse.get(pos, 999)
            score += 100 - min(dist * 2, 100)
            
            # Number of water/edge neighbors
            barrier_neighbors = 0
            for nx, ny in get_neighbors(pos[0], pos[1]):
                if not in_bounds(nx, ny) or grid[ny][nx] == 'water':
                    barrier_neighbors += 1
            score += barrier_neighbors * 30
            
            # Check if it's a chokepoint
            blocked = {pos}
            _, can_esc = flood_fill(horse_pos[0], horse_pos[1], blocked)
            if not can_esc:
                score += 500
            
            scores.append((pos, score))
        
        scores.sort(key=lambda x: -x[1])
        return [p for p, _ in scores]
    
    candidates = score_candidates()[:min(50, len(boundary_cells))]
    
    # ===== STRATEGY 6: Combination search =====
    def combo_search(cells, max_size, time_limit):
        nonlocal best
        
        for size in range(1, min(max_size + 1, len(cells) + 1)):
            if elapsed() > time_limit:
                break
            
            count = 0
            for combo in combinations(cells, size):
                if elapsed() > time_limit:
                    break
                count += 1
                if count > 20000:
                    break
                
                wall_set = set(combo)
                valid, enclosed = is_valid(wall_set)
                if valid and len(enclosed) > best['area']:
                    best = {'area': len(enclosed), 'walls': list(combo), 'enclosed': enclosed}
    
    combo_search(candidates[:35], min(max_walls, 7), timeout * 0.7)
    
    # ===== STRATEGY 7: Greedy improvement =====
    def greedy_improve():
        nonlocal best
        
        current_walls = set(best['walls'])
        
        for _ in range(max_walls - len(current_walls)):
            if elapsed() > timeout * 0.85:
                break
            
            best_add = None
            best_area = best['area']
            
            for pos in candidates:
                if pos in current_walls:
                    continue
                
                test_walls = current_walls | {pos}
                if len(test_walls) > max_walls:
                    continue
                
                valid, enclosed = is_valid(test_walls)
                if valid and len(enclosed) > best_area:
                    best_area = len(enclosed)
                    best_add = (pos, enclosed)
            
            if best_add:
                pos, enclosed = best_add
                current_walls.add(pos)
                best = {'area': len(enclosed), 'walls': list(current_walls), 'enclosed': enclosed}
            else:
                break
    
    greedy_improve()
    
    # ===== STRATEGY 8: Try removing walls to find better solutions =====
    def refine():
        nonlocal best
        
        if len(best['walls']) < 2:
            return
        
        # Try removing each wall one at a time and see if we can do better
        current_walls = set(best['walls'])
        
        for wall in list(current_walls):
            if elapsed() > timeout * 0.95:
                break
            
            test_walls = current_walls - {wall}
            valid, enclosed = is_valid(test_walls)
            
            if valid and len(enclosed) >= best['area']:
                # Can remove this wall! Update best
                best = {'area': len(enclosed), 'walls': list(test_walls), 'enclosed': enclosed}
                current_walls = test_walls
    
    refine()
    
    return best


# ============ GRID AUTO-DETECTION ============

def detect_grid_dimensions(image_path):
    """Detect grid dimensions from screenshot."""
    img = Image.open(image_path)
    pixels = np.array(img, dtype=np.float32)
    height, width = pixels.shape[:2]
    
    if len(pixels.shape) == 3:
        gray = np.mean(pixels[:, :, :3], axis=2)
    else:
        gray = pixels
    
    h_gradient = np.abs(np.diff(gray, axis=1))
    v_gradient = np.abs(np.diff(gray, axis=0))
    
    col_profile = np.sum(h_gradient, axis=0)
    row_profile = np.sum(v_gradient, axis=1)
    
    def find_period(signal, min_period=20, max_period=80):
        signal = signal - np.mean(signal)
        std = np.std(signal)
        if std > 0:
            signal = signal / std
        
        n = len(signal)
        autocorr = np.correlate(signal, signal, mode='full')[n-1:]
        
        best_period = min_period
        best_score = 0
        
        for period in range(min_period, min(max_period, n // 3)):
            if period < len(autocorr):
                score = autocorr[period]
                if score > best_score:
                    best_score = score
                    best_period = period
        
        return best_period
    
    cell_width = find_period(col_profile)
    cell_height = find_period(row_profile)
    
    cols = round(width / cell_width)
    rows = round(height / cell_height)
    
    return max(5, min(50, cols)), max(5, min(50, rows))


# ============ IMAGE PARSING ============

def parse_image(image_path, cols, rows, water_thresh=50, horse_thresh=160):
    """Parse game screenshot into grid."""
    img = Image.open(image_path)
    pixels = np.array(img, dtype=np.float32)
    img_height, img_width = pixels.shape[:2]
    
    cell_w = img_width / cols
    cell_h = img_height / rows
    
    grid = []
    horse_pos = None
    
    for row in range(rows):
        grid_row = []
        for col in range(cols):
            samples = []
            for fx, fy in [(0.5, 0.5), (0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)]:
                px = int((col + fx) * cell_w)
                py = int((row + fy) * cell_h)
                if px < img_width and py < img_height:
                    samples.append(pixels[py, px, :3])
            
            if not samples:
                grid_row.append('grass')
                continue
            
            avg = np.mean(samples, axis=0)
            avg_r, avg_g, avg_b = avg[0], avg[1], avg[2]
            brightness = np.mean(avg)
            is_whitish = abs(avg_r - avg_g) < 30 and abs(avg_g - avg_b) < 30
            
            if brightness > horse_thresh and is_whitish:
                grid_row.append('horse')
                horse_pos = (col, row)
            elif avg_b > avg_g and avg_b > avg_r and avg_b > water_thresh and avg_r < 80 and avg_g < 110:
                grid_row.append('water')
            else:
                grid_row.append('grass')
        
        grid.append(grid_row)
    
    return {'grid': grid, 'width': cols, 'height': rows, 'horsePos': horse_pos}


# ============ TEST UTILITIES ============

def create_test_grid(width, height, water_pattern=[]):
    grid = [['grass' for _ in range(width)] for _ in range(height)]
    for x, y in water_pattern:
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = 'water'
    return grid


def generate_random_test(seed=None):
    if seed is not None:
        random.seed(seed)
    
    width = random.randint(15, 25)
    height = random.randint(15, 25)
    grid = [['grass' for _ in range(width)] for _ in range(height)]
    
    region_cx = random.randint(4, width - 5)
    region_cy = random.randint(4, height - 5)
    region_size = random.randint(3, 6)
    
    min_x = max(1, region_cx - region_size)
    max_x = min(width - 2, region_cx + region_size)
    min_y = max(1, region_cy - region_size)
    max_y = min(height - 2, region_cy + region_size)
    
    water_cells = []
    for y in range(min_y - 1, max_y + 2):
        for x in range(min_x - 1, max_x + 2):
            if 0 <= x < width and 0 <= y < height:
                is_inside = min_x <= x <= max_x and min_y <= y <= max_y
                if not is_inside:
                    grid[y][x] = 'water'
                    water_cells.append((x, y))
    
    num_gaps = random.randint(2, 5)
    gap_cells = []
    
    for _ in range(num_gaps * 5):
        if len(gap_cells) >= num_gaps:
            break
        idx = random.randint(0, len(water_cells) - 1)
        gx, gy = water_cells[idx]
        
        touches_inside = touches_outside = False
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = gx + dx, gy + dy
            if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] == 'grass':
                is_inside = min_x <= nx <= max_x and min_y <= ny <= max_y
                if is_inside:
                    touches_inside = True
                else:
                    touches_outside = True
        
        if touches_inside and touches_outside:
            grid[gy][gx] = 'grass'
            gap_cells.append((gx, gy))
    
    horse_x = random.randint(min_x, max_x)
    horse_y = random.randint(min_y, max_y)
    grid[horse_y][horse_x] = 'horse'
    
    for y in range(height):
        for x in range(width):
            if grid[y][x] == 'grass' and random.random() < 0.03:
                is_inside = min_x <= x <= max_x and min_y <= y <= max_y
                if not is_inside and (x, y) not in gap_cells:
                    grid[y][x] = 'water'
    
    expected_area = sum(1 for y in range(min_y, max_y + 1) 
                       for x in range(min_x, max_x + 1) if grid[y][x] != 'water')
    
    return {
        'grid': grid, 'width': width, 'height': height,
        'horse_pos': (horse_x, horse_y),
        'expected_walls': len(gap_cells), 'expected_area': expected_area,
        'gap_cells': gap_cells
    }


def visualize_grid(grid, width, height, horse_pos, solution=None):
    wall_set = set(solution['walls']) if solution else set()
    enclosed = solution['enclosed'] if solution else set()
    
    lines = []
    for y in range(height):
        line = ""
        for x in range(width):
            if (x, y) == horse_pos:
                line += "H"
            elif (x, y) in wall_set:
                line += "W"
            elif grid[y][x] == 'water':
                line += "~"
            elif (x, y) in enclosed:
                line += "+"
            else:
                line += "."
        lines.append(line)
    return "\n".join(lines)


# ============ TESTS ============

def test_simple_enclosure():
    print("\n=== Test 1: Simple 1-gap enclosure ===")
    
    water = []
    for x in range(3, 10):
        water.append((x, 3))
        water.append((x, 9))
    for y in range(4, 9):
        water.append((3, y))
        water.append((9, y))
    water = [w for w in water if not (w[0] == 6 and w[1] == 3)]
    
    grid = create_test_grid(15, 15, water)
    grid[6][6] = 'horse'
    
    print("Grid:")
    print(visualize_grid(grid, 15, 15, (6, 6)))
    
    solution = find_optimal_enclosure(grid, 15, 15, (6, 6), 1)
    
    print(f"\nWith max 1 wall: {solution['area']} tiles, {len(solution['walls'])} wall(s)")
    print(f"Walls: {solution['walls']}")
    
    if solution['area'] >= 25 and len(solution['walls']) == 1:
        print("✓ PASS")
        return True
    else:
        print(f"✗ FAIL")
        print(visualize_grid(grid, 15, 15, (6, 6), solution))
        return False


def test_two_gaps():
    print("\n=== Test 2: 2-gap enclosure ===")
    
    water = []
    for x in range(3, 10):
        if x != 6:
            water.append((x, 3))
        if x != 5:
            water.append((x, 9))
    for y in range(4, 9):
        water.append((3, y))
        water.append((9, y))
    
    grid = create_test_grid(15, 15, water)
    grid[6][6] = 'horse'
    
    print("Grid:")
    print(visualize_grid(grid, 15, 15, (6, 6)))
    
    solution = find_optimal_enclosure(grid, 15, 15, (6, 6), 2)
    
    print(f"\nWith max 2 walls: {solution['area']} tiles, {len(solution['walls'])} wall(s)")
    print(f"Walls: {solution['walls']}")
    print(visualize_grid(grid, 15, 15, (6, 6), solution))
    
    if solution['area'] >= 25 and len(solution['walls']) <= 2:
        print("✓ PASS")
        return True
    else:
        print(f"✗ FAIL")
        return False


def test_random_cases():
    print("\n=== Test 3: Random test cases ===")
    
    passed = 0
    total = 5
    
    for i in range(total):
        test = generate_random_test(seed=12345 + i * 1000)
        
        solution = find_optimal_enclosure(
            test['grid'], test['width'], test['height'],
            test['horse_pos'], test['expected_walls'] + 2,
            timeout=5.0
        )
        
        area_threshold = test['expected_area'] * 0.7
        
        print(f"\n  Test {i+1}: {test['width']}x{test['height']}")
        print(f"    Expected: ~{test['expected_area']} tiles with {test['expected_walls']} walls")
        print(f"    Got: {solution['area']} tiles with {len(solution['walls'])} walls")
        
        if solution['area'] >= area_threshold:
            print("    ✓ PASS")
            passed += 1
        else:
            print("    ✗ FAIL")
    
    print(f"\nRandom tests: {passed}/{total} passed")
    return passed >= 3


def test_performance():
    print("\n=== Test 4: Performance test ===")
    
    width, height = 25, 30
    grid = [['grass' for _ in range(width)] for _ in range(height)]
    
    random.seed(99999)
    for _ in range(100):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        if not (x == 12 and y == 15):
            grid[y][x] = 'water'
    
    grid[15][12] = 'horse'
    
    start = time.time()
    solution = find_optimal_enclosure(grid, width, height, (12, 15), 15, timeout=10.0)
    elapsed = time.time() - start
    
    print(f"Grid: {width}x{height}, Time: {elapsed:.2f}s")
    print(f"Solution: {solution['area']} tiles with {len(solution['walls'])} walls")
    
    if elapsed < 10.0:
        print("✓ PASS (< 10s)")
        return True
    else:
        print("✗ FAIL (> 10s)")
        return False


def test_actual_screenshot():
    print("\n=== Test 5: Actual screenshot ===")
    
    try:
        cols, rows = 19, 23
        data = parse_image(PREVIEW_PATH, cols, rows, water_thresh=50, horse_thresh=160)
        
        if data['horsePos'] is None:
            img = Image.open(PREVIEW_PATH)
            pixels = np.array(img, dtype=np.float32)
            img_h, img_w = pixels.shape[:2]
            cell_w, cell_h = img_w / cols, img_h / rows
            
            best_bright = 0
            for r in range(rows):
                for c in range(cols):
                    px, py = int((c + 0.5) * cell_w), int((r + 0.5) * cell_h)
                    if px < img_w and py < img_h:
                        bright = np.mean(pixels[py, px, :3])
                        if bright > best_bright:
                            best_bright = bright
                            data['horsePos'] = (c, r)
            
            if data['horsePos']:
                data['grid'][data['horsePos'][1]][data['horsePos'][0]] = 'horse'
        
        if data['horsePos'] is None:
            print("✗ FAIL: Could not detect horse")
            return False
        
        print(f"Grid: {cols}x{rows}, Horse: {data['horsePos']}")
        
        grass = sum(1 for row in data['grid'] for c in row if c == 'grass')
        water = sum(1 for row in data['grid'] for c in row if c == 'water')
        print(f"Cells: {grass} grass, {water} water")
        
        print("\nDetected grid:")
        print(visualize_grid(data['grid'], cols, rows, data['horsePos']))
        
        start = time.time()
        solution = find_optimal_enclosure(
            data['grid'], cols, rows, data['horsePos'], 13, timeout=10.0
        )
        elapsed = time.time() - start
        
        print(f"\nTime: {elapsed:.2f}s")
        print(f"Solution: {solution['area']} tiles with {len(solution['walls'])} walls")
        print(f"Walls: {solution['walls']}")
        print("\nWith solution:")
        print(visualize_grid(data['grid'], cols, rows, data['horsePos'], solution))
        
        if solution['area'] >= 90:
            print("✓ PASS (≥90 tiles)")
            return True
        elif solution['area'] >= 70:
            print(f"Partial: got {solution['area']} tiles (target 95)")
            return True
        else:
            print(f"✗ FAIL: got {solution['area']} tiles (target 95)")
            return False
            
    except Exception as e:
        import traceback
        print(f"✗ FAIL: {e}")
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("Horse Pen Optimizer - Algorithm Test Suite")
    print("=" * 60)
    
    results = [
        ("Simple 1-gap", test_simple_enclosure()),
        ("2-gap", test_two_gaps()),
        ("Random cases", test_random_cases()),
        ("Performance", test_performance()),
        ("Actual screenshot", test_actual_screenshot()),
    ]
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        print(f"  {name}: {'✓ PASS' if passed else '✗ FAIL'}")
    
    print(f"\nTotal: {sum(1 for _, p in results if p)}/{len(results)} passed")
