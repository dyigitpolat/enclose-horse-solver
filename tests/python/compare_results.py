"""
Generate reference values from Python for JS comparison.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FIXTURES_DIR = os.path.join(ROOT_DIR, "tests", "fixtures")
sys.path.insert(0, ROOT_DIR)

from hp_image import parse_image_to_grid


def test_image(path):
    pg = parse_image_to_grid(path)
    return {
        'path': path,
        'grid': f'{pg.width}x{pg.height}',
        'water': pg.debug['water_count'],
        'grass': pg.debug['grass_count'],
        'cherries': len(pg.cherry_cells),
        'horse': pg.horse,
        'method': pg.debug.get('horse_method'),
        'water_thresh': pg.debug.get('color_clusters', {}).get('water_threshold', {}),
    }

if __name__ == '__main__':
    import glob
    
    # Get all p4 and p5 test images
    test_images = sorted(glob.glob(os.path.join(FIXTURES_DIR, 'p4*.png')) + glob.glob(os.path.join(FIXTURES_DIR, 'p5*.png')))
    
    print('='*70)
    print('PYTHON REFERENCE VALUES')
    print('='*70)
    
    results = {}
    for img in test_images:
        try:
            result = test_image(img)
            name = os.path.basename(img)
            results[name] = result
            print(f'\n{name}:')
            print(f'  Grid:     {result["grid"]}')
            print(f'  Horse:    {result["horse"]}')
            print(f'  Water:    {result["water"]}')
            print(f'  Grass:    {result["grass"]}')
            print(f'  Cherries: {result["cherries"]}')
            print(f'  Method:   {result["method"]}')
            print(f'  Water thresholds: bMin={result["water_thresh"].get("b_min", 0):.1f}, rMax={result["water_thresh"].get("r_max", 0):.1f}, gMax={result["water_thresh"].get("g_max", 0):.1f}')
        except Exception as e:
            print(f'\n{os.path.basename(img)}: ERROR - {e}')
    
    # Comparisons
    print('\n' + '='*70)
    print('COMPARISONS TO ORIGINALS')
    print('='*70)
    
    def compare(img1, img2):
        if img1 in results and img2 in results:
            r1, r2 = results[img1], results[img2]
            print(f'\n{img2}:')
            grid_ok = r1["grid"] == r2["grid"]
            horse_ok = r1["horse"] == r2["horse"]
            water_ok = r1["water"] == r2["water"]
            grass_ok = r1["grass"] == r2["grass"]
            cherry_ok = r1["cherries"] == r2["cherries"]
            
            print(f'  Grid:     {r2["grid"]} {"✓" if grid_ok else "✗ MISMATCH"}')
            print(f'  Horse:    {r2["horse"]} {"✓" if horse_ok else "✗ MISMATCH"}')
                print(f'  Water:    {r2["water"]} (expected {r1["water"]}) {"✓" if water_ok else "✗ MISMATCH"}')
                print(f'  Grass:    {r2["grass"]} (expected {r1["grass"]}) {"✓" if grass_ok else "✗ MISMATCH"}')
            
            if r1["cherries"] > 0 or r2["cherries"] > 0:
                print(f'  Cherries: {r2["cherries"]} {"✓" if cherry_ok else "✗ MISMATCH"}')
            
            all_ok = grid_ok and horse_ok and cherry_ok and water_ok and grass_ok
            return all_ok
        return False
    
    # Compare all p4 variations to p4.png
    if 'p4.png' in results:
        print('\n--- p4.png variations ---')
        p4_tests = [img for img in results.keys() if img.startswith('p4_')]
        for img in sorted(p4_tests):
            compare('p4.png', img)
    
    # Compare all p5 variations to p5.png
    if 'p5.png' in results:
        print('\n--- p5.png variations ---')
        p5_tests = [img for img in results.keys() if img.startswith('p5_')]
        for img in sorted(p5_tests):
            compare('p5.png', img)
    
    print('\n' + '='*70)
    print('COPY THESE VALUES TO COMPARE WITH JAVASCRIPT')
    print('='*70)

