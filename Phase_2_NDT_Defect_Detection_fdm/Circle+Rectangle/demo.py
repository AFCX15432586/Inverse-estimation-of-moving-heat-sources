#!/usr/bin/env python3

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demo():
    print("="*70)
    print("NDT System V2 - Quick Demo")
    print("FEM-Guided Two-Stage Defect Detection")
    print("="*70)
    
    # test packages
    print("\ntest packages...")
    try:
        import numpy as np
        import scipy
        import matplotlib
        print("All packages have been added")
    except ImportError as e:
        print(f"These package are missing: {e}")
        print("\nPlease run: pip install -r requirements.txt")
        sys.exit(1)
    
    # output dir
    demo_dir = "NDT_Demo"
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)
    
    print(f"\nAll files will be saved to: {demo_dir}/")
    
    # 1. data init
    print("\n" + "-"*70)
    print("Generate demonstration data")
    print("-"*70)
    
    from ndt_data_v2 import solve_fdm_full, visualize_data, compute_laser_trajectory
    
    start = time.time()
    x, y, t, U, K, K_raw = solve_fdm_full(nx=80, ny=80, nt_save=50)

    U_noisy = U + 0.02 * np.std(U) * np.random.randn(*U.shape)

    np.savez(
        os.path.join(demo_dir, 'ndt_data_full.npz'),
        x=x, y=y, t=t,
        U_true=U, U_measured=U_noisy,
        K_true=K, K_raw=K_raw,
        laser_cx=compute_laser_trajectory(t)[0],
        laser_cy=compute_laser_trajectory(t)[1]
    )
    
    visualize_data(x, y, t, U_noisy, K, K_raw, demo_dir)
    
    print(f"\n✓ data generated ({time.time()-start:.1f}s)")
    
    # 2. compare baseline vs measured
    print("\n" + "-"*70)
    print("Compare baseline vs measured")
    print("-"*70)
    
    from ndt_fem_solver import compare_baseline_vs_measured
    
    start = time.time()
    data = np.load(os.path.join(demo_dir, 'ndt_data_full.npz'))
    residual, grad_diff, U_baseline = compare_baseline_vs_measured(data, demo_dir)
    
    print(f"\nbaseline vs measured completed ({time.time()-start:.1f}s)")
    
    # 3.Two-stage defect detection
    print("\n" + "-"*70)
    print("Two-stage defect detection")
    print("-"*70)
    
    from ndt_defect_localization import full_pipeline
    
    start = time.time()
    K_optimal, defect_mask = full_pipeline(
        os.path.join(demo_dir, 'ndt_data_full.npz'),
        demo_dir,
        visualize=True
    )
    
    print(f"\nDefect detection completed ({time.time()-start:.1f}s)")
    
    # 4. Generate visualization
    print("\n" + "-"*70)
    print("Generate visualization")
    print("-"*70)
    
    from ndt_visualizer import NDTVisualizer
    
    visualizer = NDTVisualizer(demo_dir)
    visualizer.plot_overview()
    visualizer.plot_temporal_evolution()
    visualizer.generate_report()
    
    # final message
    print("\n" + "="*70)
    print("The demonstration is complete!")
    print("="*70)
    
    result = np.load(os.path.join(demo_dir, 'ndt_result_v2.npz'), allow_pickle=True)
    
    print(f"\nKey results:")
    print(f"  IoU: {result['iou']:.3f}")
    print(f"  K value error: {result['k_error']:.4f}")
    print(f"  Predicting the K value: {result['opt_result'][0]:.4f} (true: 0.20)")
    
    print(f"\ngenerated file:")
    files = [
        'K_distribution.png',
        'temperature_evolution.png',
        'laser_trajectory.png',
        'baseline_comparison.png',
        'ndt_results_v2.png',
        'overview_dashboard.png',
        'temporal_evolution.png',
        'detection_report.md'
    ]
    
    for fname in files:
        fpath = os.path.join(demo_dir, fname)
        if os.path.exists(fpath):
            print(f"  ✓ {fname}")
    
    print(f"\nAll files have been saved to: {demo_dir}/")
    print(f"\nHint:")
    print(f"  - View the complete report: cat {demo_dir}/detection_report.md")
    print(f"  - View the overview chart: {demo_dir}/overview_dashboard.png")
    print(f"\nAttempt full-scale operation:")
    print(f"  python ndt_main.py --mode all --nx 80 --ny 80")

if __name__ == "__main__":
    try:
        demo()
    except KeyboardInterrupt:
        print("\n\nKey interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nerror: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
