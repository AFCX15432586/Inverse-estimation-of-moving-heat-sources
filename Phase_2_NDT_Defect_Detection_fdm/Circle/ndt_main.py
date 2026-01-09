"""
NDT System V2 - Main Runner
============================
How To Use:
    python ndt_main.py --mode all        # Complete the entire process
    python ndt_main.py --mode data       # Generate only data
    python ndt_main.py --mode detect     # Only perform detection (requires existing data)
"""

import argparse
import os
import sys
import time
import numpy as np

def run_data_generation(save_dir='NDT_Data_V2', nx=80, ny=80, nt_save=100):

    print("\n" + "="*70)
    print("Step 1/3: data init")
    print("="*70)
    
    from ndt_data_v2 import solve_fdm_full, visualize_data, compute_laser_trajectory
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # init data
    x, y, t, U, K, K_raw = solve_fdm_full(nx=nx, ny=ny, nt_save=nt_save)
    
    # check
    if np.isnan(U).any():
        print("\nerror: NaN")
        return False
    
    # noise
    noise_level = 0.02
    U_noisy = U + noise_level * np.std(U) * np.random.randn(*U.shape)
    
    # save
    print(f"\nsave data to {save_dir}...")
    np.savez(
        os.path.join(save_dir, 'ndt_data_full.npz'),
        x=x, y=y, t=t,
        U_true=U,
        U_measured=U_noisy,
        K_true=K,
        K_raw=K_raw,
        laser_cx=compute_laser_trajectory(t)[0],
        laser_cy=compute_laser_trajectory(t)[1]
    )
    
    # visualize
    visualize_data(x, y, t, U_noisy, K, K_raw, save_dir)
    
    print("\n✓ Data generation is complete!")
    return True

def run_baseline_comparison(save_dir='NDT_Data_V2'):

    print("\n" + "="*70)
    print("Step 2/3: Baseline comparison")
    print("="*70)
    
    from ndt_fem_solver import compare_baseline_vs_measured
    
    data_file = os.path.join(save_dir, 'ndt_data_full.npz')
    if not os.path.exists(data_file):
        print(f"no data file exists: {data_file}")
        return False
    
    data = np.load(data_file)
    residual, grad_diff, U_baseline = compare_baseline_vs_measured(data, save_dir)
    
    # save baseline
    np.savez(
        os.path.join(save_dir, 'baseline_solution.npz'),
        U_baseline=U_baseline,
        residual=residual,
        grad_diff=grad_diff
    )
    
    print("\n✓ Baseline comparison completed!")
    return True

def run_defect_detection(save_dir='NDT_Data_V2'):

    print("\n" + "="*70)
    print("Step 3/3: Defect detecting")
    print("="*70)
    
    from ndt_defect_localization import full_pipeline
    
    data_file = os.path.join(save_dir, 'ndt_data_full.npz')
    if not os.path.exists(data_file):
        print(f"error: no data file exists: {data_file}")
        return False
    
    K_optimal, defect_mask = full_pipeline(data_file, save_dir, visualize=True)
    
    print("\n✓ Defect detection is complete!")
    return True

def print_summary(save_dir='NDT_Data_V2'):

    print("\n" + "="*70)
    print("Result Summary")
    print("="*70)
    
    result_file = os.path.join(save_dir, 'ndt_result_v2.npz')
    if not os.path.exists(result_file):
        print("ndt_result_v2.npz does not exist.")
        return
    
    result = np.load(result_file, allow_pickle=True)
    
    print(f"\nPositional accuracy:")
    print(f"  IoU: {result['iou']:.3f}")
    
    print(f"\nK value accuracy:")
    print(f"  Predict K (defect): {result['opt_result'][0]:.4f}")
    print(f"  absolute error: {result['k_error']:.4f}")
    
    print(f"\nThe generated file:")
    for fname in ['ndt_data_full.npz', 'baseline_comparison.png', 
                  'ndt_results_v2.png', 'ndt_result_v2.npz']:
        fpath = os.path.join(save_dir, fname)
        if os.path.exists(fpath):
            print(f"  ✓ {fname}")
        else:
            print(f"  ✗ {fname}")

def main():
    parser = argparse.ArgumentParser(description='NDT System V2')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'data', 'baseline', 'detect'],
                       help='running mode')
    parser.add_argument('--save-dir', type=str, default='NDT_Data_V2',
                       help='save directory')
    parser.add_argument('--nx', type=int, default=80,
                       help='Number of grid points in the x direction')
    parser.add_argument('--ny', type=int, default=80,
                       help='Number of grid points in the y direction')
    parser.add_argument('--nt-save', type=int, default=100,
                       help='Number of saved time snapshots')
    
    args = parser.parse_args()
    
    print("="*70)
    print("NDT System V2 - FEM leadership + two-stage inspection")
    print("="*70)
    print(f"mode: {args.mode}")
    print(f"save directory: {args.save_dir}")
    
    start_time = time.time()
    success = True
    
    if args.mode in ['all', 'data']:
        success = run_data_generation(args.save_dir, args.nx, args.ny, args.nt_save)
        if not success:
            sys.exit(1)
    
    if args.mode in ['all', 'baseline']:
        success = run_baseline_comparison(args.save_dir)
        if not success:
            sys.exit(1)
    
    if args.mode in ['all', 'detect']:
        success = run_defect_detection(args.save_dir)
        if not success:
            sys.exit(1)
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"Using time: {elapsed:.2f} seconds")
    print("="*70)
    
    if args.mode == 'all':
        print_summary(args.save_dir)
    
    print("\n✓ ALL Completed!")

if __name__ == "__main__":
    main()
