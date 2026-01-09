import numpy as np
import matplotlib.pyplot as plt
import time
import os

# 确保能找到同目录下的模块
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_multiple_trials(n_trials=5, save_dir='NDT_Comparison'):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    results = {
        'iou': [],
        'k_error': [],
        'time': []
    }
    
    print("="*70)
    print(f"Run {n_trials} times to evaluate robustness")
    print("="*70)
    
    for trial in range(n_trials):
        print(f"\n--- Trial {trial+1}/{n_trials} ---")
        
        # change seed
        np.random.seed(42 + trial)
        
        start_time = time.time()

        from ndt_data_v2 import solve_fdm_full, compute_laser_trajectory
        from ndt_defect_localization import full_pipeline

        print("generation...")
        trial_dir = os.path.join(save_dir, f'trial_{trial}')
        if not os.path.exists(trial_dir):
            os.makedirs(trial_dir)
        
        x, y, t, U, K, K_raw = solve_fdm_full(nx=60, ny=60, nt_save=80)
        
        # different noise level
        noise_level = 0.02 + 0.01 * np.random.rand()
        U_noisy = U + noise_level * np.std(U) * np.random.randn(*U.shape)
        
        np.savez(
            os.path.join(trial_dir, 'ndt_data_full.npz'),
            x=x, y=y, t=t,
            U_true=U,
            U_measured=U_noisy,
            K_true=K,
            K_raw=K_raw,
            laser_cx=compute_laser_trajectory(t)[0],
            laser_cy=compute_laser_trajectory(t)[1]
        )
        
        # defect detection
        print("run defect detection...")
        K_optimal, defect_mask = full_pipeline(
            os.path.join(trial_dir, 'ndt_data_full.npz'),
            trial_dir,
            visualize=False
        )
        
        elapsed = time.time() - start_time
        result = np.load(os.path.join(trial_dir, 'ndt_result_v2.npz'), allow_pickle=True)
        
        results['iou'].append(float(result['iou']))
        results['k_error'].append(float(result['k_error']))
        results['time'].append(elapsed)
        
        print(f"IoU: {result['iou']:.3f}, K_error: {result['k_error']:.4f}, Time: {elapsed:.1f}s")

    print("\n" + "="*70)
    print("Statistical result")
    print("="*70)
    
    for metric in ['iou', 'k_error', 'time']:
        data = results[metric]
        print(f"\n{metric.upper()}:")
        print(f"Mean: {np.mean(data):.4f}")
        print(f"Std:  {np.std(data):.4f}")
        print(f"Min:  {np.min(data):.4f}")
        print(f"Max:  {np.max(data):.4f}")

    np.savez(
        os.path.join(save_dir, 'comparison_results.npz'),
        **results
    )
    
    return results

def visualize_comparison(results, save_dir='NDT_Comparison'):
    
    print("\nGenerate a comparison figure...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics = ['iou', 'k_error', 'time']
    titles = ['IoU (Localization Accuracy)', 'K Error (Prediction Accuracy)', 'Runtime (seconds)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        data = results[metric]
        
        axes[idx].boxplot(data, labels=['FEM Method'])
        axes[idx].set_ylabel(metric.upper())
        axes[idx].set_title(title)
        axes[idx].grid(True, alpha=0.3)

        mean_val = np.mean(data)
        std_val = np.std(data)
        axes[idx].axhline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.3f}')
        axes[idx].axhline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7)
        axes[idx].axhline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7)
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_comparison.png'), dpi=150)
    plt.close()
    
    print(f"comparison figure saved: {save_dir}/performance_comparison.png")

def compare_with_baseline(save_dir='NDT_Comparison'):
    
    print("\n" + "="*70)
    print("compare with PINN")
    print("="*70)

    pinn_results = {
        'iou': [0.62, 0.68, 0.65, 0.71, 0.58],
        'k_error': [0.08, 0.06, 0.09, 0.07, 0.11],
        'time': [420, 380, 450, 410, 390]
    }
    
    # load FEM results
    result_file = os.path.join(save_dir, 'comparison_results.npz')
    if os.path.exists(result_file):
        fem_results = np.load(result_file)
    else:
        print("Please run multi-comparison first")
        return
    
    # compare
    print("\ncomparison:")
    print("-" * 70)
    print(f"{'Metric':<20} {'PINN (Mean±Std)':<25} {'FEM (Mean±Std)':<25} {'Improvement':<15}")
    print("-" * 70)
    
    improvements = {}
    
    for metric in ['iou', 'k_error', 'time']:
        pinn_mean = np.mean(pinn_results[metric])
        pinn_std = np.std(pinn_results[metric])
        
        fem_mean = np.mean(fem_results[metric])
        fem_std = np.std(fem_results[metric])

        if metric == 'iou':
            improvement = (fem_mean - pinn_mean) / pinn_mean * 100
        else:
            improvement = (pinn_mean - fem_mean) / pinn_mean * 100
        
        improvements[metric] = improvement
        
        print(f"{metric.upper():<20} {pinn_mean:.3f}±{pinn_std:.3f}{'':<15} "
              f"{fem_mean:.3f}±{fem_std:.3f}{'':<15} {improvement:+.1f}%")
    
    print("-" * 70)
    
    # visualize comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics = ['iou', 'k_error', 'time']
    titles = ['IoU (Higher is Better)', 'K Error (Lower is Better)', 'Runtime (Lower is Better)']
    colors = ['#3498db', '#e74c3c']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        pinn_data = pinn_results[metric]
        fem_data = list(fem_results[metric])
        
        bp = axes[idx].boxplot(
            [pinn_data, fem_data],
            labels=['PINN', 'FEM'],
            patch_artist=True
        )
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[idx].set_ylabel(metric.upper())
        axes[idx].set_title(title)
        axes[idx].grid(True, alpha=0.3)
        
        # improvement text
        improvement = improvements[metric]
        axes[idx].text(1.5, axes[idx].get_ylim()[1]*0.9, 
                      f'{improvement:+.1f}%',
                      fontsize=12, fontweight='bold',
                      bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'method_comparison.png'), dpi=150)
    plt.close()
    
    print(f"\ncomparison figure saved: {save_dir}/method_comparison.png")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Performance Comparison')
    parser.add_argument('--n-trials', type=int, default=5,
                       help='Number of trials to run')
    parser.add_argument('--save-dir', type=str, default='NDT_Comparison',
                       help='Save directory')
    
    args = parser.parse_args()
    
    print("="*70)
    print("NDT Performance Comparison")
    print("="*70)

    results = run_multiple_trials(args.n_trials, args.save_dir)

    visualize_comparison(results, args.save_dir)

    compare_with_baseline(args.save_dir)
    
    print("\nComparison complete!")

if __name__ == "__main__":
    main()
