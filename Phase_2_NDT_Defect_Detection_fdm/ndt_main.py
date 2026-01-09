"""
NDT System V2 - Main Runner
============================
集成整个NDT系统的主脚本

使用方法:
    python ndt_main.py --mode all        # 运行完整流程
    python ndt_main.py --mode data       # 仅生成数据
    python ndt_main.py --mode detect     # 仅运行检测（需要已有数据）
"""

import argparse
import os
import sys
import time
import numpy as np

def run_data_generation(save_dir='NDT_Data_V2', nx=80, ny=80, nt_save=100):
    """运行数据生成"""
    print("\n" + "="*70)
    print("步骤 1/3: 数据生成")
    print("="*70)
    
    from ndt_data_v2 import solve_fdm_full, visualize_data, compute_laser_trajectory
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 生成数据
    x, y, t, U, K, K_raw = solve_fdm_full(nx=nx, ny=ny, nt_save=nt_save)
    
    # 检查
    if np.isnan(U).any():
        print("\n❌ 错误: 检测到NaN！")
        return False
    
    # 添加噪声
    noise_level = 0.02
    U_noisy = U + noise_level * np.std(U) * np.random.randn(*U.shape)
    
    # 保存
    print(f"\n保存数据到 {save_dir}...")
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
    
    # 可视化
    visualize_data(x, y, t, U_noisy, K, K_raw, save_dir)
    
    print("\n✓ 数据生成完成！")
    return True

def run_baseline_comparison(save_dir='NDT_Data_V2'):
    """运行基准对比"""
    print("\n" + "="*70)
    print("步骤 2/3: 基准对比")
    print("="*70)
    
    from ndt_fem_solver import compare_baseline_vs_measured
    
    data_file = os.path.join(save_dir, 'ndt_data_full.npz')
    if not os.path.exists(data_file):
        print(f"❌ 错误: 数据文件不存在: {data_file}")
        return False
    
    data = np.load(data_file)
    residual, grad_diff, U_baseline = compare_baseline_vs_measured(data, save_dir)
    
    # 保存基准解
    np.savez(
        os.path.join(save_dir, 'baseline_solution.npz'),
        U_baseline=U_baseline,
        residual=residual,
        grad_diff=grad_diff
    )
    
    print("\n✓ 基准对比完成！")
    return True

def run_defect_detection(save_dir='NDT_Data_V2'):
    """运行缺陷检测"""
    print("\n" + "="*70)
    print("步骤 3/3: 缺陷检测")
    print("="*70)
    
    from ndt_defect_localization import full_pipeline
    
    data_file = os.path.join(save_dir, 'ndt_data_full.npz')
    if not os.path.exists(data_file):
        print(f"❌ 错误: 数据文件不存在: {data_file}")
        return False
    
    K_optimal, defect_mask = full_pipeline(data_file, save_dir, visualize=True)
    
    print("\n✓ 缺陷检测完成！")
    return True

def print_summary(save_dir='NDT_Data_V2'):
    """打印结果摘要"""
    print("\n" + "="*70)
    print("结果摘要")
    print("="*70)
    
    result_file = os.path.join(save_dir, 'ndt_result_v2.npz')
    if not os.path.exists(result_file):
        print("结果文件不存在")
        return
    
    result = np.load(result_file, allow_pickle=True)
    
    print(f"\n定位精度:")
    print(f"  IoU: {result['iou']:.3f}")
    
    print(f"\nK值精度:")
    print(f"  预测K (缺陷): {result['opt_result'][0]:.4f}")
    print(f"  绝对误差: {result['k_error']:.4f}")
    
    print(f"\n生成的文件:")
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
                       help='运行模式')
    parser.add_argument('--save-dir', type=str, default='NDT_Data_V2',
                       help='保存目录')
    parser.add_argument('--nx', type=int, default=80,
                       help='x方向网格数')
    parser.add_argument('--ny', type=int, default=80,
                       help='y方向网格数')
    parser.add_argument('--nt-save', type=int, default=100,
                       help='保存的时间快照数')
    
    args = parser.parse_args()
    
    print("="*70)
    print("NDT System V2 - FEM先导 + 两阶段检测")
    print("="*70)
    print(f"模式: {args.mode}")
    print(f"保存目录: {args.save_dir}")
    
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
    print(f"总用时: {elapsed:.2f} 秒")
    print("="*70)
    
    if args.mode == 'all':
        print_summary(args.save_dir)
    
    print("\n✓ 全部完成！")

if __name__ == "__main__":
    main()
