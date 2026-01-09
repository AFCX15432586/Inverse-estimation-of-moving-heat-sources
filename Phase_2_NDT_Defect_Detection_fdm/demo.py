#!/usr/bin/env python3
"""
Quick Demo - NDT System V2
===========================
快速演示新系统的能力

这个脚本会:
1. 生成小规模数据（快速）
2. 运行完整检测流程
3. 展示关键结果
"""

import os
import sys
import time

# 确保能找到同目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demo():
    print("="*70)
    print("NDT System V2 - Quick Demo")
    print("FEM-Guided Two-Stage Defect Detection")
    print("="*70)
    
    # 检查依赖
    print("\n检查依赖...")
    try:
        import numpy as np
        import scipy
        import matplotlib
        print("  ✓ 所有依赖已安装")
    except ImportError as e:
        print(f"  ✗ 缺少依赖: {e}")
        print("\n请运行: pip install -r requirements.txt")
        sys.exit(1)
    
    # 创建输出目录
    demo_dir = "NDT_Demo"
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)
    
    print(f"\n演示数据将保存到: {demo_dir}/")
    
    # 步骤1: 数据生成
    print("\n" + "-"*70)
    print("步骤 1: 生成演示数据（小规模，快速）")
    print("-"*70)
    
    from ndt_data_v2 import solve_fdm_full, visualize_data, compute_laser_trajectory
    
    start = time.time()
    x, y, t, U, K, K_raw = solve_fdm_full(nx=40, ny=40, nt_save=50)
    
    # 添加噪声
    U_noisy = U + 0.02 * np.std(U) * np.random.randn(*U.shape)
    
    # 保存
    np.savez(
        os.path.join(demo_dir, 'ndt_data_full.npz'),
        x=x, y=y, t=t,
        U_true=U, U_measured=U_noisy,
        K_true=K, K_raw=K_raw,
        laser_cx=compute_laser_trajectory(t)[0],
        laser_cy=compute_laser_trajectory(t)[1]
    )
    
    visualize_data(x, y, t, U_noisy, K, K_raw, demo_dir)
    
    print(f"\n✓ 数据生成完成 ({time.time()-start:.1f}s)")
    
    # 步骤2: 基准对比
    print("\n" + "-"*70)
    print("步骤 2: 求解K=1基准并对比")
    print("-"*70)
    
    from ndt_fem_solver import compare_baseline_vs_measured
    
    start = time.time()
    data = np.load(os.path.join(demo_dir, 'ndt_data_full.npz'))
    residual, grad_diff, U_baseline = compare_baseline_vs_measured(data, demo_dir)
    
    print(f"\n✓ 基准对比完成 ({time.time()-start:.1f}s)")
    
    # 步骤3: 缺陷检测
    print("\n" + "-"*70)
    print("步骤 3: 两阶段缺陷检测")
    print("-"*70)
    
    from ndt_defect_localization import full_pipeline
    
    start = time.time()
    K_optimal, defect_mask = full_pipeline(
        os.path.join(demo_dir, 'ndt_data_full.npz'),
        demo_dir,
        visualize=True
    )
    
    print(f"\n✓ 缺陷检测完成 ({time.time()-start:.1f}s)")
    
    # 步骤4: 生成可视化
    print("\n" + "-"*70)
    print("步骤 4: 生成详细报告和可视化")
    print("-"*70)
    
    from ndt_visualizer import NDTVisualizer
    
    visualizer = NDTVisualizer(demo_dir)
    visualizer.plot_overview()
    visualizer.plot_temporal_evolution()
    visualizer.generate_report()
    
    # 最终总结
    print("\n" + "="*70)
    print("演示完成！")
    print("="*70)
    
    result = np.load(os.path.join(demo_dir, 'ndt_result_v2.npz'), allow_pickle=True)
    
    print(f"\n📊 关键结果:")
    print(f"  • 定位精度 (IoU): {result['iou']:.3f}")
    print(f"  • K值误差: {result['k_error']:.4f}")
    print(f"  • 预测K值: {result['opt_result'][0]:.4f} (真实值: ~0.20)")
    
    print(f"\n📁 生成的文件:")
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
    
    print(f"\n📍 所有结果保存在: {demo_dir}/")
    print(f"\n💡 提示:")
    print(f"  - 查看完整报告: cat {demo_dir}/detection_report.md")
    print(f"  - 查看总览图: {demo_dir}/overview_dashboard.png")
    print(f"\n🚀 尝试完整规模:")
    print(f"  python ndt_main.py --mode all --nx 80 --ny 80")

if __name__ == "__main__":
    try:
        demo()
    except KeyboardInterrupt:
        print("\n\n⚠ 演示被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
