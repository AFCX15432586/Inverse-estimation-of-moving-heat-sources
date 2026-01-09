#!/usr/bin/env python3
"""
Import Verification Script
==========================
验证所有模块是否可以正确导入
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*70)
print("验证所有模块导入...")
print("="*70)

try:
    print("\n1. 测试 ndt_data_v2...")
    from ndt_data_v2 import solve_fdm_full, k_distribution_ground_truth, laser_source_known
    print("   ✓ ndt_data_v2 导入成功")
except Exception as e:
    print(f"   ✗ ndt_data_v2 导入失败: {e}")
    sys.exit(1)

try:
    print("\n2. 测试 ndt_fem_solver...")
    from ndt_fem_solver import FEMSolver, compare_baseline_vs_measured
    print("   ✓ ndt_fem_solver 导入成功")
except Exception as e:
    print(f"   ✗ ndt_fem_solver 导入失败: {e}")
    sys.exit(1)

try:
    print("\n3. 测试 ndt_defect_localization...")
    from ndt_defect_localization import DefectLocalizer, DefectQuantifier, full_pipeline
    print("   ✓ ndt_defect_localization 导入成功")
except Exception as e:
    print(f"   ✗ ndt_defect_localization 导入失败: {e}")
    sys.exit(1)

try:
    print("\n4. 测试 ndt_visualizer...")
    from ndt_visualizer import NDTVisualizer
    print("   ✓ ndt_visualizer 导入成功")
except Exception as e:
    print(f"   ✗ ndt_visualizer 导入失败: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✓ 所有模块导入成功！")
print("="*70)
print("\n可以安全运行:")
print("  python demo.py")
print("  python test_system.py")
print("  python ndt_main.py")
print("  python ndt_comparison.py")
print("="*70)
