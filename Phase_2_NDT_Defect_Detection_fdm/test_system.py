"""
Quick Test Script
=================
快速测试所有模块是否正常工作

使用小规模参数以加快测试速度
"""

import numpy as np
import os
import sys

# 确保能找到同目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试所有导入"""
    print("\n[1/5] 测试模块导入...")
    try:
        from ndt_data_v2 import solve_fdm_full
        from ndt_fem_solver import FEMSolver, compare_baseline_vs_measured
        from ndt_defect_localization import DefectLocalizer, DefectQuantifier
        print("  ✓ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"  ✗ 导入失败: {e}")
        return False

def test_data_generation():
    """测试数据生成"""
    print("\n[2/5] 测试数据生成（小规模）...")
    try:
        from ndt_data_v2 import solve_fdm_full
        
        # 使用小规模参数
        x, y, t, U, K, K_raw = solve_fdm_full(nx=30, ny=30, nt_save=20)
        
        # 检查
        assert not np.isnan(U).any(), "检测到NaN"
        assert U.shape == (20, 30, 30), "形状不正确"
        assert K.shape == (30, 30), "K形状不正确"
        
        print("  ✓ 数据生成成功")
        print(f"    U范围: [{np.min(U):.2f}, {np.max(U):.2f}]")
        print(f"    K范围: [{np.min(K):.2f}, {np.max(K):.2f}]")
        
        return True, (x, y, t, U, K, K_raw)
    except Exception as e:
        print(f"  ✗ 数据生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_fem_solver(data):
    """测试FEM求解器"""
    print("\n[3/5] 测试FEM求解器...")
    try:
        from ndt_fem_solver import FEMSolver
        
        x, y, t, U, K, K_raw = data
        
        dt = t[1] - t[0]
        solver = FEMSolver(x, y, dt=dt)
        
        # 求解k=1情况
        K_baseline = np.ones_like(solver.X)
        U_baseline = solver.solve_transient(K_baseline, t[:5], verbose=False)  # 只解5步
        
        assert not np.isnan(U_baseline).any(), "检测到NaN"
        assert U_baseline.shape == (5, 30, 30), "形状不正确"
        
        print("  ✓ FEM求解器工作正常")
        print(f"    基准解范围: [{np.min(U_baseline):.2f}, {np.max(U_baseline):.2f}]")
        
        return True, U_baseline
    except Exception as e:
        print(f"  ✗ FEM求解器失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_defect_localization(data, U_baseline):
    """测试缺陷定位"""
    print("\n[4/5] 测试缺陷定位...")
    try:
        from ndt_defect_localization import DefectLocalizer
        
        x, y, t, U, K, K_raw = data
        
        # 添加噪声
        U_noisy = U[:5] + 0.02 * np.std(U) * np.random.randn(5, 30, 30)
        
        localizer = DefectLocalizer(x, y, t[:5], U_noisy, threshold_percentile=80)
        defect_mask, anomaly_map = localizer.locate_defects(U_baseline)
        
        assert defect_mask.shape == (30, 30), "掩码形状不正确"
        assert np.sum(defect_mask) > 0, "未检测到缺陷"
        
        print("  ✓ 缺陷定位工作正常")
        print(f"    检测到缺陷点数: {np.sum(defect_mask)}")
        print(f"    异常图范围: [{np.min(anomaly_map):.3f}, {np.max(anomaly_map):.3f}]")
        
        return True, defect_mask
    except Exception as e:
        print(f"  ✗ 缺陷定位失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_defect_quantification(data, defect_mask):
    """测试缺陷量化"""
    print("\n[5/5] 测试缺陷量化（简化版）...")
    try:
        from ndt_defect_localization import DefectQuantifier
        
        x, y, t, U, K, K_raw = data
        
        # 添加噪声
        U_noisy = U[:5] + 0.02 * np.std(U) * np.random.randn(5, 30, 30)
        
        # 使用小的缺陷mask测试
        test_mask = np.zeros((30, 30), dtype=bool)
        test_mask[10:15, 10:15] = True  # 小区域
        
        quantifier = DefectQuantifier(x, y, t[:5], U_noisy, test_mask)
        
        print("  ✓ 缺陷量化模块初始化成功")
        print("  ⚠ 跳过完整优化（耗时较长）")
        
        return True
    except Exception as e:
        print(f"  ✗ 缺陷量化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """运行所有测试"""
    print("="*70)
    print("NDT System V2 - Quick Test")
    print("="*70)
    print("使用小规模参数进行快速测试...")
    
    # 测试1: 导入
    if not test_imports():
        print("\n❌ 测试失败：模块导入错误")
        return False
    
    # 测试2: 数据生成
    success, data = test_data_generation()
    if not success:
        print("\n❌ 测试失败：数据生成错误")
        return False
    
    # 测试3: FEM求解器
    success, U_baseline = test_fem_solver(data)
    if not success:
        print("\n❌ 测试失败：FEM求解器错误")
        return False
    
    # 测试4: 缺陷定位
    success, defect_mask = test_defect_localization(data, U_baseline)
    if not success:
        print("\n❌ 测试失败：缺陷定位错误")
        return False
    
    # 测试5: 缺陷量化
    success = test_defect_quantification(data, defect_mask)
    if not success:
        print("\n❌ 测试失败：缺陷量化错误")
        return False
    
    print("\n" + "="*70)
    print("✅ 所有测试通过！")
    print("="*70)
    print("\n系统已准备就绪，可以运行完整流程:")
    print("  python ndt_main.py --mode all")
    print("\n或运行性能对比:")
    print("  python ndt_comparison.py --n-trials 3")
    
    return True

if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠ 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
