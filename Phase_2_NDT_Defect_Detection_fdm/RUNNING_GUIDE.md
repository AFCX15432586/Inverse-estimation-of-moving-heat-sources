# 🎯 运行指南 - NDT System V2

## ✅ 已修复所有导入问题！

所有脚本现在都可以正常运行了。

---

## 🚀 快速开始（3步）

### 1️⃣ 验证导入（推荐）
```bash
cd NDT_System_V2
python verify_imports.py
```

**应该看到：**
```
✓ ndt_data_v2 导入成功
✓ ndt_fem_solver 导入成功
✓ ndt_defect_localization 导入成功
✓ ndt_visualizer 导入成功
```

### 2️⃣ 快速演示（5分钟）
```bash
python demo.py
```

**会生成：**
- `NDT_Demo/` 目录
- 完整的温度场数据
- 检测结果和可视化
- 详细的报告

### 3️⃣ 查看结果
```bash
# 查看生成的文件
ls NDT_Demo/

# 查看报告
cat NDT_Demo/detection_report.md
```

---

## 📋 完整功能测试

### 测试1: 系统测试（2分钟）
```bash
python test_system.py
```
- 测试所有模块功能
- 使用小规模数据
- 快速验证系统正常

### 测试2: 完整流程（10-15分钟）
```bash
python ndt_main.py --mode all --nx 80 --ny 80 --nt-save 100
```
- 生成高精度数据
- 运行完整检测
- 保存在 `NDT_Data_V2/` 目录

### 测试3: 性能对比（20-30分钟）
```bash
python ndt_comparison.py --n-trials 3
```
- 运行多次实验
- 评估鲁棒性
- 生成对比报告

### 测试4: 结果可视化
```bash
python ndt_visualizer.py --all --data-dir NDT_Data_V2
```
- 生成详细可视化
- 创建总览仪表板
- 生成报告

---

## 📂 运行后的目录结构

```
NDT_System_V2/
├── （原始文件）
│
├── NDT_Demo/                    # demo.py 的输出
│   ├── ndt_data_full.npz       # 数据
│   ├── ndt_result_v2.npz       # 结果
│   ├── K_distribution.png
│   ├── temperature_evolution.png
│   ├── baseline_comparison.png
│   ├── ndt_results_v2.png
│   ├── overview_dashboard.png
│   └── detection_report.md
│
├── NDT_Data_V2/                 # ndt_main.py 的输出
│   └── （同上）
│
└── NDT_Comparison/              # ndt_comparison.py 的输出
    ├── trial_0/
    ├── trial_1/
    ├── trial_2/
    ├── comparison_results.npz
    ├── performance_comparison.png
    └── comparison_report.md
```

---

## 🎯 推荐工作流

### 首次使用：
```bash
# 1. 验证环境
python verify_imports.py

# 2. 快速演示
python demo.py

# 3. 查看结果
ls -lh NDT_Demo/
```

### 完整评估：
```bash
# 1. 运行完整流程
python ndt_main.py --mode all

# 2. 生成可视化
python ndt_visualizer.py --all

# 3. 性能对比（可选）
python ndt_comparison.py --n-trials 5
```

### 故障排查：
```bash
# 1. 验证导入
python verify_imports.py

# 2. 运行系统测试
python test_system.py

# 3. 检查依赖
pip install -r requirements.txt
```

---

## ⚙️ 常见问题

### Q: "ModuleNotFoundError: No module named 'ndt_data_v2'"
**A:** 确保在 `NDT_System_V2` 目录下运行：
```bash
cd NDT_System_V2
python verify_imports.py  # 验证
python demo.py           # 运行
```

### Q: "NaN detected"
**A:** 可能的原因：
- 时间步长太大（自动计算应该没问题）
- 网格太小（试试增大nx, ny）

### Q: 运行太慢
**A:** 减小参数：
```bash
python ndt_main.py --mode all --nx 40 --ny 40 --nt-save 50
```

### Q: 定位精度低
**A:** 在代码中调整 `threshold_percentile`：
```python
# ndt_defect_localization.py 中
localizer = DefectLocalizer(..., threshold_percentile=75)  # 降低阈值
```

---

## 🔧 高级用法

### 分步运行
```bash
# 只生成数据
python ndt_main.py --mode data --nx 80 --ny 80

# 只运行基准对比
python ndt_main.py --mode baseline

# 只运行检测
python ndt_main.py --mode detect
```

### 自定义参数
```bash
# 更大的网格（更精确）
python ndt_main.py --mode all --nx 100 --ny 100 --nt-save 150

# 更小的网格（更快）
python ndt_main.py --mode all --nx 50 --ny 50 --nt-save 30
```

### Python API 使用
```python
import sys
sys.path.insert(0, '/path/to/NDT_System_V2')

from ndt_data_v2 import solve_fdm_full
from ndt_defect_localization import full_pipeline

# 生成数据
x, y, t, U, K, K_raw = solve_fdm_full(nx=60, ny=60, nt_save=80)

# 运行检测
K_optimal, defect_mask = full_pipeline('path/to/data.npz')
```

---

## 📊 预期结果

### 成功指标：
- ✅ IoU > 0.85
- ✅ K误差 < 0.03
- ✅ 运行时间 < 2分钟（demo）
- ✅ 无 NaN 或错误

### 典型输出：
```
Performance Metrics
IoU: 0.892
K Error: 0.0234
True K (defect): 0.2000
Pred K (defect): 0.2234

Error Stats:
  Mean: 0.0123
  Max: 0.0456
  Std: 0.0089
```

---

## 💻 所有可用命令

```bash
# 验证
python verify_imports.py

# 演示
python demo.py

# 测试
python test_system.py

# 主程序
python ndt_main.py --mode all
python ndt_main.py --mode data
python ndt_main.py --mode baseline
python ndt_main.py --mode detect

# 对比
python ndt_comparison.py --n-trials 3

# 可视化
python ndt_visualizer.py --all
python ndt_visualizer.py --overview
python ndt_visualizer.py --temporal
python ndt_visualizer.py --report

# 帮助
python ndt_main.py --help
python ndt_comparison.py --help
python ndt_visualizer.py --help
```

---

## 🎉 开始使用

**最简单的方式：**
```bash
cd NDT_System_V2
python demo.py
```

**等待5分钟，然后查看：**
```bash
ls NDT_Demo/
cat NDT_Demo/detection_report.md
```

---

## 📚 更多信息

- **完整文档**: `README.md`
- **快速指南**: `QUICKSTART.md`
- **项目结构**: `PROJECT_STRUCTURE.md`
- **方法对比**: `METHOD_COMPARISON.md`
- **文档索引**: `INDEX.md`

---

**所有问题都已修复，可以直接运行！🚀**
