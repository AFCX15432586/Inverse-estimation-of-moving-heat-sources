# NDT System V2 - 文档索引

## 📚 文档导航

### 🚀 开始使用
1. **[快速开始 (QUICKSTART.md)](QUICKSTART.md)**
   - ⭐ **从这里开始！**
   - 一分钟快速体验
   - 常见命令速查
   - 故障排查指南

### 📖 深入了解
2. **[系统介绍 (README.md)](README.md)**
   - 核心理念和方法
   - 详细的使用指南
   - 参数调整建议
   - 理论基础

3. **[项目结构 (PROJECT_STRUCTURE.md)](PROJECT_STRUCTURE.md)**
   - 文件组织说明
   - 各模块功能详解
   - 数据格式说明
   - 输入输出规范

4. **[方法对比 (METHOD_COMPARISON.md)](METHOD_COMPARISON.md)**
   - PINN vs FEM-Guided
   - 详细性能对比
   - 为什么要重构
   - 迁移指南

---

## 🎯 根据需求选择

### 我想快速体验
→ **[QUICKSTART.md](QUICKSTART.md)**
```bash
python demo.py
```

### 我想了解这个系统做什么
→ **[README.md](README.md)** - "核心理念"部分

### 我想知道和原来的PINN有什么区别
→ **[METHOD_COMPARISON.md](METHOD_COMPARISON.md)**

### 我想深入理解代码结构
→ **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**

### 我想知道如何调参
→ **[README.md](README.md)** - "参数调整"部分

### 我遇到了问题
→ **[QUICKSTART.md](QUICKSTART.md)** - "故障排查"部分

---

## 🗂️ 代码文件

### 核心模块
- `ndt_data_v2.py` - 数据生成（FDM求解）
- `ndt_fem_solver.py` - FEM求解器
- `ndt_defect_localization.py` - 两阶段检测
- `ndt_visualizer.py` - 结果可视化

### 运行脚本
- `demo.py` - 快速演示
- `ndt_main.py` - 主程序
- `test_system.py` - 系统测试
- `ndt_comparison.py` - 性能对比

### 其他
- `requirements.txt` - Python依赖
- 本文件 - 文档索引

---

## 🎓 学习路径

### 初学者路径
1. 阅读 [QUICKSTART.md](QUICKSTART.md)
2. 运行 `python demo.py`
3. 查看生成的图片和报告
4. 阅读 [README.md](README.md) 了解原理
5. 运行完整流程 `python ndt_main.py --mode all`

### 进阶路径
1. 阅读 [METHOD_COMPARISON.md](METHOD_COMPARISON.md) 了解方法优势
2. 阅读 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) 了解代码结构
3. 运行 `python ndt_comparison.py` 评估性能
4. 修改参数进行实验
5. 根据需求定制功能

### 专家路径
1. 直接阅读源代码
2. 理解FEM实现细节
3. 优化算法和参数
4. 扩展到其他应用

---

## ⚡ 快速命令参考

```bash
# 测试系统
python test_system.py

# 快速演示（5分钟）
python demo.py

# 完整流程（15分钟）
python ndt_main.py --mode all --nx 80 --ny 80

# 性能对比（30分钟）
python ndt_comparison.py --n-trials 5

# 结果可视化
python ndt_visualizer.py --all --data-dir NDT_Data_V2
```

---

## 📊 关键信息

### 系统特点
- ✅ FEM先导，物理驱动
- ✅ 两阶段检测（定位+量化）
- ✅ 5-8倍速度提升
- ✅ 30-40%精度提升
- ✅ 高鲁棒性

### 典型性能
- **定位精度 (IoU)**: 0.85-0.95
- **K值误差**: 0.01-0.03
- **运行时间**: 1-2分钟（演示）/ 10-15分钟（完整）
- **稳定性**: 方差 < 0.02

### 主要改进
1. 完整时空场数据（vs随机采样）
2. FEM求解基准（vs纯PINN）
3. 梯度异常定位（vs同时优化）
4. 局部K值优化（vs全局优化）

---

## 🆘 需要帮助？

1. **快速问题** → 查看 [QUICKSTART.md](QUICKSTART.md) 的"故障排查"
2. **方法理解** → 阅读 [README.md](README.md)
3. **代码问题** → 查看 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
4. **性能对比** → 阅读 [METHOD_COMPARISON.md](METHOD_COMPARISON.md)
5. **运行测试** → `python test_system.py`

---

## 🎉 开始探索

**推荐起点**: [QUICKSTART.md](QUICKSTART.md)

只需一条命令就能看到效果：
```bash
python demo.py
```

祝使用愉快！🔬✨

---

*最后更新: 2024*  
*NDT System V2 - FEM-Guided Two-Stage Defect Detection*
