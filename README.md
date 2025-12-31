Project_Thermal_AI/
│
├── README.md                     # 项目总说明书 (包含演进故事)
├── requirements.txt              # 依赖库 (numpy, matplotlib, torch)
│
├── Phase_1_Source_Inversion/     # 第一阶段：热源反演 (基础研究)
│   ├── source_01_data_gen_2d.py  # 生成移动热源数据 (已知 k, 未知 f)
│   ├── source_02_inverse_pinn_2d.py # PINN 训练代码
│   └── source_03_eval_2d.py      # 结果评估：对比真实热源 vs 预测热源
│
└── Phase_2_NDT_Defect_Detection/ # 第二阶段：无损检测 (工程落地)
    ├── source_01_ndt_data.py     # 虚拟实验台：生成含缺陷材料的数据 (已知 f, 未知 k)
    ├── source_02_ndt_pinn.py     # 进阶 PINN：Fourier Feature + 参数辨识
    └── source_03_ndt_eval.py     # 探伤报告：对比真实缺陷 vs AI 预测缺陷

    # AI-Driven Thermal Analysis: From Source Inversion to NDT

## 💡 项目演进与设计思路 (Project Evolution)

本项目展示了基于物理信息神经网络 (PINN) 的热传导反问题研究。项目并非凭空产生，而是基于我们团队对物理问题的深度迭代与工业化升级，分为两个阶段：

### 🔄 Phase 1: 移动热源反演 (基础研究)
* **任务设定**：在已知介质导热属性 $k(x,y)$ 的情况下，通过稀疏的温度场测量，反推移动热源 $f(t,x,y)$ 的时空分布。
* **核心价值**：验证了 PINN 处理时变物理场和动态梯度的能力。
* **数学本质**：$\text{Known } k, u \rightarrow \text{Find } f$

### 🚀 Phase 2: 激光无损检测 (工业落地)
* **思维跃迁**：我们将物理方程中的“已知量”与“未知量”互换，解锁了 **无损检测 (NDT)** 场景。
* **任务设定**：模拟激光主动热像仪。已知我们控制的激光扫描轨迹 $f(t,x,y)$，通过观测表面温度 $u$，反推材料内部不可见的导热系数分布 $k(x,y)$，从而定位气孔和裂纹。
* **技术升级**：引入 **Fourier Feature Embeddings (傅里叶特征嵌入)**，解决了神经网络难以捕捉微小缺陷边界（高频信号）的问题。
* **数学本质**：$\text{Known } f, u \rightarrow \text{Find } k$ (缺陷识别)

---
Phase 2：激光扫描无损检测 (NDT) 深度解析

1. 物理背景与检测原理

主动热成像无损检测（Active Thermography NDT）是航空航天和精密制造中的关键技术。其物理过程如下：

激励输入（Excitation）：使用高能激光束在材料表面进行预设轨迹（如圆周或扫描式）的快速加热。由于激光轨迹由我们控制，因此热源项 $f(t, x, y)$ 是精确已知的。

物理响应（Response）：热量在材料内部传导。如果材料内部存在缺陷（如气泡、裂纹、脱胶），由于空气的导热系数 $k$ 远低于金属基体，热流会在缺陷处受阻。

观测数据（Observation）：红外热像仪记录表面温度场 $u(t, x, y)$ 的变化。

核心任务：PINN 模型需要通过表面的温度波动，像“X光”一样反推出材料内部不可见的导热系数分布 $k(x, y)$，从而定位缺陷。

2. 数学映射：从方程到反演

在热传导方程 $\frac{\partial u}{\partial t} - \nabla \cdot (k \nabla u) = f$ 中，Phase 2 的已知量和未知量发生了本质变化：

已知量 (Knowns)：

$f(t, x, y)$：激光热源（作为物理约束的驱动项）。

$u(t, x, y)$：带噪声的测量温度（作为数据拟合项）。

未知量 (Unknowns)：

$k(x, y)$：材料的空间分布属性（这是我们的反演目标）。

3. 技术核心：进阶版神经网络架构

为了应对 NDT 场景中缺陷边界突变（从 1.0 骤降到 0.2）带来的挑战，我们在架构上做了两大创新：

A. 傅里叶特征嵌入 (Fourier Feature Embedding)

神经网络存在“频谱偏差（Spectral Bias）”，倾向于学习平滑函数，难以捕捉尖锐的几何边缘。

方案：引入 $\gamma(x) = [\cos(2\pi Bx), \sin(2\pi Bx)]^T$ 映射层。

作用：通过将低维坐标映射到高频特征空间，强制网络学习高频信息，使反演出的缺陷边缘更锐利，避免“过度平滑”。

B. 双头解耦架构 (Decoupled Heads)

U-Head：接收 $(t, x, y)$，负责重建温度场，起到滤波和插值的作用。

K-Head：仅接收 $(x, y)$。这是一个重要的物理约束——材料属性不随时间改变。通过限制其输入维度，我们强行剥离了时间噪声对介质属性判断的干扰。

4. 损失函数：物理驱动的自监督

训练过程不再依赖标注好的 $k$（我们并没有真实的缺陷图给网络看），而是通过 PDE 残差 实现自监督：

$$Loss = \omega_1 L_{Data}(u) + \omega_2 L_{PDE}(u, k, f)$$

$L_{Data}$：确保网络预测的温度与红外测量的实验数据一致。

$L_{PDE}$：要求预测的 $k$ 和 $u$ 必须在激光源 $f$ 的驱动下满足热学定律。如果某个位置 $k$ 设置错了，产生的温度演化将无法满足物理方程，从而产生巨大的惩罚。

5. 评价指标的科学意义

在报告中，我们引入了超越 MSE 的工业级指标：

SSIM (结构相似性)：衡量 AI 反演出的缺陷“长得像不像”原件。它对形状、轮廓的识别非常敏感。

Relative L2 Error：衡量反演数值的精度。在复杂的反问题中，低于 10% 的误差通常被认为是极具应用价值的。

Cross-section Analysis (切片分析)：通过展示特定截面上的 $k$ 曲线对比，直观展示 AI 对“材质突变点”的定位精度。

总结：Phase 2 的工程价值

Phase 2 将 PINN 从一个单纯的“函数拟合器”转变为一个虚拟传感器。在实际工业中，这意味着我们可以仅凭表面红外图像，就定量地反推出内部损伤的深度、大小和形状，这正是“AI+工业”结合的典型范式。






## 🛠️ 环境配置
conda activate pinn

Phase 1: 运行热源反演


cd Phase_1_Source_Inversion

python source_01_data_gen_2d.py    # 生成数据
python source_02_inverse_pinn_2d.py # 训练模型 (反演 f)
python source_03_eval_2d.py        # 绘图评估


Phase 2: 运行无损检测 (NDT)


cd Phase_2_NDT_Defect_Detection
python source_01_ndt_data.py       # 生成含缺陷的虚拟实验数据
python source_02_ndt_pinn.py       # 训练 NDT 模型 (反演 k)
python source_03_ndt_eval.py       # 生成探伤报告