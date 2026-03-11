# EFMNode 仿真模式使用指南

## 概述

EFMNode 现在支持接入 **GalaxeaManipSim** 仿真环境，无需真机即可测试完整流程。

### 架构

```
┌─────────────────────────────────────────────────────────┐
│                    EFMNode                               │
│  ┌─────────────┐         ┌─────────────────────────┐    │
│  │ VLA 推理     │         │   Scheduler             │    │
│  │ (可选)      │         │   - 轨迹管理            │    │
│  └─────────────┘         │   - 指令管理            │    │
│                          └───────────┬─────────────┘    │
│                                      │                   │
│                          ┌───────────▼─────────────┐    │
│                          │   Bridge 接口           │    │
│                          │   - Ros2Bridge (真机)   │    │
│                          │   - SimBridge (仿真)    │    │
│                          └───────────┬─────────────┘    │
└──────────────────────────────────────│──────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                   │
              ┌─────▼─────┐    ┌───────▼────────┐         │
              │ ROS2      │    │ GalaxeaManipSim│         │
              │ 真机硬件   │    │ (Sapien 物理)   │         │
              └───────────┘    └────────────────┘         │
```

---

## 快速开始

### 1. 安装依赖

```bash
cd ~/efmnode

# 安装 GalaxeaManipSim
pip install -e ../GalaxeaManipSim

# 下载仿真资产（如果还没下载）
cd ../GalaxeaManipSim
gdown https://drive.google.com/file/d/1ZvtCv1H4FLrse_ElUWzsVDt8xRK4CyaC/
unzip robotwin_models.zip
mv robotwin_models galaxea_sim/assets/
```

### 2. 运行仿真测试

```bash
# 使用 pseudo random policy 测试（推荐）
python scripts/run_sim.py --random-policy --render

# 指定环境
python scripts/run_sim.py \
    --env R1ProBlocksStackEasy-v0 \
    --random-policy \
    --steps 1000

# 无头模式（适合服务器）
python scripts/run_sim.py --random-policy --steps 500
```

---

## 配置选项

### config.toml

```toml
[simulation]
enabled = false                    # 启用仿真模式
env_name = "R1ProBlocksStackEasy-v0"  # 环境名称
use_random_policy = true           # 使用随机策略测试
random_seed = 42                   # 随机种子
headless = true                    # 无头模式
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--env` | Gym 环境名称 | `R1ProBlocksStackEasy-v0` |
| `--random-policy` | 使用 pseudo random policy | `False` |
| `--seed` | 随机种子 | `42` |
| `--steps` | 运行步数 | `500` |
| `--render` | 显示可视化窗口 | `False` |
| `--freq` | 控制频率 (Hz) | `15.0` |

---

## 支持的环境

### R1 Pro

| 环境 | 任务 |
|---|---|
| `R1ProBlocksStackEasy-v0` | 堆方块（简单） |
| `R1ProBlocksStackHard-v0` | 堆方块（困难） |
| `R1ProDualBottlesPickEasy-v0` | 抓双瓶子 |
| `R1ProBlocksStackEasy-v0` | 堆方块 |

### R1

| 环境 | 任务 |
|---|---|
| `R1DualBottlesPickEasy-v0` | 抓双瓶子 |
| `R1BlocksStackEasy-v0` | 堆方块 |
| `R1MugHangingEasy-v0` | 挂杯子 |
| `R1DualShoesPlace-v0` | 放鞋子 |

### R1 Lite

| 环境 | 任务 |
|---|---|
| `R1LiteBlocksStackEasy-v0` | 堆方块 |
| `R1LiteDualBottlesPickEasy-v0` | 抓双瓶子 |

---

## Pseudo Random Policy

### 特性

- **可复现**: 固定 seed 保证结果可重复
- **平滑探索**: 随机游走避免动作突变
- **关节限制**: 在安全范围内运动
- **夹爪控制**: 随机开合

### 动作范围

```
左臂：初始位置 ±0.3 rad
右臂：初始位置 ±0.3 rad
夹爪：0 ~ 1 (全关 ~ 全开)
```

### 代码示例

```python
from core.communication.sim_bridge import SimBridge, PseudoRandomPolicy
import gymnasium as gym
import galaxea_sim.envs

# 创建环境
env = gym.make("R1ProBlocksStackEasy-v0", headless=False)

# 创建随机策略
policy = PseudoRandomPolicy(env.unwrapped, seed=42, noise_scale=0.1)

# 运行
obs, _ = env.reset()
for i in range(200):
    action = policy.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
```

---

## 接入你自己的策略

### 方式 1: 修改 run_sim.py

```python
# 在 run_sim.py 的 run_simulation 函数中
while step_count < num_steps:
    obs_time, obs = bridge.gather_obs()
    
    if obs is not None:
        # 你的策略
        action = your_policy(obs)
        bridge.publish_action(action)
```

### 方式 2: 使用 Scheduler

```python
# 继承 Scheduler 并覆盖 inference 方法
class MyScheduler(Scheduler):
    def inference(self, obs):
        # 你的推理逻辑
        actions = my_model.predict(obs)
        return actions
```

### 方式 3: 通过配置启用 VLA

```toml
[simulation]
enabled = true
use_random_policy = false  # 关闭随机策略

[model]
ckpt_dir = "/path/to/your/vla/model"
```

---

## 数据记录

### 录制仿真数据

```bash
# 修改 run_sim.py 添加数据保存
import h5py

# 在循环中保存
with h5py.File('demo.h5', 'w') as f:
    f.create_dataset('observations', data=obs_buffer)
    f.create_dataset('actions', data=action_buffer)
```

### 转换为 LeRobot 格式

```bash
python -m galaxea_sim.scripts.convert_single_galaxea_sim_to_lerobot \
    --task R1ProBlocksStackEasy-v0 \
    --tag random_policy \
    --robot r1_pro
```

---

## 故障排除

### 问题：`ModuleNotFoundError: No module named 'galaxea_sim'`

```bash
# 确保已安装 GalaxeaManipSim
cd ~/GalaxeaManipSim
pip install -e .
```

### 问题：`Asset not found`

```bash
# 下载仿真资产
cd ~/GalaxeaManipSim
gdown https://drive.google.com/file/d/1ZvtCv1H4FLrse_ElUWzsVDt8xRK4CyaC/
unzip robotwin_models.zip
mv robotwin_models galaxea_sim/assets/
```

### 问题：显示窗口失败

```bash
# 确保有 DISPLAY 环境变量
export DISPLAY=:0

# 或使用无头模式
python scripts/run_sim.py --random-policy --steps 500
```

### 问题：动作维度不匹配

检查机器人配置：
- **R1/R1 Lite**: 6 DOF 手臂 → 动作维度 14
- **R1 Pro**: 7 DOF 手臂 → 动作维度 16

---

## 下一步

1. **测试随机策略**: `python scripts/run_sim.py --random-policy --render`
2. **接入 VLA 模型**: 配置 `model.ckpt_dir` 并关闭 `use_random_policy`
3. **收集演示数据**: 使用随机策略或专家策略生成训练数据
4. **训练行为克隆**: 用收集的数据训练策略网络

---

## 参考

- [GalaxeaManipSim README](../GalaxeaManipSim/README.md)
- [EFMNode README](README.md)
- [随机策略数据收集](RANDOM_POLICY_GUIDE.md)
