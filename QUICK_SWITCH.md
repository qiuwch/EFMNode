# 一键切换真机/仿真模式

## 快速切换

### 方式 1: 修改 config.toml (推荐)

```toml
[robot]
run_mode = "sim"   # 仿真模式
# run_mode = "real"  # 真机模式
```

然后运行：
```bash
python scripts/run_sim.py
```

### 方式 2: 命令行参数（临时覆盖）

```bash
# 仿真模式（忽略 config.toml）
python scripts/run_sim.py --mode sim --random-policy --render

# 真机模式（忽略 config.toml）
python scripts/run_sim.py --mode real
```

---

## 配置对比

| 配置项 | 仿真模式 | 真机模式 |
|---|---|---|
| `run_mode` | `sim` | `real` |
| 桥接层 | SimBridge | Ros2Bridge |
| 数据源 | GalaxeaManipSim | ROS2 Topics |
| 需要 ROS2 | ❌ | ✅ |
| 需要 GPU | ❌ (可选) | ✅ (VLA 推理) |

---

## 完整示例

### 仿真模式测试

```bash
# 基础测试（随机策略）
python scripts/run_sim.py --mode sim --random-policy

# 带可视化
python scripts/run_sim.py --mode sim --random-policy --render

# 指定环境和步数
python scripts/run_sim.py \
    --mode sim \
    --env R1ProBlocksStackEasy-v0 \
    --steps 1000 \
    --random-policy

# 指定随机种子（可复现）
python scripts/run_sim.py --mode sim --random-policy --seed 123
```

### 真机模式运行

```bash
# 基础运行（需要 ROS2）
python scripts/run_sim.py --mode real

# 指定模型路径
python scripts/run_sim.py \
    --mode real \
    --model-path /path/to/vla/model

# 指定控制频率
python scripts/run_sim.py --mode real --freq 15
```

---

## config.toml 完整配置

```toml
# ============================================
# 运行模式 (一键切换)
# ============================================
[robot]
run_mode = "sim"           # "sim" 或 "real"
hardware = "R1_PRO"

enable_publish = [
    "left_gripper",
    "right_gripper",
    "right_ee_pose",
    "torso",
]

# ============================================
# 仿真模式配置 (run_mode = "sim" 时使用)
# ============================================
[simulation]
enabled = true                      # 自动根据 run_mode 设置
env_name = "R1ProBlocksStackEasy-v0"
use_random_policy = true            # 测试用随机策略
random_seed = 42                    # 可复现
headless = true                     # 无头模式

# ============================================
# 基础配置
# ============================================
[basic]
use_ehi = false
control_frequency = 15.0
step_mode = "async"
action_steps = 32

# ============================================
# 模型配置
# ============================================
[model]
ckpt_dir = "/path/to/model"
processor = "default"
use_trt = false
is_torch_compile = false

# ============================================
# 指令配置
# ============================================
[instruction]
use_vlm = false
bbox_as_instruction = false
image_condition_lang_prefix = false
pp_lower_half = false
image_as_condition = false
```

---

## 代码中的模式判断

```python
# scheduler/scheduler.py 中自动处理
def _setup_ros2_bridge(self):
    run_mode = self.schedule_config.get('robot', {}).get('run_mode', 'real')
    
    if run_mode == "sim":
        # 使用 SimBridge (仿真)
        self.ros2_bridge = SimBridge(...)
    else:
        # 使用 Ros2Bridge (真机)
        self.ros2_bridge = Ros2Bridge(...)
```

---

## 常见场景

### 场景 1: 本地开发测试

```toml
[robot]
run_mode = "sim"

[simulation]
use_random_policy = true
headless = false  # 显示窗口
```

```bash
python scripts/run_sim.py --render
```

### 场景 2: 服务器批量测试

```toml
[robot]
run_mode = "sim"

[simulation]
use_random_policy = true
headless = true  # 无头模式
```

```bash
python scripts/run_sim.py --steps 1000
```

### 场景 3: 真机部署

```toml
[robot]
run_mode = "real"

[model]
ckpt_dir = "/path/to/trained/model"
```

```bash
python scripts/run_sim.py
```

### 场景 4: 快速对比实验

```bash
# 仿真测试
python scripts/run_sim.py --mode sim --random-policy --steps 500 --seed 42

# 真机测试（相同种子）
python scripts/run_sim.py --mode real --steps 500
```

---

## 注意事项

### 仿真模式
- ✅ 无需 ROS2
- ✅ 无需真机硬件
- ✅ 可复现（固定 seed）
- ⚠️ 需要安装 GalaxeaManipSim

### 真机模式
- ✅ 真实机器人控制
- ✅ 完整 ROS2 功能
- ⚠️ 需要 ROS2 环境
- ⚠️ 需要真机硬件

---

## 故障排除

### 问题：切换模式后报错

```bash
# 检查配置
cat config.toml | grep run_mode

# 清除缓存
rm -rf __pycache__ .mypy_cache

# 重新启动
python scripts/run_sim.py --mode sim
```

### 问题：仿真模式找不到 GalaxeaManipSim

```bash
# 确保已安装
pip install -e ../GalaxeaManipSim

# 验证安装
python -c "import galaxea_sim; print(galaxea_sim.__file__)"
```

### 问题：真机模式 ROS2 连接失败

```bash
# 检查 ROS2
ros2 topic list

# 检查网络
ping <robot-ip>

# 重新 sourcing
source /opt/ros/humble/setup.bash
```

---

## 相关文件

- `config.toml` - 主配置文件
- `scheduler/scheduler.py` - 模式切换逻辑
- `scripts/run_sim.py` - 运行脚本
- `core/communication/sim_bridge.py` - 仿真桥接
- `core/communication/ros2_bridge.py` - 真机桥接
