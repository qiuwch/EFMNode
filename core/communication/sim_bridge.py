"""
仿真环境桥接层 - 替代 Ros2Bridge

将 EFMNode 连接到 GalaxeaManipSim 仿真环境，支持 pseudo random policy 测试
"""

import numpy as np
import time
import gymnasium as gym
from typing import Optional, Callable, Dict, Any
from dataclasses import asdict
from loguru import logger

# 导入 EFMNode 数据类型
from utils.message.datatype import RobotAction, ExecutionMode
from utils.message.message_convert import array_to_joint_state, array_to_pose_stamped

# 导入仿真环境
try:
    import galaxea_sim.envs
    from galaxea_sim.envs.base.bimanual_manipulation import BimanualManipulationEnv
    GALAXEA_SIM_AVAILABLE = True
except ImportError:
    GALAXEA_SIM_AVAILABLE = False
    logger.warning("GalaxeaSim not available, install with: pip install -e ../GalaxeaManipSim")


class PseudoRandomPolicy:
    """
    Pseudo Random Policy - 用于测试的随机策略
    
    特点：
    - 可复现（支持 seed）
    - 动作平滑（避免突变）
    - 关节限制内运动
    """
    
    def __init__(self, env: BimanualManipulationEnv, seed: int = 42, noise_scale: float = 0.1):
        self.env = env
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(seed)
        
        # 获取机器人配置
        self.left_arm_indices = env.left_arm_joint_indices
        self.right_arm_indices = env.right_arm_joint_indices
        self.left_gripper_indices = env.left_gripper_joint_indices
        self.right_gripper_indices = env.right_gripper_indices
        
        self.left_arm_dim = len(self.left_arm_indices)
        self.right_arm_dim = len(self.right_arm_indices)
        
        # 初始位置作为基准
        self.init_qpos = env.init_qpos.copy()
        self.left_init = self.init_qpos[self.left_arm_indices]
        self.right_init = self.init_qpos[self.right_arm_indices]
        
        # 随机游走状态
        self.current_left = self.left_init.copy()
        self.current_right = self.right_init.copy()
        self.walk_step = 0.03  # 每步最大变化
        
        # 动作范围限制
        self.left_bounds = (self.left_init - 0.3, self.left_init + 0.3)
        self.right_bounds = (self.right_init - 0.3, self.right_init + 0.3)
        
        logger.info(f"PseudoRandomPolicy initialized (seed={seed})")
        logger.info(f"Left arm DOF: {self.left_arm_dim}, Right arm DOF: {self.right_arm_dim}")
    
    def get_action(self, obs: Optional[Dict] = None) -> np.ndarray:
        """
        生成随机动作
        
        使用随机游走策略，在初始位置附近平滑探索
        """
        # 随机游走更新
        self.current_left += self.rng.uniform(-self.walk_step, self.walk_step, self.left_arm_dim)
        self.current_right += self.rng.uniform(-self.walk_step, self.walk_step, self.right_arm_dim)
        
        # 限制在范围内
        self.current_left = np.clip(self.current_left, self.left_bounds[0], self.left_bounds[1])
        self.current_right = np.clip(self.current_right, self.right_bounds[0], self.right_bounds[1])
        
        # 夹爪动作（随机开合）
        left_gripper = np.clip(self.rng.uniform(0, 1), 0, 1)
        right_gripper = np.clip(self.rng.uniform(0, 1), 0, 1)
        
        # 合并动作向量
        action = np.concatenate([
            self.current_left,
            [left_gripper],
            self.current_right,
            [right_gripper]
        ]).astype(np.float32)
        
        return action
    
    def reset(self):
        """重置策略状态"""
        self.current_left = self.left_init.copy()
        self.current_right = self.right_init.copy()
        logger.info("PseudoRandomPolicy reset")


class SimBridge:
    """
    仿真环境桥接层
    
    提供与 Ros2Bridge 相同的接口，但连接到 GalaxeaManipSim 仿真环境
    """
    
    def __init__(
        self, 
        config: Dict, 
        cfg: Dict, 
        env_name: str = "R1ProBlocksStackEasy-v0",
        use_random_policy: bool = False,
        random_seed: int = 42,
        headless: bool = False,
    ):
        if not GALAXEA_SIM_AVAILABLE:
            raise ImportError("GalaxeaSim is required for simulation mode")
        
        self.config = config
        self.cfg = cfg
        self.hardware = config["robot"]["hardware"]
        self.enable_publish = config["robot"]["enable_publish"]
        self.env_name = env_name
        self.use_random_policy = use_random_policy
        self.headless = headless
        
        # 创建仿真环境
        logger.info(f"Creating simulation environment: {env_name}")
        self.env = gym.make(env_name, headless=headless, obs_mode="image")
        self.unwrapped_env = self.env.unwrapped
        assert isinstance(self.unwrapped_env, BimanualManipulationEnv)
        
        # 创建随机策略（可选）
        if use_random_policy:
            self.policy = PseudoRandomPolicy(self.unwrapped_env, seed=random_seed)
            logger.info("Pseudo random policy enabled")
        else:
            self.policy = None
        
        # 初始化观测缓冲区
        self.obs_buffer = {}
        self.subscribers = {}
        self.publishers = {}
        self._running = True
        
        # 初始观测
        self.current_obs, self.info = self.env.reset()
        if self.policy:
            self.policy.reset()
        
        logger.info(f"SimBridge initialized successfully")
    
    def is_running(self) -> bool:
        """检查是否运行中"""
        return self._running
    
    def gather_obs(self):
        """
        获取观测数据
        
        返回格式与 Ros2Bridge 兼容:
        (obs_time, obs_dict)
        obs_dict = {"images": {...}, "state": {...}}
        """
        # 提取图像
        upper_obs = self.current_obs.get('upper_body_observations', {})
        
        images = {}
        for key in ['head_rgb', 'left_wrist_rgb', 'right_wrist_rgb']:
            if key in upper_obs:
                img = upper_obs[key]
                if isinstance(img, np.ndarray):
                    # 转换为 CHW 格式
                    if img.ndim == 3 and img.shape[-1] == 3:
                        img = img.transpose(2, 0, 1)
                    images[key] = img
        
        # 提取状态
        state = {}
        
        # 手臂关节位置
        if 'left_arm_joint_position' in upper_obs:
            state['left_arm'] = upper_obs['left_arm_joint_position']
        if 'right_arm_joint_position' in upper_obs:
            state['right_arm'] = upper_obs['right_arm_joint_position']
        
        # 夹爪位置
        if 'left_arm_gripper_position' in upper_obs:
            state['left_gripper'] = upper_obs['left_arm_gripper_position']
        if 'right_arm_gripper_position' in upper_obs:
            state['right_gripper'] = upper_obs['right_arm_gripper_position']
        
        # 末端位姿
        if 'left_arm_ee_pose' in upper_obs:
            state['left_ee_pose'] = upper_obs['left_arm_ee_pose']
        if 'right_arm_ee_pose' in upper_obs:
            state['right_ee_pose'] = upper_obs['right_arm_ee_pose']
        
        # 躯干
        lower_obs = self.current_obs.get('lower_body_observations', {})
        if 'torso_joint_position' in lower_obs:
            state['torso'] = lower_obs['torso_joint_position']
        
        obs_dict = {
            "images": images,
            "state": state,
        }
        
        return None, obs_dict
    
    def publish_action(self, action: RobotAction):
        """
        发布动作到仿真环境
        
        将 RobotAction 转换为仿真环境的动作向量并执行 step
        """
        # 从 RobotAction 提取动作
        action_list = []
        
        # 左臂
        if action.left_arm is not None:
            left_arm = np.array(action.left_arm.position)
        elif hasattr(action, 'left_arm_joints'):
            left_arm = np.array(action.left_arm_joints)
        else:
            left_arm = np.zeros(self.unwrapped_env.left_arm_joint_indices.shape[0])
        action_list.append(left_arm)
        
        # 左夹爪
        if action.left_gripper is not None:
            left_grip = np.array(action.left_gripper.position)[0:1]
        else:
            left_grip = np.array([0.5])
        action_list.append(left_grip)
        
        # 右臂
        if action.right_arm is not None:
            right_arm = np.array(action.right_arm.position)
        elif hasattr(action, 'right_arm_joints'):
            right_arm = np.array(action.right_arm_joints)
        else:
            right_arm = np.zeros(self.unwrapped_env.right_arm_joint_indices.shape[0])
        action_list.append(right_arm)
        
        # 右夹爪
        if action.right_gripper is not None:
            right_grip = np.array(action.right_gripper.position)[0:1]
        else:
            right_grip = np.array([0.5])
        action_list.append(right_grip)
        
        # 合并动作向量
        action_vec = np.concatenate(action_list).astype(np.float32)
        
        # 执行仿真步骤
        self.current_obs, reward, terminated, truncated, info = self.env.step(action_vec)
    
    def step_with_random_policy(self):
        """
        使用随机策略执行一步
        
        用于测试模式，自动调用 random policy 生成动作
        """
        if self.policy is None:
            logger.warning("Random policy not enabled")
            return
        
        # 获取当前观测
        _, obs = self.gather_obs()
        
        # 生成随机动作
        action_vec = self.policy.get_action(obs)
        
        # 转换为 RobotAction 格式
        left_dim = len(self.unwrapped_env.left_arm_joint_indices)
        right_dim = len(self.unwrapped_env.right_arm_joint_indices)
        
        action = RobotAction(
            left_arm=array_to_joint_state(action_vec[:left_dim]),
            left_gripper=array_to_joint_state(action_vec[left_dim:left_dim+1]),
            right_arm=array_to_joint_state(action_vec[left_dim+1:left_dim+1+right_dim]),
            right_gripper=array_to_joint_state(action_vec[left_dim+1+right_dim:left_dim+1+right_dim+1]),
        )
        
        # 发布动作
        self.publish_action(action)
    
    def reset(self):
        """重置仿真环境和策略"""
        self.current_obs, self.info = self.env.reset()
        if self.policy:
            self.policy.reset()
        logger.info("Simulation environment reset")
    
    def register_subscription(self, msg_type, topic: str, callback: Callable):
        """仿真模式不需要 ROS2 订阅"""
        logger.debug(f"SimBridge: ignoring subscription to {topic}")
    
    def register_publish_callback(self, frequency: float, callback: Callable):
        """仿真模式使用 step 驱动，不需要定时器"""
        logger.debug(f"SimBridge: ignoring publish callback at {frequency}Hz")
    
    def now(self) -> float:
        """获取当前时间"""
        return time.time()
    
    def destroy(self):
        """清理资源"""
        self._running = False
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()
        logger.info("SimBridge destroyed")
    
    def render(self):
        """渲染环境（如果非 headless）"""
        if not self.headless:
            self.env.render()
    
    def get_episode_info(self) -> Dict:
        """获取当前 episode 信息"""
        return self.info.copy()
