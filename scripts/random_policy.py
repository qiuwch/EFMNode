#!/usr/bin/env python3
"""
高级随机策略 - 支持多种随机模式

用法:
    # 基础随机
    python scripts/random_policy.py --env-name R1ProBlocksStackEasy-v0 --mode random
    
    # 随机游走（更平滑）
    python scripts/random_policy.py --env-name R1ProBlocksStackEasy-v0 --mode random-walk
    
    # 带噪声的 PD 控制
    python scripts/random_policy.py --env-name R1ProBlocksStackEasy-v0 --mode noisy-pd
"""

import numpy as np
from typing import Optional, Literal
from loguru import logger


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck 过程噪声（用于更自然的随机探索）"""
    
    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, 
                 sigma: float = 0.2, dt: float = 1.0):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.size) * self.mu
    
    def sample(self) -> np.ndarray:
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x)) * np.sqrt(self.dt)
        self.state = x + dx
        return self.state


class RandomPolicy:
    """
    高级随机策略
    
    支持模式:
    - random: 完全随机动作
    - random-walk: 随机游走（平滑）
    - noisy-pd: 带噪声的 PD 控制到随机目标
    - sinusoidal: 正弦波探索
    """
    
    def __init__(
        self,
        env,
        mode: Literal["random", "random-walk", "noisy-pd", "sinusoidal"] = "random-walk",
        noise_scale: float = 0.1,
        target_update_freq: int = 50,
    ):
        self.env = env
        self.unwrapped = env.unwrapped
        self.mode = mode
        self.noise_scale = noise_scale
        self.target_update_freq = target_update_freq
        
        # 获取关节信息
        self.left_arm_indices = self.unwrapped.left_arm_joint_indices
        self.right_arm_indices = self.unwrapped.right_arm_joint_indices
        self.left_gripper_indices = self.unwrapped.left_gripper_joint_indices
        self.right_gripper_indices = self.unwrapped.right_gripper_indices
        
        self.left_arm_dim = len(self.left_arm_indices)
        self.right_arm_dim = len(self.right_arm_indices)
        
        self.action_dim = self.left_arm_dim + self.right_arm_dim + 2
        
        # 初始位置
        self.init_qpos = self.unwrapped.init_qpos.copy()
        self.left_init = self.init_qpos[self.left_arm_indices]
        self.right_init = self.init_qpos[self.right_arm_indices]
        
        # 随机游走状态
        self.current_left = self.left_init.copy()
        self.current_right = self.right_init.copy()
        self.walk_step_size = 0.05
        
        # OU 噪声
        self.ou_noise = OrnsteinUhlenbeckNoise(
            size=self.action_dim,
            mu=0.0,
            theta=0.15,
            sigma=noise_scale
        )
        
        # 随机目标（用于 noisy-pd 模式）
        self.target_left = self.left_init.copy()
        self.target_right = self.right_init.copy()
        self.target_step = 0
        
        # 正弦波参数
        self.sinusoidal_phase = np.zeros(self.action_dim)
        self.sinusoidal_freq = np.random.uniform(0.5, 2.0, self.action_dim)
        self.sinusoidal_amp = np.ones(self.action_dim) * 0.1
        
        logger.info(f"RandomPolicy initialized with mode: {mode}")
        logger.info(f"Action dim: {self.action_dim}")
    
    def get_action(self, obs: dict = None) -> np.ndarray:
        """根据当前模式生成动作"""
        
        if self.mode == "random":
            return self._random_action()
        elif self.mode == "random-walk":
            return self._random_walk_action()
        elif self.mode == "noisy-pd":
            return self._noisy_pd_action()
        elif self.mode == "sinusoidal":
            return self._sinusoidal_action()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _random_action(self) -> np.ndarray:
        """完全随机动作"""
        left = self.left_init + np.random.uniform(-0.15, 0.15, self.left_arm_dim)
        right = self.right_init + np.random.uniform(-0.15, 0.15, self.right_arm_dim)
        left_grip = np.clip(np.random.uniform(0, 1), 0, 1)
        right_grip = np.clip(np.random.uniform(0, 1), 0, 1)
        
        return np.concatenate([left, [left_grip], right, [right_grip]]).astype(np.float32)
    
    def _random_walk_action(self) -> np.ndarray:
        """随机游走（更平滑的随机性）"""
        # 更新当前位置
        self.current_left += np.random.uniform(-self.walk_step_size, self.walk_step_size, self.left_arm_dim)
        self.current_right += np.random.uniform(-self.walk_step_size, self.walk_step_size, self.right_arm_dim)
        
        # 限制范围
        self.current_left = np.clip(self.current_left, self.left_init - 0.3, self.left_init + 0.3)
        self.current_right = np.clip(self.current_right, self.right_init - 0.3, self.right_init + 0.3)
        
        # 夹爪随机
        left_grip = np.clip(0.5 + np.random.uniform(-0.3, 0.3), 0, 1)
        right_grip = np.clip(0.5 + np.random.uniform(-0.3, 0.3), 0, 1)
        
        return np.concatenate([
            self.current_left, [left_grip], self.current_right, [right_grip]
        ]).astype(np.float32)
    
    def _noisy_pd_action(self) -> np.ndarray:
        """带噪声的 PD 控制到随机目标"""
        self.target_step += 1
        
        # 定期更新目标
        if self.target_step % self.target_update_freq == 0:
            self.target_left = self.left_init + np.random.uniform(-0.2, 0.2, self.left_arm_dim)
            self.target_right = self.right_init + np.random.uniform(-0.2, 0.2, self.right_arm_dim)
        
        # 获取当前状态
        qpos = self.unwrapped.robot.get_qpos()
        current_left = qpos[self.left_arm_indices]
        current_right = qpos[self.right_arm_indices]
        
        # PD 控制
        kp = 0.5
        left_action = current_left + kp * (self.target_left - current_left)
        right_action = current_right + kp * (self.target_right - current_right)
        
        # 添加 OU 噪声
        noise = self.ou_noise.sample()
        left_action += noise[:self.left_arm_dim] * self.noise_scale
        right_action += noise[self.left_arm_dim:self.left_arm_dim + self.right_arm_dim] * self.noise_scale
        
        # 夹爪
        left_grip = np.clip(0.5 + noise[self.left_arm_dim + self.right_arm_dim] * 0.5, 0, 1)
        right_grip = np.clip(0.5 + noise[-1] * 0.5, 0, 1)
        
        return np.concatenate([left_action, [left_grip], right_action, [right_grip]]).astype(np.float32)
    
    def _sinusoidal_action(self) -> np.ndarray:
        """正弦波探索"""
        t = time.time() if 'time' in globals() else 0
        
        left = self.left_init + self.sinusoidal_amp[:self.left_arm_dim] * np.sin(
            2 * np.pi * self.sinusoidal_freq[:self.left_arm_dim] * t + self.sinusoidal_phase[:self.left_arm_dim]
        )
        right = self.right_init + self.sinusoidal_amp[self.left_arm_dim:self.left_arm_dim + self.right_arm_dim] * np.sin(
            2 * np.pi * self.sinusoidal_freq[self.left_arm_dim:self.left_arm_dim + self.right_arm_dim] * t + 
            self.sinusoidal_phase[self.left_arm_dim:self.left_arm_dim + self.right_arm_dim]
        )
        
        left_grip = 0.5 + 0.3 * np.sin(2 * np.pi * 0.5 * t)
        right_grip = 0.5 + 0.3 * np.sin(2 * np.pi * 0.5 * t + np.pi)
        
        return np.concatenate([
            left, [np.clip(left_grip, 0, 1)], right, [np.clip(right_grip, 0, 1)]
        ]).astype(np.float32)
    
    def reset(self):
        """重置策略状态"""
        self.current_left = self.left_init.copy()
        self.current_right = self.right_init.copy()
        self.ou_noise.reset()
        self.target_step = 0
        logger.info("RandomPolicy reset")


def main():
    """测试随机策略"""
    import gymnasium as gym
    import galaxea_sim.envs
    
    # 创建环境
    env = gym.make("R1ProBlocksStackEasy-v0", headless=False)
    
    # 创建策略
    policy = RandomPolicy(env, mode="random-walk", noise_scale=0.2)
    
    # 运行测试
    obs, _ = env.reset()
    policy.reset()
    
    for i in range(100):
        action = policy.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            obs, _ = env.reset()
            policy.reset()
    
    env.close()
    logger.info("Test completed")


if __name__ == "__main__":
    main()
