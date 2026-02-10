#!/usr/bin/env python
# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Evaluate fine-tuned VLA policies on Franka-Factory-MCXCardBlockInsert task.

Supports:
- Pi-Zero (LeRobot): tshiamor/pizero-mcx-card
- GR00T N1.5 (LeRobot): tshiamor/groot-n15-mcx-card
- OpenVLA: tshiamor/openvla-mcx-card

Usage:
    # Pi-Zero
    ./isaaclab.sh -p scripts/eval/eval_vla_policy.py --task Franka-Factory-MCXCardBlockInsert-Mimic-v0 \
        --policy pizero --model tshiamor/pizero-mcx-card --enable_cameras

    # GR00T
    ./isaaclab.sh -p scripts/eval/eval_vla_policy.py --task Franka-Factory-MCXCardBlockInsert-Mimic-v0 \
        --policy groot --model tshiamor/groot-n15-mcx-card --enable_cameras

    # OpenVLA
    ./isaaclab.sh -p scripts/eval/eval_vla_policy.py --task Franka-Factory-MCXCardBlockInsert-Mimic-v0 \
        --policy openvla --model tshiamor/openvla-mcx-card --enable_cameras
"""

import argparse
from isaaclab.app import AppLauncher

# CLI arguments
parser = argparse.ArgumentParser(description="Evaluate VLA policies on MCX Card task")
parser.add_argument("--task", type=str, required=True, help="Task name")
parser.add_argument("--policy", type=str, required=True, choices=["pizero", "groot", "openvla"])
parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
parser.add_argument(
    "--task_instruction",
    type=str,
    default="pick up the blue block and place it in the first card's closest slot",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports after simulator launch
import gymnasium as gym
import torch
import numpy as np
from PIL import Image

import franka_factory  # noqa: F401


class VLAPolicyWrapper:
    """Wrapper for VLA policies to provide unified interface."""

    def __init__(self, policy_type: str, model_id: str, device: str = "cuda"):
        self.policy_type = policy_type
        self.model_id = model_id
        self.device = device
        self.policy = None
        self.processor = None

        self._load_policy()

    def _load_policy(self):
        """Load policy from HuggingFace."""
        print(f"Loading {self.policy_type} policy from {self.model_id}...")

        if self.policy_type == "pizero":
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy
            from transformers import AutoTokenizer
            self.policy = PI0Policy.from_pretrained(self.model_id)
            self.policy.to(self.device)
            self.policy.eval()
            # Load tokenizer for language
            self.tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

        elif self.policy_type == "groot":
            from lerobot.policies.groot.modeling_groot import GrootPolicy
            from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors
            self.policy = GrootPolicy.from_pretrained(self.model_id)
            self.policy.to(self.device)
            self.policy.eval()
            # Create preprocessor for GR00T
            self.preprocessor, self.postprocessor = make_groot_pre_post_processors(self.policy.config)

        elif self.policy_type == "openvla":
            from transformers import AutoModelForVision2Seq, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.policy = AutoModelForVision2Seq.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
            self.policy.to(self.device)
            self.policy.eval()

        print(f"Policy loaded successfully!")

    def get_action(self, obs: dict, task_instruction: str = None) -> torch.Tensor:
        """Get action from policy given observation.

        Args:
            obs: Dictionary with 'wrist_rgb' (H,W,C), 'state' (proprioception)
            task_instruction: Language instruction for the task

        Returns:
            action: Tensor of shape (7,) - 6D pose + 1D gripper
        """
        if self.policy_type == "pizero":
            return self._get_action_pizero(obs, task_instruction)
        elif self.policy_type == "groot":
            return self._get_action_groot(obs, task_instruction)
        elif self.policy_type == "openvla":
            return self._get_action_openvla(obs, task_instruction)

    def _get_action_pizero(self, obs: dict, task_instruction: str) -> torch.Tensor:
        """Get action from Pi-Zero policy."""
        # Prepare image: normalize to [-1, 1] as expected by PaliGemma
        # Image should be (B, C, H, W) with values in [-1, 1]
        wrist_rgb = obs["wrist_rgb"].astype(np.float32) / 255.0  # [0, 1]
        wrist_rgb = (wrist_rgb * 2) - 1  # [-1, 1]
        wrist_tensor = torch.from_numpy(wrist_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        # Prepare state
        state_tensor = torch.from_numpy(obs["state"]).unsqueeze(0).float().to(self.device)

        # Tokenize language instruction (add newline as required by PaliGemma)
        task = task_instruction if task_instruction.endswith("\n") else f"{task_instruction}\n"
        tokenized = self.tokenizer(
            task,
            return_tensors="pt",
            padding="max_length",
            max_length=self.policy.config.tokenizer_max_length,
            truncation=True,
        )

        # Prepare observation in the format expected by Pi-Zero
        lerobot_obs = {
            "observation.images.wrist_rgb": wrist_tensor,
            "observation.state": state_tensor,
            "observation.language.tokens": tokenized["input_ids"].to(self.device),
            "observation.language.attention_mask": tokenized["attention_mask"].bool().to(self.device),
        }

        with torch.no_grad():
            action = self.policy.select_action(lerobot_obs)

        return action.squeeze(0)

    def _get_action_groot(self, obs: dict, task_instruction: str) -> torch.Tensor:
        """Get action from GR00T N1.5 policy."""
        from PIL import Image

        # Convert wrist RGB to PIL Image for preprocessor
        wrist_image = Image.fromarray(obs["wrist_rgb"].astype(np.uint8))

        # Prepare observation in the format expected by the preprocessor
        # The preprocessor expects data similar to dataset format
        raw_obs = {
            "observation.images.wrist_rgb": wrist_image,
            "observation.state": obs["state"],
            "task": task_instruction,
        }

        # Apply preprocessor to convert to GR00T format
        processed = self.preprocessor(raw_obs)

        # Move tensors to device
        for k, v in processed.items():
            if isinstance(v, torch.Tensor):
                processed[k] = v.to(self.device)

        with torch.no_grad():
            action = self.policy.select_action(processed)

        return action.squeeze(0)

    def _get_action_openvla(self, obs: dict, task_instruction: str) -> torch.Tensor:
        """Get action from OpenVLA policy."""
        # Convert to PIL Image
        image = Image.fromarray(obs["wrist_rgb"].astype(np.uint8))

        # Process inputs
        inputs = self.processor(
            images=image,
            text=task_instruction,
            return_tensors="pt",
        ).to(self.device)

        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        with torch.no_grad():
            # Generate action tokens
            generated_ids = self.policy.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

            # Decode action
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            action = self._parse_openvla_action(generated_text)

        return torch.from_numpy(action).float().to(self.device)

    def _parse_openvla_action(self, text: str, action_dim: int = 7) -> np.ndarray:
        """Parse action from OpenVLA generated text."""
        import re
        try:
            numbers = re.findall(r"[-+]?\d*\.?\d+", text)
            if len(numbers) >= action_dim:
                return np.array([float(n) for n in numbers[:action_dim]], dtype=np.float32)
            else:
                print(f"Warning: Could not parse action from: {text[:100]}")
                return np.zeros(action_dim, dtype=np.float32)
        except Exception as e:
            print(f"Error parsing action: {e}")
            return np.zeros(action_dim, dtype=np.float32)


def prepare_observation(env_obs: dict, device: str = "cuda") -> dict:
    """Convert Isaac Lab observation to VLA policy format.

    Isaac Lab obs structure:
        policy.joint_pos: (N, 9) joint positions
        policy.joint_vel: (N, 9) joint velocities
        policy.eef_pos: (N, 3) end-effector position
        policy.eef_quat: (N, 4) end-effector quaternion
        policy.gripper_pos: (N, 2) gripper finger positions
        policy.wrist_cam: (N, H, W, 3) wrist camera RGB
        policy.table_cam: (N, H, W, 3) table camera RGB

    VLA policy expects:
        wrist_rgb: (H, W, 3) uint8 RGB image
        state: (18,) proprioceptive state [joint_pos(9), joint_vel(9)] or similar
    """
    policy_obs = env_obs["policy"]

    # Get wrist camera image (first env)
    wrist_rgb = policy_obs["wrist_cam"][0].cpu().numpy()  # (H, W, 3)
    if wrist_rgb.dtype != np.uint8:
        wrist_rgb = (wrist_rgb * 255).astype(np.uint8)

    # Construct proprioceptive state
    # Concatenate: joint_pos (9) + joint_vel (9) = 18 dims
    joint_pos = policy_obs["joint_pos"][0].cpu().numpy()
    joint_vel = policy_obs["joint_vel"][0].cpu().numpy()
    state = np.concatenate([joint_pos, joint_vel]).astype(np.float32)

    return {
        "wrist_rgb": wrist_rgb,
        "state": state,
    }


def run_evaluation():
    """Main evaluation loop."""
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    # Parse environment config
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    # Create environment
    print(f"Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Load VLA policy
    vla_policy = VLAPolicyWrapper(
        policy_type=args_cli.policy,
        model_id=args_cli.model,
        device=args_cli.device,
    )

    # Run evaluation episodes
    successes = []
    rewards_all = []

    print(f"\nStarting evaluation: {args_cli.episodes} episodes, max {args_cli.max_steps} steps")
    print(f"Task: {args_cli.task_instruction}\n")

    for ep in range(args_cli.episodes):
        # Reset environment
        obs, info = env.reset()
        episode_reward = 0.0
        success = False

        for step in range(args_cli.max_steps):
            # Prepare observation for VLA policy
            vla_obs = prepare_observation(obs, args_cli.device)

            # Get action from policy
            action = vla_policy.get_action(vla_obs, args_cli.task_instruction)

            # Ensure action has correct shape for environment
            # Environment expects: arm_action (6D pose) + gripper_action (1D binary)
            if action.dim() == 1:
                action = action.unsqueeze(0)  # Add batch dimension

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward.mean().item()

            if terminated.any() or truncated.any():
                # Check success from info if available
                if "success" in info:
                    success = info["success"].any().item()
                break

        successes.append(success)
        rewards_all.append(episode_reward)
        print(f"Episode {ep + 1}/{args_cli.episodes}: reward={episode_reward:.2f}, success={success}")

    # Print summary
    success_rate = sum(successes) / len(successes) * 100
    avg_reward = np.mean(rewards_all)

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Policy: {args_cli.policy} ({args_cli.model})")
    print(f"Task: {args_cli.task}")
    print(f"Episodes: {args_cli.episodes}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"{'='*60}")

    env.close()


def main():
    """Main entry point."""
    run_evaluation()
    simulation_app.close()


if __name__ == "__main__":
    main()
