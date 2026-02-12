# Franka Factory Tasks

Factory manipulation tasks for the Franka Panda robot using IsaacLab.

## Overview

This extension provides precision manipulation tasks including:
- **Peg Insertion**: Insert a peg into a hole
- **Gear Mesh**: Place a gear onto a shaft and mesh with adjacent gears
- **Nut Threading**: Thread a nut onto a bolt

## Installation

1. Make sure IsaacLab is installed and working
2. Install this extension:

```bash
cd /home/tshiamo/SIMULATION_MANIPULATION/franka_factory
python -m pip install -e source/franka_factory
```

## Available Environments

| Environment ID | Description |
|---------------|-------------|
| `Franka-Factory-PegInsert-Direct-v0` | Peg insertion task |
| `Franka-Factory-GearMesh-Direct-v0` | Gear meshing task |
| `Franka-Factory-NutThread-Direct-v0` | Nut threading task |

## Usage

### List Environments

```bash
cd /home/tshiamo/IsaacLab
./isaaclab.sh -p /home/tshiamo/SIMULATION_MANIPULATION/franka_factory/scripts/list_envs.py
```

### Run with Random Actions

```bash
./isaaclab.sh -p scripts/environments/random_agent.py --task Franka-Factory-PegInsert-Direct-v0 --num_envs 4
```

### Train with RL Games

```bash
./isaaclab.sh -p /home/tshiamo/SIMULATION_MANIPULATION/franka_factory/scripts/rl_games/train.py --task Franka-Factory-PegInsert-Direct-v0 --headless
```

### Play Trained Model

```bash
./isaaclab.sh -p /home/tshiamo/SIMULATION_MANIPULATION/franka_factory/scripts/rl_games/play.py --task Franka-Factory-PegInsert-Direct-v0 --checkpoint <path_to_checkpoint> --num_envs 4
```

## Project Structure

```
franka_factory/
├── README.md
├── scripts/
│   ├── list_envs.py
│   └── rl_games/
│       ├── train.py
│       └── play.py
└── source/
    └── franka_factory/
        ├── config/
        │   └── extension.toml
        ├── pyproject.toml
        ├── setup.py
        └── franka_factory/
            ├── __init__.py
            └── tasks/
                └── direct/
                    └── factory/
                        ├── __init__.py
                        ├── factory_env.py
                        ├── factory_env_cfg.py
                        ├── factory_control.py
                        ├── factory_tasks_cfg.py
                        ├── factory_utils.py
                        └── agents/
                            └── rl_games_ppo_cfg.yaml
```

## Robot Configuration

- **Robot**: Franka Panda (7 DoF arm + 2 DoF gripper)
- **Control**: Operational space control with torque commands
- **Gripper**: Parallel fingers (mimic joint)

## Training Results

Typical training results after 200 epochs:
- Best reward: ~370
- Training time: ~1.5 hours (GPU dependent)

---

## Imitation Learning Tasks

This extension also provides teleoperation environments for imitation learning
data collection using CloudXR/Vision Pro or other input devices.

### Available Teleoperation Environments

| Environment ID | Description |
|---------------|-------------|
| `Franka-Factory-PegInsert-Teleop-v0` | Peg insertion with dual cameras |
| `Franka-Factory-GearMesh-Teleop-v0` | Gear meshing with dual cameras |
| `Franka-Factory-NutThread-Teleop-v0` | Nut threading with dual cameras |
| `Franka-Factory-MCXCardBlockInsert-Teleop-v0` | Insert block into MCX card slot |

### Camera Setup

The teleoperation environments include dual cameras for visuomotor learning:
- **wrist_cam**: Mounted on the end-effector (panda_hand) for close-up manipulation view
- **table_cam**: Fixed overhead/side view for context awareness

### Teleoperation (Practice)

Test the environment with keyboard control before recording:

```bash
cd /home/tshiamo/IsaacLab
./isaaclab.sh -p /home/tshiamo/SIMULATION_MANIPULATION/franka_factory/scripts/teleop/teleop_agent.py \
    --task Franka-Factory-PegInsert-Teleop-v0
```

With CloudXR/Vision Pro hand tracking:

```bash
./isaaclab.sh -p /home/tshiamo/SIMULATION_MANIPULATION/franka_factory/scripts/teleop/teleop_agent.py \
    --task Franka-Factory-PegInsert-Teleop-v0 \
    --teleop_device handtracking
```

### Recording Demonstrations

Record demonstrations for imitation learning:

```bash
# With keyboard
./isaaclab.sh -p /home/tshiamo/SIMULATION_MANIPULATION/franka_factory/scripts/teleop/record_demos.py \
    --task Franka-Factory-PegInsert-Teleop-v0 \
    --dataset_file ./datasets/franka_peg_demos.hdf5

# With CloudXR/Vision Pro
./isaaclab.sh -p /home/tshiamo/SIMULATION_MANIPULATION/franka_factory/scripts/teleop/record_demos.py \
    --task Franka-Factory-PegInsert-Teleop-v0 \
    --teleop_device handtracking \
    --dataset_file ./datasets/franka_peg_demos.hdf5

# Record specific number of demos
./isaaclab.sh -p /home/tshiamo/SIMULATION_MANIPULATION/franka_factory/scripts/teleop/record_demos.py \
    --task Franka-Factory-PegInsert-Teleop-v0 \
    --num_demos 50 \
    --dataset_file ./datasets/franka_peg_demos.hdf5
```

### Keyboard Controls

When using keyboard teleoperation:
- **WASD**: Move end-effector in X/Y plane
- **Q/E**: Move end-effector up/down (Z axis)
- **Arrow keys**: Rotate end-effector
- **Space**: Toggle gripper open/close
- **R**: Reset environment

### CloudXR/Vision Pro Controls

When using hand tracking:
- **Pinch gesture**: Start/stop recording
- Hand position controls end-effector position
- Hand orientation controls end-effector rotation

### Project Structure (with Imitation Learning)

```
franka_factory/
├── README.md
├── scripts/
│   ├── list_envs.py
│   ├── rl_games/
│   │   ├── train.py
│   │   └── play.py
│   └── teleop/
│       ├── record_demos.py          # Record demonstrations via teleoperation
│       ├── replay_demos.py          # Interactive action-based replay
│       ├── replay_demos_resetmethod.py  # State-based replay for video generation
│       ├── replay_demos_statebased.py   # Alternative state-based replay
│       └── teleop_agent.py          # Teleoperation testing (no recording)
└── source/
    └── franka_factory/
        └── franka_factory/
            └── tasks/
                ├── direct/
                │   └── factory/
                └── imitation/
                    ├── __init__.py
                    ├── factory_teleop_env_cfg.py
                    ├── agents/
                    └── mdp/
                        ├── observations.py
                        ├── events.py
                        └── terminations.py
```

## MCX Card Block Insert Task

The `Franka-Factory-MCXCardBlockInsert-Teleop-v0` task features:
- **Blue block**: 10cm × 2cm × 1cm cuboid (pickup object)
- **MCX Cards**: 5 network cards arranged in a line along Y axis, spaced 5cm apart
- **Target marker**: Green semi-transparent cube showing where to place the block
- **Target platform**: Gray platform that catches the block when dropped
- **Success condition**: Block placed on target platform (within 4cm X, 3cm Y, 3cm Z tolerance)

### Scene Layout

| Element | Position (X, Y, Z) | Description |
|---------|-------------------|-------------|
| Card 1 | [0.5, 0.15, 0.13] | First MCX card |
| Card 2 | [0.5, 0.20, 0.13] | Second MCX card |
| Card 3 | [0.5, 0.25, 0.13] | Third MCX card |
| Card 4 | [0.5, 0.30, 0.13] | Fourth MCX card |
| Card 5 | [0.5, 0.35, 0.13] | Fifth MCX card |
| Target Marker | [0.45, 0.15, 0.13] | Green cube (visual guide) |
| Target Platform | [0.45, 0.15, 0.07] | Gray platform (catches block) |

### Running MCX Card Task

```bash
# Teleop with keyboard
./isaaclab.sh -p scripts/teleop/teleop_agent.py --task Franka-Factory-MCXCardBlockInsert-Teleop-v0

# Teleop with CloudXR/Vision Pro
./isaaclab.sh -p scripts/teleop/teleop_agent.py --task Franka-Factory-MCXCardBlockInsert-Teleop-v0 --teleop_device handtracking

# Record demonstrations (15 demos)
./isaaclab.sh -p scripts/teleop/record_demos.py --task Franka-Factory-MCXCardBlockInsert-Teleop-v0 --num_demos 15 --dataset_file ./demos/mcx_card_demos.hdf5
```

### Recording Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_demos` | 0 (unlimited) | Number of demos to record |
| `--num_success_steps` | 5 | Steps block must stay in target zone |
| `--step_hz` | 30 | Environment stepping rate |
| `--dataset_file` | `./datasets/franka_factory_demos.hdf5` | Output file path |

### Replaying Demonstrations

After recording, you can replay demonstrations to verify quality. There are multiple replay methods available:

#### Interactive Replay (Action-Based)

```bash
# Replay all demos with visual feedback
./isaaclab.sh -p scripts/teleop/replay_demos.py \
    --task Franka-Factory-MCXCardBlockInsert-Teleop-v0 \
    --dataset_file ./demos/mcx_card_demos.hdf5

# Replay specific episode (e.g., episode 5)
./isaaclab.sh -p scripts/teleop/replay_demos.py \
    --task Franka-Factory-MCXCardBlockInsert-Teleop-v0 \
    --dataset_file ./demos/mcx_card_demos.hdf5 \
    --episode 5

# Replay at half speed (easier to watch)
./isaaclab.sh -p scripts/teleop/replay_demos.py \
    --task Franka-Factory-MCXCardBlockInsert-Teleop-v0 \
    --dataset_file ./demos/mcx_card_demos.hdf5 \
    --speed 0.5
```

#### State-Based Replay (Recommended for Video Generation)

For accurate visual replay where objects appear in their exact recorded positions, use the
state-based replay script with `scene.reset_to()`:

```bash
# Generate video of episode 0 using state-based replay
./isaaclab.sh -p scripts/teleop/replay_demos_resetmethod.py \
    --task Franka-Factory-MCXCardBlockInsert-Teleop-v0 \
    --dataset_file ./demos/mcx_card_demos.hdf5 \
    --episode 0 \
    --output_dir ./videos \
    --camera table_cam \
    --headless

# Generate videos for ALL episodes
./isaaclab.sh -p scripts/teleop/replay_demos_resetmethod.py \
    --task Franka-Factory-MCXCardBlockInsert-Teleop-v0 \
    --dataset_file ./demos/mcx_card_demos.hdf5 \
    --all \
    --output_dir ./videos \
    --camera table_cam \
    --headless

# Generate video with wrist camera view
./isaaclab.sh -p scripts/teleop/replay_demos_resetmethod.py \
    --task Franka-Factory-MCXCardBlockInsert-Teleop-v0 \
    --dataset_file ./demos/mcx_card_demos.hdf5 \
    --episode 0 \
    --output_dir ./videos \
    --camera wrist_cam \
    --headless
```

**Important:** The `replay_demos_resetmethod.py` script uses `scene.reset_to()` which directly
sets robot joint states and object poses from the HDF5 data. This ensures manipulated objects
(like the blue block) appear in their exact recorded positions, which is essential for
generating accurate training videos for visuomotor policies.

#### Replay Script Comparison

| Script | Method | Use Case |
|--------|--------|----------|
| `replay_demos.py` | Action replay | Interactive viewing, debugging |
| `replay_demos_resetmethod.py` | State replay via `reset_to()` | Video generation, visuomotor training data |
| `replay_demos_statebased.py` | Direct state setting | Alternative state-based replay |

### Quick Dataset Inspection (No Simulation)

Inspect the HDF5 dataset without launching the simulator:

```bash
python -c "
import h5py
with h5py.File('./demos/mcx_card_demos.hdf5', 'r') as f:
    print('Episodes:', list(f['data'].keys()))
    for ep in list(f['data'].keys())[:5]:
        print(f'  {ep}: {len(f[\"data\"][ep][\"actions\"])} steps')
"
```

### HDF5 Data Structure

The recorded demonstrations contain:

```
data/
├── demo_0/
│   ├── actions                    # (N, 7) - Recorded actions
│   ├── obs/
│   │   ├── actions               # (N, 7) - Action observations
│   │   ├── eef_pos               # (N, 3) - End-effector position
│   │   ├── eef_quat              # (N, 4) - End-effector orientation
│   │   ├── gripper_pos           # (N, 2) - Gripper finger positions
│   │   ├── joint_pos             # (N, 9) - Robot joint positions
│   │   └── joint_vel             # (N, 9) - Robot joint velocities
│   └── states/
│       ├── articulation/
│       │   └── robot/
│       │       ├── joint_position    # (N, 9) - Robot joint states
│       │       ├── joint_velocity    # (N, 9) - Robot joint velocities
│       │       ├── root_pose         # (N, 7) - Robot base pose
│       │       └── root_velocity     # (N, 6) - Robot base velocity
│       └── rigid_object/
│           └── block/
│               ├── root_pose         # (N, 7) - Block pose (x,y,z,qw,qx,qy,qz)
│               └── root_velocity     # (N, 6) - Block velocity
├── demo_1/
│   └── ...
```

### Video Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--episode` | 0 | Episode index to replay (ignored if `--all` is set) |
| `--all` | False | Replay all episodes in the dataset |
| `--output_dir` | `./videos` | Directory for output videos |
| `--fps` | 30 | Video frame rate |
| `--camera` | `table_cam` | Camera to use (`table_cam` or `wrist_cam`) |
| `--headless` | False | Run without GUI (faster) |

---

## VLA Policy Evaluation

Evaluate fine-tuned Vision-Language-Action (VLA) policies on the MCX Card Block Insert task.

### Supported Models

| Policy | Model ID | Description |
|--------|----------|-------------|
| Pi-Zero | `tshiamor/pizero-mcx-card` | LeRobot Pi-Zero (3.5B params) fine-tuned on MCX card demos |
| GR00T N1.5 | `tshiamor/groot-n15-mcx-card` | NVIDIA GR00T N1.5-3B fine-tuned on MCX card demos |
| GR00T N1.6 | `~/groot_data/finetune_output_n16` | NVIDIA GR00T N1.6-3B locally fine-tuned on MCX card demos |
| OpenVLA | `tshiamor/openvla-mcx-card` | OpenVLA fine-tuned on MCX card demos |

### Prerequisites

#### 1. Clone required repositories

```bash
# LeRobot (required for Pi-Zero and GR00T N1.5)
git clone https://github.com/huggingface/lerobot.git ~/lerobot

# Isaac-GR00T (required for GR00T N1.6 fine-tuning and inference)
git clone https://github.com/NVIDIA/Isaac-GR00T.git ~/Isaac-GR00T
```

#### 2. Install dependencies in IsaacLab environment

```bash
conda activate isaaclab

# Install LeRobot (for Pi-Zero, GR00T N1.5)
pip install -e ~/lerobot

# Install Isaac-GR00T (for GR00T N1.6)
cd ~/Isaac-GR00T && pip install -e .

# For OpenVLA: Install timm (compatible version)
pip install "timm>=0.9.10,<1.0.0"

# For Flash Attention (recommended for speed)
pip install flash-attn --no-build-isolation

# Fix NumPy compatibility for pinocchio
pip install "numpy<2"
```

#### 3. Transformers version (per model)

Different models require different `transformers` versions. The batch eval script
(`run_all_evals.sh`) handles this automatically. For manual runs:

```bash
# Pi-Zero / GR00T N1.6: transformers 4.51.3
pip install transformers==4.51.3

# GR00T N1.5: transformers 4.57.1
pip install transformers==4.57.1

# OpenVLA: transformers 4.45.0
pip install transformers==4.45.0
```

#### 4. GR00T N1.6 local fine-tuning (optional)

To fine-tune GR00T N1.6 on your own data, see the training pipeline:
```bash
bash scripts/data_pipeline/brev_train_groot_n16.sh
```
The fine-tuned model will be saved to `~/groot_data/finetune_output_n16/`.

### Running Evaluation

```bash
cd ~/SIMULATION_MANIPULATION/franka_factory

# Pi-Zero evaluation
python scripts/eval/eval_vla_policy.py \
    --task Franka-Factory-MCXCardBlockInsert-Mimic-v0 \
    --policy pizero \
    --model tshiamor/pizero-mcx-card \
    --episodes 10 \
    --max_steps 100 \
    --enable_cameras

# GR00T N1.5 evaluation
python scripts/eval/eval_vla_policy.py \
    --task Franka-Factory-MCXCardBlockInsert-Mimic-v0 \
    --policy groot \
    --model tshiamor/groot-n15-mcx-card \
    --episodes 10 \
    --max_steps 100 \
    --enable_cameras

# GR00T N1.6 evaluation (locally fine-tuned)
python scripts/eval/eval_vla_policy.py \
    --task Franka-Factory-MCXCardBlockInsert-Mimic-v0 \
    --policy groot_n16 \
    --model ~/groot_data/finetune_output_n16 \
    --episodes 10 \
    --max_steps 100 \
    --enable_cameras

# OpenVLA evaluation
python scripts/eval/eval_vla_policy.py \
    --task Franka-Factory-MCXCardBlockInsert-Mimic-v0 \
    --policy openvla \
    --model tshiamor/openvla-mcx-card \
    --episodes 10 \
    --max_steps 100 \
    --enable_cameras

# Headless mode (faster, no GUI)
python scripts/eval/eval_vla_policy.py \
    --task Franka-Factory-MCXCardBlockInsert-Mimic-v0 \
    --policy pizero \
    --model tshiamor/pizero-mcx-card \
    --episodes 10 \
    --enable_cameras \
    --headless

# Run ALL models in sequence (batch)
bash scripts/eval/run_all_evals.sh --episodes 10 --max_steps 2400
```

### Evaluation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--task` | Required | Task environment name |
| `--policy` | Required | Policy type: `pizero`, `groot`, `groot_n16`, or `openvla` |
| `--model` | Required | HuggingFace model ID or local path |
| `--num_envs` | 1 | Number of parallel environments |
| `--episodes` | 10 | Number of evaluation episodes |
| `--max_steps` | 500 | Maximum steps per episode |
| `--task_instruction` | "pick up the blue block..." | Language instruction for the task |
| `--enable_cameras` | False | Enable camera sensors (required for VLA) |
| `--headless` | False | Run without GUI |

### Observation Format

The evaluation script converts Isaac Lab observations to VLA policy format:

**Isaac Lab Observations:**
- `policy.joint_pos`: (N, 9) joint positions
- `policy.joint_vel`: (N, 9) joint velocities
- `policy.eef_pos`: (N, 3) end-effector position
- `policy.eef_quat`: (N, 4) end-effector quaternion
- `policy.gripper_pos`: (N, 2) gripper finger positions
- `policy.wrist_cam`: (N, 200, 200, 3) wrist camera RGB
- `policy.table_cam`: (N, 200, 200, 3) table camera RGB

**VLA Policy Input:**
- `wrist_rgb`: (224, 224, 3) resized wrist camera image
- `state`: (18,) proprioceptive state [joint_pos(9), joint_vel(9)]

### Action Format

VLA policies output 7D actions:
- **arm_action**: 6D relative pose (position + rotation)
- **gripper_action**: 1D binary gripper command

### Training VLA Models

See `scripts/data_pipeline/` for training scripts:

```bash
# Train Pi-Zero on cloud GPU (requires HF_TOKEN)
export HF_TOKEN="your_huggingface_token"
bash scripts/data_pipeline/brev_train_pizero.sh

# Train GR00T N1.5 on cloud GPU
bash scripts/data_pipeline/brev_train_groot.sh
```

### Troubleshooting

**NumPy version conflict:**
```
ImportError: A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```
Fix: `pip install "numpy<2"`

**Flash Attention not found:**
```
ImportError: FlashAttention2 has been toggled on, but it cannot be used
```
Fix: `pip install flash-attn --no-build-isolation`

**TIMM version error (OpenVLA):**
```
NotImplementedError: TIMM Version must be >= 0.9.10 and < 1.0.0
```
Fix: `pip install "timm>=0.9.10,<1.0.0"`

**Transformers version error (Pi-Zero):**
```
ValueError: An incorrect transformer version is used
```
Fix: `pip install "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"`
