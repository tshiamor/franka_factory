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
│       ├── record_demos.py
│       ├── replay_demos.py
│       └── teleop_agent.py
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

After recording, you can replay demonstrations to verify quality:

```bash
# Replay all demos
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
