# PPO-based Autonomous Navigation for Quadcopters
[![license](https://img.shields.io/badge/license-AGPL%203.0-%23F65314?style=flat-square)](LICENSE)

This repository contains an implementation of Proximal Policy Optimization (PPO) for autonomous navigation in a corridor environment with a quadcopter. There are blocks having circular opening for the drone to go through for each 4 meters. The expectation is that the agent navigates through these openings without colliding with blocks. **This project currently runs only on Windows since Unreal environments were packaged for Windows.**

### 🛠️ Libraries & Tools

- [OpenAI - Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)
- [OpenAI - Gym](https://github.com/openai/gym)
- [Microsoft AirSim](https://github.com/microsoft/AirSim)
- [Unreal Engine 4](https://www.unrealengine.com/en-US/)

## Overview

The training environment has 14 sections with different textures and hole positions. The agent starts at these sections randomly. The starting point of the agent is also random within a specific region in the yz-plane.

### Observation Space

- State is in the form of a RGB image taken by the front camera of the agent.
- Image shape: 50 x 50 x 3

<img src="https://user-images.githubusercontent.com/53112883/134379178-d1866c81-8e2c-43b1-8296-ba93a93ee4bb.png" width="600"/>

### Action Space

- There are 9 discrete actions.

## Environment setup to run the codes
#️⃣ **1. Clone the repository**

```
git clone https://github.com/bilalkabas/PPO-based-Autonomous-Navigation-for-Quadcopters
```

#️⃣ **2. From Anaconda command prompt, create a new conda environment**

I recommend you to use [Anaconda or Miniconda](https://www.anaconda.com/products/individual-d) to create a virtual environment.

```
conda create -n ppo_drone python==3.8
```

#️⃣ 3. **Install required libraries**

```
conda activate ppo_drone
pip install -r requirements.txt
```

#️⃣ 4. **(Optional) Install Pytorch for GPU**

> You must have a CUDA supported NVIDIA GPU.


<details>
<summary>Details for installation</summary>

- [Install CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
- [Install Pytorch with the compatible CUDA version](https://pytorch.org/get-started/locally/)

For this project, I used CUDA 11.0 and the following conda installation command to install Pytorch:

```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```

</details>

#️⃣ **4. Edit `settings.json`**

Content of the settings.json should be as below:

> The `setting.json` file is located at `Documents\AirSim` folder.

```json
{
    "SettingsVersion": 1.2,
    "LocalHostIp": "127.0.0.1",
    "SimMode": "Multirotor",
    "ClockSpeed": 10,
    "ViewMode": "FlyWithMe",
    "Vehicles": {
        "drone0": {
            "VehicleType": "SimpleFlight",
            "X": 0.0,
            "Y": 0.0,
            "Z": -1.3,
            "Yaw": 0.0
        }
    },
    "CameraDefaults": {
        "CaptureSettings": [
            {
                "ImageType": 0,
                "Width": 99 ,
                "Height": 99,
                "FOV_Degrees": 62.2
            },
            {
                "ImageType": 2,
                "Width": 99,
                "Height": 99,
                "FOV_Degrees": 62.2
            }
        ]
    }
  }
```

## How to run the training?
Make sure you followed the instructions above to setup the environment.

#️⃣ **1. Download the training environment**

Go to the [releases](https://github.com/bilalkabas/DQN-based-Autonomous-Navigation-for-Quadcopters/releases) and download `TrainEnv.zip`. After downloading completed, extract it.


#️⃣ **2. Now, you can open up environment's executable file and start the training**

So, inside the repository
```
python main.py
```

## How to run the pretrained model?
Make sure you followed the instructions above to setup the environment.

#️⃣ **1. Download the test environment**

Go to the [releases](https://github.com/bilalkabas/DQN-based-Autonomous-Navigation-for-Quadcopters/releases) and download `TestEnv.zip`. After downloading completed, extract it.


#️⃣ **2. Now, you can open up environment's executable file and start the training**

So, inside the repository
```
python policy_run.py
```

## Training results
The trained model in `saved_policy` folder was trained for 280k steps.

![Picture2](https://user-images.githubusercontent.com/53112883/134378107-5ba81690-8307-42aa-aca1-cc3555565d26.png)


## Author

- [Bilal Kabas](https://github.com/bilalkabas)

## License

This project is licensed under the [GNU Affero General Public License](LICENSE).
