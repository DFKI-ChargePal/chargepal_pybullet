## ChargePal PyBullet

This package contains a simulation of primitive manipulation tasks using PyBullet. The interface to interact with the simulations follows the OpenAI Gym interface. The implementation is related to this [tutorial](https://www.gymlibrary.ml/pages/environment_creation/#creating-a-package).
This allows the package to be integrated into other projects, for example to train reinforcement learning agents.

### Dependencies

The simulation uses URDF-files to describe the robot model. The robot bringup-files are placed in the ROS [package](https://git.ni.dfki.de/chargepal/chargepal_description) `chargepal_description`. Furthermore, is the [Universal Robots UR10e](https://github.com/fmauch/universal_robot) arm part of the simulation.

### Installation 

The following instructions are only a suggestion to install this repository. There are many ways to use this package. Feel free to deviate from the instructions.

##### Create a ROS workspace:

```shell
mkdir -p ~/ros_ws/src && cd ~/ros_ws
catkin init
```
##### Install ROS-packages:
```
cd ~/ros_ws/src
git clone -b calibration_devel https://github.com/fmauch/universal_robot.git

mkdir chargepal && cd ~/ros_ws/src/chargepal
git clone git@git.ni.dfki.de:chargepal/chargepal_description.git
git clone git@git.ni.dfki.de:chargepal/manipulation/chargepal_pybullet.git

# set soft-link
cd ~/ros_ws/src/chargepal/chargepal_description
ln -s ../../universal_robot/ur_description

cd ~/ros_ws
catkin build
```

##### Install Python Dependencies:
```shell
curl -O https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
bash Anaconda3-2021.05-Linux-x86_64.sh 
```

##### Create Package:
```shell
cd ~/ros_ws/src/chargepal/
conda create --name chargepal
conda activate chargepal

pip install -e chargepal_pybullet
```

### Usage

```python
import gym


N_EPISODES = 2

def main(env_name: str) -> None:

    env = gym.make("gym_chargepal:" + env_name)
    env.action_space.seed(42)
    env.render()
    obs = env.reset()

    episode = 1
    ep_return = 0.0
    print(f'Start test with environment: {env_name}')
    print(f'Start episode: {episode}')
    while True:

        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action=action)
        ep_return += reward

        env.render()
        # reset environment
        if done:
            print(f'Finish episode {episode} with return: {ep_return}')
            episode += 1
            ep_return = 0.0
            if episode > N_EPISODES:
                break
            print(f'\n\nStart episode: {episode}')
            obs = env.reset()

    env.close()


if __name__ == '__main__':
    env_names = [
        'ChargePal-P2P-1D-PositionControl-v0',
        'ChargePal-P2P-3D-PositionControl-v0',
        'ChargePal-P2P-6D-PositionControl-v0',
    ]

    for env_ in env_names:
        main(env_)


```
