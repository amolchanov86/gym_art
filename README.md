# GymArt

Additional environments for the [OpenAI Gym](https://gym.openai.com/).

Currently contains the following environments:
- [QuadrotorEnv](https://github.com/amolchanov86/gym_art/blob/master/gym_art/quadrotor/quadrotor.py). 
  This is an environment for quadrotor stabilization at the origin. The env has static episode length that depends on episode time in sec `ep_time`, `sim_freq` that is integration freq, and `sim_steps` - number of simulation time steps after action is applied. For example: for `ep_time=5`, `sim_freq=200` and `sim_steps=2` the environment will have 500 steps.

## Requirements
- python 3.X

## Installation
Clone it:
```sh
mkdir ~/prj
cd ~/prj
git clone https://github.com/amolchanov86/gym_art.git
```

Add it to your `.bashrc`
```sh
export PYTHONPATH=$PYTHONPATH:~/prj/gym_art
```
## Experiments

### Test default quadrotor with the [Mellinger nonlinear controller](http://www-personal.acfr.usyd.edu.au/spns/cdm/papers/Mellinger.pdf)
```sh
cd ~/prj/gym_art/gym_art/quadrotor
./quadrotor.py
```

### Test CrazyFlie quadrotor with the [Mellinger nonlinear controller](http://www-personal.acfr.usyd.edu.au/spns/cdm/papers/Mellinger.pdf)
```sh
cd ~/prj/gym_art/gym_art/quadrotor
./quadrotor.py -q crazyflie
```
see `--help` option if you want to experiment with other parameters.

## Remarks:
- I tested everything with the [Garage's](https://github.com/rlworkgroup/garage/) anaconda environment. Hence, please, install Garage if something goes wrong.
- The current `default` version of the quadrotor model is roughly approximates AscTech Hummingbird quadrotor.
All supported models:
  - [Hummingbird](http://www.asctec.de/en/uav-uas-drones-rpas-roav/asctec-hummingbird/). Parameters were borrowed in large from the [Rotors Simulator](https://github.com/ethz-asl/rotors_simulator). See [Hummingbird URDF](https://github.com/ethz-asl/rotors_simulator/blob/master/rotors_description/urdf/hummingbird.xacro) for more details.
  - [CrazyFlie](http://mikehamer.info/assets/papers/Crazyflie%20Modelling.pdf)


