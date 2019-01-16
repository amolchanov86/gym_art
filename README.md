# gym_art

Additional environments for the [OpenAI Gym](https://gym.openai.com/).

Currently contains the following environments:
- [QuadrotorEnv](https://github.com/amolchanov86/gym_art/blob/master/gym_art/quadrotor/quadrotor_modular.py). 
  This is an environment for quadrotor stabilization at the origin. Has static episode length (typically 100 time steps).

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

## Test with [Mellinger nonlinear controller](http://www-personal.acfr.usyd.edu.au/spns/cdm/papers/Mellinger.pdf)
```sh
cd ~/prj/gym_art/gym_art/quadrotor
./quadrotor_modular.py
```

## Remarks:
- currently I am refactoring it a bit. The new version will be called `quadrotor.py`


