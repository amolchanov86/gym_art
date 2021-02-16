import sys
import gym
import random
import numpy as np

from copy import deepcopy
from utils.utils import log


class ReplayBuffer(object):
    def __init__(self, control_frequency, cp_step_size=0.5, buffer_size=1e6):
        self.control_frequency = control_frequency
        self.cp_step_size_sec = cp_step_size  # how often (seconds) a checkpoint is saved
        self.cp_step_size_freq = self.cp_step_size_sec * self.control_frequency
        self.buffer_size = buffer_size
        self.buffer_idx = 0
        self.buffer = []
        self.checkpoint_history = []
        self._cp_history_size = int(3.0 / cp_step_size)  # keep only checkpoints from the last 3 seconds

    def save_checkpoint(self, cp):
        '''
        Save a checkpoint every X steps so that we may load it later if a collision was found. This is NOT the same as the buffer
        Checkpoints are added to the buffer only if we find a collision and want to replay that event later on
        :param cp: A tuple of (env, obs)
        '''
        self.checkpoint_history.append(cp)
        self.checkpoint_history = self.checkpoint_history[:self._cp_history_size]

    def clear_checkpoints(self):
        self.checkpoint_history = []

    def write_cp_to_buffer(self, seconds_ago=0.5):
        '''
        A collision was found and we want to load the corresponding checkpoint from X seconds ago into the buffer to be sampled later on
        '''
        steps_ago = int(seconds_ago / self.cp_step_size_sec)
        try:
            env, obs = self.checkpoint_history[-steps_ago]
        except IndexError:
            log.info("tried to get checkpoint out of bounds of checkpoint_history in the replay buffer")
            sys.exit(1)

        env.saved_in_replay_buffer = True

        if len(self.buffer) < self.buffer_size:
            self.buffer.append((env, obs))
        else:
            self.buffer[self.buffer_idx] = (env, obs)  # override existing event
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size

    def sample_event(self):
        '''
        Sample an event to replay
        '''
        return random.choice(self.buffer)

    def __len__(self):
        return len(self.buffer)


class ExperienceReplayWrapper(gym.Wrapper):
    def __init__(self, env, eps=1.0):
        gym.Wrapper.__init__(self, env)
        self.replay_buffer = ReplayBuffer(env.envs[0].control_freq)
        self.epsilon = eps

        self.env.unwrapped.experience_replay_interface = self

    def step(self, action):
        obs, rewards, dones, infos = self.env.step(action)

        if self.env.use_replay_buffer and self.env.activate_replay_buffer and not self.env.saved_in_replay_buffer and\
                self.env.envs[0].tick % self.replay_buffer.cp_step_size_freq == 0:
            #  remove attributes that prevent self from being pickleable
            for e in self.env.envs:
                del e.dynamics.thrust_noise

            if self.env.scene:
                chase_cam_backup = deepcopy(self.scene.chase_cam)
                self.env.scene = None
                self.deleted_scene = True

            self.replay_buffer.save_checkpoint((deepcopy(self.env), obs))

            if self.env.deleted_scene:
                self.env.reset_scene_multi()
                self.env.scene.chase_cam = chase_cam_backup
            self.env.reset_thrust_noise()

        if self.env.collision_occurred and self.env.use_replay_buffer and self.env.activate_replay_buffer\
                and self.env.envs[0].tick >= self.env.collisions_grace_period_seconds * self.env.envs[0].control_freq and not self.saved_in_replay_buffer:
            self.replay_buffer.write_cp_to_buffer(seconds_ago=1.5)
            self.env.collision_occurred = False

        if any(dones):
            obs = self.reset()

        return obs, rewards, dones, infos

    def reset(self):
        if self.env.use_replay_buffer:
            self.replay_buffer.clear_checkpoints()
        if np.random.uniform(0, 1) < self.epsilon and self.replay_buffer and self.env.activate_replay_buffer\
                and len(self.replay_buffer) > 0:
            env, obs = self.replay_buffer.sample_event()
            self.env = deepcopy(env)
            #  reset attributes that were deleted when first pickled
            self.env.reset_scene_multi()
            self.env.reset_thrust_noise()
            return obs
        else:
            obs = self.env.reset()
            self.env.saved_in_replay_buffer = False
            return obs



