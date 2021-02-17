import time
from unittest import TestCase
import numpy as np

from gym_art.quadrotor_multi.quad_experience_replay import ExperienceReplayWrapper
from gym_art.quadrotor_multi.quadrotor_multi import QuadrotorEnvMulti


def create_env(num_agents, use_numba=False, use_replay_buffer=False, episode_duration=7):
    quad = 'Crazyflie'
    dyn_randomize_every = dyn_randomization_ratio = None

    episode_duration = episode_duration  # seconds

    raw_control = raw_control_zero_middle = True

    sampler_1 = None
    if dyn_randomization_ratio is not None:
        sampler_1 = dict(type='RelativeSampler', noise_ratio=dyn_randomization_ratio, sampler='normal')

    sense_noise = 'default'

    dynamics_change = dict(noise=dict(thrust_noise_ratio=0.05), damp=dict(vel=0, omega_quadratic=0))

    env = QuadrotorEnvMulti(
        num_agents=num_agents,
        dynamics_params=quad, raw_control=raw_control, raw_control_zero_middle=raw_control_zero_middle,
        dynamics_randomize_every=dyn_randomize_every, dynamics_change=dynamics_change, dyn_sampler_1=sampler_1,
        sense_noise=sense_noise, init_random_state=True, ep_time=episode_duration, quads_use_numba=use_numba,
        use_replay_buffer=use_replay_buffer,
    )
    return env


class TestMultiEnv(TestCase):
    def test_basic(self):
        num_agents = 2
        env = create_env(num_agents, use_numba=False)

        self.assertTrue(hasattr(env, 'num_agents'))
        self.assertEqual(env.num_agents, num_agents)

        obs = env.reset()
        self.assertIsNotNone(obs)

        for i in range(1000):
            obs, rewards, dones, infos = env.step([env.action_space.sample() for i in range(num_agents)])
            try:
                self.assertIsInstance(obs, list)
            except:
                self.assertIsInstance(obs, np.ndarray)

            self.assertIsInstance(rewards, list)
            self.assertIsInstance(dones, list)
            self.assertIsInstance(infos, list)

        env.close()

    def test_render(self):
        num_agents = 16
        env = create_env(num_agents, use_numba=True)
        env.render_speed = 1.0

        env.reset()
        time.sleep(0.1)

        num_steps = 0
        render_n_frames = 500

        render_start = None
        while num_steps < render_n_frames:
            obs, rewards, dones, infos = env.step([env.action_space.sample() for _ in range(num_agents)])
            num_steps += 1
            # print('Rewards: ', rewards, "\nCollisions: \n", env.collisions, "\nDistances: \n", env.dist)
            env.render()

            if num_steps <= 1:
                render_start = time.time()

        render_took = time.time() - render_start
        print(f'Rendering of {render_n_frames} frames took {render_took:.3f} sec')

        env.close()


class TestReplayBuffer(TestCase):
    def test_replay(self):
        num_agents = 16
        replay_buffer_sample_prob = 1.0
        env = create_env(num_agents, use_numba=False, use_replay_buffer=replay_buffer_sample_prob > 0, episode_duration=5)
        env.render_speed = 1.0
        env = ExperienceReplayWrapper(env, replay_buffer_sample_prob=replay_buffer_sample_prob)

        env.reset()
        time.sleep(0.01)

        num_steps = 0
        render_n_frames = 1500

        while num_steps < render_n_frames:
            obs, rewards, dones, infos = env.step([env.action_space.sample() for _ in range(num_agents)])
            num_steps += 1
            # print('Rewards: ', rewards, "\nCollisions: \n", env.collisions, "\nDistances: \n", env.dist)
            env.render()
            # this env self-resets

        env.close()
