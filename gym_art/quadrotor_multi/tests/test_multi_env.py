import time
from unittest import TestCase

from gym_art.quadrotor_multi.quadrotor_multi import QuadrotorEnvMulti


def create_env(num_agents, use_numba=False):
    quad = 'Crazyflie'
    dyn_randomize_every = dyn_randomization_ratio = None

    episode_duration = 7  # seconds

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
    )
    return env


class TestMultiEnv(TestCase):
    def test_basic(self):
        num_agents = 2
        env = create_env(num_agents)

        self.assertTrue(hasattr(env, 'num_agents'))
        self.assertEqual(env.num_agents, num_agents)

        obs = env.reset()
        self.assertIsNotNone(obs)

        obs, rewards, dones, infos = env.step([env.action_space.sample() for i in range(num_agents)])
        self.assertIsInstance(obs, list)
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
