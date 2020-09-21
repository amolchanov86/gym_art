import time
from unittest import TestCase

import numpy

from gym_art.quadrotor_multi.tests.test_multi_env import create_env


class TestOpt(TestCase):
    def test_optimized_env(self):
        num_agents = 4
        env = create_env(num_agents, use_numba=True)

        env.reset()
        time.sleep(0.1)

        num_steps = 0
        while num_steps < 100:
            obs, rewards, dones, infos = env.step([env.action_space.sample() for _ in range(num_agents)])
            num_steps += 1
            print('Rewards: ', rewards, "\nCollisions: \n", env.collisions, "\nDistances: \n", env.dist)

        env.close()

    @staticmethod
    def step_env(use_numba, steps):
        num_agents = 4
        env = create_env(num_agents, use_numba=use_numba)
        env.reset()
        num_steps = 0

        # warmup
        for i in range(20):
            obs, rewards, dones, infos = env.step([env.action_space.sample() for _ in range(num_agents)])
            num_steps += 1

        print('Measuring time, numba:', use_numba)
        start = time.time()
        for i in range(steps):
            obs, rewards, dones, infos = env.step([env.action_space.sample() for _ in range(num_agents)])
            if all(dones):
                env.reset()

        elapsed_sec = time.time() - start
        fps = (num_agents * steps) / elapsed_sec
        return fps, elapsed_sec

    def test_performance_difference(self):
        steps = 1000
        fps, elapsed_sec = self.step_env(use_numba=False, steps=steps)
        fps_numba, elapsed_sec_numba = self.step_env(use_numba=True, steps=steps)

        print('Regular: ', fps, elapsed_sec)
        print('Numba: ', fps_numba, elapsed_sec_numba)

    def test_step_opt(self):
        num_agents = 4
        env = create_env(num_agents)
        env.reset()

        dynamics = env.envs[0].dynamics

        dt = 0.005
        thrusts = numpy.float64([0.77263618, 0.5426721, 0.5024945, 0.66090029])
        thrust_noise = dynamics.thrust_noise.noise()

        import copy
        dynamics_copy = copy.deepcopy(dynamics)
        dynamics_copy_numba = copy.deepcopy(dynamics)

        dynamics.step1(thrusts, dt, thrust_noise)
        dynamics_copy.step1(thrusts, dt, thrust_noise)
        dynamics_copy_numba.step1_numba(thrusts, dt, thrust_noise)

        def pos_vel_acc(d):
            return d.pos, d.vel, d.acc

        p1, v1, a1 = pos_vel_acc(dynamics)
        p2, v2, a2 = pos_vel_acc(dynamics_copy)
        p3, v3, a3 = pos_vel_acc(dynamics_copy_numba)

        self.assertTrue(numpy.allclose(p1, p2))
        self.assertTrue(numpy.allclose(v1, v2))
        self.assertTrue(numpy.allclose(a1, a2))

        self.assertTrue(numpy.allclose(p1, p3))
        self.assertTrue(numpy.allclose(v1, v3))
        self.assertTrue(numpy.allclose(a1, a3))

        env.close()
