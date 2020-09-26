import time
from unittest import TestCase
import numpy.random as nr

import numpy

from gym_art.quadrotor_multi.tests.test_multi_env import create_env
from gym_art.quadrotor_multi.numba_utils import OUNoiseNumba
from gym_art.quadrotor_multi.quad_utils import OUNoise
from gym_art.quadrotor_multi.sensor_noise import SensorNoise


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

    def test_step_and_noise_opt(self):
        for _ in range(3):
            num_agents = 4
            env = create_env(num_agents)
            env.reset()

            dynamics = env.envs[0].dynamics

            dt = 0.005
            thrust_noise_ratio = 0.05
            thrusts = numpy.random.random(4)

            import copy
            dynamics_copy = copy.deepcopy(dynamics)
            dynamics_copy_numba = copy.deepcopy(dynamics)

            dynamics.thrust_noise = OUNoise(4, sigma=0.2 * thrust_noise_ratio, use_seed=True)
            dynamics_copy_numba.thrust_noise = OUNoiseNumba(4, sigma=0.2 * thrust_noise_ratio, use_seed=True)
            thrust_noise = thrust_noise_copy = dynamics.thrust_noise.noise()
            thrust_noise_numba = dynamics_copy_numba.thrust_noise.noise()

            dynamics.step1(thrusts, dt, thrust_noise)
            dynamics_copy.step1(thrusts, dt, thrust_noise_copy)
            dynamics_copy_numba.step1_numba(thrusts, dt, thrust_noise_numba)

            def pos_vel_acc_tor(d):
                return d.pos, d.vel, d.acc, d.torque

            def rot_omega_accm(d):
                return d.rot, d.omega, d.accelerometer

            p1, v1, a1, t1 = pos_vel_acc_tor(dynamics)
            p2, v2, a2, t2 = pos_vel_acc_tor(dynamics_copy)
            p3, v3, a3, t3 = pos_vel_acc_tor(dynamics_copy_numba)

            self.assertTrue(numpy.allclose(p1, p2))
            self.assertTrue(numpy.allclose(v1, v2))
            self.assertTrue(numpy.allclose(a1, a2))
            self.assertTrue(numpy.allclose(t1, t2))

            self.assertTrue(numpy.allclose(p1, p3))
            self.assertTrue(numpy.allclose(v1, v3))
            self.assertTrue(numpy.allclose(a1, a3))
            self.assertTrue(numpy.allclose(t1, t3))

            # the below test is to check if add_noise is returning the same value
            r1, o1, accm1 = rot_omega_accm(dynamics)
            r2, o2, accm2 = rot_omega_accm(dynamics_copy_numba)

            sense_noise = SensorNoise(bypass=False, use_numba=False)
            sense_noise_numba = SensorNoise(bypass=False, use_numba=True)

            new_p1, new_v1, new_r1, new_o1, new_a1 = sense_noise.add_noise(p1, v1, r1, o1, accm1, dt)
            new_p2, new_v2, new_r2, new_o2, new_a2 = sense_noise_numba.add_noise_numba(p2, v2, r2, o2, accm2, dt)

            self.assertTrue(numpy.allclose(new_p1, new_p2))
            self.assertTrue(numpy.allclose(new_v1, new_v2))
            self.assertTrue(numpy.allclose(new_a1, new_a2))
            self.assertTrue(numpy.allclose(new_o1, new_o2))
            self.assertTrue(numpy.allclose(new_r1, new_r2))
            env.close()
