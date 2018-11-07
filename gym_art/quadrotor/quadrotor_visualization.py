import numpy as np
from numpy.linalg import norm

import gym_art.quadrotor.rendering3d as r3d
from gym_art.quadrotor.quad_utils import *

# for visualization.
# a rough attempt at a reasonable third-person camera
# that looks "over the quadrotor's shoulder" from behind
class ChaseCamera(object):
    def __init__(self):
        self.view_dist = 4

    def reset(self, goal, pos, vel):
        self.goal = goal
        self.pos_smooth = pos
        self.vel_smooth = vel
        self.right_smooth, _ = normalize(cross(vel, npa(0, 0, 1)))

    def step(self, pos, vel):
        # lowpass filter
        ap = 0.6
        av = 0.999
        ar = 0.9
        self.pos_smooth = ap * self.pos_smooth + (1 - ap) * pos
        self.vel_smooth = av * self.vel_smooth + (1 - av) * vel

        veln, n = normalize(self.vel_smooth)
        up = npa(0, 0, 1)
        ideal_vel, _ = normalize(self.goal - self.pos_smooth)
        if True or np.abs(veln[2]) > 0.95 or n < 0.01 or np.dot(veln, ideal_vel) < 0.7:
            # look towards goal even though we are not heading there
            right, _ = normalize(cross(ideal_vel, up))
        else:
            right, _ = normalize(cross(veln, up))
        self.right_smooth = ar * self.right_smooth + (1 - ar) * right

    # return eye, center, up suitable for gluLookAt
    def look_at(self):
        up = npa(0, 0, 1)
        back, _ = normalize(cross(self.right_smooth, up))
        to_eye, _ = normalize(0.9 * back + 0.3 * self.right_smooth)
        eye = self.pos_smooth + self.view_dist * (to_eye + 0.3 * up)
        center = self.pos_smooth
        return eye, center, up


# for visualization.
# In case we have vertical control only we use a side view camera
class SideCamera(object):
    def __init__(self):
        self.view_dist = 4

    def reset(self, goal, pos, vel):
        self.goal = goal
        self.pos_smooth = pos
        self.vel_smooth = vel
        self.right_smooth, _ = normalize(cross(vel, npa(0, 0, 1)))

    def step(self, pos, vel):
        # lowpass filter
        ap = 0.6
        av = 0.999
        ar = 0.9
        self.pos_smooth = ap * self.pos_smooth + (1 - ap) * pos
        self.vel_smooth = av * self.vel_smooth + (1 - av) * vel

        veln, n = normalize(self.vel_smooth)
        up = npa(0, 0, 1)
        ideal_vel, _ = normalize(self.goal - self.pos_smooth)
        if True or np.abs(veln[2]) > 0.95 or n < 0.01 or np.dot(veln, ideal_vel) < 0.7:
            # look towards goal even though we are not heading there
            right, _ = normalize(cross(ideal_vel, up))
        else:
            right, _ = normalize(cross(veln, up))
        self.right_smooth = ar * self.right_smooth + (1 - ar) * right

    # return eye, center, up suitable for gluLookAt
    def look_at(self):
        up = npa(0, 0, 1)
        back, _ = normalize(cross(self.right_smooth, up))
        to_eye, _ = normalize(0.9 * back + 0.3 * self.right_smooth)
        # eye = self.pos_smooth + self.view_dist * (to_eye + 0.3 * up)
        eye = self.pos_smooth + self.view_dist * np.array([0, 1, 0])
        center = self.pos_smooth
        return eye, center, up


# determine where to put the obstacles such that no two obstacles intersect
# and compute the list of obstacles to collision check at each 2d tile.
def _place_obstacles(np_random, N, box, radius_range, our_radius, tries=5):

    t = np.linspace(0, box, TILES+1)[:-1]
    scale = box / float(TILES)
    x, y = np.meshgrid(t, t)
    pts = np.zeros((N, 2))
    dist = x + np.inf

    radii = np_random.uniform(*radius_range, size=N)
    radii = np.sort(radii)[::-1]
    test_list = [[] for i in range(TILES**2)]

    for i in range(N):
        rad = radii[i]
        ok = np.where(dist.flat > rad)[0]
        if len(ok) == 0:
            if tries == 1:
                print("Warning: only able to place {}/{} obstacles. "
                    "Increase box, decrease radius, or decrease N.")
                return pts[:i,:], radii[:i]
            else:
                return _place_obstacles(N, box, radius_range, tries-1)
        pt = np.unravel_index(np_random.choice(ok), dist.shape)
        pt = scale * np.array(pt)
        d = np.sqrt((x - pt[1])**2 + (y - pt[0])**2) - rad
        # big slop factor for tile size, off-by-one errors, etc
        for ind1d in np.where(d.flat <= 2*our_radius + scale)[0]:
            test_list[ind1d].append(i)
        dist = np.minimum(dist, d)
        pts[i,:] = pt - box/2.0

    # very coarse to allow for binning bugs
    test_list = np.array(test_list).reshape((TILES, TILES))
    #amt_free = sum(len(a) == 0 for a in test_list.flat) / float(test_list.size)
    #print(amt_free * 100, "pct free space")
    return pts, radii, test_list


# generate N obstacles w/ randomized primitive, size, color, TODO texture
# arena: boundaries of world in xy plane
# our_radius: quadrotor's radius
def _random_obstacles(np_random, N, arena, our_radius):
    arena = float(arena)
    # all primitives should be tightly bound by unit circle in xy plane
    boxside = np.sqrt(2)
    box = r3d.box(boxside, boxside, boxside)
    sphere = r3d.sphere(radius=1.0, facets=16)
    cylinder = r3d.cylinder(radius=1.0, height=2.0, sections=32)
    # TODO cone-sphere collision
    #cone = r3d.cone(radius=0.5, height=1.0, sections=32)
    primitives = [box, sphere, cylinder]

    bodies = []
    max_radius = 2.0
    positions, radii, test_list = _place_obstacles(
        np_random, N, arena, (0.5, max_radius), our_radius)
    for i in range(N):
        primitive = np_random.choice(primitives)
        tex_type = r3d.random_textype()
        tex_dark = 0.5 * np_random.uniform()
        tex_light = 0.5 * np_random.uniform() + 0.5
        color = 0.5 * np_random.uniform(size=3)
        heightscl = np.random.uniform(0.5, 2.0)
        height = heightscl * 2.0 * radii[i]
        z = (0 if primitive is cylinder else
            (height/2.0 if primitive is sphere else
            (height*boxside/4.0 if primitive is box
            else np.nan)))
        translation = np.append(positions[i,:], z)
        matrix = np.matmul(r3d.translate(translation), r3d.scale(radii[i]))
        matrix = np.matmul(matrix, np.diag([1, 1, heightscl, 1]))
        body = r3d.Transform(matrix,
            #r3d.ProceduralTexture(tex_type, (tex_dark, tex_light), primitive))
                r3d.Color(color, primitive))
        bodies.append(body)

    return ObstacleMap(arena, bodies, test_list)


# main class for non-visual aspects of the obstacle map.
class ObstacleMap(object):
    def __init__(self, box, bodies, test_lists):
        self.box = box
        self.bodies = bodies
        self.test = test_lists

    def detect_collision(self, dynamics):
        pos = dynamics.pos
        if pos[2] <= dynamics.arm:
            print("collided with terrain")
            return True
        r, c = self.coord2tile(*dynamics.pos[:2])
        if r < 0 or c < 0 or r >= TILES or c >= TILES:
            print("collided with wall")
            return True
        if self.test is not None:
            radius = dynamics.arm + 0.1
            return any(self.bodies[k].collide_sphere(pos, radius)
                for k in self.test[r,c])
        return False

    def sample_start(self, np_random):
        pad = 4
        band = TILES // 8
        return self.sample_freespace((pad, pad + band), np_random)

    def sample_goal(self, np_random):
        pad = 4
        band = TILES // 8
        return self.sample_freespace((-(pad + band), -pad), np_random)

    def sample_freespace(self, rowrange, np_random):
        rfree, cfree = np.where(np.vectorize(lambda t: len(t) == 0)(
            self.test[rowrange[0]:rowrange[1],:]))
        choice = np_random.choice(len(rfree))
        r, c = rfree[choice], cfree[choice]
        r += rowrange[0]
        x, y = self.tile2coord(r, c)
        z = np_random.uniform(1.0, 3.0)
        return np.array([x, y, z])

    def tile2coord(self, r, c):
        #TODO consider moving origin to corner of world
        scale = self.box / float(TILES)
        return scale * np.array([r,c]) - self.box / 2.0

    def coord2tile(self, x, y):
        scale = float(TILES) / self.box
        return np.int32(scale * (np.array([x,y]) + self.box / 2.0))


# using our rendering3d.py to draw the scene in 3D.
# this class deals both with map and mapless cases.
class Quadrotor3DScene(object):
    def __init__(self, np_random, quad_arm, w, h,
        obstacles=True, visible=True, resizable=True, goal_diameter=None, viewpoint='chase'):

        self.window_target = r3d.WindowTarget(w, h, resizable=resizable)
        self.obs_target = r3d.FBOTarget(64, 64)
        self.cam1p = r3d.Camera(fov=90.0)
        self.cam3p = r3d.Camera(fov=45.0)

        self.viepoint = viewpoint
        if self.viepoint == 'chase':
            self.chase_cam = ChaseCamera()
        elif self.viepoint == 'side':
            self.chase_cam = SideCamera()
        self.world_box = 40.0

        diameter = 2 * quad_arm
        if goal_diameter:
            self.goal_diameter = goal_diameter
        else:
            self.goal_diameter = diameter
        self.quad_transform = self._quadrotor_3dmodel(diameter)

        self.shadow_transform = r3d.transform_and_color(
            np.eye(4), (0, 0, 0, 0.4), r3d.circle(0.75*diameter, 32))

        # TODO make floor size or walls to indicate world_box
        floor = r3d.ProceduralTexture(r3d.random_textype(), (0.15, 0.25),
            r3d.rect((1000, 1000), (0, 100), (0, 100)))

        self.goal_transform = r3d.transform_and_color(np.eye(4),
            (0.85, 0.55, 0), r3d.sphere(self.goal_diameter/2, 18))

        self.map = None
        bodies = [r3d.BackToFront([floor, self.shadow_transform]),
            self.goal_transform, self.quad_transform]

        if obstacles:
            N = 20
            self.map = _random_obstacles(np_random, N, self.world_box, quad_arm)
            bodies += self.map.bodies

        world = r3d.World(bodies)
        batch = r3d.Batch()
        world.build(batch)

        self.scene = r3d.Scene(batches=[batch], bgcolor=(0,0,0))
        self.scene.initialize()

    def _quadrotor_3dmodel(self, diam):
        r = diam / 2
        prop_r = 0.3 * diam
        prop_h = prop_r / 15.0

        # "X" propeller configuration, start fwd left, go clockwise
        rr = r * np.sqrt(2)/2
        deltas = ((rr, rr, 0), (rr, -rr, 0), (-rr, -rr, 0), (-rr, rr, 0))
        colors = ((1,0,0), (1,0,0), (0,1,0), (0,1,0))
        def disc(translation, color):
            color = 0.5 * np.array(list(color)) + 0.2
            disc = r3d.transform_and_color(r3d.translate(translation), color,
                r3d.cylinder(prop_r, prop_h, 32))
            return disc
        props = [disc(d, c) for d, c in zip(deltas, colors)]

        arm_thicc = diam / 20.0
        arm_color = (0.6, 0.6, 0.6)
        arms = r3d.transform_and_color(
            np.matmul(r3d.translate((0, 0, -arm_thicc)), r3d.rotz(np.pi / 4)), arm_color,
            [r3d.box(diam/10, diam, arm_thicc), r3d.box(diam, diam/10, arm_thicc)])

        arrow = r3d.Color((0.2, 0.3, 0.9), r3d.arrow(0.12*prop_r, 2.5*prop_r, 16))

        bodies = props + [arms, arrow]
        self.have_state = False
        return r3d.Transform(np.eye(4), bodies)

    # TODO allow resampling obstacles?
    def reset(self, goal, dynamics):
        self.goal_transform.set_transform(r3d.translate(goal[0:3]))
        self.chase_cam.reset(goal[0:3], dynamics.pos, dynamics.vel)
        self.update_state(dynamics)

    def update_state(self, dynamics):
        self.have_state = True
        self.fpv_lookat = dynamics.look_at()
        self.chase_cam.step(dynamics.pos, dynamics.vel)

        matrix = r3d.trans_and_rot(dynamics.pos, dynamics.rot)
        self.quad_transform.set_transform_nocollide(matrix)

        shadow_pos = 0 + dynamics.pos
        shadow_pos[2] = 0.001 # avoid z-fighting
        matrix = r3d.translate(shadow_pos)
        self.shadow_transform.set_transform_nocollide(matrix)

        if self.map is not None:
            collided = self.map.detect_collision(dynamics)
        else:
            collided = dynamics.pos[2] <= dynamics.arm
        return collided

    def render_chase(self):
        assert self.have_state
        self.cam3p.look_at(*self.chase_cam.look_at())
        #self.cam3p.look_at(*self.fpv_lookat)
        r3d.draw(self.scene, self.cam3p, self.window_target)

    def render_obs(self):
        assert self.have_state
        self.cam1p.look_at(*self.fpv_lookat)
        r3d.draw(self.scene, self.cam1p, self.obs_target)
        return self.obs_target.read()