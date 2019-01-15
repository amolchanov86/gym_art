#!/usr/bin/env python

# Computing inertias of bodies.
# Coordinate frame:
# x - forward; y - left; z - up
# The same coord frame is used for quads
# All default inertias of objects are with respect to COM
# Source of inertias: https://en.wikipedia.org/wiki/List_of_moments_of_inertia

import numpy as np
import copy

def rotate_I(I, R):
    """
    Rotating inertia tensor I
    R - rotation matrix
    """
    return R @ I @ R.T

def translate_I(I, m, xyz):
    """
    Offsetting inertia tensor I by [x,y,z].T
    relative to COM
    """
    x,y,z = xyz[0], xyz[1], xyz[2]
    I_new = np.zeros([3,3])
    I_new[0][0] = I[0][0] + m * (y**2 + z**2)
    I_new[1][1] = I[1][1] + m * (x**2 + z**2)
    I_new[2][2] = I[2][2] + m * (x**2 + y**2)
    I_new[0][1] = I_new[1][0] = I[0][1] + m * x * y
    I_new[0][2] = I_new[2][0] = I[0][1] + m * x * z
    I_new[1][2] = I_new[2][1] = I[1][2] + m * y * z
    return I_new

def deg2rad(deg):
    return deg / 180. * np.pi

class SphereLink():
    """
    Box object
    """
    def __init__(self, r, m=None, density=None):
        """
        m = mass
        dx = dy = dz = diameter = 2 * r
        """
        self.r = r
        if m is None:
            self.m = self.compute_m(density)
        else:
            self.m = m
    @property
    def I_com(self):
        r = self.r
        return np.array([
            [2/5. * self.m * r **2, 0., 0.],
            [0., 2/5. * self.m * r **2, 0.],
            [0., 0., 2/5. * self.m * r **2],
        ])

    def compute_m(self, density):
        return density * 4./3. * np.pi * self.r ** 3


class BoxLink():
    """
    Box object
    """
    def __init__(self, l, w, h, m=None, density=None):
        """
        m = mass
        dx = length = l
        dy = width = l
        dz = height = h
        """
        self.l, self.w, self.h = l, w, h
        if m is None:
            self.m = self.compute_m(density)
        else:
            self.m = m
    @property
    def I_com(self):
        l ,w, h = self.l, self.w, self.h
        return np.array([
            [1/12. * self.m * (h**2 + w**2), 0., 0.],
            [0., 1/12. * self.m * (l**2 +  h**2), 0.],
            [0., 0., 1/12. * self.m * (w**2 + l**2)],
        ])
    
    def compute_m(self, density):
        return density * self.l * self.w * self.h

class RodLink():
    """
    Rod == Horizontal Cylinder
    """
    def __init__(self, l, r=0.002, m=None, density=None):
        """
        m = mass
        dx = length
        dy = dz = diameter == height
        """
        self.l = l
        self.r = r
        if m is None:
            self.m = self.compute_m(density)
        else:
            self.m = m
    @property
    def I_com(self):
        return np.array([
            [1/12. * self.m * self.l**2, 0., 0.],
            [0., 0., 0.],
            [0., 0., 1/12. * self.m * self.l**2],
        ])

    def compute_m(self, density):
        return density * np.pi * self.l * self.r ** 2

class CylinderLink():
    """
    Vertical Cylinder
    """
    def __init__(self, h, r, m=None, density=None):
        """
        m = mass
        dz = height = h
        dy = dx = 2*radius = 2*r = diameter
        """
        self.h, self.r = h, r
        if m is None:
            self.m = self.compute_m(density)
        else:
            self.m = m
    
    @property
    def I_com(self):
        h, r = self.h, self.r
        return np.array([
            [1/12. * self.m * (3*r**2 + h**2), 0., 0.],
            [0., 1/12. * self.m * (3*r**2 + h**2), 0.],
            [0., 0., 0.5 * self.m * r**2],
        ])
    def compute_m(self, density):
        return density * np.pi * self.h * self.r ** 2

class LinkPose(object):
    def __init__(self, R=None, xyz=None, alpha_deg=None):
        """
        One can provide either:
        R - rotation matrix or 
        alpha - angle of roation in a xy (horizontal) plane [degrees]
        xyz - offset
        """
        if xyz is not None:
            self.xyz = np.array(xyz)
        else:
            self.xyz = np.zeros(3)
        if R is not None:
            self.R = R
        elif alpha_deg:
            alpha = deg2rad(alpha_deg)
            self.R = np.array([
                [np.cos(alpha), -np.sin(alpha), 0.],
                [np.sin(alpha), np.cos(alpha), 0.],
                [0., 0., 1.]
            ])
        else:
            self.R = np.eye(3)


class QuadLink(object):
    """
    Quadrotor body set to compute inertia.
    Initial coordinate system assumes being in the middle of the central body.
    arm_angle == |/ , i.e. between the x axis and the axis of the arm
    """
    def __init__(self):
        # PARAMETERS
        self.motors_num = 4
        self.params = {}
        self.params["body"] = {"l": 0.03, "w": 0.03, "h": 0.004, "m": 0.005}
        self.params["payload"] = {"l": 0.035, "w": 0.02, "h": 0.008, "m": 0.01}
        self.params["arms"] = {"l":0.022, "w":0.005, "h":0.005, "m":0.001}
        # self.params["arms"] = {"w":0.005, "h":0.005, "m":0.001}
        self.params["motors"] = {"h":0.02, "r":0.0035, "m":0.0015}

        self.params["arms_pos"] = {"angle": 45., "z": 0.}

        self.params["payload_pos"] = {"xy": [0., 0.]}
        self.params["motor_pos"] = {"xyz": [0.065/2, 0.065/2, 0.]}

        # Printing all params
        print("######################################################")
        print("QUAD PARAMETERS:")
        [print(key,":", val) for key,val in self.params.items()]
        print("######################################################")
        
        # Dependent parameters
        self.arm_angle = deg2rad(self.params["arms_pos"]["angle"])
        self.motor_xyz = np.array(self.params["motor_pos"]["xyz"])
        delta_y = self.motor_xyz[1] - self.params["body"]["w"] / 2.
        self.arm_length =  delta_y / np.sin(self.arm_angle)
        if "l" not in self.params["arms"]:
            self.params["arms"]["l"] = self.arm_length
        # print("Arm length: ", self.arm_length, "angle: ", self.arm_angle)

        # Vectors of coordinates of the COMs of arms, s.t. their ends will be exactly at motors locations
        self.arm_xyz = np.array([ self.motor_xyz[0] - delta_y /(2 * np.tan(self.arm_angle)),
                                 self.motor_xyz[1] - delta_y / 2,
                                 self.params["arms_pos"]["z"] ])
        

        # X signs according to clockwise starting front-right
        self.x_sign = np.array([1, -1, -1, 1])
        self.y_sign = np.array([1, 1, -1, -1])
        self.sign_mx = np.array([self.x_sign, self.y_sign, np.array([1., 1., 1., 1.])])
        self.motors_coord = self.sign_mx * self.motor_xyz[:, None]
        self.arm_angles = [
            -self.arm_angle, 
             self.arm_angle, 
            -self.arm_angle, 
             self.arm_angle]
        self.arms_coord = self.sign_mx * self.arm_xyz[:, None]

        # First defining the bodies
        # In the list bodies are counting clockwise: front_right, back_right, back_left, front_left
        self.body =  BoxLink(**self.params["body"]) # Central body 
        self.payload = BoxLink(**self.params["payload"]) # Could include battery
        self.arms  = [BoxLink(**self.params["arms"]) for i in range(self.motors_num)] # Just arms
        self.motors =  [CylinderLink(**self.params["motors"]) for i in range(self.motors_num)] # The motors itself
        # self.props =  [CylinderLink(h=0.002, r=0.045, m=0.0001) for i in range(self.motors_num)] # Propellers
        
        self.links = [self.body, self.payload] + self.arms + self.motors

        print("######################################################")
        print("Inertias:")
        [print(link.I_com, "\n") for link in self.links]
        print("######################################################")

        # Defining locations of all bodies
        self.body_pose = LinkPose()
        self.payload_pose = LinkPose(xyz=list(self.params["payload_pos"]["xy"]) + [(self.body.h + self.payload.h) / 2])
        self.arms_pose = [LinkPose(alpha_deg=self.arm_angles[i], xyz=self.arms_coord[:, i]) 
                            for i in range(self.motors_num)]
        self.motors_pos = [LinkPose(xyz=self.motors_coord[:, i]) 
                            for i in range(self.motors_num)]
        
        self.poses = [self.body_pose, self.payload_pose] + self.arms_pose + self.motors_pos

        # Recomputing the center of mass of the new system of bodies
        masses = [link.m for link in self.links]
        self.com = sum([ masses[i] * pose.xyz for i, pose in enumerate(self.poses)]) / self.m

        # Recomputing corrections on posess with the respect to the new system
        self.poses_init = copy.deepcopy(self.poses)
        for pose in self.poses:
            pose.xyz -= self.com

        # Computing inertias
        self.links_I = []
        for link_i, link in enumerate(self.links):
            I_rot = rotate_I(I=link.I_com, R=self.poses[link_i].R)
            I_trans = translate_I(I=I_rot, m=link.m, xyz=self.poses[link_i].xyz)
            self.links_I.append(I_trans)
        
        # Total inertia
        self.I_com = sum(self.links_I)
    
    @property
    def m(self):
        return np.sum([link.m for link in self.links]) 


if __name__ == "__main__":
    quad = QuadLink()
    print("Quad inertia: ", quad.I_com)
    print("Quad mass:", quad.m)
    print("Quad arm_xyz:", quad.arm_xyz)
    print("Quad COM: ", quad.com)