#!/usr/bin/env python
'''
This python script is trying to regress the physics params (dimensions and mass) of 
the Crazyflie model, given the urdf file and a ground truth inertia.
Inputs: ground truth inertia, urdf file
Outputs: dimensions and masses of each link as defined in the urdf file 
Ground truth inertia can be found on: http://mikehamer.info/assets/papers/Crazyflie%20Modelling.pdf
Currently, I assume I know how the model is defined in the urdf file. 
For example, I know what links are there and their relative position.
'''
import matplotlib.pyplot as plt
import numpy as np
import math

## some useful constants
cos45 = math.cos(math.radians(45))
sin45 = math.sin(math.radians(45))

## Position class for easier readibility
## use x, y, z 
class Position:
	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z


class BoxLink:
	def __init__(self, length, width, height, mass):
		self.length = length
		self.width = width
		self.height = height
		self.mass = mass
		self.computeInertia()

	def setParams(self, length, width, height, mass=0.0, density=0.0):
		self.length = length
		self.width = width
		self.height = height
		self.mass = mass
		self.computeInertia()

	## set the position of the link relative to the
	## center of the base link 
	def setGlobalPosition(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
		self.x = x
		self.y = y 
		self.z = z
		self.roll = roll
		self.pitch = pitch 
		self.yaw = yaw


	## set the position of the COM relative to the 
	## COM of the whole object
	def setRelativePositionCOM(self, x, y, z):
		self.com_X = x
		self.com_Y = y 
		self.com_Z = z 
		self.computeInertiaCOM()

	## inertia wrt to its center
	def computeInertia(self):
		self.ixx = 0.0833333 * self.mass * (self.width**2 + self.height**2)
		self.ixy = 0.0
		self.ixz = 0.0
		self.iyx = 0.0
		self.iyy = 0.0833333 * self.mass * (self.length**2 + self.height**2)
		self.iyz = 0.0
		self.izx = 0.0
		self.izy = 0.0
		self.izz = 0.0833333 * self.mass * (self.length**2 + self.width**2)


	def getInertia(self):
		return (self.ixx, self.ixy, self.ixz, self.iyy, self.iyz, self.izz)

	## calculate the inertia of single link wrt the COM using the parallel axis theorem
	def computeInertiaCOM(self):
		rot = np.array([[math.cos(math.radians(self.yaw)), -math.sin(math.radians(self.yaw)), 0.0], \
						[math.sin(math.radians(self.yaw)), math.cos(math.radians(self.yaw)), 0.0], \
						[0.0, 0.0, 1.0]])

		inertia_COM = np.array([[self.ixx, self.ixy, self.ixz],\
								[self.iyx, self.iyy, self.iyz],\
								[self.izx, self.izy, self.izz]])
		inertia_COM = np.matmul(np.matmul(rot, inertia_COM), rot.T)

		self.ixx_COM = inertia_COM[0][0] + self.mass * (self.com_Y**2 + self.com_Z**2)
		self.ixy_COM = inertia_COM[0][1] - self.mass * self.com_X * self.com_Y 
		self.ixz_COM = inertia_COM[0][2] - self.mass * self.com_X * self.com_Z 
		self.iyx_COM = self.ixy_COM
		self.iyy_COM = inertia_COM[1][1] + self.mass * (self.com_X**2 + self.com_Z**2)
		self.iyz_COM = inertia_COM[1][2] - self.mass * self.com_Y * self.com_Z
		self.izx_COM = self.ixz_COM
		self.izy_COM = self.iyz_COM 
		self.izz_COM = inertia_COM[2][2] + self.mass * (self.com_X**2 + self.com_Y**2)

	def getInertiaCOM(self):
		return (self.ixx_COM, self.ixy_COM, self.ixz_COM, self.iyy_COM, self.iyz_COM, self.izz_COM)

	## return the inertia as a numpy array
	def getInertiaCOM_numpy(self):
		return np.array([self.ixx_COM, self.ixy_COM, self.ixz_COM, self.iyy_COM, self.iyz_COM, self.izz_COM])


class CylinderLink:
	def __init__(self, length, radius, mass):
		self.length = length 
		self.radius = radius 
		self.mass = mass 
		self.computeInertia() 

	def setParams(self, length, radius, mass):
		self.length = length 
		self.radius = radius 
		self.mass = mass 
		self.computeInertia() 	

	## set the COM position of the link relative to the
	## center of the base link 
	def setGlobalPosition(self, x, y, z):
		self.x = x
		self.y = y 
		self.z = z

	## set the position of the COM relative to the 
	## COM of the whole object
	def setRelativePositionCOM(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
		self.com_X = x
		self.com_Y = y 
		self.com_Z = z
		self.roll = roll 
		self.pitch = pitch
		self.yaw = yaw 
		self.computeInertiaCOM()  

	def computeInertia(self):
		self.ixx = self.mass * (3 * self.radius**2 + self.length**2)/12 
		self.ixy = 0.0 
		self.ixz = 0.0 
		self.iyx = 0.0
		self.iyy = self.mass * (3 * self.radius**2 + self.length**2)/12 
		self.iyz = 0.0
		self.izx = 0.0
		self.izy = 0.0 
		self.izz = (self.mass * self.radius**2)/2 

	def getInertia(self):
		return (self.ixx, self.ixy, self.ixz, self.iyy, self.iyz, self.izz)

	## calculate the inertia of single link wrt the COM using the parallel axis theorem
	def computeInertiaCOM(self):
		rot = np.array([[math.cos(math.radians(self.yaw)), -math.sin(math.radians(self.yaw)), 0.0], \
						[math.sin(math.radians(self.yaw)), math.cos(math.radians(self.yaw)), 0.0], \
						[0.0, 0.0, 1.0]])

		inertia_COM = np.array([[self.ixx, self.ixy, self.ixz],\
								[self.iyx, self.iyy, self.iyz],\
								[self.izx, self.izy, self.izz]])
		inertia_COM = np.matmul(np.matmul(rot, inertia_COM), rot.T)

		self.ixx_COM = inertia_COM[0][0] + self.mass * (self.com_Y**2 + self.com_Z**2)
		self.ixy_COM = inertia_COM[0][1] - self.mass * self.com_X * self.com_Y 
		self.ixz_COM = inertia_COM[0][2] - self.mass * self.com_X * self.com_Z 
		self.iyx_COM = self.ixy_COM
		self.iyy_COM = inertia_COM[1][1] + self.mass * (self.com_X**2 + self.com_Z**2)
		self.iyz_COM = inertia_COM[1][2] - self.mass * self.com_Y * self.com_Z
		self.izx_COM = self.ixz_COM
		self.izy_COM = self.iyz_COM 
		self.izz_COM = inertia_COM[2][2] + self.mass * (self.com_X**2 + self.com_Y**2)

	def getInertiaCOM(self):
		return (self.ixx_COM, self.ixy_COM, self.ixz_COM, self.iyy_COM, self.iyz_COM, self.izz_COM)

	## return the inertia as a numpy array
	def getInertiaCOM_numpy(self):
		return np.array([self.ixx_COM, self.ixy_COM, self.ixz_COM, self.iyy_COM, self.iyz_COM, self.izz_COM])

def get_I(i):
	return np.array([
		[i[0], i[1], i[2]],
		[i[1], i[3], i[4]],
		[i[2], i[4], i[5]]
	])

## defins a model structure that matches the one in Gazebo
class Structure:
	def __init__(self):
		# define the links (with default params, approaximation)
		self.center_box_link = BoxLink(0.03, 0.03, 0.004, 0.005)
		self.battery_link = BoxLink(0.035, 0.02, 0.008, 0.01)
		self.motor_arm_link_front_right = BoxLink(0.022, 0.005, 0.005, 0.001)
		self.motor_arm_link_back_right = BoxLink(0.022, 0.005, 0.005, 0.001)
		self.motor_arm_link_back_left = BoxLink(0.022, 0.005, 0.005, 0.001)
		self.motor_arm_link_front_left = BoxLink(0.022, 0.005, 0.005, 0.001)
		self.motor_link_front_right = CylinderLink(0.02, 0.0035, 0.0015)
		self.motor_link_back_right = CylinderLink(0.02, 0.0035, 0.0015)
		self.motor_link_back_left = CylinderLink(0.02, 0.0035, 0.0015)
		self.motor_link_front_left = CylinderLink(0.02, 0.0035, 0.0015)

		self.motor_offset = 0.0
		self.setBatteryOffset(0.0, 0.0) # inputs are battery offsets

		[print(get_I(link.getInertia()), "\n") for link in self.getLinksInList() ]

	def getLinksInList(self):
		return [self.center_box_link, self.battery_link, self.motor_arm_link_front_right, self.motor_arm_link_front_left, \
				self.motor_arm_link_back_right, self.motor_arm_link_back_left, self.motor_link_front_right, self.motor_link_front_left, \
				self.motor_link_back_right, self.motor_link_back_left]

	## batter offset is the only thing that depends on external parameters
	def setBatteryOffset(self, battery_offset_x, battery_offset_y):
		## center box
		p = Position(0.0, 0.0, 0.0)
		self.center_box_link.setGlobalPosition(p.x, p.y, p.z)

		## battery
		p = Position(self.center_box_link.x + battery_offset_x, 
					 self.center_box_link.y + battery_offset_y, 
					 self.center_box_link.z + (self.center_box_link.height + self.battery_link.height)/2)
		self.battery_link.setGlobalPosition(p.x, p.y, p.z)

		## the front right arm and motor
		p = Position(self.center_box_link.x + (self.center_box_link.length + self.motor_arm_link_front_right.length*sin45)/2, 
					 self.center_box_link.y - (self.center_box_link.width + self.motor_arm_link_front_right.length*cos45)/2, 
					 self.center_box_link.z + 0.0)
		self.motor_arm_link_front_right.setGlobalPosition(p.x, p.y, p.z, 0, 0, -45) 

		p = Position(self.motor_arm_link_front_right.x + (self.motor_arm_link_front_right.length/2 + self.motor_link_front_right.radius)*sin45, 
					 self.motor_arm_link_front_right.y - (self.motor_arm_link_front_right.length/2 + self.motor_link_front_right.radius)*cos45,
					 self.motor_arm_link_front_right.z + self.motor_offset)
		self.motor_link_front_right.setGlobalPosition(p.x, p.y, p.z)

		## the back right arm and motor
		p = Position(self.center_box_link.x - (self.center_box_link.length + self.motor_arm_link_back_right.length*sin45)/2, 
					 self.center_box_link.y - (self.center_box_link.width + self.motor_arm_link_back_right.length*cos45)/2, 
					 self.center_box_link.z + 0.0)
		self.motor_arm_link_back_right.setGlobalPosition(p.x, p.y, p.z, 0, 0, 45)

		p = Position(self.motor_arm_link_back_right.x - (self.motor_arm_link_back_right.length/2 + self.motor_link_back_right.radius)*sin45, 
					 self.motor_arm_link_back_right.y - (self.motor_arm_link_back_right.length/2 + self.motor_link_back_right.radius)*cos45,
					 self.motor_arm_link_back_right.z + self.motor_offset)
		self.motor_link_back_right.setGlobalPosition(p.x, p.y, p.z)

		## the back left arm
		p = Position(self.center_box_link.x - (self.center_box_link.length + self.motor_arm_link_back_left.length*sin45)/2, 
					 self.center_box_link.y + (self.center_box_link.width + self.motor_arm_link_back_left.length*cos45)/2, 
					 self.center_box_link.z + 0.0)
		self.motor_arm_link_back_left.setGlobalPosition(p.x, p.y, p.z, 0, 0, -45)

		p = Position(self.motor_arm_link_back_left.x - (self.motor_arm_link_back_left.length/2 + self.motor_link_back_left.radius)*sin45, 
					 self.motor_arm_link_back_left.y + (self.motor_arm_link_back_left.length/2 + self.motor_link_back_left.radius)*cos45,
					 self.motor_arm_link_back_left.z + self.motor_offset)
		self.motor_link_back_left.setGlobalPosition(p.x, p.y, p.z)

		## the front left arm 
		p = Position(self.center_box_link.x + (self.center_box_link.length + self.motor_arm_link_front_left.length*sin45)/2, 
					 self.center_box_link.y + (self.center_box_link.width + self.motor_arm_link_front_left.length*cos45)/2, 
					 self.center_box_link.z + 0.0)
		self.motor_arm_link_front_left.setGlobalPosition(p.x, p.y, p.z, 0, 0, 45)

		p = Position(self.motor_arm_link_front_left.x + (self.motor_arm_link_front_left.length/2 + self.motor_link_front_left.radius)*sin45, 
					 self.motor_arm_link_front_left.y + (self.motor_arm_link_front_left.length/2 + self.motor_link_front_left.radius)*cos45,
					 self.motor_arm_link_front_left.z + self.motor_offset)
		self.motor_link_front_left.setGlobalPosition(p.x, p.y, p.z)

		links = self.getLinksInList()
		self.cog_x, self.cog_y, self.cog_z = getCOM(links)
		print("COM:", self.cog_x, self.cog_y, self.cog_z)

	def setCenterBoxParams(self, width=0.03, height=0.004, mass=0.005):
		self.center_box_link.setParams(width, width, height, mass)

	def setBatterLinkParams(self, length=0.02, width=0.02, height=0.008, mass=0.01):
		self.battery_link.setParams(length, width, height, mass)

	def setMotorArmParams(self, length=0.022, width=0.005, mass=0.001):
		self.motor_arm_link_front_left.setParams(length, width, width, mass) 
		self.motor_arm_link_front_right.setParams(length, width, width, mass) 
		self.motor_arm_link_back_left.setParams(length, width, width, mass) 
		self.motor_arm_link_back_right.setParams(length, width, width, mass) 

	def setMotorParams(self, radius=0.0035, length=0.02, mass=0.0015):
		self.motor_link_front_right.setParams(length, radius, mass)
		self.motor_link_back_left.setParams(length, radius, mass)
		self.motor_link_back_right.setParams(length, radius, mass)
		self.motor_link_front_left.setParams(length, radius, mass)

	def getInertia(self):
		combined_inertia = np.zeros(6)
		links = self.getLinksInList()

		for link in links:
			p = Position(link.x - self.cog_x, link.y - self.cog_y, link.z - self.cog_z)
			link.setRelativePositionCOM(p.x, p.y, p.z)
			### sum the COM inertia
			combined_inertia += link.getInertiaCOM_numpy()

		return combined_inertia

## calculate the COM of the list of links that get passed in
def getCOM(links):
	total_w = 0.0
	x_w = 0.0
	y_w = 0.0 
	z_w = 0.0
	for link in links:
		## sum up the total weights 
		total_w += link.mass 
		x_w += link.x * link.mass 
		y_w += link.y * link.mass 
		z_w += link.z * link.mass 

	print("total mass:",total_w)
	return (x_w/total_w, y_w/total_w, z_w/total_w)


if __name__ == "__main__":
	quad = Structure()
	print("CrazyFie Inertia: ", quad.getInertia())
