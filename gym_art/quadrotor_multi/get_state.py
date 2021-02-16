import numpy as np

## NOTE: the state_* methods are static because otherwise getattr memorizes self

def state_xyz_vxyz_R_omega(self):
    if self.use_numba:
        pos, vel, rot, omega, acc = self.sense_noise.add_noise_numba(
            self.dynamics.pos,
            self.dynamics.vel,
            self.dynamics.rot,
            self.dynamics.omega,
            self.dynamics.accelerometer,
            self.dt
        )
    else:
        pos, vel, rot, omega, acc = self.sense_noise.add_noise(
            pos=self.dynamics.pos,
            vel=self.dynamics.vel,
            rot=self.dynamics.rot,
            omega=self.dynamics.omega,
            acc=self.dynamics.accelerometer,
            dt=self.dt
        )
    # return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, (pos[2],)])
    return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega])

def state_xyz_vxyz_R_omega_wall(self):
    if self.use_numba:
        pos, vel, rot, omega, acc = self.sense_noise.add_noise_numba(
            self.dynamics.pos,
            self.dynamics.vel,
            self.dynamics.rot,
            self.dynamics.omega,
            self.dynamics.accelerometer,
            self.dt
        )
    else:
        pos, vel, rot, omega, acc = self.sense_noise.add_noise(
            pos=self.dynamics.pos,
            vel=self.dynamics.vel,
            rot=self.dynamics.rot,
            omega=self.dynamics.omega,
            acc=self.dynamics.accelerometer,
            dt=self.dt
        )
    # return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, (pos[2],)])
    wall_box_0 = np.clip(pos - self.room_box[0], a_min=0.0, a_max=5.0)
    wall_box_1 = np.clip(self.room_box[1] - pos, a_min=0.0, a_max=5.0)
    return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, wall_box_0, wall_box_1])

def state_xyz_vxyz_tx3_R_omega(self):        
    pos, vel, rot, omega, acc = self.sense_noise.add_noise(
        pos=self.dynamics.pos,
        vel=self.dynamics.vel,
        rot=self.dynamics.rot,
        omega=self.dynamics.omega,
        acc=self.dynamics.accelerometer,
        dt=self.dt
    )
    
    ## incoporating previous 3 states
    if self.tick == 0:
        self.vel_3 = np.array(vel, np.zeros(3), np.zeros(3))
        self.pos_3 = np.array(pos, np.zeros(3), np.zeros(3))
    elif self.tick == 1:
        self.vel_3[1] = self.vel_3[0]
        self.vel_3[0] = vel 

        self.pos_3[1] = self.pos_3[0]
        self.pos_3[0] = pos
    else:
        self.vel_3[1:3] = self.vel_3[0:2]
        self.vel_3[0] = vel

        self.pos_3[1:3] = self.pos_3[0:2]
        self.pos_3[0] = pos 
    
    #return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, (pos[2],)])
    return np.concatenate([pos_3[0] - self.goal[:3],pos_3[1] - self.goal[:3],pos_3[2] - self.goal[:3],vel_3[0],vel_3[1],vel_3[2], rot.flatten(), omega])

def state_vxyz_tx3_xyz_R_omega(self):        
    pos, vel, rot, omega, acc = self.sense_noise.add_noise(
        pos=self.dynamics.pos,
        vel=self.dynamics.vel,
        rot=self.dynamics.rot,
        omega=self.dynamics.omega,
        acc=self.dynamics.accelerometer,
        dt=self.dt
    )
    
    ## incoporating previous 3 states
    if self.tick == 0:
        self.vel_3 = np.array(vel, np.zeros(3), np.zeros(3))
    elif self.tick == 1:
        self.vel_3[1] = self.vel_3[0]
        self.vel_3[0] = vel 
    else:
        self.vel_3[1:3] = self.vel_3[0:2]
        self.vel_3[0] = vel

    #return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, (pos[2],)])
    return np.concatenate([pos - self.goal[:3],vel_3[0],vel_3[1],vel_3[2], rot.flatten(), omega])

def state_xyz_tx3_vxyz_R_omega(self):        
    pos, vel, rot, omega, acc = self.sense_noise.add_noise(
        pos=self.dynamics.pos,
        vel=self.dynamics.vel,
        rot=self.dynamics.rot,
        omega=self.dynamics.omega,
        acc=self.dynamics.accelerometer,
        dt=self.dt
    )
    
    ## incoporating previous 3 states
    if self.tick == 0:
        self.pos_3 = np.array(pos, np.zeros(3), np.zeros(3))
    elif self.tick == 1:
        self.pos_3[1] = self.pos_3[0]
        self.pos_3[0] = pos
    else:
        self.pos_3[1:3] = self.pos_3[0:2]
        self.pos_3[0] = pos        
    #return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, (pos[2],)])
    return np.concatenate([pos_3[0] - self.goal[:3],pos_3[1] - self.goal[:3],pos_3[2] - self.goal[:3],vel, rot.flatten(), omega])


def state_xyz_tx2_vxyz_R_omega(self):        
    pos, vel, rot, omega, acc = self.sense_noise.add_noise(
        pos=self.dynamics.pos,
        vel=self.dynamics.vel,
        rot=self.dynamics.rot,
        omega=self.dynamics.omega,
        acc=self.dynamics.accelerometer,
        dt=self.dt
    )
    
    ## incoporating previous 2 states
    if self.tick == 0:
        self.pos_2 = np.array(pos, np.zeros(3))
    else:
        self.pos_2[1] = self.pos_2[0]
        self.pos_2[0] = pos

    #return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, (pos[2],)])
    return np.concatenate([pos_2[0] - self.goal[:3],pos_2[1] - self.goal[:3],vel, rot.flatten(), omega])

def state_vxyz_tx2_xyz_R_omega(self):        
    pos, vel, rot, omega, acc = self.sense_noise.add_noise(
        pos=self.dynamics.pos,
        vel=self.dynamics.vel,
        rot=self.dynamics.rot,
        omega=self.dynamics.omega,
        acc=self.dynamics.accelerometer,
        dt=self.dt
    )
    
    ## incoporating previous 2 states
    if self.tick == 0:
        self.vel_2 = np.array(vel, np.zeros(3))
    else:
        self.vel_2[1] = self.vel_2[0]
        self.vel_2[0] = vel        
    #return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, (pos[2],)])
    return np.concatenate([pos - self.goal[:3],vel_2[0],vel_2[1],rot.flatten(), omega])    


def state_xyz_vxyz_R_omega_h(self):        
    pos, vel, rot, omega, acc = self.sense_noise.add_noise(
        pos=self.dynamics.pos,
        vel=self.dynamics.vel,
        rot=self.dynamics.rot,
        omega=self.dynamics.omega,
        acc=self.dynamics.accelerometer,
        dt=self.dt
    )
    return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, (pos[2],)])
    # return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega])


def state_xyzr_vxyzr_R_omega(self):
    """
    xyz and Vxyz are given in a body frame
    """        
    pos, vel, rot, omega, acc = self.sense_noise.add_noise(
        pos=self.dynamics.pos,
        vel=self.dynamics.vel,
        rot=self.dynamics.rot,
        omega=self.dynamics.omega,
        acc=self.dynamics.accelerometer,
        dt=self.dt
    )
    e_xyz_rel = self.dynamics.rot.T @ (pos - self.goal[:3])
    vel_rel = self.dynamics.rot.T @ vel
    return np.concatenate([e_xyz_rel, vel_rel, rot.flatten(), omega])


def state_xyzr_vxyzr_R_omega_tx1(self):
    pos, vel, rot, omega, acc = self.sense_noise.add_noise(
        pos=self.dynamics.pos,
        vel=self.dynamics.vel,
        rot=self.dynamics.rot,
        omega=self.dynamics.omega,
        acc=self.dynamics.accelerometer,
        dt=self.dt
    )
    e_xyz_rel = self.dynamics.rot.T @ (pos - self.goal[:3])
    vel_rel = self.dynamics.rot.T @ vel

    if self.tick == 0:
        vel_2 = np.array(vel, np.zeros(3))
        pos_2 = np.array(pos, np.zeros(3))
        rot_2 = np.array(rot, np.eye(3))
        omega_2 = np.array(omega, np.zeros(3))
    else:
        vel_2 = np.array(vel, vel_2[0])
        pos_2 = np.array(pos, pos_2[0])
        rot_2 = np.array(rot, rot_2[0])
        omega_2 = np.array(omega, omega_2[0])        
    
    #return np.concatenate([e_xyz_rel, vel_rel, rot.flatten(), omega])
    return np.concatenate([pos_2[0], pos_2[1], vel_2[0],vel_2[1], rot_2[0].flatten(), rot_2[1].flatten(), omega_2[0],omega_2[0]])


def state_xyzr_vxyzr_R_omega_action_tx1(self):
    pos, vel, rot, omega, acc = self.sense_noise.add_noise(
        pos=self.dynamics.pos,
        vel=self.dynamics.vel,
        rot=self.dynamics.rot,
        omega=self.dynamics.omega,
        acc=self.dynamics.accelerometer,
        dt=self.dt
    )
    e_xyz_rel = self.dynamics.rot.T @ (pos - self.goal[:3])
    vel_rel = self.dynamics.rot.T @ vel

    if self.tick == 0:
        vel_2 = np.array(vel, np.zeros(3))
        pos_2 = np.array(pos, np.zeros(3))
        rot_2 = np.array(rot, np.eye(3))
        omega_2 = np.array(omega, np.zeros(3))
    else:
        vel_2 = np.array(vel, vel_2[0])
        pos_2 = np.array(pos, pos_2[0])
        rot_2 = np.array(rot, rot_2[0])
        omega_2 = np.array(omega, omega_2[0])      
    actions_2 = copy.deepcopy(self.actions)        
    
    #return np.concatenate([e_xyz_rel, vel_rel, rot.flatten(), omega])
    return np.concatenate([pos_2[0], pos_2[1], vel_2[0],vel_2[1], rot_2[0].flatten(), rot_2[1].flatten(), omega_2[0],omega_2[0],actions_2[0],actions_2[1]])


def state_xyzr_vxyzr_R_omega_h(self):
    """
    xyz and Vxyz are given in a body frame
    """        
    pos, vel, rot, omega, acc = self.sense_noise.add_noise(
        pos=self.dynamics.pos,
        vel=self.dynamics.vel,
        rot=self.dynamics.rot,
        omega=self.dynamics.omega,
        acc=self.dynamics.accelerometer,
        dt=self.dt
    )
    e_xyz_rel = self.dynamics.rot.T @ (pos - self.goal[:3])
    vel_rel = self.dynamics.rot.T @ vel
    return np.concatenate([e_xyz_rel, vel_rel, rot.flatten(), omega, (pos[2],)])


def state_xyz_vxyz_R_omega_acc_act(self):        
    pos, vel, rot, omega, acc = self.sense_noise.add_noise(
        pos=self.dynamics.pos,
        vel=self.dynamics.vel,
        rot=self.dynamics.rot,
        omega=self.dynamics.omega,
        acc=self.dynamics.accelerometer,
        dt=self.dt
    )
    # return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, (pos[2],)])
    return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, acc, self.actions[1]])


def state_xyz_vxyz_R_omega_act(self):        
    pos, vel, rot, omega, acc = self.sense_noise.add_noise(
        pos=self.dynamics.pos,
        vel=self.dynamics.vel,
        rot=self.dynamics.rot,
        omega=self.dynamics.omega,
        acc=self.dynamics.accelerometer,
        dt=self.dt
    )
    # return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, (pos[2],)])
    return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, self.actions[1]])


def state_act_tx2_xyz_vxyz_R_omega(self):
    pos, vel, rot, omega, acc = self.sense_noise.add_noise(
        pos=self.dynamics.pos,
        vel=self.dynamics.vel,
        rot=self.dynamics.rot,
        omega=self.dynamics.omega,
        acc=self.dynamics.accelerometer,
        dt=self.dt
    )
    actions_2 = copy.deepcopy(self.actions)
    # return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, (pos[2],)])
    return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, actions_2[0],actions_2[1]])    


def state_xyz_vxyz_quat_omega(self):
    self.quat = R2quat(self.dynamics.rot)
    pos, vel, quat, omega, acc = self.sense_noise.add_noise(
        pos=self.dynamics.pos,
        vel=self.dynamics.vel,
        rot=self.quat,
        omega=self.dynamics.omega,
        acc=self.dynamics.accelerometer,
        dt=self.dt
    )
    return np.concatenate([pos - self.goal[:3], vel, quat, omega])


def state_xyzr_vxyzr_quat_omega(self):
    """
    xyz and Vxyz are given in a body frame
    """   
    self.quat = R2quat(self.dynamics.rot)
    pos, vel, quat, omega, acc = self.sense_noise.add_noise(
        pos=self.dynamics.pos,
        vel=self.dynamics.vel,
        rot=self.quat,
        omega=self.dynamics.omega,
        acc=self.dynamics.accelerometer,
        dt=self.dt
    )
    e_xyz_rel = self.dynamics.rot.T @ (pos - self.goal[:3])
    vel_rel = self.dynamics.rot.T @ vel
    return np.concatenate([e_xyz_rel, vel_rel, quat, omega])


def state_xyzr_vxyzr_quat_omega_h(self):
    """
    xyz and Vxyz are given in a body frame
    """   
    self.quat = R2quat(self.dynamics.rot)
    pos, vel, quat, omega, acc = self.sense_noise.add_noise(
        pos=self.dynamics.pos,
        vel=self.dynamics.vel,
        rot=self.quat,
        omega=self.dynamics.omega,
        acc=self.dynamics.accelerometer,
        dt=self.dt
    )
    e_xyz_rel = self.dynamics.rot.T @ (pos - self.goal[:3])
    vel_rel = self.dynamics.rot.T @ vel
    return np.concatenate([e_xyz_rel, vel_rel, quat, omega, (pos[2],)])


def state_xyzr_vxyzr_R_omega_t2w(self):
    pos, vel, rot, omega, acc = self.sense_noise.add_noise(
        pos=self.dynamics.pos,
        vel=self.dynamics.vel,
        rot=self.dynamics.rot,
        omega=self.dynamics.omega,
        acc=self.dynamics.accelerometer,
        dt=self.dt
    )
    ## Adding noise to t2w and scale it to [0, 1]
    noisy_t2w = self.dynamics.thrust_to_weight + \
                normal(loc=0., scale=abs((self.t2w_std/2)*self.dynamics.thrust_to_weight), size=1)
    noisy_t2w = np.clip(noisy_t2w, a_min=self.t2w_min, a_max=self.t2w_max)
    noisy_t2w = (noisy_t2w-self.t2w_min) / (self.t2w_max-self.t2w_min)

    e_xyz_rel = self.dynamics.rot.T @ (pos - self.goal[:3])
    vel_rel = self.dynamics.rot.T @ vel
    return np.concatenate([e_xyz_rel,vel_rel, rot.flatten(), omega, noisy_t2w])


def state_xyz_vxyz_R_omega_t2w(self):
    pos, vel, rot, omega, acc = self.sense_noise.add_noise(
        pos=self.dynamics.pos,
        vel=self.dynamics.vel,
        rot=self.dynamics.rot,
        omega=self.dynamics.omega,
        acc=self.dynamics.accelerometer,
        dt=self.dt
    )
    ## Adding noise to t2w and scale it to [0, 1]
    noisy_t2w = self.dynamics.thrust_to_weight + \
                normal(loc=0., scale=abs((self.t2w_std/2)*self.dynamics.thrust_to_weight), size=1)
    noisy_t2w = np.clip(noisy_t2w, a_min=self.t2w_min, a_max=self.t2w_max)
    noisy_t2w = (noisy_t2w-self.t2w_min) / (self.t2w_max-self.t2w_min)
    
    return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, noisy_t2w])


def state_xyz_vxyz_R_omega_t2w_t2t(self):
    pos, vel, rot, omega, acc = self.sense_noise.add_noise(
        pos=self.dynamics.pos,
        vel=self.dynamics.vel,
        rot=self.dynamics.rot,
        omega=self.dynamics.omega,
        acc=self.dynamics.accelerometer,
        dt=self.dt
    )
    ## Adding noise to t2w and scale it to [0, 1]
    noisy_t2w = self.dynamics.thrust_to_weight + \
                normal(loc=0., scale=abs((self.t2w_std/2)*self.dynamics.thrust_to_weight), size=1)
    noisy_t2w = np.clip(noisy_t2w, a_min=self.t2w_min, a_max=self.t2w_max)
    noisy_t2w = (noisy_t2w-self.t2w_min) / (self.t2w_max-self.t2w_min)

    ## Adding noise to t2t and scaling it to [0, 1]
    noisy_t2t = self.dynamics.torque_to_thrust + \
                normal(loc=0., scale=abs((self.t2t_std/2)*self.dynamics.torque_to_thrust), size=1)
    noisy_t2t = np.clip(noisy_t2t, a_min=self.t2t_min, a_max=self.t2t_max)
    noisy_t2t = (noisy_t2t-self.t2t_min) / (self.t2t_max-self.t2t_min)
    
    return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, noisy_t2w, noisy_t2t])


def state_xyz_vxyz_R_omega_t2w_t2t_l(self):
    pos, vel, rot, omega, acc = self.sense_noise.add_noise(
        pos=self.dynamics.pos,
        vel=self.dynamics.vel,
        rot=self.dynamics.rot,
        omega=self.dynamics.omega,
        acc=self.dynamics.accelerometer,
        dt=self.dt
    )
    ## Adding noise to t2w and scale it to [0, 1]
    noisy_t2w = self.dynamics.thrust_to_weight + \
                normal(loc=0., scale=abs((self.t2w_std/2)*self.dynamics.thrust_to_weight), size=1)
    noisy_t2w = np.clip(noisy_t2w, a_min=self.t2w_min, a_max=self.t2w_max)
    noisy_t2w = (noisy_t2w-self.t2w_min) / (self.t2w_max-self.t2w_min)

    ## Adding noise to t2t and scaling it to [0, 1]
    noisy_t2t = self.dynamics.torque_to_thrust + \
                normal(loc=0., scale=abs((self.t2t_std/2)*self.dynamics.torque_to_thrust), size=1)
    noisy_t2t = np.clip(noisy_t2t, a_min=self.t2t_min, a_max=self.t2t_max)
    noisy_t2t = (noisy_t2t-self.t2t_min) / (self.t2t_max-self.t2t_min)

    ## Adding noise to l (the distance from center to a motor)
    noise_l = self.dynamics.model.params["arms"]["l"] / 2 + \
                normal(loc=0., scale=0.005, size=1)
    
    return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, noisy_t2w, noisy_t2t, noise_l])


def state_xyz_vxyz_euler_omega(self):
    self.euler = t3d.euler.mat2euler(self.dynamics.rot)
    pos, vel, quat, omega, acc = self.sense_noise.add_noise(
        pos=self.dynamics.pos,
        vel=self.dynamics.vel,
        rot=self.euler,
        omega=self.dynamics.omega,
        acc=self.dynamics.accelerometer,
        dt=self.dt
    )       
    return np.concatenate([pos - self.goal[:3], vel, euler, omega])


def state_xyz_xyzi_vxyz_R_omega_t2w(self):
    pos, vel, rot, omega, acc = self.sense_noise.add_noise(
        pos=self.dynamics.pos,
        vel=self.dynamics.vel,
        rot=self.dynamics.rot,
        omega=self.dynamics.omega,
        acc=self.dynamics.accelerometer,
        dt=self.dt
    )
    ## Adding noise to t2w and scale it to [0, 1]
    noisy_t2w = self.dynamics.thrust_to_weight + \
                normal(loc=0., scale=abs((self.t2w_std/2)*self.dynamics.thrust_to_weight), size=1)
    noisy_t2w = np.clip(noisy_t2w, a_min=self.t2w_min, a_max=self.t2w_max)
    noisy_t2w = (noisy_t2w-self.t2w_min) / (self.t2w_max-self.t2w_min)

    ## Integrating the position error
    pos_err = pos - self.goal[:3]
    if self.tick == 0:
        self.accumulative_pos_err = pos_err
    else:
        self.accumulative_pos_err = self.accumulative_pos_err * 0.9 + pos_err
    ## prevent the accumulative error from exploding at the beginning of the training
    self.accumulative_pos_err = np.clip(self.accumulative_pos_err, a_min=-self.room_size, a_max=self.room_size)

    return np.concatenate([pos_err, self.accumulative_pos_err, vel, rot.flatten(), omega, noisy_t2w])