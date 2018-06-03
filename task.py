import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
              
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, 
                              runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 
                                                                              10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
    # Set the emphase on the Z dimension by only focusing on this dimension [2] instead all the    dimensions(:3) 
    # adding reward as a fraction of the z-velocity (source Forum)
        reward = 0.8 * self.sim.v[2]
    
    #reward when the agent is closed to the target pose  (Source Forum)
        DistanceToTarget = abs(self.sim.pose[2] - self.target_pos[2])
        if DistanceToTarget < 5.: 
            reward +=((10- DistanceToTarget)**2)# Boosting reward when getting to mid route
        else:
            reward -= 1.
    # penalize the downward movement relative to the starting position
    #if self.sim.pose[2] < self.init_pose[2]:
        #reward -= 5
    
    #Clip the reward (Source Forum)
        reward = np.clip(reward, -1, 1, out=None)
        return reward
    
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            #Penalize the crashes 
            if done and self.sim.time < self.sim.runtime: 
                reward -= 1.
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done
    

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state