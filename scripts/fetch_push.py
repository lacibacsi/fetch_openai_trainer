#! /usr/env/bin python

'''
    Fetch task environment based (taken from) Unit 8 from the construct open AI course
'''

# generic, system, ros, and gym
import math
import rospy
from gym import utils
from gym import spaces
from gym.envs.registration import register

# robot env
import fetch_env
from cube_positions import Obj_Pos

# others
import numpy as np


max_episode_steps = 1000  # Can be any Value

# this register part is always outside the class -> generic, static
register(
    id='FetchPush-v0',
    entry_point='fetch_push:FetchPushEnv',
    max_episode_steps=max_episode_steps,
)


class FetchPushEnv(fetch_env.FetchEnv, utils.EZPickle):

    def __init__(self):
        self.obj_positions = Obj_Pos(object_name="demo_cube")

        # this method populates all local variables
        self.get_params()

        # TODO: why do we need this?
        fetch_env.FetchEnv.__init__(self)
        utils.EzPickle.__init__(self)

        self.gazebo.unpauseSim()

        # description - explanation why?
        self.action_space = spaces.Box(
            low=self.position_joints_min,
            high=self.position_joints_max, shape=(self.n_actions,),
            dtype=np.float32
        )

        # setting up observation space
        observations_high_dist = np.array([self.max_distance])
        observations_low_dist = np.array([0.0])

        observations_high_speed = np.array([self.max_speed])
        observations_low_speed = np.array([0.0])

        observations_ee_z_max = np.array([self.ee_z_max])
        observations_ee_z_min = np.array([self.ee_z_min])

        high = np.concatenate(
            [observations_high_dist, observations_high_speed, observations_ee_z_max])
        low = np.concatenate(
            [observations_low_dist, observations_low_speed, observations_ee_z_min])

        self.observation_space = spaces.Box(low, high)

        # reading current pos? dummy call?
        obs = self._get_obs()

    # ----------------------------------------------------
    # Local methods for the Task Environment
    # ----------------------------------------------------
    def get_params(self):
        '''
            get configuration parameters
            called by the constuctor            
        '''

        self.sim_time = rospy.get_time()
        self.n_actions = 7
        self.n_observations = 3
        self.position_ee_max = 10.0
        self.position_ee_min = -10.0
        self.position_joints_max = 2.16
        self.position_joints_min = -2.16

        self.init_pos = {"joint0": 0.0,
                         "joint1": -0.8,
                         "joint2": 0.0,
                         "joint3": 1.6,
                         "joint4": 0.0,
                         "joint5": 0.8,
                         "joint6": 0.0}

        self.setup_ee_pos = {"x": 0.598,
                             "y": 0.005,
                             "z": 0.9}

        self.position_delta = 0.1
        self.step_punishment = -1
        self.closer_reward = 10
        self.impossible_movement_punishement = -100
        self.reached_goal_reward = 100

        self.max_distance = 3.0
        self.max_speed = 1.0
        self.ee_z_max = 1.0
        # Normal z pos of cube minus its height/2
        self.ee_z_min = 0.3

    def calc_dist(self, p1, p2):
        '''
            returns the distance between 2 points            
            d = ((2 - 1)2 + (1 - 1)2 + (2 - 0)2)1/2
            or
            d = square root( (x1 -x2)^2 + (y1 - y2)^2 + (z1 - z2)^2 )
        '''
        x_d = math.pow(p1[0] - p2[0], 2)
        y_d = math.pow(p1[1] - p2[1], 2)
        z_d = math.pow(p1[2] - p2[2], 2)
        d = math.sqrt(x_d + y_d + z_d)

        return d

    def get_elapsed_time(self):
        '''
            Returns the elapsed time since the beginning of the simulation -- (ed) or since the last time asked
            Then maintains the current time as "previous time" to calculate the elapsed time again
        '''

        # this does not seem to be correct - this returns the elapsed time since last checked,

        current_time = rospy.get_time()
        dt = self.sim_time - current_time
        self.sim_time = current_time
        return dt

    # ----------------------------------------------------
    # Implementation of Robot Environment virtual methods
    # ----------------------------------------------------
    def _set_init_pose(self):
        '''
            Sets the Robot in its init pose
            The Simulation will be unpaused for this purpose.
        '''
        self.gazebo.unpauseSim()
        if not self.set_trajectory_joints(self.init_pos):
            assert False, "Initialisation is failed...."

    def _init_env_variables(self):
        '''
            Inits variables needed to be initialised each time we reset at the start
            of an episode.
        '''
        # TODO ??????
        rospy.logdebug("Init Env Variables...")
        rospy.logdebug("Init Env Variables...END")

    def _compute_reward(self, observations, done):
        '''
            Calculates the reward to give based on the observations given.
            Reward moving the cube
            Punish movint to unreachable positions
            Calculate the reward: binary => 1 for success, 0 for failure
        '''

        distance = observations[0]
        speed = observations[1]
        ee_z_pos = observations[2]

        # checking if the last action was successful, otherwise penalize
        done_fail = not(self.movement_result)

        # it moved the cube? how?
        done_success = speed >= self.max_speed

        if done_fail:
            # We punish that it tries to move where moveit cant reach
            reward = self.impossible_movement_punishement
        else:
            if done_success:
                # It moved the cube?? how?
                reward = -1*self.impossible_movement_punishement
            else:
                if ee_z_pos < self.ee_z_min or ee_z_pos >= self.ee_z_max:
                    print("Punish, ee z too low or high..."+str(ee_z_pos))
                    reward = self.impossible_movement_punishement / 4.0
                else:
                    # It didnt move the cube. We reward it by getting closer
                    print("Reward for getting closer")
                    reward = 1.0 / distance

        return reward

    def _set_action(self, action):
        '''
            Applies the given action to the simulation.
        '''

        # convert the action to joint position and apply to trajectory
        self.new_pos = {"joint0": action[0],
                        "joint1": action[1],
                        "joint2": action[2],
                        "joint3": action[3],
                        "joint4": action[4],
                        "joint5": action[5],
                        "joint6": action[6]}
        self.movement_result = self.set_trajectory_joints(self.new_pos)

    def _get_obs(self):
        '''
            It returns the Position of the TCP/EndEffector as observation.
            And the speed of cube
            Orientation for the moment is not considered
        '''
        self.gazebo.unpauseSim()

        # ee pose is returned by the robot /gazebo env...
        grip_pose = self.get_ee_pose()
        ee_array_pose = [grip_pose.position.x,
                         grip_pose.position.y, grip_pose.position.z]

        # the pose of the cube/box on a table
        object_data = self.obj_positions.get_states()

        # distance of actuator / end effector and the cube
        object_pos = object_data[3:]
        distance_from_cube = self.calc_dist(object_pos, ee_array_pose)

        # speed of the cube
        object_velo = object_data[-3:]
        speed = np.linalg.norm(object_velo)

        # IMPORTANT
        # We state as observations the distance form cube, the speed of cube and the z postion of the end effector
        observations_obj = np.array(
            [distance_from_cube, speed, ee_array_pose[2]])

        return observations_obj

    def _is_done(self, observations):
        '''
            Checks if episode done based on observations given.
            If the latest Action didnt succeed, it means that tha position asked was imposible therefore the episode must end.
            It will also end if it reaches its goal.
        '''

        #distance = observations[0]
        speed = observations[1]

        # checking if the last result is false -> set by set_action when applying the position to trajectory
        done_fail = not(self.movement_result)

        done_sucess = speed >= self.max_speed

        print(">>>>>>>>>>>>>>>>done_fail="+str(done_fail) +
              ",done_sucess="+str(done_sucess))

        # If it moved or the arm couldnt reach a position asked for it stops
        return done_fail or done_sucess
