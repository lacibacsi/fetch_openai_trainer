#! /usr/bin/env python

import rospy
from openai_ros import robot_gazebo_env

#from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
#from nav_msgs.msg import Odometry
import geometry_msgs.msg


JOINT_STATES_SUBSCRIBER = '/joint_states'


class FetchEnv(robot_gazebo_env.RobotGazeboEnv):
    '''
        Fetch robot Robot Environment
        Based on RobotEnv template from the ConstructSim Unit7 file
    '''

    def __init__(self):
        '''
            Initializes a new Fetch Robot environment. Will be the parent (super) for Task environment
        '''

        # subscribing and reading current joint state
        self.joint_states_sub = rospy.Subscriber(
            JOINT_STATES_SUBSCRIBER, JointState, self.joints_callback)
        self.joints = JointState()

        # Internal Vars
        reset_controls_bool = False

        # Variables that we give through the constructor.
        # TODO? why do we pass empty namespace?
        self.controllers_list = []
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(FetchEnv, self).__init__(controllers_list=self.controllers_list,
                                       robot_name_space=self.robot_name_space,
                                       reset_controls=reset_controls_bool)

    # ----------------------------------------------------
    # Methods needed by the Gazebo Environment
    # ----------------------------------------------------

    def _check_all_systems_ready(self):
        '''
            Checks that all the sensors, publishers and other simulation systems are
            operational.
        '''

        # similar to cartpole, need to check if all sensors are ready.
        self._check_all_sensors_ready()
        return True

    # ----------------------------------------------------
    # local methods
    # ----------------------------------------------------

    def _check_all_sensors_ready(self):
        self._check_joint_states_ready()

        rospy.logdebug("ALL SENSORS READY")

    def _check_joint_states_ready(self):
        '''
            Checking if the joint state topic that is subscribed to in the constructor is being populated

        '''

        self.joints = None

        while self.joints in None and not rospy.is_shutdown():
            try:
                self.joints = rospy.wait_for_message(
                    JOINT_STATES_SUBSCRIBER, JointState, timeout=1.0)
                rospy.logdebug(
                    "Current /joint_states READY=>" + str(self.joints))
            except:  # pylint: disable=bare-except
                # generic catch all, TODO catch only timeout and some other reasonable exceptions?
                # TODO: avoid magic strings, there are soooo many in these sources
                rospy.logerr(
                    "Current /joint_states not ready yet, retrying for getting joint_states")

        return self.joints

    def joints_callback(self, data):
        '''
            Method that the saubscriber calls - reads published joint state. 
            Just store it so it can be resused
        '''
        self.joints = data

    # this is unused - or will it be used in the task env?
    def get_joints(self):
        return self.joints

    def set_trajectory_ee(self, action):
        '''
            Sets the enf effector position and orientation
        '''

        ee_pose = geometry_msgs.msg.Pose()
        ee_pose.position.x = action[0]
        ee_pose.position.y = action[1]
        ee_pose.position.z = action[2]
        ee_pose.orientation.x = 0.0
        ee_pose.orientation.y = 0.0
        ee_pose.orientation.z = 0.0
        ee_pose.orientation.w = 1.0
        self.fetch_commander_obj.move_ee_to_pose(ee_pose)

        return True

    def set_trajectory_joints(self, initial_qpos):
        '''
            Helper function.
            Wraps an action vector of joint angles into a JointTrajectory message.
            The velocities, accelerations, and effort do not control the arm motion
        '''

        position = [None] * 7
        position[0] = initial_qpos["joint0"]
        position[1] = initial_qpos["joint1"]
        position[2] = initial_qpos["joint2"]
        position[3] = initial_qpos["joint3"]
        position[4] = initial_qpos["joint4"]
        position[5] = initial_qpos["joint5"]
        position[6] = initial_qpos["joint6"]
        #position = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        try:
            self.fetch_commander_obj.move_joints_traj(position)
            result = True
        except Exception as ex:  # pylint: disable=broad-except
            print(ex)
            result = False

        return result

    def get_ee_pose(self):

        gripper_pose = self.fetch_commander_obj.get_ee_pose()

        return gripper_pose

    def get_ee_rpy(self):

        gripper_rpy = self.fetch_commander_obj.get_ee_rpy()

        return gripper_rpy

    # Methods that the Task Environment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        '''
            Inits variables needed to be initialised each time we reset at the start
            of an episode.
        '''
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        '''
            Calculates the reward to give based on the observations given.
        '''
        raise NotImplementedError()

    def _set_action(self, action):
        '''
            Applies the given action to the simulation.
        '''
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        '''
            Checks if episode done based on observations given.
        '''
        raise NotImplementedError()
    