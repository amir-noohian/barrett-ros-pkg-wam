#!/usr/bin/python3

import numpy as np
import rospy
from wam_msgs.msg import RTJointPos, RTJointVel, WAMJointState
from wam_srvs.srv import JointMove
import json
import glob
from wam_srvs.srv import Hold
from numpy import linalg as LA
import time
import math


class WAM(object):
    """abstract WAM arm definitions"""
    def __init__(self):
        self.pos = []
        self.vel = []
        self.joint_state_data = []
        self.joint_angle_bound = np.array([[-2.0, 2.0], [-1.8, 1.8], [-2.0, 2.0], [-0.7, 0.7], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]) # the reference angle is the zero pose.
        self.num_joints = 7
        self.joint = rospy.ServiceProxy('/wam/hold_joint_pos', Hold) 
        self.collect = False
        self._init_joint_states_listener()

        # initialize publisher for jnt_pos_cmd and jnt_vel_cmd
        self.jnt_vel_pub = rospy.Publisher('/wam/jnt_vel_cmd', RTJointVel, queue_size=1)
        self.jnt_pos_pub = rospy.Publisher('/wam/jnt_pos_cmd', RTJointPos, queue_size=1)

        self.pos_home = self._read_home_pos()
        print("home pos received:", self.pos_home)
        

    def _init_joint_states_listener(self):
        """set up joint states listener from WAM control computer"""
        rospy.Subscriber('/wam/joint_states', WAMJointState, self._cb_joint_state)
        # rospy.spin() # i am not sure whether we should use it here or not

    def _cb_joint_state(self, data : WAMJointState):
        self.pos = np.array(data.position)
        self.vel = np.array(data.velocity)
        # print(data)
        # print(data.header.stamp.nsecs*1e-9)
        joint_state = {'time': data.header.stamp.secs+data.header.stamp.nsecs*1e-9,
                'position' : data.position,
                'velocity' : data.velocity,
                'effort' : data.effort,
                'gravity' : data.gravity}
        if self.collect:
            self.joint_state_data.append(joint_state)
            self.collect = False

    def _wait_for_joint_states(self):
        while len(self.pos) == 0:
            rospy.sleep(0.001)
    
    def _read_home_pos(self):
        self._wait_for_joint_states()
        return self.pos.copy()
    
    def joint_move(self, pos_goal: np.ndarray):
        """Move WAM to a desired position.
        q is a numpy array of length DoF that specifies the joint angles
        """
        # Communicate with /wam/joint_move service on control computer
        rospy.wait_for_service('/wam/joint_move')
        try:
            joint_move_service = rospy.ServiceProxy('/wam/joint_move', JointMove)
            joint_move_service(pos_goal)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
    
    def joint_pos_cmd(self, pos_goal: np.ndarray):
                     
        msg = RTJointPos()
        msg.joints = pos_goal
        msg.rate_limits = np.array([500.0]*7)
        self.jnt_pos_pub.publish(msg)

    def joint_vel_cmd(self, vel_goal: np.ndarray):
                   
        msg = RTJointVel()
        msg.velocities = vel_goal
        self.jnt_vel_pub.publish(msg)
        print("velocity applied")

    def go_home(self):
        self._wait_for_joint_states()
        self.joint_move(self.pos_home)
    
    def go_zero(self):
        self._wait_for_joint_states()
        self.joint_move([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def stop(self):
        self.joint_vel_cmd([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def check_joint_bound(self):
        current_joint_angle = self.pos

        for i in range(self.num_joints):
            if current_joint_angle[i] > 0.05 + self.joint_angle_bound[i][1]:
                print('Joint {} out of bound.'.format(i+1))
                return True
            elif current_joint_angle[i] < -0.05 + self.joint_angle_bound[i][0]:
                print('JOint {} out of bound.'.format(i+1))
                return True
            
        return False

class DataRecorder(object):
    def __init__(self, robot, joint_num):
        self.robot = robot
        self.joint_num = joint_num
        self.robot._wait_for_joint_states()
        self.MAX_TIME = 10
        self.control_frequency = 500
        self.rate = rospy.Rate(self.control_frequency)

    def joint_traj(self, t, f=0.1, A=[1.8, 0.6]):
        f *= 2 * math.pi
        q = np.array([0, A[0] * np.sin(f*t), 0, A[1] * np.cos(f*t), 0, 0, 0])
        # qdot = f * np.array([0, A[0] * np.cos(f*t), 0, A[1] * np.cos(f*t), 0, 0, 0])
        return q

    # def joint_traj1(self, t, T=10):
    #     L = 5
        
    #     x = [0.707990384021940, -0.0150146020506483,    0.000280191811242826,   -0.150424710183653, -0.542831263598882, 0.0285165790096300, -0.0379420078788927,    -0.000370896589483716,  -0.126234977048364, 0.110684006942012,  0.0679857584856881, -0.000160746280917686,  3.59730863728628e-05,   -0.608666198402881, 0.540805213111737,  -0.0433137281308812,    -0.000693832290851609,  -0.000400179294006102,  0.466199557165230,  -0.363779259613263]
        
    #     A2 = x[0:L]
    #     B2 = x[L:2*L]
    #     A4= x[2*L:3*L]
    #     B4 = x[3*L:4*L]
    #     omega = 2 * math.pi/T

    #     theta2=0
    #     theta4=0
    #     thetad2=0
    #     thetad4=0
    #     thetadd2=0
    #     thetadd4=0

    #     for i in range(L):
    #         theta2 = theta2 + (A2[i] * np.sin(omega*(i+1)*t) - B2[i] * np.cos(omega*(i+1)*t))/(omega*(i+1))
    #         theta4 = theta4 + (A4[i] * np.sin(omega*(i+1)*t) - B4[i] * np.cos(omega*(i+1)*t))/(omega*(i+1))

    #         thetad2 = thetad2 + (A2[i] * np.cos(omega*(i+1)*t) + B2[i] * np.sin(omega*(i+1)*t))
    #         thetad4 = thetad4 + (A4[i] * np.cos(omega*(i+1)*t) + B4[i] * np.sin(omega*(i+1)*t))

    #         thetadd2 = thetadd2 + (-A2[i] * np.sin(omega*(i+1)*t) + B2[i] * np.cos(omega*(i+1)*t)) * (omega*(i+1))
    #         thetadd4 = thetadd4 + (-A4[i] * np.sin(omega*(i+1)*t) + B4[i] * np.cos(omega*(i+1)*t)) * (omega*(i+1))
        
    #     factor = 1.0
    #     q = np.array([0, theta2, 0, theta4, 0, 0, 0]) * factor
    #     qdot = np.array([0, thetad2, 0, thetad4, 0, 0, 0]) * factor
    #     qdotdot = np.array([0, thetadd2, 0, thetadd4, 0, 0, 0]) * factor

    #     return q, qdot


    def write_data(self, name = "dynamics"):
        data_id = len(glob.glob('/home/wam/data/gravity_data/{}_*.json'.format(name)))
        out_file = open("/home/wam/data/gravity_data/{}_{}.json".format(name, data_id), "w")
        json.dump(self.robot.joint_state_data, out_file)
        out_file.close()
        print('trajectory colleced and saved')
        self.robot.joint_state_data = []

    def collect_dynamics(self):

        self.robot.go_zero()
        rospy.sleep(2)
        # self.robot.joint(False)

        initial_angle = self.joint_traj(0)
        self.robot.joint_move(initial_angle)
        rospy.sleep(2)
        # self.robot.joint(False)
        self.robot.collect = True

        for t in range(20):
            joint_angle = self.joint_traj(t/2)
            self.robot.joint_move(joint_angle)
            rospy.sleep(5)
            # self.robot.joint(False)
            rospy.sleep(10)
            self.robot.collect = True
            if robot.check_joint_bound():
                break

        print('movement done.')
        self.robot.stop()
        rospy.sleep(2)
        self.robot.go_home()
        rospy.sleep(2)
        # self.robot.joint(False)
        
        self.write_data("gravity")


if __name__ == '__main__':
    rospy.init_node("data_collection_node")
    
    robot = WAM()
    recorder = DataRecorder(robot, 7)
    recorder.collect_dynamics()
    rospy.spin() # I am not sure whether we should use it here or not : Faezeh: you need it to make the callbacks running on highest frq as possible!

