#!/usr/bin/python3

import numpy as np
import rospy
from wam_msgs.msg import RTJointPos, RTJointVel, Gravity
from sensor_msgs.msg import JointState
from wam_srvs.srv import JointMove
import json
import glob
from wam_srvs.srv import Hold
from numpy import linalg as LA
import time
import math
import argparse
import os


class WAM(object):
    """abstract WAM arm definitions"""
    def __init__(self):
        self.pos = []
        self.vel = []
        self.joint_state_data = []
        self.joint_gravity_data = []
        self.joint_angle_bound = np.array([[-2.0, 2.0], [-1.8, 1.8], [-2.0, 2.0], [-0.7, 1.7], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]) # the reference angle is the zero pose.
        self.num_joints = 7
        self.joint = rospy.ServiceProxy('/wam/hold_joint_pos', Hold) 
        self.collect = False
        self._init_joint_states_listener()
        self._init_joint_gravity_listener()

        # initialize publisher for jnt_pos_cmd and jnt_vel_cmd
        self.jnt_vel_pub = rospy.Publisher('/wam/jnt_vel_cmd', RTJointVel, queue_size=1)
        self.jnt_pos_pub = rospy.Publisher('/wam/jnt_pos_cmd', RTJointPos, queue_size=1)

        self.pos_home = self._read_home_pos()
        print("home pos received:", self.pos_home)
        

    def _init_joint_states_listener(self):
        """set up joint states listener from WAM control computer"""
        rospy.Subscriber('/wam/joint_states', JointState, self._cb_joint_state)
        # rospy.spin() # i am not sure whether we should use it here or not
    
    def _init_joint_gravity_listener(self):
        rospy.Subscriber('/wam/gravity', Gravity, self._cb_joint_gravity)

    def _cb_joint_state(self, data : JointState):
        self.pos = np.array(data.position)
        self.vel = np.array(data.velocity)
        # print(data)
        # print(data.header.stamp.nsecs*1e-9)
        joint_state = {'time': data.header.stamp.secs+data.header.stamp.nsecs*1e-9,
                'position' : data.position,
                'velocity' : data.velocity,
                'effort' : data.effort}
        if self.collect:
            self.joint_state_data.append(joint_state)
    
    def _cb_joint_gravity(self, data : Gravity):
        self.gravity = np.array(data.data)
        joint_gravity = {'gravity': data.data}
        if self.collect:
            self.joint_gravity_data.append(joint_gravity)

    def _wait_for_joint_states(self):
        while len(self.pos) == 0:
            rospy.sleep(0.001)

    def _wait_for_joint_gravity(self):
        while len(self.gravity) == 0:
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
        print("position: ", pos_goal)
        print("position applied")

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

    def joint_traj_eval(self, t, f=0.05, A=0.3):
        f *= 2 * math.pi
        q = A * np.array([np.sin(f*t), np.sin(f*t), np.sin(f*t), np.sin(f*t), 0, 0, 0])
        qdot = A * f * np.array([np.cos(f*t), np.cos(f*t), np.cos(f*t), np.cos(f*t), 0, 0, 0])
        return q, qdot

    def joint_traj_excite(self, t, T=10):
        L = 5
        
        x = [0.707990384021940, -0.0150146020506483,    0.000280191811242826,   -0.150424710183653, -0.542831263598882, 0.0285165790096300, -0.0379420078788927,    -0.000370896589483716,  -0.126234977048364, 0.110684006942012,  0.0679857584856881, -0.000160746280917686,  3.59730863728628e-05,   -0.608666198402881, 0.540805213111737,  -0.0433137281308812,    -0.000693832290851609,  -0.000400179294006102,  0.466199557165230,  -0.363779259613263]
        
        A2 = x[0:L]
        B2 = x[L:2*L]
        A4= x[2*L:3*L]
        B4 = x[3*L:4*L]
        omega = 2 * math.pi/T

        theta2=0
        theta4=0
        thetad2=0
        thetad4=0
        thetadd2=0
        thetadd4=0

        for i in range(L):
            theta2 = theta2 + (A2[i] * np.sin(omega*(i+1)*t) - B2[i] * np.cos(omega*(i+1)*t))/(omega*(i+1))
            theta4 = theta4 + (A4[i] * np.sin(omega*(i+1)*t) - B4[i] * np.cos(omega*(i+1)*t))/(omega*(i+1))

            thetad2 = thetad2 + (A2[i] * np.cos(omega*(i+1)*t) + B2[i] * np.sin(omega*(i+1)*t))
            thetad4 = thetad4 + (A4[i] * np.cos(omega*(i+1)*t) + B4[i] * np.sin(omega*(i+1)*t))

            thetadd2 = thetadd2 + (-A2[i] * np.sin(omega*(i+1)*t) + B2[i] * np.cos(omega*(i+1)*t)) * (omega*(i+1))
            thetadd4 = thetadd4 + (-A4[i] * np.sin(omega*(i+1)*t) + B4[i] * np.cos(omega*(i+1)*t)) * (omega*(i+1))
        
        factor = 0.5
        q = np.array([0, theta2, 0, theta4, 0, 0, 0]) * factor
        qdot = np.array([0, thetad2, 0, thetad4, 0, 0, 0]) * factor
        qdotdot = np.array([0, thetadd2, 0, thetadd4, 0, 0, 0]) * factor

        return q, qdot

    def write_data(self, machine):
        home_dir = os.path.expanduser("~")
        base_dir = os.path.join(home_dir, "amir", "data", f"{machine}_data")
        os.makedirs(base_dir, exist_ok=True)

        data_id = len(glob.glob(os.path.join(base_dir, 'state_*.json')))
        with open(os.path.join(base_dir, f"state_{data_id}.json"), "w") as out_file_state:
            json.dump(self.robot.joint_state_data, out_file_state)
        with open(os.path.join(base_dir, f"gravity_{data_id}.json"), "w") as out_file_gravity:
            json.dump(self.robot.joint_gravity_data, out_file_gravity)

        print(f'trajectory collected and saved in {base_dir}')
        self.robot.joint_state_data = []
        self.robot.joint_gravity_data = []

    def collect_dynamics(self, machine):

        self.robot.go_zero()
        rospy.sleep(2)
        self.robot.joint(False)

        initial_angle, initial_velocity = self.joint_traj_excite(0.0)
        self.robot.joint_move(initial_angle)
        rospy.sleep(2)
        self.robot.joint(False)

        start_time = time.time()
        self.robot.collect = True
        while True:
            t = time.time() - start_time
            joint_angle, joint_velocity = self.joint_traj_excite(t)
            self.robot.joint_vel_cmd(joint_velocity)
            if robot.check_joint_bound() or t > self.MAX_TIME:
                break
            self.rate.sleep()
        print('movement done.')
        self.robot.collect = False
        self.robot.stop()
        rospy.sleep(2)
        self.robot.go_home()
        rospy.sleep(2)
        self.robot.joint(False)
        
        self.write_data(machine)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', choices=['slax', 'zeus'], required=True, help="Machine name: slax or zeus")
    args = parser.parse_args()

    rospy.init_node("data_collection_node")
    
    robot = WAM()
    recorder = DataRecorder(robot, 7)

    # Pass the machine type to collect_dynamics
    recorder.collect_dynamics(machine=args.machine)

    rospy.spin()  # Optional if you're using asynchronous subscribers

