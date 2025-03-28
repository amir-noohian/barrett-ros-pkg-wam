#!/usr/bin/python3

import numpy as np
import rospy
from wam_msgs.msg import RTJointPos, RTJointVel, Gravity, CartForce, DynamicCartForce
from sensor_msgs.msg import JointState
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
        self.data = []
        self.gravity = []
        self.cart_force = []
        self.dynamic_cart_force = []
        # self.joint_gravity_data = []
        self.joint_angle_bound = np.array([[-2.0, 2.0], [-1.8, 1.8], [-2.0, 2.0], [-0.7, 2.7], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]) # the reference angle is the zero pose.
        self.num_joints = 7
        self.joint = rospy.ServiceProxy('/wam/hold_joint_pos', Hold) 
        self.collect = False
        self._init_listeners()
        # self._init_joint_states_listener()
        # self._init_joint_gravity_listener()

        # initialize publisher for jnt_pos_cmd and jnt_vel_cmd
        self.jnt_vel_pub = rospy.Publisher('/wam/jnt_vel_cmd', RTJointVel, queue_size=1)
        self.jnt_pos_pub = rospy.Publisher('/wam/jnt_pos_cmd', RTJointPos, queue_size=1)

        self.pos_home = self._read_home_pos()
        print("home pos received:", self.pos_home)
        
    def _init_listeners(self):
        rospy.Subscriber('/wam/joint_states', JointState, self._cb_joint_state)
        rospy.Subscriber('/wam/gravity', Gravity, self._cb_joint_gravity)
        rospy.Subscriber('/wam/cart_force', CartForce, self._cb_cart_force)
        rospy.Subscriber('/wam/dynamic_cart_force', DynamicCartForce, self._cb_dynamic_cart_force)
    
    '''
    def _init_joint_states_listener(self):
        """set up joint states listener from WAM control computer"""
        rospy.Subscriber('/wam/joint_states', JointState, self._cb_joint_state)
        # rospy.spin() # i am not sure whether we should use it here or not
    
    def _init_joint_gravity_listener(self):
        rospy.Subscriber('/wam/gravity', Gravity, self._cb_joint_gravity)
    '''

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
            self.data.append(joint_state)
    
    def _cb_joint_gravity(self, data : Gravity):
        self.gravity = np.array(data.data)
        # joint_gravity = {'gravity': data.data}
        if self.collect:
            self.data[-1]['gravity'] = data.data
    
    def _cb_cart_force(self, data : CartForce):
        self.cart_force = np.array(data.data)
        if self.collect:
            self.data[-1]['cart force'] = data.data

    def _cb_dynamic_cart_force(self, data : DynamicCartForce):
        self.dynamic_cart_force = np.array(data.data)
        if self.collect:
            self.data[-1]['dynamic cart force'] = data.data

    def _wait_for_joint_states(self):
        while len(self.pos) == 0:
            rospy.sleep(0.001)

    def _wait_for_joint_gravity(self):
        while len(self.gravity) == 0:
            rospy.sleep(0.001)
    
    def _wait_for_cart_force(self):
        while len(self.cart_force) == 0:
            rospy.sleep(0.001)
    
    def _wait_for_dynamic_cart_force(self):
        while len(self.dynamic_cart_force) == 0:
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
        self.robot._wait_for_joint_gravity()
        self.robot._wait_for_cart_force()
        self.robot._wait_for_dynamic_cart_force()
        self.MAX_TIME = 10
        self.control_frequency = 500
        self.rate = rospy.Rate(self.control_frequency)

    def joint_traj_eval(self, t, f=0.1, A=0.5):
        f *= 2 * math.pi
        q = A * np.array([np.sin(f*t), np.sin(f*t), np.sin(f*t), np.sin(f*t), 0, 0, 0])
        qdot = A * f * np.array([np.cos(f*t), np.cos(f*t), np.cos(f*t), np.cos(f*t), 0, 0, 0])
        return q, qdot

    def joint_traj_evalforce(self, t, f=0.2):
        f *= 2 * math.pi
        q = np.array([
            0.5 * np.sin(f * t),  # First joint
            0.2 * np.sin(f * t) + 0.7,  # Second joint
            0.0,                  # Third joint
            0.4 * np.sin(f * t) + 1.7,  # Fourth joint
            0.0, 0.0, 0.0         # Other joints
        ])
        
        qdot = f * np.array([
            0.5 * np.cos(f * t),  # First joint
            0.2 * np.cos(f * t),  # Second joint
            0.0,                  # Third joint
            0.4 * np.cos(f * t),  # Fourth joint
            0.0, 0.0, 0.0         # Other joints
        ])

        return q, qdot


    def joint_traj_excite(self, t, T=10):
        T = self.MAX_TIME
        omega = 2 * math.pi / T

        x = [
            0.142931485845318, 0.403056793608812, -0.0951265598500125, 0.0279672132702271,
            -0.478828932874835, 0.455978106817742, -1.05974697416844, 0.112804088134919,
            -0.185668061378723, 0.413555164727720, 0.530203919904277, -0.660070426818711,
            0.0894194285768867, 0.0468877045347822, -0.00644062611615576, -0.371509518883150,
            0.922807985135430, 0.0314035658425545, -0.418107145858326, 0.0208222867704066,
            1.20600325575271, -1.24481714047118, 0.0276291510531659, -0.0721859307327485,
            0.0833706640095264, 0.236543881042100, -0.570373040751604, 0.0431757358312745,
            0.0362636392347866, 0.125924087204133, -0.0549509789949885, -0.0471098863580389,
            0.199259591416666, 0.402023372647745, -0.499222098708336, 0.0700455463292073,
            0.144867295002918, -0.175010341580362, -1.00836054274791, 0.839738611702069
        ]

        L = 5

        A1 = x[0:L]
        B1 = x[L:2*L]
        A2 = x[2*L:3*L]
        B2 = x[3*L:4*L]
        A3 = x[4*L:5*L]
        B3 = x[5*L:6*L]
        A4 = x[6*L:7*L]
        B4 = x[7*L:8*L]

        theta1 = np.zeros_like(t)
        theta2 = np.zeros_like(t)
        theta3 = np.zeros_like(t)
        theta4 = np.zeros_like(t)

        thetad1 = np.zeros_like(t)
        thetad2 = np.zeros_like(t)
        thetad3 = np.zeros_like(t)
        thetad4 = np.zeros_like(t)

        thetadd1 = np.zeros_like(t)
        thetadd2 = np.zeros_like(t)
        thetadd3 = np.zeros_like(t)
        thetadd4 = np.zeros_like(t)

        for i in range(1, L+1):
            theta1 += (A1[i-1] * np.sin(omega * i * t) - B1[i-1] * np.cos(omega * i * t)) / (omega * i)
            theta2 += (A2[i-1] * np.sin(omega * i * t) - B2[i-1] * np.cos(omega * i * t)) / (omega * i)
            theta3 += (A3[i-1] * np.sin(omega * i * t) - B3[i-1] * np.cos(omega * i * t)) / (omega * i)
            theta4 += (A4[i-1] * np.sin(omega * i * t) - B4[i-1] * np.cos(omega * i * t)) / (omega * i)

            thetad1 += A1[i-1] * np.cos(omega * i * t) + B1[i-1] * np.sin(omega * i * t)
            thetad2 += A2[i-1] * np.cos(omega * i * t) + B2[i-1] * np.sin(omega * i * t)
            thetad3 += A3[i-1] * np.cos(omega * i * t) + B3[i-1] * np.sin(omega * i * t)
            thetad4 += A4[i-1] * np.cos(omega * i * t) + B4[i-1] * np.sin(omega * i * t)

            thetadd1 += (-A1[i-1] * np.sin(omega * i * t) + B1[i-1] * np.cos(omega * i * t)) * (omega * i)
            thetadd2 += (-A2[i-1] * np.sin(omega * i * t) + B2[i-1] * np.cos(omega * i * t)) * (omega * i)
            thetadd3 += (-A3[i-1] * np.sin(omega * i * t) + B3[i-1] * np.cos(omega * i * t)) * (omega * i)
            thetadd4 += (-A4[i-1] * np.sin(omega * i * t) + B4[i-1] * np.cos(omega * i * t)) * (omega * i)

        factor = 0.9
        q = np.array([theta1, theta2, theta3*0.7, theta4, 0.0, 0.0, 0.0]) * factor
        qdot = np.array([thetad1, thetad2, thetad3*0.7, thetad4, 0.0, 0.0, 0.0]) * factor
        # qdotdot = np.array([thetadd1, thetadd2, thetadd3*0.7, thetadd4, 0.0, 0.0, 0.0]) * factor

        return q, qdot


    def write_data(self):
        data_id = len(glob.glob('/home/wam/data/slax_data/data_*.json'))
        out_file_data = open("/home/wam/data/slax_data/data_{}.json".format(data_id), "w")
        # out_file_gravity = open("/home/wam/data/slax_data/gravity_{}.json".format(data_id), "w")
        json.dump(self.robot.data, out_file_data)
        # json.dump(self.robot.joint_gravity_data, out_file_gravity)
        out_file_data.close()
        # out_file_gravity.close()
        print('trajectory colleced and saved')
        self.robot.data = []
        # self.robot.joint_gravity_data = []

    def collect_data(self):

        self.robot.go_zero()
        rospy.sleep(2)
        self.robot.joint(False)

        initial_angle, initial_velocity = self.joint_traj_evalforce(0.0)
        self.robot.joint_move(initial_angle)
        rospy.sleep(2)
        self.robot.joint(False)

        start_time = time.time()
        self.robot.collect = True
        while True:
            t = time.time() - start_time
            joint_angle, joint_velocity = self.joint_traj_evalforce(t)
            self.robot.joint_vel_cmd(joint_velocity)
            if robot.check_joint_bound() or t > self.MAX_TIME:
                break
            self.rate.sleep()
        print('movement done.')
        self.robot.collect = False
        self.robot.stop()
        # self.robot.joint_move(self.robot.pos)
        rospy.sleep(2)
        self.robot.go_home()
        rospy.sleep(2)
        self.robot.joint(False)
        
        self.write_data()


if __name__ == '__main__':
    rospy.init_node("data_collection_node")
    
    robot = WAM()
    recorder = DataRecorder(robot, 7)
    recorder.collect_data()
    # rospy.spin()
