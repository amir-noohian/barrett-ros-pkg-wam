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
        self.joint_angle_bound = np.array([[-2.4, 2.4], [-1.8, 1.8], [-2.5, 2.5], [-0.8, 3.0], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]) # the reference angle is the zero pose.
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
        self.MAX_TIME = 60
        self.control_frequency = 500
        self.rate = rospy.Rate(self.control_frequency)

    def joint_traj_eval(self, t, f=0.3, A=0.5):
        f *= 2 * math.pi
        q = A * np.array([np.sin(f*t), np.sin(f*t), np.sin(f*t), np.sin(f*t), 0, 0, 0])
        qdot = A * f * np.array([np.cos(f*t), np.cos(f*t), np.cos(f*t), np.cos(f*t), 0, 0, 0])

        mask = np.array([0, 0, 0, 1, 0, 0, 0])
        q = q * mask
        qdot = qdot * mask

        return q, qdot

    def joint_traj_excite(self, t, omega_f=0.314):
        """
        Compute joint positions and velocities for a 4-DOF trajectory,
        and expand them to 7-DOF by zero-padding the last 3 joints.

        Parameters:
            t           : Time
            omega_f     : Base frequency (default 0.314)
            factor      : Output scaling factor

        Returns:
            q     : (7,) joint positions
            qdot  : (7,) joint velocities
        """
        import numpy as np

        # Full 77-element Fourier coefficient vector
        a_b_q0_full = np.array([
            -0.22453075, -0.06509415,  0.0084825,   0.7463596,   0.04482256,
            0.15292831,  0.17945343,  0.13698774,  0.15005761,  0.40428706,
            -0.06912397,  0.64039236,  0.41406122, -0.05345818,  0.46419411,
            0.27601025,  0.55370932,  0.065848,    0.04071695,  0.41272949,
            -0.17373824,  0.89269347,  0.45028845, -0.26077073,  0.10794414,
            -0.16322598,  0.22051963, -0.13005116,  0.38388683,  0.93214031,
            0.36874565,  0.11058303, -0.13207445,  0.39799543,  0.28705749,
            0.09974301, -0.09950086,  0.77226083, -0.28470714,  0.06468834,
            0.0187677,  -0.05244391, -0.40604485,  0.43133299, -0.57178278,
            -0.12185623,  0.51881781,  0.27173848,  0.15961459, -0.29374854,
            0.2251367,  -0.29917266,  0.09730888,  0.24762081,  0.07477825,
            -0.11031733,  0.08532278,  0.21811307, -0.10056108,  0.56254872,
            -0.02235595,  0.34816564, -0.24721517, -0.1987875,  -0.29918626,
            0.19194917, -0.39180923, -0.60461852,  0.00473514, -0.24540649,
            -0.24181945,  0.03114698, -0.08510632,  1.54641938, -1.67238949,
            -0.17570127,  0.5569734
        ])

        K_full = 7
        K_used = 4
        L = 5

        # Split the full vector
        a = a_b_q0_full[:K_full * L].reshape(K_full, L)
        b = a_b_q0_full[K_full * L:2 * K_full * L].reshape(K_full, L)
        q0_full = a_b_q0_full[2 * K_full * L:]

        omega_f_l = np.arange(1, L + 1) * omega_f

        # Initialize 4-DOF
        q = np.copy(q0_full[:K_used])
        qdot = np.zeros(K_used)

        for k in range(K_used):
            for l in range(L):
                w = omega_f_l[l]
                sin_wt = np.sin(w * t)
                cos_wt = np.cos(w * t)

                q[k] += (a[k, l] * sin_wt - b[k, l] * cos_wt) / w
                qdot[k] += a[k, l] * cos_wt + b[k, l] * sin_wt

        factor = np.array([1,1,1,0.8,1,1,1])

        # Pad with zeros to make 7-DOF
        q_full = np.concatenate([q, np.zeros(3)]) * factor
        qdot_full = np.concatenate([qdot, np.zeros(3)]) * factor

        return q_full, qdot_full



    def write_data(self, machine):
        import os
        import glob

        home_dir = os.path.expanduser("~")
        base_dir = os.path.join(home_dir, "amir", "data", f"{machine}_data")
        os.makedirs(base_dir, exist_ok=True)

        data_id = len(glob.glob(os.path.join(base_dir, 'rbtlog_*.dat')))
        file_path = os.path.join(base_dir, f"rbtlog_{data_id}.dat")

        with open(file_path, "w") as f:
            for entry in self.robot.joint_state_data:
                time = entry['time']
                position = entry['position']
                effort = entry['effort']
                # Combine into one row: [time, pos0, ..., posN, torque0, ..., torqueN]
                line = [time] + list(position) + list(effort)
                f.write(" ".join(map(str, line)) + "\n")

        print(f"Trajectory saved to {file_path}")

        # Clear data for next round
        self.robot.joint_state_data = []

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

