import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

def load_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def extract_trajectory(traj_list):
    times = []
    positions = []
    velocities = []

    for item in traj_list:
        times.append(item["time"])
        positions.append(item["position"])
        velocities.append(item["velocity"])

    return np.array(times), np.array(positions), np.array(velocities)

def extract_joint_data(values, joint_index):
    return [v[joint_index] for v in values]

def plot_joint_trajectories(data, num_joints=4):
    ref_time, ref_pos, ref_vel = extract_trajectory(data["reference_trajectory"])


    meas_time = np.array([item["time"] for item in data["joint_states"]])
    meas_time = meas_time - meas_time[0]
    meas_pos = [item["position"] for item in data["joint_states"]]
    meas_vel = [item["velocity"] for item in data["joint_states"]]

    fig, axes = plt.subplots(2, num_joints, figsize=(5*num_joints, 8), sharex='col')
    fig.suptitle("Reference vs Measured Joint Trajectories")

    for j in range(num_joints):
        # Position subplot
        axes[0, j].plot(ref_time, extract_joint_data(ref_pos, j), label="ref", linestyle='--')
        axes[0, j].plot(meas_time, extract_joint_data(meas_pos, j), label="meas")
        axes[0, j].set_title(f"Joint {j+1} Position")
        axes[0, j].set_ylabel("Position [rad]")
        axes[0, j].legend()
        axes[0, j].grid()

        # Velocity subplot
        axes[1, j].plot(ref_time, extract_joint_data(ref_vel, j), label="ref", linestyle='--')
        axes[1, j].plot(meas_time, extract_joint_data(meas_vel, j), label="meas")
        axes[1, j].set_title(f"Joint {j+1} Velocity")
        axes[1, j].set_xlabel("Time [s]")
        axes[1, j].set_ylabel("Velocity [rad/s]")
        axes[1, j].legend()
        axes[1, j].grid()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help="Path to the log_*.json file")
    args = parser.parse_args()

    data = load_data(args.file)
    plot_joint_trajectories(data)
