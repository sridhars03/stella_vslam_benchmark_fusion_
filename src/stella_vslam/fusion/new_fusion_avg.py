import numpy as np
import pandas as pd

# Placeholder ATE values for ORB-SLAM3 and OpenVSLAM
ate_orb = 0.147318  # Replace with actual ATE for ORB-SLAM3
ate_openvslam = 0.049798  # Replace with actual ATE for OpenVSLAM

# Calculate weights based on ATE
weight_orb = 1 / ate_orb
weight_openvslam = 1 / ate_openvslam

# Normalize weights
total_weight = weight_orb + weight_openvslam
normalized_weight_orb = weight_orb / total_weight
normalized_weight_openvslam = weight_openvslam / total_weight
print(normalized_weight_orb, normalized_weight_openvslam)

# Read the synchronized trajectory files
def read_synchronized_trajectory(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    return data

# Replace 'synchronized_trajectory_orb.txt' and 'synchronized_trajectory_openvslam.txt' with your actual file paths
trajectory_orb = read_synchronized_trajectory('/home/sridhar03/Downloads/new_traj_sync/mh05/synchronized_trajectory_orb.txt')
trajectory_openvslam = read_synchronized_trajectory('/home/sridhar03/Downloads/new_traj_sync/mh05/synchronized_trajectory_openvslam.txt')

# Extract the trajectory data (excluding the timestamp)
trajectory_orb_data = trajectory_orb.iloc[:, 1:].values
trajectory_openvslam_data = trajectory_openvslam.iloc[:, 1:].values

# Extract the timestamps from OpenVSLAM trajectory
timestamps_openvslam = trajectory_openvslam.iloc[:, 0].values

# Ensure the number of samples match
assert trajectory_orb_data.shape == trajectory_openvslam_data.shape, "Trajectory shapes do not match"

# Compute the fused trajectory using weighted average
fused_trajectory = (normalized_weight_orb * trajectory_orb_data + normalized_weight_openvslam * trajectory_openvslam_data)

# Append the timestamps to the fused trajectory
fused_trajectory_with_timestamps = np.column_stack((timestamps_openvslam, fused_trajectory))

# Optionally, save the fused trajectory to a new file
fused_trajectory_df = pd.DataFrame(fused_trajectory_with_timestamps)
fused_trajectory_df.to_csv('/home/sridhar03/Downloads/new_traj_sync/mh05/fused_trajectory.txt', sep=' ', header=False, index=False)
