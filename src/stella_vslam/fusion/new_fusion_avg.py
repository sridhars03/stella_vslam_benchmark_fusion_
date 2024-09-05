#imports
import numpy as np
import pandas as pd

#TODO get ATE as arguments?
ate_orb = 0.147318  #will be replaced with actual ATE for ORB-SLAM3
ate_openvslam = 0.049798  #will be replaced with actual ATE for OpenVSLAM

# Calc weights based on ATE
weight_orb = 1 / ate_orb
weight_openvslam = 1 / ate_openvslam

# Normalize weights
total_weight = weight_orb + weight_openvslam
normalized_weight_orb = weight_orb / total_weight
normalized_weight_openvslam = weight_openvslam / total_weight
print("Normalized weights of orb:{}, openvslam:{}".format(normalized_weight_orb, normalized_weight_openvslam))

#TODO put it in utils.py and import here
# Read the synchronized trajectory files
def read_synchronized_trajectory(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    return data

#TODO get as parse arguments - reading the traj files
trajectory_orb = read_synchronized_trajectory('/home/sridhar03/Downloads/new_traj_sync/mh05/synchronized_trajectory_orb.txt')
trajectory_openvslam = read_synchronized_trajectory('/home/sridhar03/Downloads/new_traj_sync/mh05/synchronized_trajectory_openvslam.txt')

#Extract only the trajectory data(excluding timestamps)
trajectory_orb_data = trajectory_orb.iloc[:, 1:].values
trajectory_openvslam_data = trajectory_openvslam.iloc[:, 1:].values

#TODO report why/why not using either of timestamps matters?
#Extract the timestamps from OpenVSLAM trajectory
timestamps_openvslam = trajectory_openvslam.iloc[:, 0].values

#Ensure the number of samples match
assert trajectory_orb_data.shape == trajectory_openvslam_data.shape, "Trajectory shapes do not match"

#Compute the fused trajectory using weighted average
fused_trajectory = (normalized_weight_orb * trajectory_orb_data + normalized_weight_openvslam * trajectory_openvslam_data)

#Append the timestamps to the fused trajectory
fused_trajectory_with_timestamps = np.column_stack((timestamps_openvslam, fused_trajectory))

#TODO get as parse arguments or save where its run
#Save the synced trajectories
fused_trajectory_df = pd.DataFrame(fused_trajectory_with_timestamps)
fused_trajectory_df.to_csv('/home/sridhar03/Downloads/new_traj_sync/mh05/fused_trajectory.txt', sep=' ', header=False, index=False)
