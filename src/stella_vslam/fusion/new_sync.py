import numpy as np
import pandas as pd

def read_trajectory(file_path):
    # Read the trajectory file, ensuring timestamps are read as floats
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, dtype={0: float})
    data.columns = ['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
    return data

def find_nearest_unique_timestamps(reference_timestamps, target_timestamps):
    # Initialize an array to store the indices of the nearest unique timestamps
    nearest_indices = []
    used_indices = set()
    
    for ref_time in reference_timestamps:
        # Calculate the absolute differences between the reference timestamp and all target timestamps
        differences = np.abs(target_timestamps - ref_time)
        
        # Mask out the already used indices by setting their differences to a large number
        differences[list(used_indices)] = np.inf
        
        # Find the index of the minimum difference
        nearest_index = np.argmin(differences)
        nearest_indices.append(nearest_index)
        
        # Mark this index as used
        used_indices.add(nearest_index)
    
    return nearest_indices

trajectory_orb = read_trajectory('/home/sridhar03/Downloads/new_traj_sync/v101/trajectory_orb.txt')
trajectory_openvslam = read_trajectory('/home/sridhar03/Downloads/new_traj_sync/v101/frame_trajectory.txt')

#######################new change to generalize to 10 digits
trajectory_orb['timestamp'] = trajectory_orb['timestamp'].apply(lambda x: x / 1e9 if len(str(int(x))) > 10 else x)

# Extract timestamps
timestamps_orb = trajectory_orb['timestamp'].values
timestamps_openvslam = trajectory_openvslam['timestamp'].values

# Find nearest timestamps in OpenVSLAM for ORB-SLAM3 timestamps
nearest_indices = find_nearest_unique_timestamps(timestamps_orb, timestamps_openvslam)

synchronized_openvslam = trajectory_openvslam.iloc[nearest_indices].reset_index(drop=True)
synchronized_orb = trajectory_orb.reset_index(drop=True)

# Save it
synchronized_orb.to_csv('/home/sridhar03/Downloads/new_traj_sync/mh05/synchronized_trajectory_orb.txt', sep=' ', header=False, index=False)
synchronized_openvslam.to_csv('/home/sridhar03/Downloads/new_traj_sync/mh05/synchronized_trajectory_openvslam.txt', sep=' ', header=False, index=False)
