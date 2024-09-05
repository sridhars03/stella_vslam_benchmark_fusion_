#imports
import numpy as np
import pandas as pd

#TODO put it in utils.py and import here
def read_trajectory(file_path):
    # Read the trajectory file, ensuring timestamps are read as floats
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, dtype={0: float})
    data.columns = ['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
    return data

def find_nearest_unique_timestamps(reference_timestamps, target_timestamps):
    #array to store the indices of the nearest unique timestamps
    nearest_indices = []
    used_indices = set()
    
    for ref_time in reference_timestamps:
        #Calc the absolute differences between the reference timestamp and all target timestamps
        differences = np.abs(target_timestamps - ref_time)
        
        #Mask out the already used indices by setting their differences to a large number
        differences[list(used_indices)] = np.inf
        
        #Find the index of the minimum difference
        nearest_index = np.argmin(differences)
        nearest_indices.append(nearest_index)
        
        #Mark this index as used
        used_indices.add(nearest_index)
    
    return nearest_indices


#TODO get as parse arguments - reading the traj files
trajectory_openvslam = read_trajectory('/home/sridhar03/Downloads/new_traj_sync/mh05/synchronized_trajectory_openvslam.txt')
trajectory_gt = read_trajectory('/home/sridhar03/Downloads/new_traj_sync/mh05/data.tum')

#TODO should include generalizing line?
# #######################new change to generalize to 10 digits
# trajectory_orb['timestamp'] = trajectory_orb['timestamp'].apply(lambda x: x / 1e9 if len(str(int(x))) > 10 else x)

#TODO put it in utils.py and import here
# Extract timestamps
timestamps_gt = trajectory_gt['timestamp'].values
timestamps_openvslam = trajectory_openvslam['timestamp'].values

#for now - its for two SLAM outputs
# Find nearest timestamps in OpenVSLAM for ORB-SLAM3 timestamps
nearest_indices = find_nearest_unique_timestamps(timestamps_openvslam,timestamps_gt)

synchronized_gt = trajectory_gt.iloc[nearest_indices].reset_index(drop=True)

#TODO get as parse arguments or save where its run
#Save the synced trajectories
synchronized_gt.to_csv('/home/sridhar03/Downloads/new_traj_sync/mh05/synchronized_gt.txt', sep=' ',header=False,index=False)
print("**************")
