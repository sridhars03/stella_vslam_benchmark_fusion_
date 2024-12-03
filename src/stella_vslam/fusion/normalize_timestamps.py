import numpy as np

def extract_and_normalize_timestamps(file_path):
    """
    >>>>>>normalize timestamps to [0, 1]
    Returns:
        np.ndarray = Normalized timestamps.
    """
    
    data = np.loadtxt(file_path, usecols=0)
    min_val, max_val = data.min(), data.max()
    normalized_timestamps = (data - min_val) / (max_val - min_val)
    
    return normalized_timestamps

#trajectories
openvslam_file = "/home/sridhar03/Downloads/new_traj_sync/mh01/rnnfinal/synchronized_trajectory_openvslam.txt"
orb3_file = "/home/sridhar03/Downloads/new_traj_sync/mh01/rnnfinal/synchronized_trajectory_orb.txt"

#Normalize timestamps
openvslam_timestamps = extract_and_normalize_timestamps(openvslam_file)
orb3_timestamps = extract_and_normalize_timestamps(orb3_file)

#Verify results
# print("OpenVSLAM normalized timestamps:\n", openvslam_timestamps)
# print("ORB-SLAM3 normalized timestamps:\n", orb3_timestamps)




############# SAVING ########
def replace_and_save_normalized_timestamps(input_file, output_file):

    data = np.loadtxt(input_file)

    timestamps = data[:, 0]
    min_val, max_val = timestamps.min(), timestamps.max()
    normalized_timestamps = (timestamps - min_val) / (max_val - min_val)

    data[:,0] = normalized_timestamps

    np.savetxt(output_file, data, fmt='%f', delimiter=' ')
    print(f"Normalized timestamps saved to: {output_file}")

openvslam_input = "/home/sridhar03/Downloads/new_traj_sync/mh02/rnnfinal/synchronized_trajectory_openvslam.txt"
openvslam_output = "/home/sridhar03/Downloads/new_traj_sync/mh02/rnnfinal/normalized_openvslam.txt"
replace_and_save_normalized_timestamps(openvslam_input, openvslam_output)

orb3_input = "/home/sridhar03/Downloads/new_traj_sync/mh02/rnnfinal/synchronized_trajectory_orb.txt"
orb3_output = "/home/sridhar03/Downloads/new_traj_sync/mh02/rnnfinal/normalized_orb3.txt"
replace_and_save_normalized_timestamps(orb3_input, orb3_output)