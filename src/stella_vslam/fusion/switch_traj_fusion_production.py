#imports
import numpy as np
import pandas as pd


#TODO here include angle also-not just distance
#Function to calculate ATE (between ground truth and estimated trajectory)
def calculate_ate(trajectory_segment, ground_truth_segment):
    differences = trajectory_segment - ground_truth_segment
    squared_errors = np.sum(differences[:, :3]**2, axis=1)
    ate = np.sqrt(np.mean(squared_errors))
    return ate

#TODO put this func in just one py file like "utils" and import in others
# Read the synchronized trajectory files
def read_synchronized_trajectory(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    return data

#TODO test multiple values
sequence_length = 10  # Number of timestamps to compare at each step

#TODO get path as parse arguments
synced_GT = read_synchronized_trajectory('/home/sridhar03/Downloads/new_traj_sync/mh05/synchronized_gt.txt')
trajectory_orb = read_synchronized_trajectory('/home/sridhar03/Downloads/new_traj_sync/mh05/synchronized_trajectory_orb.txt')
trajectory_openvslam = read_synchronized_trajectory('/home/sridhar03/Downloads/new_traj_sync/mh05/synchronized_trajectory_openvslam.txt')

#TODO put this func in just one py file like "utils" and import in others
#Extract the timestamps from OpenVSLAM & ORB trajectory
timestamps_openvslam = trajectory_openvslam.iloc[:, 0].values
timestamps_orb = trajectory_orb.iloc[:, 0].values

#TODO put this func in just one py file like "utils" and import in others
# Extract the trajectory data (excluding the timestamp)
synced_GT_data = synced_GT.iloc[:, 1:].values
trajectory_orb_data = trajectory_orb.iloc[:, 1:].values
trajectory_openvslam_data = trajectory_openvslam.iloc[:, 1:].values

#Initialize a dummy fused trajectory
fused_trajectory = np.zeros_like(trajectory_orb_data)
num_samples=len(fused_trajectory)

#lower the better trajectory 
ate_orb_high_count,ate_high_openvslam_count=0,0

#iterate through the synced traj acc to sequence_length
for i in range(0, num_samples, sequence_length):
    end_index = min(i + sequence_length, num_samples)  #Ensure we don't go out of bounds
    
    # Extract these segments
    orb_segment = trajectory_orb_data[i:end_index]
    openvslam_segment = trajectory_openvslam_data[i:end_index]
    gt_segment = synced_GT_data[i:end_index]
    
    # Calculate ATE
    ate_orb = calculate_ate(orb_segment, gt_segment)
    ate_openvslam = calculate_ate(openvslam_segment, gt_segment)
    
    # Choose the segment with the smaller ATE
    if ate_openvslam >= ate_orb:
        fused_trajectory[i:end_index] = orb_segment
        ate_high_openvslam_count +=1
    else:
        fused_trajectory[i:end_index] = openvslam_segment
        ate_orb_high_count +=1

#will prolly chuck it- mention indifference in report
#timestamp adding logic
#1.see lower ape and add that timestamp with columnstack
#2.for timebeing add openvslam(as im using that to sync GT)
#3.follow colab's timestamp adding logic

#LOGIC #1
if ate_orb_high_count <= ate_high_openvslam_count:
    fused_trajectory_with_timestamps = np.column_stack((timestamps_orb, fused_trajectory))
else:
    fused_trajectory_with_timestamps = np.column_stack((timestamps_openvslam, fused_trajectory))

#LOGIC #2
# fused_trajectory_with_timestamps = np.column_stack((timestamps_openvslam, fused_trajectory))

#LOGIC #3


#TODO get path as argument or save wherever it is run
#Save the fused trajectory to a new file
fused_trajectory_df = pd.DataFrame(fused_trajectory_with_timestamps)
fused_trajectory_df.to_csv('/home/sridhar03/Downloads/new_traj_sync/mh05/fused_trajectory_switch.txt', sep=' ', header=False, index=False)

print("*****Unit test successful")