import numpy as np

def extract_xyz(file_path):
    data = np.loadtxt(file_path) 
    xyz_data = data[:, 1:4]
    return xyz_data

def get_stat(xyz_data):
    mean_values = np.mean(xyz_data, axis=0)  
    min_values = np.min(xyz_data, axis=0)   
    max_values = np.max(xyz_data, axis=0)    
    return mean_values, min_values, max_values

def main():
    file_path = '/home/sridhar03/Downloads/new_traj_sync/mh04/data.tum'

    #reads + extracts x,y,z columns in your trajectory file
    xyz_data=extract_xyz(file_path)

    #get statistics
    mean_xyz,min_xyz,max_xyz=get_stat(xyz_data)

    print(f"Mean(x, y, z): {mean_xyz}")
    print(f"Min(x, y, z): {min_xyz}")
    print(f"Max(x, y, z): {max_xyz}")

if __name__ =="__main__":
    main()