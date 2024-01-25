import h5py

def list_datasets(hdf5_file_path):
    with h5py.File(hdf5_file_path, 'r') as h5_file:
        print("Datasets in the HDF5 file:")
        for name in h5_file:
            print(name)

# Replace with your actual file path
hdf5_file_path = '/home/malek/Documents/CARLA/datasets/CORL2017ImitationLearningData/AgentHuman/SeqTrain/data_03663.h5'
list_datasets(hdf5_file_path)
