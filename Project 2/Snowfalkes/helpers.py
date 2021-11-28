# Import the used libraries
import pandas as pd
import os


class MASCDB_classes:
    
    def __init__(self, dir_path):

        # Get the paths for the hydro training sets
        hydro_cam0_path = os.path.join(dir_path, "hydro_trainingset/hydro_trainingset_cam0.pkl")
        hydro_cam1_path = os.path.join(dir_path, "hydro_trainingset/hydro_trainingset_cam1.pkl")
        hydro_cam2_path = os.path.join(dir_path, "hydro_trainingset/hydro_trainingset_cam2.pkl")

        # Get the paths for the riming training sets
        riming_cam0_path = os.path.join(dir_path, "riming_trainingset/riming_trainingset_cam0.pkl")
        riming_cam1_path = os.path.join(dir_path, "riming_trainingset/riming_trainingset_cam1.pkl")
        riming_cam2_path = os.path.join(dir_path, "riming_trainingset/riming_trainingset_cam2.pkl")

        # Read the dataframes for hydro classes
        self.hydro_cam0 = pd.read_pickle(hydro_cam0_path)
        self.hydro_cam1 = pd.read_pickle(hydro_cam1_path)
        self.hydro_cam2 = pd.read_pickle(hydro_cam2_path)

        # Read the dataframes for riming classes
        self.riming_cam0 = pd.read_pickle(riming_cam0_path)
        self.riming_cam1 = pd.read_pickle(riming_cam1_path)
        self.riming_cam2 = pd.read_pickle(riming_cam2_path)


    def get_class_cam(self, classifier, cam):
        if classifier == "riming":
            if cam == 0:
                class_cam = self.riming_cam0
            elif cam == 1:
                class_cam = self.riming_cam1
            elif cam == 2:
                class_cam = self.riming_cam2
            else:
                raise ValueError("Wrong cam, it should be equal to: 0, 1 or 2.")
        elif classifier == "hydro":
            if cam == 0:
                class_cam = self.hydro_cam0
            elif cam == 1:
                class_cam = self.hydro_cam1
            elif cam == 2:
                class_cam = self.hydro_cam2
            else:
                raise ValueError("Wrong cam, it should be equal to: 0, 1 or 2.")
        else:
            raise ValueError("Wrong classifier, it should be either: 'riming' or 'hydro'.")
        return class_cam


    def find_flake_id(self, classifier, cam, cam_data):
        # Get the classifier cam
        class_cam = self.get_class_cam(classifier, cam)

        # Get the sub data frame of cam_data containing flake_id of class_cam
        sub_cam_data = cam_data[cam_data['flake_id'].isin(class_cam['flake_id'])]

        # Return the result
        return sub_cam_data

    def get_input(self, classifier, data):
        # Create the input
        # Add for each cam
        # cam0
        input = self.find_flake_id(classifier, 0, data.cam0)
        
        #cam1
        input = pd.concat([input, self.find_flake_id(classifier, 1, data.cam1)])
        
        #cam2
        input = pd.concat([input, self.find_flake_id(classifier, 1, data.cam2)])
       
        return input



def numpy_helpers(df, cols):
    """
        Get a numpy array out of the dataframe df.

    Args:
        df (DataFrame): Considered data frame.
        cols (string): The name of the columns that we want in numpy array format.

    Returns:
        nympay array: numpy array of the columns from our dataframe df.
    """
    np_array = df[cols].to_numpy()
    return np_array


