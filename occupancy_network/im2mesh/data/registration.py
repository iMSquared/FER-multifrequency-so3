import os
import logging
from torch.utils import data
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
from im2mesh.data.fields import PointCloudField


logger = logging.getLogger(__name__)



class Shapes3dDatasetRegistration(data.Dataset):
    ''' 3D Shapes dataset class. '''

    def __init__(self, dataset_folder, fields, split, categories, rotation_folder):
        ''' Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            no_except (bool): no exception
            transform (callable): transformation applied to data points
        '''
        # Attributes
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.rotation_folder = rotation_folder

        # SO3
        self.fields["inputs2"] = deepcopy(self.fields["inputs"])


        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [c for c in categories
                          if os.path.isdir(os.path.join(dataset_folder, c))]

        # Read metadata file
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.load(f, Loader=yaml.FullLoader)
        else:
            self.metadata = {
                c: {'id': c, 'name': 'n/a'} for c in categories
            } 
        
        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx

        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                logger.warning('Category %s does not exist in dataset.' % c)

            split_file = os.path.join(subpath, split + '.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')
            
            self.models += [
                {'category': c, 'model': m}
                for m in models_c
            ]

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.models)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        category = self.models[idx]['category']
        model = self.models[idx]['model']
        c_idx = self.metadata[category]['idx']

        model_path = os.path.join(self.dataset_folder, category, model)
        rotation_path = os.path.join(self.rotation_folder, category, model, 'random_rotations.npz')
        rotation = np.load(rotation_path)

        data = {}

        data["R_gt"] = rotation['so3'].T

        for field_name, field in self.fields.items():
            field_data = field.load(model_path, idx, c_idx)
            
            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if v.shape[-1] == 3:
                            r_so3 = rotation['so3']
                            v_so3 = np.matmul(v, r_so3)
                    if k is None:
                        data[field_name] = v
                        data[f"{field_name}_so3"] = v_so3
                    else:
                        data['%s.%s' % (field_name, k)] = v
                        data['%s.%s_so3' % (field_name, k)] = v_so3
            else:
                data[field_name] = field_data

        return data

    def get_model_dict(self, idx):
        return self.models[idx]

    def test_model_complete(self, category, model):
        ''' Tests if model is complete.

        Args:
            model (str): modelname
        '''
        model_path = os.path.join(self.dataset_folder, category, model)
        files = os.listdir(model_path)
        for field_name, field in self.fields.items():
            if not field.check_complete(files):
                logger.warn('Field "%s" is incomplete: %s'
                            % (field_name, model_path))
                return False

        return True
