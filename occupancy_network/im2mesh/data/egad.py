import os
import logging
from torch.utils import data
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from typing import Dict, List
from im2mesh.data.core import Field

# from custom_metric_utils import visualize_point_cloud, split_by_occupancy

logger = logging.getLogger(__name__)


class EGADDataset(data.Dataset):
    ''' EGAD dataset class.
    '''

    def __init__(self, dataset_folder: str, 
                       fields: Dict[str, Field], 
                       split: str, 
                       categories: List[str],
                       rotation_folder: str = None,
                       rotation_augment: str = 'aligned'):
        ''' Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (Dict[str, Field]): dictionary of fields
            split (str): which split is used
            categories (List[str]): list of categories to use. Should not be None
            rotation_folder (str): Directory to save rotation files
            rotation_augment (str): Rotation augmentation method
        '''
        # Attributes
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.rotation_folder = rotation_folder
        self.rotation_augment = rotation_augment

        # Category must be specified for egad dataset
        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [c for c in categories
                          if os.path.isdir(os.path.join(dataset_folder, c))]

        # Read metadata file
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.load(f, Loader=yaml.FullLoader)
        
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
                for m in models_c if m != ""
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
        if self.rotation_folder is not None and self.rotation_augment != 'aligned':
            rotation_path = os.path.join(self.rotation_folder, category, model, 'random_rotations.npz')
            rotation = np.load(rotation_path)
        data = {}

        for field_name, field in self.fields.items():
            field_data = field.load(model_path, idx, c_idx)
            
            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if v.shape[-1] == 3:
                        if self.rotation_augment in ['so3', 'pca']:
                            r = rotation['so3']
                            v = np.matmul(v, r)
                        elif self.rotation_augment == 'z':
                            r = R.from_euler('z', rotation['z'], degrees=True).as_matrix()
                            v = np.matmul(v, r).astype(np.float32)

                    if k is None:
                        data[field_name] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data

        # Debug
        # in_points, out_points = split_by_occupancy(data["points"], data["points.occ"])
        # visualize_point_cloud([in_points, out_points], save_path="./debug.png", save=True, lower_lim=-1, upper_lim=1)

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