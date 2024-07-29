from tokenize import group
from zipfile import is_zipfile
from config import RAW_DATASET_ROOT_FOLDER

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from abc import *
from pathlib import Path
import pickle

class Dataset(metaclass=ABCMeta):
    def __init__(self,args):
        self.args=args
        self.dataset_code=args.dataset_code
        self.min_rating=args.min_rating
        self.min_uc=args.min_uc
        self.min_sc=args.min_sc
        self.split=args.split
        assert self.min_uc>=2,'Need at least 2 ratings per user for validation and test'
 
    def load_dataset(self):
        dataset_path=self._get_preprocessed_dataset_path()
        dataset=pickle.load(dataset_path.open('rb'))
        return dataset

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}_min_rating{}-min_uc{}-min_sc{}-split{}' \
            .format(self.dataset_code, self.min_rating, self.min_uc, self.min_sc, self.split)
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')