from typing import List, Optional, Any
import numpy as np
import lmdb
import os

from .base_dataset import BaseDataset

__all__ = ['LMDBDataset']

class LMDBDataset(BaseDataset):
    """Data iterator for ocr datasets including ICDAR15 dataset. 
    The annotaiton format is required to aligned to paddle, which can be done using the `converter.py` script.

    Args:
        is_train: whether the dataset is for training
        data_dir: data root directory for lmdb dataset(s)
        shuffle: Optional, if not given, shuffle = is_train
        transform_pipeline: list of dict, key - transform class name, value - a dict of param config.
                    e.g., [{'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}}]
            -       if None, default transform pipeline for text detection will be taken.
        kwargs: additional info, used in data transformation, possible keys:
            - character_dict_path
            

    Returns:
        data (tuple): Depending on the transform pipeline, __get_item__ returns a tuple for the specified data item. 
        You can specify the `output_columns` arg to order the output data for dataloader.

    Notes: 
        1. Dataset file structure should follow:
            data_dir
            ├── dataset01
                ├── data.mdb
                ├── lock.mdb
            ├── dataset02
                ├── data.mdb
                ├── lock.mdb
            ├── ... 
    """
    def __init__(self, 
            is_train: bool = True, 
            data_dir: str = '', 
            sample_ratio: float = 1.0, 
            shuffle: Optional[bool] = None,
            **kwargs: Any
            ):

        self.data_dir = data_dir
        assert isinstance(shuffle, bool), f'type error of {shuffle}'
        shuffle = shuffle if shuffle is not None else is_train

        self.lmdb_sets = self.load_list_of_hierarchical_lmdb_dataset(data_dir)
        
        if len(self.lmdb_sets) == 0:
            raise ValueError(f"Cannot find any lmdb dataset in `{data_dir}`.")
        
        self.data_idx_order_list = self.get_dataset_idx_orders(sample_ratio, shuffle)
        self._output_columns = ["image", "label"]

    @property
    def output_columns(self):
        return self._output_columns
    
    @output_columns.setter
    def output_columns(self, columns):
        for x in columns:
            self._output_columns.append(x)
        self._output_columns = list(set(self._output_columns))

    def load_list_of_hierarchical_lmdb_dataset(self, data_dir):
        if isinstance(data_dir, str):
            results = self.load_hierarchical_lmdb_dataset(data_dir)
        elif isinstance(data_dir, list):
            results = {}
            for sub_data_dir in data_dir:
                start_idx = len(results)
                lmdb_sets = self.load_hierarchical_lmdb_dataset(sub_data_dir, start_idx)
                results.update(lmdb_sets)
        else:
            results = {}
            
        return results

    def load_hierarchical_lmdb_dataset(self, data_dir, start_idx=0):
        
        lmdb_sets = {}
        dataset_idx = start_idx
        for rootdir, dirs, _ in os.walk(data_dir + '/'):
            if not dirs:
                env = lmdb.open(rootdir, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
                txn = env.begin(write=False)
                data_size = int(txn.get('num-samples'.encode()))
                lmdb_sets[dataset_idx] = {
                    "rootdir":rootdir,
                    "env":env,
                    "txn":txn,
                    "data_size":data_size
                    }
                dataset_idx += 1
        return lmdb_sets

    def get_dataset_idx_orders(self, sample_ratio, shuffle):
        n_lmdbs = len(self.lmdb_sets)
        total_sample_num = 0
        for idx in range(n_lmdbs):
            total_sample_num += self.lmdb_sets[idx]['data_size']
        data_idx_order_list = np.zeros((total_sample_num, 2))
        beg_idx = 0
        for idx in range(n_lmdbs):
            tmp_sample_num = self.lmdb_sets[idx]['data_size']
            end_idx = beg_idx + tmp_sample_num
            data_idx_order_list[beg_idx:end_idx, 0] = idx
            data_idx_order_list[beg_idx:end_idx, 1] \
                = list(range(tmp_sample_num))
            data_idx_order_list[beg_idx:end_idx, 1] += 1
            beg_idx = beg_idx + tmp_sample_num
            
        if shuffle:
            np.random.shuffle(data_idx_order_list)

        data_idx_order_list = data_idx_order_list[:round(len(data_idx_order_list) * sample_ratio)]

        return data_idx_order_list

    def get_lmdb_sample_info(self, txn, idx):
        label_key = 'label-%09d'.encode() % idx
        label = txn.get(label_key)
        if label is None:
            return None
        label = label.decode('utf-8')
        img_key = 'image-%09d'.encode() % idx
        imgbuf = txn.get(img_key)
        return imgbuf, label

    def __getitem__(self, idx):
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        sample_info = self.get_lmdb_sample_info(self.lmdb_sets[int(lmdb_idx)]['txn'],
                                                int(file_idx))
        if sample_info is None:
            random_idx = np.random.randint(self.__len__())
            return self.__getitem__(random_idx)
        
        data = {
            "image": sample_info[0],
            "label": sample_info[1]
        }
        
        # convert dict to tuple, with extra dummy output
        output = list()
        for k in self.output_columns:
            if k in data:
                output.append(data[k])
            else:
                output.append(0)  # dummy output

        return output

    def __len__(self):
        return self.data_idx_order_list.shape[0]
