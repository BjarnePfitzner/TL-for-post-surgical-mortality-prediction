import os
import logging
import numpy as np

import pandas as pd

from src.data.abstract_dataset import Dataset


DATA_PATH = {
    'pre': os.path.abspath("./data/transfer_learning_dataset_cleaned_height_weight.csv"),
}
DATA_INFOS_PATH = os.path.abspath("./src/data/tl_dataset/tl_dataset_infos.csv")


class TLDataset(Dataset):
    def __init__(self, feature_set):
        super(TLDataset, self).__init__(DATA_PATH[feature_set], DATA_INFOS_PATH)

        self.X_tl = None
        self.Y_tl = None

    def read_csv(self):
        return pd.read_csv(self.path), None

    def get_tl_data(self):
        """Return validation X and endpoints Y if parse() has been called"""
        if self.X_tl is None:
            raise RuntimeError('Please call parse() before accessing data and make sure you have specified validation '
                               'data')
        return self.X_tl, self.Y_tl

    def parse(self, drop_columns=True, feature_set=None, drop_missing_value=0, external_validation=False,
              split_col=None, tl_organ_system='Liver'):
        """Parse dataframe according to parameters and fill X and Y class attributes

                Parameters
                ----------
                drop_columns : bool, default=True
                    Drop the columns/features determined as drop columns in data infos
                feature_set : list, optional
                    List including any of "pre", "intra", "post", "dyn", defining the feature set to parse
                drop_missing_value : int, optional
                    Drop rows missing this percentage of columns
                """
        self.feature_set = feature_set
        complete_data, _ = self.read_csv()

        # Assert the length of the intersection of data and data infos
        assert len(
            set(complete_data.columns).intersection(set(self.get_all_features() + self.get_all_endpoints()))) == len(
            set(complete_data.columns)), "Column set doesn't match: " + \
                                         str([col for col in complete_data.columns if
                                              col not in set(complete_data.columns).intersection(
                                                  set(self.get_all_features() + self.get_all_endpoints()))])

        drop_columns_list = []

        if feature_set is not None:
            drop_columns_list.extend(list(self.data_infos.loc[
                                              ~self.data_infos['endpoint'] & ~self.data_infos['input_time'].isin(
                                                  feature_set), 'column_name']))

        if drop_columns:
            drop_columns_list.extend(list(self.data_infos.loc[self.data_infos['drop'], 'column_name']))
            # Remove the updated values from esophagus_info_updated
            logging.debug(list(set(self.data_infos.column_name.values).difference(set(complete_data.columns.values))))
            difference = list(set(self.data_infos.column_name.values).difference(set(complete_data.columns.values)))
            self.data_infos = self.data_infos[~self.data_infos["column_name"].isin(difference)]

        # Perform column dropping
        complete_data.drop(columns=drop_columns_list, inplace=True, errors='ignore')

        if drop_missing_value > 0:
            # Calculate the minimum amount of columns that have to contain a value
            min_count = int(((100 - drop_missing_value) / 100) * complete_data.shape[1] + 1)

            # Drop rows not meeting threshold
            complete_data = complete_data.dropna(axis=0, thresh=min_count)

        # Extract features and endpoints
        self.X = complete_data[self.data_infos.loc[
            ~(self.data_infos['endpoint'] | self.data_infos['column_name'].isin(drop_columns_list)), 'column_name']]
        self.Y = complete_data[self.data_infos.loc[
            self.data_infos['endpoint'] & ~self.data_infos['column_name'].isin(drop_columns_list), 'column_name']]

        logging.info(f'Total number of patients: {len(self.X)}')

        self.X_tl = self.X[self.X['organ'] == tl_organ_system]
        self.Y_tl = self.Y.loc[self.X_tl.index]

        self.X = self.X[self.X['organ'] != tl_organ_system]
        self.Y = self.Y.loc[self.X.index]

        self.X.reset_index(drop=True, inplace=True)
        self.Y.reset_index(drop=True, inplace=True)
        self.X_tl.reset_index(drop=True, inplace=True)
        self.Y_tl.reset_index(drop=True, inplace=True)
