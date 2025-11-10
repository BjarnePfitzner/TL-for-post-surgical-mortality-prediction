import logging

from hydra.utils import instantiate

from src.data.tl_dataset.tl_dataset import TLDataset
from src.data.preprocess import get_preprocessed_data


def get_data_from_name(name, feature_set=None):
    if name == 'transfer_learning':
        return TLDataset(feature_set[-1])
    else:
        raise NotImplementedError


def print_class_imbalance(Y, labels):
    logging.info(f'Class distributions for {len(Y)} data points:')
    for y_col in Y:
        if y_col not in labels:
            continue
        logging.info(f'Endpoint {y_col}:')
        abs_value_counts = Y[y_col].value_counts()
        rel_value_counts = Y[y_col].value_counts(normalize=True)
        for i in range(len(abs_value_counts.index)):
            logging.info(f'\tClass "{abs_value_counts.index[i]}":\t{abs_value_counts.iloc[i]} ({rel_value_counts.iloc[i]:.3f})')
        logging.info('\n')


def load_dataset(dataset_cfg):
    # Get DataInformation object for the specified task
    data = get_data_from_name(dataset_cfg.name, dataset_cfg.feature_set)

    # Parse data
    data.parse(drop_columns=dataset_cfg.drop_features,
               feature_set=dataset_cfg.feature_set,
               drop_missing_value=dataset_cfg.drop_rows_missing_col_fraction,
               external_validation=dataset_cfg.external_test_set,
               split_col=None,
               tl_organ_system=dataset_cfg.tl_organ_system)

    # Preprocess data
    (X, X_tl), (Y, Y_tl) = get_preprocessed_data(data,
                                                 fs_operations=dataset_cfg.fs_operations,
                                                 missing_threshold=dataset_cfg.drop_cols_missing_row_fraction,
                                                 correlation_threshold=dataset_cfg.correlation_threshold,
                                                 imputer=instantiate(dataset_cfg.imputer))

    return (X, X_tl), (Y, Y_tl)
