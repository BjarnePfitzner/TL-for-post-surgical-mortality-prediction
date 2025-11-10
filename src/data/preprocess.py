import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from src.utils.feature_selector import FeatureSelector
from src.data.abstract_dataset import Dataset


def get_preprocessed_data(data: Dataset,
                          fs_operations=None,
                          missing_threshold=0.5,
                          correlation_threshold=0.95,
                          imputer=IterativeImputer(max_iter=10, verbose=False),
                          normalise='standard'):
    """Return X and Y after applying specified preprocessing steps

    Parameters
    ----------
    validation
    data : Dataset
        The parsed data
    fs_operations : list, default=['single_unique', 'collinear']
        The feature selection operations to perform with FeatureSelector instance
    missing_threshold : float, default=0.5
        The threshold for removing features with missing values
    correlation_threshold : float, default=0.95
        The threshold for removing collinear features
    impute : bool, default=True
        Whether to impute missing values
    normalise : bool, default=True
        Whether to normalise numerical values
    verbose : bool, default=False
        Turns on verbose output
    validation : bool
        Whether to process the validation dataset
    Returns
    -------
    X : DataFrame
        The feature values
    Y : DataFrame
        The endpoints
    """
    if fs_operations is None:
        fs_operations = ['single_unique', 'collinear']

    X, Y = data.get_data()
    X_tl, Y_tl = data.get_tl_data()
    n_tl_samples = X_tl.shape[0]
    # concat base and tl data for proper One-Hot encoding
    X = pd.concat([X, X_tl], ignore_index=True)
    Y = pd.concat([Y, Y_tl], ignore_index=True)

    # Apply FeatureSelector functionality
    if len(fs_operations) > 0:
        fs = FeatureSelector()
        if 'single_unique' in fs_operations:
            fs.identify_single_unique(X)
            logging.debug(fs.record_single_unique)
        if 'missing' in fs_operations:
            fs.identify_missing(X, missing_threshold=missing_threshold)
            logging.debug(fs.record_missing)
        if 'collinear' in fs_operations:
            fs.identify_collinear(X, correlation_threshold=correlation_threshold)
            logging.debug(fs.record_collinear)
        X = fs.remove(X, fs_operations, one_hot=False)

    # Fix strings in Binary columns
    for binary_col in data.get_binary_features():
        if binary_col in X.columns:
            unique_vals = list(np.unique(X[binary_col].values))
            assert len(unique_vals) <= 2, f'Binary column {binary_col} has more than 2 unique values: {unique_vals}'
            if len(unique_vals) != 1 and unique_vals != [0, 1]:
                logging.debug(f'Renaming entries from {binary_col}: {unique_vals[0]} -> 0; {unique_vals[1]} -> 1')
                X[binary_col].replace({unique_vals[0]: 0,
                                       unique_vals[1]: 1}, inplace=True)

    categorical_features = [col for col in X.columns if col in data.get_categorical_features()]

    # One-hot-encode categorical features
    X = pd.get_dummies(X, columns=categorical_features, dummy_na=False)

    X_numerical = X[[col for col in X.columns if col in data.get_numerical_features()]]
    X_binary = X.drop(columns=[col for col in X.columns if col in data.get_numerical_features()])

    X_numerical_feature_names = X_numerical.columns
    # Interpolate numerical features
    if imputer is not None:
        logging.debug('Running Imputer...')
        X_numerical = imputer.fit_transform(X_numerical)
        X_numerical = pd.DataFrame(X_numerical, columns=X_numerical_feature_names)
    else:
        X_numerical = X_numerical.dropna(axis=1)
        X_numerical_feature_names = X_numerical.columns

    if normalise is not None:
        logging.debug(f'Normalising numerical features using {normalise} scaling...')
        if normalise == 'standard':
            # Normalise numerical features
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        X_numerical = scaler.fit_transform(X_numerical)
        X_numerical = pd.DataFrame(X_numerical, columns=X_numerical_feature_names)

    # Build X and y arrays
    X = pd.concat([X_binary, X_numerical], axis=1)

    X_tl = X.tail(n_tl_samples)
    Y_tl = Y.tail(n_tl_samples)

    X = X.head(len(X) - n_tl_samples)
    Y = Y.head(len(Y) - n_tl_samples)

    logging.info(f'Final Dataset dimensions = {X.shape}')
    logging.info(f'Final TL Dataset dimensions = {X_tl.shape}')

    return (X, X_tl), (Y, Y_tl)
