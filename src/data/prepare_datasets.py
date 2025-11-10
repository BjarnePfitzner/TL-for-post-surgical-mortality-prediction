import collections
import logging

import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


def _element_fn(X, y):
    return collections.OrderedDict(
        x=X, y=y)


def _get_class_weights(y: pd.Series, n_classes=2):
    return [len(y) / (n_classes * sum(y == cls)) for cls in range(n_classes)]


def convert_np_data_to_tf_ds(X, y, batch_size, reshuffle, drop_remainder, seed):
    return (tf.data.Dataset.from_tensor_slices((X, y))
            .map(_element_fn)
            .shuffle(len(y), reshuffle_each_iteration=reshuffle, seed=seed)
            .batch(batch_size, drop_remainder=drop_remainder)
            .prefetch(AUTOTUNE))


def prepare_nested_kfold_dss(X, y, cfg):
    class_weights = _get_class_weights(y)
    logging.info(f'Class weights: {class_weights}')

    outer_cv = StratifiedKFold(n_splits=cfg.training.outer_k_folds, shuffle=True, random_state=cfg.zeed)
    inner_cv = StratifiedKFold(n_splits=cfg.training.inner_k_folds, shuffle=True, random_state=cfg.zeed)

    all_datasets = []
    for outer_train_idx, outer_test_idx in outer_cv.split(X, y):
        outer_train_val_X, outer_train_val_y = X.iloc[outer_train_idx], y.iloc[outer_train_idx]
        outer_test_X, outer_test_y = X.iloc[outer_test_idx], y.iloc[outer_test_idx]
        outer_train_X, outer_val_X, outer_train_y, outer_val_y = train_test_split(
            outer_train_val_X, outer_train_val_y,
            train_size=1 - cfg.training.val_fraction,
            stratify=outer_train_val_y,
            random_state=cfg.zeed
        )
        outer_train_ds = convert_np_data_to_tf_ds(outer_train_X, outer_train_y, cfg.training.batch_size,
                                                  reshuffle=True, drop_remainder=(cfg.dp.type == 'LDP'),
                                                  seed=cfg.zeed)
        outer_val_ds = convert_np_data_to_tf_ds(outer_val_X, outer_val_y, cfg.evaluation.batch_size,
                                                reshuffle=False, drop_remainder=False, seed=cfg.zeed)
        outer_test_ds = convert_np_data_to_tf_ds(outer_test_X, outer_test_y, cfg.evaluation.batch_size,
                                                 reshuffle=False, drop_remainder=False, seed=cfg.zeed)
        cv_datasets = []
        for inner_train_idx, inner_test_idx in inner_cv.split(outer_train_X, outer_train_y):
            inner_train_val_X, inner_train_val_y = outer_train_X.iloc[inner_train_idx], outer_train_y.iloc[inner_train_idx]
            inner_test_X, inner_test_y = outer_train_X.iloc[inner_test_idx], outer_train_y.iloc[inner_test_idx]
            inner_train_X, inner_val_X, inner_train_y, inner_val_y = train_test_split(
                inner_train_val_X, inner_train_val_y,
                train_size=1 - cfg.training.val_fraction,
                stratify=inner_train_val_y,
                random_state=cfg.zeed
            )
            inner_train_ds = convert_np_data_to_tf_ds(inner_train_X, inner_train_y, cfg.training.batch_size,
                                                      reshuffle=True, drop_remainder=(cfg.dp.type == 'LDP'),
                                                      seed=cfg.zeed)
            inner_val_ds = convert_np_data_to_tf_ds(inner_val_X, inner_val_y, cfg.evaluation.batch_size,
                                                    reshuffle=False, drop_remainder=False, seed=cfg.zeed)
            inner_test_ds = convert_np_data_to_tf_ds(inner_val_X, inner_val_y, cfg.evaluation.batch_size,
                                                     reshuffle=False, drop_remainder=False, seed=cfg.zeed)
            cv_datasets.append({'train': inner_train_ds,
                                'val': inner_val_ds,
                                'test': inner_test_ds})
        all_datasets.append((cv_datasets,
                             {'train': outer_train_ds,
                              'val': outer_val_ds,
                              'test': outer_test_ds})
                            )

    logging.info(f'The data is split into {len(outer_train_y)} for CV, {len(outer_val_y)} for outer validation and {len(outer_test_y)} for outer testing.')
    logging.info(f'The CV data is split into {len(inner_train_y)} for training, {len(inner_val_y)} for validation and {len(inner_test_y)} testing.')

    if cfg.dp.type == 'LDP':
        inner_train_size = int(len(inner_train_y) / cfg.training.batch_size) * cfg.training.batch_size
        outer_train_size = int(len(outer_train_y) / cfg.training.batch_size) * cfg.training.batch_size
    else:
        inner_train_size = len(inner_train_y)
        outer_train_size = len(outer_train_y)

    return all_datasets, (inner_train_size, outer_train_size), class_weights


def prepare_kfold_centralised_dss(X, y, cfg):
    class_weights = _get_class_weights(y)
    logging.info(f'Class weights: {class_weights}')

    train_cv_X, test_X, train_cv_y, test_y = train_test_split(
        X, y,
        train_size=1 - cfg.dataset.test_fraction - cfg.dataset.val_fraction,
        stratify=y,
        random_state=cfg.zeed
    )

    final_val_X, final_test_X, final_val_y, final_test_y = train_test_split(
        test_X, test_y,
        train_size=cfg.dataset.val_fraction / (cfg.dataset.test_fraction + cfg.dataset.val_fraction),
        stratify=test_y,
        random_state=cfg.zeed
    )
    logging.info(f'Splitting {len(X)} samples into {len(train_cv_X)} for the CV, '
                 f'{len(final_val_X)} for final early stopping and {len(final_test_X)} for final testing.')

    train_ds = (tf.data.Dataset.from_tensor_slices((train_cv_X, train_cv_y))
                .map(_element_fn)
                .shuffle(len(test_y), reshuffle_each_iteration=True, seed=cfg.zeed)
                .batch(cfg.training.batch_size)
                .prefetch(AUTOTUNE))
    val_ds = (tf.data.Dataset.from_tensor_slices((final_val_X, final_val_y))
               .map(_element_fn)
               .shuffle(len(final_val_y), reshuffle_each_iteration=False, seed=cfg.zeed)
               .batch(cfg.evaluation.batch_size)
               .prefetch(AUTOTUNE))
    test_ds = (tf.data.Dataset.from_tensor_slices((final_test_X, final_test_y))
               .map(_element_fn)
               .shuffle(len(final_test_y), reshuffle_each_iteration=False, seed=cfg.zeed)
               .batch(cfg.evaluation.batch_size)
               .prefetch(AUTOTUNE))

    train_dss = []
    es_dss = []
    val_dss = []
    skf = StratifiedKFold(n_splits=cfg.training.k_folds, shuffle=True, random_state=cfg.zeed)
    for train_idx, val_idx in skf.split(train_cv_X, train_cv_y):
        train_X, train_y = train_cv_X.iloc[train_idx], train_cv_y.iloc[train_idx]
        val_X, val_y = train_cv_X.iloc[val_idx], train_cv_y.iloc[val_idx]
        train_X, es_X, train_y, es_y = train_test_split(
            train_X, train_y,
            train_size=1 - (1 / (cfg.training.k_folds - 1)),
            stratify=train_y,
            random_state=cfg.zeed
        )
        train_dss.append((train_X, train_y))
        es_dss.append((es_X, es_y))
        val_dss.append((val_X, val_y))

    logging.info(f'The CV data is split into {len(train_y)} for training and {len(es_y)} and {len(val_y)} for early stopping and testing.')

    if cfg.dp.type == 'LDP':
        train_size = int(len(train_y) / cfg.training.batch_size) * cfg.training.batch_size
    else:
        train_size = len(train_y)

    return zip(train_dss, es_dss, val_dss), (train_ds, val_ds, test_ds), train_size, class_weights


def prepare_centralised_ds_with_external_test(X, y, test_X, test_y, cfg):
    class_weights = _get_class_weights(y)
    logging.info(f'Class weights: {class_weights}')

    n_splits = int(1 / (1 - cfg.dataset.train_fraction))
    logging.debug(f'Splitting data into {n_splits} folds, choosing fold {(cfg.zeed % n_splits) + 1}')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + (cfg.zeed // n_splits))

    for i, (train_idx, val_test_idx) in enumerate(skf.split(X, y)):
        if i == cfg.zeed % n_splits:
            train_X, val_X = X.iloc[train_idx], X.iloc[val_test_idx]
            train_y, val_y = y.iloc[train_idx], y.iloc[val_test_idx]
            break

    train_ds = (tf.data.Dataset.from_tensor_slices((train_X, train_y))
                .map(_element_fn)
                .shuffle(len(train_y), reshuffle_each_iteration=True, seed=cfg.zeed)
                .batch(cfg.training.batch_size, drop_remainder=(cfg.dp.type == 'LDP'))
                .prefetch(AUTOTUNE))
    val_ds = (tf.data.Dataset.from_tensor_slices((val_X, val_y))
              .map(_element_fn)
              .shuffle(len(val_y), reshuffle_each_iteration=False, seed=cfg.zeed)
              .batch(cfg.evaluation.batch_size)
              .prefetch(AUTOTUNE))
    test_ds = (tf.data.Dataset.from_tensor_slices((test_X, test_y))
               .map(_element_fn)
               .shuffle(len(test_y), reshuffle_each_iteration=False, seed=cfg.zeed)
               .batch(cfg.evaluation.batch_size)
               .prefetch(AUTOTUNE))

    if cfg.dp.type == 'LDP':
        train_size = int(len(train_y) / cfg.training.batch_size) * cfg.training.batch_size
    else:
        train_size = len(train_y)

    return train_ds, val_ds, test_ds, train_size, class_weights
