import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import numpy as np
import pandas as pd
import tensorflow as tf
from src.data import load_dataset, print_class_imbalance
from src.data.prepare_datasets import prepare_kfold_centralised_dss, prepare_nested_kfold_dss
from src.models import get_model
from src.training.train_centralised import train_model
from src.training.tl_baseline_performance import get_baseline_performance
from src.utils.shap import plot_shap_values

# Set memory growth for GPU devices
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Main function to run the training and evaluation process based on the provided configuration.
    """
    # Set random seeds for reproducibility
    if cfg.zeed is not None:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        random.seed(42 + cfg.zeed)
        np.random.seed(42 + cfg.zeed)
        tf.random.set_seed(42 + cfg.zeed)

    # Configure logging
    logging.basicConfig(format='%(message)s', level=logging.DEBUG if cfg.debug else logging.INFO)

    # Custom calculation of config values
    if cfg.dp.get('maximum_noise_in_average_gradient') is not None:
        cfg.dp.noise_multiplier = round(cfg.dp.maximum_noise_in_average_gradient *
                                        cfg.training.batch_size / cfg.dp.l2_norm_clip, 2)
        logging.info(f'Setting noise multiplier to {cfg.dp.noise_multiplier} to keep maximum noise level')

    if cfg.dp.get('maximum_noise_in_average_model_update') is not None:
        cfg.dp.noise_multiplier = round(cfg.dp.maximum_noise_in_average_model_update *
                                        (
                                                    cfg.training.client_sampling_prob * cfg.training.n_total_clients) / cfg.dp.l2_norm_clip,
                                        2)
        logging.info(f'Setting noise multiplier to {cfg.dp.noise_multiplier} to keep maximum noise level')

    # Initialize Weights & Biases
    if cfg.wandb.disabled:
        wandb.init(mode='disabled')
    else:
        sweep_run = wandb.init(project=(cfg.wandb.project or f'{cfg.model.name}_{cfg.dataset.name}'),
                               name=cfg.wandb.name,
                               group=cfg.wandb.group,
                               entity="bjarnepfitzner",
                               job_type=cfg.training.type,
                               config=OmegaConf.to_container(cfg, resolve=True),
                               settings=wandb.Settings(start_method="thread"))
        sweep_id = sweep_run.sweep_id or "unknown_sweep"
        sweep_run_name = sweep_run.name or sweep_run.id or "unknown_run"
        logging.info(f'sweep_id: {sweep_id}, sweep_run_name: {sweep_run_name}')
        cfg.output_folder = f'{cfg.output_folder}/{cfg.experiment_name}/{sweep_id}/{sweep_run_name}'
        os.makedirs(cfg.output_folder, exist_ok=True)

    # Save config file
    logging.info(OmegaConf.to_yaml(cfg))

    # Load dataset
    (X, X_tl), (Y, Y_tl) = load_dataset(cfg.dataset)
    y = Y[cfg.dataset.prediction_target]
    y_tl = Y_tl[cfg.dataset.prediction_target]

    # Remove NA entries and convert labels to int
    X, y = X[y.notna()], y[y.notna()].astype(int)
    X_tl, y_tl = X_tl[y_tl.notna()], y_tl[y_tl.notna()].astype(int)

    # Determine training type and prepare datasets
    if 'base_training' in cfg.training.type:
        logging.info('Using base training dataset')
        logging.info(f'Dataset dimensions: {X.shape}')
        print_class_imbalance(Y, [cfg.dataset.prediction_target])
    else:
        logging.info('Using TL dataset')
        logging.info(f'Dataset dimensions: {X_tl.shape}')
        print_class_imbalance(Y_tl, [cfg.dataset.prediction_target])

    # Adjust learning rate for Adam optimizer
    if 'Adam' in cfg.training.client_optimizer._target_:
        cfg.training.client_optimizer.learning_rate /= 10
        logging.info(
            f'Dividing client learning rate by 10 due to Adam Optimizer. Is now {cfg.training.client_optimizer.learning_rate}')

    # Setup model and loss function
    model = get_model(cfg.model, (len(X.columns),), cfg.zeed)
    initial_model_weights = model.get_weights()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
        reduction=(tf.losses.Reduction.NONE if cfg.dp.type == 'LDP' else tf.losses.Reduction.AUTO))

    all_baseline_metrics = None
    if cfg.training.type == 'base_training':
        if cfg.training.all_train_data:
            logging.info('Joining data')
            X, y = pd.concat([X, X_tl], ignore_index=True), pd.concat([y, y_tl], ignore_index=True)
            logging.info(f'New data dimensions: {X.shape}')
        all_datasets, (inner_train_size, outer_train_size), class_weights = prepare_nested_kfold_dss(X, y, cfg)
    elif cfg.training.type in ['transfer_learning_baseline', 'transfer_learning']:
        all_datasets, (inner_train_size, outer_train_size), class_weights = prepare_nested_kfold_dss(X_tl, y_tl, cfg)
        if cfg.training.type == 'transfer_learning':
            logging.info('Running Transfer Learning')
            assert cfg.get('base_model_path') and cfg.training.get('tl_type')

            trained_model = tf.keras.models.load_model(cfg.base_model_path)
            all_baseline_metrics = evaluate_baseline(cfg, trained_model, all_datasets, loss, class_weights, X.columns)

            model, initial_model_weights = setup_transfer_learning_model(cfg, trained_model, X.columns)
    else:
        raise ValueError('Training Type not known, has to be either "base_model" or "transfer_learning".')

    run_outer_loop(cfg, model, loss, initial_model_weights, all_datasets, inner_train_size, outer_train_size,
                   class_weights, all_baseline_metrics, X.columns)


def evaluate_baseline(cfg, trained_model, all_datasets, loss, class_weights, feature_columns):
    """
    Evaluate the baseline performance of the trained model.
    """
    logging.info(f'=========== Running Baseline Evaluation ===========')
    all_baseline_metrics = []
    all_baseline_shap_values = []
    all_baseline_full_shap_values = []
    all_baseline_test_x = []
    baseline_labels = []
    baseline_pred_probs = []
    for i, (_, outer_dss) in enumerate(all_datasets):
        os.makedirs(f'{cfg.output_folder}/outer_fold_{i}', exist_ok=True)
        baseline_metrics, labels, pred_probs = get_baseline_performance(trained_model, outer_dss['test'], loss,
                                                                        use_sample_weights=cfg.training.use_sample_weights,
                                                                        class_weights=class_weights)
        baseline_labels.extend(labels)
        baseline_pred_probs.extend(pred_probs)
        all_baseline_metrics.append(pd.DataFrame(baseline_metrics, index=pd.Index([i], name='Outer_Fold')))

        if cfg.evaluation.shap and trained_model is not None:
            train_x = np.array(list(outer_dss['train'].map(lambda elem: elem['x']).unbatch().as_numpy_iterator()))
            test_x = np.array(list(outer_dss['test'].map(lambda elem: elem['x']).unbatch().as_numpy_iterator()))
            try:
                shap_values, full_shap_values = plot_shap_values(trained_model, test_x, train_x, feature_columns,
                                                                 suffix=f'/baseline/outer_fold_{i}')
                all_baseline_shap_values.append(shap_values)
                all_baseline_full_shap_values.append(full_shap_values)
                all_baseline_test_x.append(test_x)
            except AssertionError as e:
                logging.info('Encountered assertion error in SHAP calculation. Not saving SHAP plots.')
                logging.info(e)

    if len(all_baseline_shap_values) > 0:
        wandb.log({
            'Baseline SHAP Values': wandb.Table(dataframe=pd.concat(all_baseline_shap_values).reset_index()),
            'Baseline SHAP Values Sum': wandb.Table(
                dataframe=pd.concat(all_baseline_shap_values).sum().to_frame(name='Baseline_SHAP_Value').T),
            'Baseline SHAP Values All': wandb.Table(
                dataframe=pd.DataFrame(np.concatenate(all_baseline_full_shap_values),
                                       columns=feature_columns)),
            'Baseline SHAP Values test_x': wandb.Table(dataframe=pd.DataFrame(np.concatenate(all_baseline_test_x),
                                                                              columns=feature_columns))
        }, step=0)
    wandb.log({'Baseline Metrics': wandb.Table(dataframe=pd.concat(all_baseline_metrics).reset_index())}, step=0)
    aggregated_baseline_metrics = {f'baseline/{key}_mean': val for key, val in
                                   pd.concat(all_baseline_metrics).mean().items()}
    aggregated_baseline_metrics.update(
        {f'baseline/{key}_std': val for key, val in pd.concat(all_baseline_metrics).std().items()})
    wandb.log(aggregated_baseline_metrics, step=0)

    wandb.log({
        f'baseline/pr_curve': wandb.plot.pr_curve(baseline_labels, baseline_pred_probs, interp_size=200,
                                                  classes_to_plot=[1]),
        f'baseline/roc_curve': wandb.plot.roc_curve(baseline_labels, baseline_pred_probs, classes_to_plot=[1])
    }, step=0)
    return all_baseline_metrics


def setup_transfer_learning_model(cfg, trained_model, feature_columns):
    """
    Setup the model for transfer learning based on the specified type.
    """
    if cfg.training.tl_type == 'full_fine_tune':
        model = trained_model
    elif cfg.training.tl_type == 'out_layer_fine_tune':
        model = trained_model
        for layer in model.layers[:-1]:
            layer.trainable = False
    elif cfg.training.tl_type == 'new_out_layer':
        model = tf.keras.models.Sequential(trained_model.layers[:-1])
        model.build((None, len(feature_columns)))
        for layer in model.layers:
            layer.trainable = False
        model.add(tf.keras.layers.Dense(
            2, activation=tf.nn.softmax,
            kernel_initializer=tf.keras.initializers.get(
                {'class_name': cfg.model.initialiser, 'config': {'seed': cfg.zeed}})))
    else:
        raise ValueError('Transfer Learning type not known.')
    model.summary(print_fn=logging.info)
    return model, model.get_weights()


def run_outer_loop(cfg, model, loss, initial_model_weights, all_datasets, inner_train_size, outer_train_size,
                   class_weights, all_baseline_metrics, feature_columns):
    """
    Run the outer loop of the nested cross-validation.
    """
    all_cv_metrics = []
    all_test_metrics = []
    all_trained_models = []
    all_shap_values = []
    all_full_shap_values = []
    all_test_x = []
    all_labels = []
    all_pred_probs = []
    all_pred_labels = []
    for i, (cv_datasets, outer_dss) in enumerate(all_datasets):
        logging.info(f'=========== Running outer fold {i} ===========')
        os.makedirs(f'{cfg.output_folder}/outer_fold_{i}', exist_ok=True)

        # Inner loop
        cv_metrics = run_inner_loop(cfg, model, loss, initial_model_weights, cv_datasets, inner_train_size,
                                    feature_columns, i)
        all_cv_metrics.append(pd.concat(cv_metrics))
        aggregated_cv_metrics = {f'cv_out_fold_{i}/{key.replace("test/", "")}_mean': val for key, val in
                                 pd.concat(cv_metrics).mean().items()}
        aggregated_cv_metrics.update({f'cv_out_fold_{i}/{key.replace("test/", "")}_std': val for key, val in
                                      pd.concat(cv_metrics).std().items()})
        wandb.log(aggregated_cv_metrics, step=0)

        # Outer test
        model.set_weights(initial_model_weights)
        trained_model, metrics, labels_and_preds = train_model(model=model, loss_object=loss,
                                                               train_ds=outer_dss['train'], val_ds=outer_dss['val'],
                                                               test_ds=outer_dss['test'],
                                                               train_size=outer_train_size, class_weights=class_weights,
                                                               cfg=cfg,
                                                               log_file=f'{cfg.output_folder}/outer_fold_{i}/metrics_final_test.csv',
                                                               log_to_wandb=False)
        all_test_metrics.append(pd.DataFrame(metrics, index=pd.Index([i], name='Outer_Fold')))
        wandb.log({f'cv_out_fold_{i}/{key}': val for key, val in metrics.items()}, step=0)

        all_trained_models.append(trained_model)

        # Save labels and pres for ROC and PR curve
        all_labels.extend(labels_and_preds['labels'])
        all_pred_probs.extend(labels_and_preds['pred_probs'])
        all_pred_labels.extend(labels_and_preds['pred_labels'])

        # SHAP evaluation
        if cfg.evaluation.shap and trained_model is not None:
            train_x = np.array(list(outer_dss['train'].map(lambda elem: elem['x']).unbatch().as_numpy_iterator()))
            test_x = np.array(list(outer_dss['test'].map(lambda elem: elem['x']).unbatch().as_numpy_iterator()))
            try:
                shap_values, full_shap_values = plot_shap_values(trained_model, test_x, train_x, feature_columns,
                                                                 suffix=f'/outer_fold_{i}')
                all_shap_values.append(shap_values)
                all_full_shap_values.append(full_shap_values)
                all_test_x.append(test_x)
            except AssertionError as e:
                logging.info('Encountered assertion error in SHAP calculation. Not saving SHAP plots.')
                logging.info(e)
    log_final_results(cfg, all_cv_metrics, all_test_metrics, all_shap_values, all_baseline_metrics, all_trained_models,
                      all_pred_labels, all_labels, all_pred_probs, all_full_shap_values, all_test_x, feature_columns)


def run_inner_loop(cfg, model, loss, initial_model_weights, cv_datasets, inner_train_size, class_weights,
                   outer_fold_idx):
    """
    Run the inner loop of the nested cross-validation.
    """
    cv_metrics = []
    for j, inner_dss in enumerate(cv_datasets):
        logging.info(f'=========== Running inner fold {j} ===========')
        model.set_weights(initial_model_weights)
        _, metrics, _ = train_model(model=model, loss_object=loss,
                                    train_ds=inner_dss['train'], val_ds=inner_dss['val'], test_ds=inner_dss['test'],
                                    train_size=inner_train_size, class_weights=class_weights, cfg=cfg,
                                    log_file=f'{cfg.output_folder}/outer_fold_{outer_fold_idx}/metrics_fold_{j}.csv',
                                    log_to_wandb=False)
        cv_metrics.append(pd.DataFrame(metrics,
                                       index=pd.MultiIndex.from_tuples([(outer_fold_idx, j)],
                                                                       names=['Outer_Fold', 'Inner_Fold'])))
    return cv_metrics


def log_final_results(cfg, all_cv_metrics, all_test_metrics, all_shap_values, all_baseline_metrics, all_trained_models,
                      all_pred_labels, all_labels, all_pred_probs, all_full_shap_values, all_test_x, feature_columns):
    """
    Log and save the final results of the cross-validation and SHAP evaluations.
    """
    wandb.log({
        'CV Metrics': wandb.Table(dataframe=pd.concat(all_cv_metrics).reset_index()),
        'Test Metrics': wandb.Table(dataframe=pd.concat(all_test_metrics).reset_index())
    }, step=0)

    wandb.log({
        'test/conf_mat': wandb.plot.confusion_matrix(preds=all_pred_labels, y_true=all_labels),
        'test/pr_curve': wandb.plot.pr_curve(all_labels, all_pred_probs, classes_to_plot=[1]),
        'test/roc_curve': wandb.plot.roc_curve(all_labels, all_pred_probs, classes_to_plot=[1])
    }, step=0)

    if all_shap_values:
        logging.info(pd.concat(all_shap_values).sum().to_frame(name='SHAP_Value').T)
        wandb.log({
            'SHAP Values': wandb.Table(dataframe=pd.concat(all_shap_values).reset_index()),
            'SHAP Values Sum': wandb.Table(dataframe=pd.concat(all_shap_values).sum().to_frame(name='SHAP_Value').T),
            'SHAP Values All': wandb.Table(
                dataframe=pd.DataFrame(np.concatenate(all_full_shap_values), columns=feature_columns)),
            'SHAP Values test_x': wandb.Table(
                dataframe=pd.DataFrame(np.concatenate(all_test_x), columns=feature_columns))
        }, step=0)

    sub_aggregated_cv_metrics = pd.concat(all_cv_metrics).reset_index().groupby('Outer_Fold').aggregate('mean')
    aggregated_all_cvs_metrics = {
        f'cvs/{key.replace("test/", "")}_means_mean': sub_aggregated_cv_metrics[key].mean()
        for key in pd.concat(all_cv_metrics).columns if 'Fold' not in key}
    aggregated_all_cvs_metrics.update({
        f'cvs/{key.replace("test/", "")}_means_std': sub_aggregated_cv_metrics[key].std()
        for key in pd.concat(all_cv_metrics).columns if 'Fold' not in key})
    wandb.log(aggregated_all_cvs_metrics, step=0)

    aggregated_test_metrics = {f'{key}_mean': val for key, val in pd.concat(all_test_metrics).mean().items()}
    aggregated_test_metrics.update({f'{key}_std': val for key, val in pd.concat(all_test_metrics).std().items()})
    wandb.log(aggregated_test_metrics, step=0)

    if all_baseline_metrics is not None:
        metric_diffs = {key: pd.concat(all_test_metrics)[f'test/{key}'] - pd.concat(all_baseline_metrics)[key]
                        for key in pd.concat(all_baseline_metrics).columns}
        aggregated_metric_diffs = {f'diffs/{key}_mean': val.mean() for key, val in metric_diffs.items()}
        aggregated_metric_diffs.update({f'diffs/{key}_std': val.std() for key, val in metric_diffs.items()})
        wandb.log(aggregated_metric_diffs, step=0)

    # Save the best model
    if cfg.save_model:
        auprc_values = [test_metrics['test/auprc'] for test_metrics in all_test_metrics]
        best_fold = np.argmax(auprc_values)
        logging.info(f'Saving model of outer fold {best_fold} to {cfg.output_folder}/trained_model')
        all_trained_models[best_fold].save(f'{cfg.output_folder}/trained_model')


if __name__ == '__main__':
    main()
