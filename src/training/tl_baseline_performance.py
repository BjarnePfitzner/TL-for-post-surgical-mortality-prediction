import tensorflow as tf

from src.utils.auc_metric_wrapper import AUCWrapper
from src.utils.f_score_metrics import F1Score, CalibratedF1Score, FBetaScore, CalibratedFBetaScore


def get_baseline_performance(model, test_ds, loss_object, use_sample_weights=False, class_weights=None):
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_metrics = {
        'accuracy': tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy'),
        'calibrated_f1': CalibratedF1Score(name='test_calibrated_f1', dtype=tf.float32),
        'calibrated_f2': CalibratedFBetaScore(name='test_calibrated_f2', beta=2.0, dtype=tf.float32),
        'f1': F1Score(name='test_f1', dtype=tf.float32),
        'f2': FBetaScore(name='test_f2', beta=2.0, dtype=tf.float32),
        'auprc': AUCWrapper(curve='PR', name='test_AUPRC', from_logits=False),
        'auroc': AUCWrapper(curve='ROC', name='test_AUROC', from_logits=False)
    }

    @tf.function
    def test_step(x, y, loss_metric, metrics):
        predictions = model(x, training=False)
        sample_weights = None
        if use_sample_weights:
            sample_weights = tf.gather(class_weights, y)
        t_loss = loss_object(y, predictions, sample_weight=sample_weights)

        loss_metric(t_loss)
        for metric in metrics.values():
            metric(y, predictions)

        return predictions

    labels = []
    pred_probs = []
    for batch in test_ds:
        labels.extend(batch['y'].numpy())
        predictions = test_step(batch['x'], batch['y'], test_loss, test_metrics)
        pred_probs.extend(predictions.numpy())

    calibrated_f1, calibrated_recall, calibrated_precision, threshold = test_metrics['calibrated_f1'].result()
    calibrated_f2, calibrated_f2_recall, calibrated_f2_precision, f2_threshold = test_metrics['calibrated_f2'].result()
    result_dict = {metric_name: metric.result().numpy() for metric_name, metric in test_metrics.items()
                   if not metric_name.startswith('calibrated')}
    result_dict.update({'calibrated_f1': calibrated_f1.numpy(),
                        'calibrated_recall': calibrated_recall.numpy(),
                        'calibrated_precision': calibrated_precision.numpy(),
                        'calibrated_threshold': threshold.numpy(),
                        'calibrated_f2': calibrated_f2.numpy(),
                        'calibrated_f2_recall': calibrated_f2_recall.numpy(),
                        'calibrated_f2_precision': calibrated_f2_precision.numpy(),
                        'calibrated_f2_threshold': f2_threshold.numpy(),
                        'loss': test_loss.result().numpy()})
    return result_dict, labels, pred_probs
