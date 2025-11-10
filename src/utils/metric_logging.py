from typing import Dict
import logging

import wandb
import numpy as np
import pandas as pd


class MetricsLogger:
    def __init__(self, output_file: str, log_to_wandb=True):
        self.metrics_df = pd.DataFrame(index=pd.UInt64Index([1]))
        self.output_file = output_file
        self.log_to_wandb = log_to_wandb

    def log(self, metrics_dict: Dict, step: int):
        for metric_key, metric_val in metrics_dict.items():
            if np.isscalar(metric_val):
                self.metrics_df.at[step, metric_key] = metric_val
        if self.log_to_wandb:
            wandb.log(metrics_dict, step=step)

    def write_metrics_to_file(self):
        logging.info('writing metrics to file')
        self.metrics_df.to_csv(self.output_file)
