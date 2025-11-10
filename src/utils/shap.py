import warnings
import logging

import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import shap


meta_system_replacements = {
    'meta_system_0': 'Oesophagus',
    'meta_system_1': 'Stomach',
    'meta_system_2': 'Colorectum',
    'meta_system_3': 'Liver',
    'meta_system_4': 'Pancreas'
}


def plot_shap_values(model, X_test, X_train, feature_names, explainer_class='deep', suffix=''):
    #X_test_df = pd.DataFrame(X_test, columns=feature_names)
    #X_train_df = pd.DataFrame(X_train, columns=feature_names)
    #X_test_df['meta_system'].replace(meta_system_replacements)
    #X_train_df['meta_system'].replace(meta_system_replacements)
    def get_shap_values(test_data, train_data):
        if explainer_class == 'deep':
            explainer = shap.DeepExplainer(model, train_data)
        else:
            explainer = shap.KernelExplainer(model, train_data[:100])
        #return explainer(test_data)
        return explainer.shap_values(test_data, check_additivity=False) # todo test this

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # Test SHAP values
        test_shap_values = get_shap_values(X_test, X_train)[0]
        fig = plt.figure()
        shap.summary_plot(test_shap_values, X_test, feature_names=feature_names, plot_type='bar', show=False, max_display=10)
        #shap.plots.bar(test_shap_values, show=False)
        plt.tight_layout()
        wandb.log({f'SHAP_Value_Bars{suffix}': wandb.Image(fig)}, step=0)
        plt.close()

        fig = plt.figure()
        shap.summary_plot(test_shap_values, X_test, feature_names=feature_names, plot_type='dot', show=False, max_display=10)
        #shap.plots.bar(test_shap_values, show=False)
        plt.tight_layout()
        wandb.log({f'SHAP_Value_Dots{suffix}': wandb.Image(fig)}, step=0)
        plt.close()

        # fig = plt.figure()
        # shap.plots.bar(test_shap_values.cohorts(X_test_df['meta_system'].abs.mean(axis=0)), show=False)
        # plt.tight_layout()
        # wandb.log({'SHAP Value Bars (By Organ)': wandb.Image(fig)})
        # plt.close()

        # fig = plt.figure()
        # shap.plots.beeswarm(test_shap_values, show=False)
        # plt.tight_layout()
        # wandb.log({'SHAP Value Beeswarm': wandb.Image(fig)})
        # plt.close()

        shap_value_df = pd.Series(np.abs(test_shap_values).mean(0), index=feature_names).to_frame().T
        if suffix != '':
            shap_value_df['Outer_Fold'] = int(suffix.split('_')[-1])
            shap_value_df.set_index('Outer_Fold', inplace=True)

        return shap_value_df, test_shap_values
