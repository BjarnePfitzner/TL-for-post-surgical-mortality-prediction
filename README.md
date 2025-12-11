# TL for Post-Surgical Mortality Prediction

## Description
Project for evaluating transfer learning on an in-house dataset containing patient data from major visceral surgery.
This code was used for our paper _Winter, Axel MD; Pfitzner, Bjarne MSc; van de Water, Robin P. MSc; Faraj, Lara; Riepe, Christoph; Hahn, Wolf-Heinrich; Krenzien, Felix MD; Schineis, Christian MD; Malinka, Thomas MD; Schöning, Wenzel MD; Denecke, Christian MD; Arnrich, Bert; Beyer, Katharina; Pratschke, Johann; Sauer, Igor M.; Maurer, Max M. MD. Overcoming the data barrier: transfer learning for 90-day mortality prediction in general surgery – a retrospective multicenter development and comparison study. International Journal of Surgery ():10.1097/JS9.0000000000003595, November 04, 2025. | DOI: 10.1097/JS9.0000000000003595_

## Installation
First, create an environment with conda (or any other environment management software you prefer) using Python 3.8 and activate it:

```bash
$ conda create --name tl_env python=3.8.0
$ conda activate tl_env
```

Then, install the required packages from the `requirements.txt` file using `pip`:

```bash
$ pip install -r requirements.txt
```

## Usage
Experiments are configured with the [Hydra](https://hydra.cc) framework using the yaml files in the top-level [config](config) folder.
[config.yaml](config%2Fconfig.yaml) holds all the base configurations and the subfolders (except for [experiment](config%2Fexperiment)) hold templates for different configurations (e.g. the [training](config%2Ftraining) folder holds the base configurations for full centralised training or transfer learning).
Specific overrides for an experiment are aggregated in the yaml files inside the [experiment](config%2Fexperiment) folder.
These are selected in the python command to run, like so:

```bash
$ python -m src.main_tl +experiment=tl_90d_base
```

Please refer to [Hydra's override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/) for an explanation on how to change specific configs from the command line.

The configurations used for the abovementioned paper can be found in [experiment/paper](config%2Fexperiment%2Ftl_final), but since the data cannot be made public for privacy reasons, they are just for reference.

We use [Weights and Biases](https://wandb.ai) for experiment tracking, so, if you set up your API key on your system, it should automatically upload the experiment results to a new project.

For running a hyperoptimisation, we use the [hydra-wandb-sweeper](https://github.com/captain-pool/hydra-wandb-sweeper) that allowed us to track the runs directly to WandB. 
You can adapt the `hydra.sweeper` configuration in [config.yaml](config%2Fconfig.yaml) to be suitable for your WandB setup, or change it to a different sweeper entirely.
We configure the search parameters in the experiment files. The auxiliary file [create_wandb_sweep_config.py](create_wandb_sweep_config.py) can be used to create a WandB sweep config file based on the hyperparameter setup in an experiment config file.

### Transfer Learning Pipeline
To perform transfer learning, we first run the "\_base" version of the experiment (e.g. using the `tl_90d_base.yaml` experiment config). This trains the model on all but one organ system (in our use case) and saves it to a file.
This file has to be provided in the follow-up config with the suffix "\_tl" (e.g. `tl_90d_tl.yaml`), where the transfer learning is applied to adapt the pre-trained model to the data of the held-out organ system.
Lastly, the "\_tl\_baseline" suffix (e.g. `tl_90d_tl_baseline.yaml`) trains a fresh model on the held-out organ system's data to provide a baseline performance for a model that did not leverage the pre-training phase.

## Support
If there are any questions or issues, please contact me at [bjarne.pfitzner@hpi.de](mailto:bjarne.pfitzner@hpi.de?subject=[GitHub]).
