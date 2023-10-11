import os
import tensorflow as tf
from train_disco_sweep import train_disco,get_params,read_data
import argparse

import pdb

"""
    Trains a prototype label distributional learning neural model (LDL-NM) which
    is an artificial neural network that jointly learns to model label distributions
    for ground truth labels, items, and annotators. The resultant model can be
    used to iteratively infer embeddings for annotators (or be used to
    conduct majority/modal voting across its memory of known annotators).

    Here is an example run `python3 train_ldlnm_sweep.py --config ./config_files/ldlnm_config.cfg --sweep_id rit_pl/jobq2_sweep/smd8rl2y --gpu_id 8`

    In order to run this, you initially need to create a Sweep on Weights and Biases. 
"""

import wandb
from wandb_creds import wandb_creds 
os.environ["WANDB_API_KEY"] = wandb_creds()


def read_wandb_sweep_id(sweep_id,params, simulation_params, gpu_tag):
    data = read_data(params)
    
    # obtain the best runs from the sweep
    api = wandb.Api()
    sweep = api.sweep(sweep_id)

    # Get best run parameters
    best_run = sweep.best_run()
    best_parameters = best_run.config
    wandb.init(config=best_parameters)
    with tf.device(gpu_tag):
        train_disco(data, simulation_params, best_parameters, params)


def main():
    ################################################################################
    # read in configuration file and extract necessary variables/constants
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Config file")
    parser.add_argument("--sweep_id", help="Sweep ID from WANDB")
    parser.add_argument("--gpu_id", help="GPU id",default=-1)

    args = parser.parse_args()
    cfg_fname = args.config
    sweep_id = args.sweep_id
    gpu_id = int(args.gpu_id)

    if gpu_id>-1:
        print(" > Using GPU ID {0}".format(gpu_id))
        os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(gpu_id)
        gpu_tag = '/GPU:0'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        gpu_tag = '/CPU:0'

    params, simulation_params,_ = get_params(cfg_fname)

    read_wandb_sweep_id(sweep_id,params, simulation_params, gpu_tag)


if __name__ == '__main__':
    main()