# Welcome to DisCo

This repo includes the code to reproduce the experiments from DisCo (Distribution from Context). 

## Abstract
Annotator disagreement is common whenever human judgment is needed for supervised learning. It is conventional to assume that one label per item represents ground truth. However, this obscures minority opinions, if present. We regard "ground truth″ as the distribution of all labels that a population of annotators could produce, if asked (and of which we only have a small sample). We next introduce DisCo (Distribution from Context), a simple neural model that learns to predict this distribution. The model takes annotator-item pairs, rather than items alone, as input, and performs inference by aggregating over all annotators. Despite its simplicity, our experiments show that, on six benchmark datasets, our model is competitive with, and frequently outperforms, other, more complex models that either do not model specific annotators or were not designed for label distribution learning


# Code House Keeping

In order to run the pipeline;
1. Download the dataset needed and preprocess it. We use the dices_990 dataset for this purpose. [Repo](https://github.com/google-research-datasets/dices-dataset).  The preprocess file for this dataset is included in the `preprocess_dices.py` file. 
2.  Setup the python environment. 
3. Setup wandb and a wandb sweep. The instructions for the sweep is included in WandB Sweeps section. Remember to change the `wandb_creds.py` file with your credentials. 
4.  Run the experiments as mentioned in `run_dices_exp.sh`. 
5. Results should automatically log to the wandb project.  

We have included a `requirements.txt` with the exact version we've used to reproduce these results. In a nutshell, you need; 
6. Tensorflow (running all experiments)
7. Basic ML packages; pandas,numpy, sckit-learn, transformers, sentence-transformers, and matplotlib. 
8. WandB https://wandb.ai/home. 

## WandB and DisCo

WandB is used as the experimental management system in this pipeline. We use WandB to log our results from each run and to do hyper parameter tuning. WandB is free to setup and Pro account is free for educational users. We use the sweeps feature of WandB for [hyper parameter tuning](https://docs.wandb.ai/guides/sweeps).

### Hyperparameter Tuning with WandB Sweeps

Weights and Biases has a Hyperparameter tuning library which acts as a central server that has the tuning parameters and to run an experiment you request a parameter from the central server. 

The [documentation](https://docs.wandb.ai/guides/sweeps) at WandB for Sweeps is quite helpful. 

![enter image description here](https://i.imgur.com/UwhpqA6.png)

#### Getting Started

You need to create a project in WandB and after you create a project, go to **Sweeps** on the sidebar. Then click on the **Create Sweep**. 

##### Creating a Sweep

You will be prompted for a YAML file for Sweep definition. Here is a sample YAML file which can be used for these experiments. [The sweep documentation at Wandb is helpful in understanding/modifying this YAML file](https://docs.wandb.ai/guides/sweeps/configuration). 

    method: random
    metric:
      goal: minimize
      name: train KL
    parameters:
      act_fx:
        values:
        - softsign
        - tanh
        - relu
        - relu6
        - elu
      drop_p:
        values:
        - 0.5
        - 0.75
        - 0.3
        - 0
        - 1
      gamma_a:
        value: 1
      gamma_i:
        value: 1
      lat_a_dim:
        value: 64
      lat_dim:
        value: 128
      lat_fusion_type:
        value: concat
      lat_i_dim:
        values:
        - 128
        - 256
        - 64
        - 1024
      learning_rate:
        distribution: constant
        value: 0.001
      opt_type:
        values:
        - adam
      update_radius:
        value: -2
      weight_init_scheme:
        values:
        - gaussian
        - orthogonal
        - uniform

## Kicking off a job

Once you create a Sweep, you will be prompted with a screen that has an agent. An example is given below;

    $ wandb agent disco_exp/dices990_sweep/3whsjqn6

Here, the sweep ID is *disco_exp/dices990_sweep/3whsjqn6**. You need to pass this to the program `train_disco_sweep.py`.

Once that is linked, you can kick off an experiment through an agent. For example, 1 GPU and it is a matter of kicking off an experiment through a shell script. Here is an example, 

    $dataset=dices990
    
    python3 train_disco_sweep.py --config ./config_files/disco_config_"$dataset".cfg --sweep_id disco_exp/dices990_sweep/3whsjqn6 --gpu_id 8
    
     
##### Number of runs per Agent

By default, the code is setup to run 10 jobs through a single agent. But you can change this at `wandb.agent(sweep_id, train, count=20)`.


# Paper

We presented our work at ACL 2023. The poster and recording of the talk is available on [our website](https://cyrilw.com/publication/weerasooriya-etal-2023-disagreement/).

To cite our work; 

```
@inproceedings{weerasooriya-etal-2023-disagreement,
 address = {Toronto, Canada},
 author = {Weerasooriya, Tharindu Cyril  and
Alexander G. Ororbia II  and
Bhensadadia, Raj  and
KhudaBukhsh, Ashiqur  and
Homan, Christopher M.},
 booktitle = {Findings of the Association for Computational Linguistics: ACL 2023},
 month = {July},
 pages = {4679--4695},
 publisher = {Association for Computational Linguistics},
 title = {Disagreement Matters: Preserving Label Diversity by Jointly Modeling Item and Annotator Label Distributions with DisCo},
 url = {https://aclanthology.org/2023.findings-acl.287},
 year = {2023}
}
```