# Hyperparameter Tuning with WandB Sweeps

Weights and Biases has a Hyperparameter tuning library which acts as a central server that has the tuning parameters and to run an experiment you request a parameter from the central server. 

The [documentation](https://docs.wandb.ai/guides/sweeps) at WandB for Sweeps is quite helpful. 

![enter image description here](https://i.imgur.com/UwhpqA6.png)

# Getting Started

You need to create a project in WandB and after you create a project, go to **Sweeps** on the sidebar. Then click on the **Create Sweep**. 

## Creating a Sweep

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

    $ wandb agent disco_exp/sbic_sweep/3whsjqn6

Here, the sweep ID is *disco_exp/sbic_sweep/3whsjqn6**. You need to pass this to the program `train_disco_sweep.py`.

Once that is linked, you can kick off an experiment through an agent. For example, 1 GPU and it is a matter of kicking off an experiment through a shell script. Here is an example, 

    $dataset=SBIC
    
    python3 train_disco_sweep.py --config ./config_files/disco_config_"$dataset".cfg --sweep_id disco_exp/sbic_sweep/3whsjqn6 --gpu_id 8
    
     
## Number of runs per Agent

By default, the code is setup to run 10 jobs through a single agent. But you can change this at `wandb.agent(sweep_id, train, count=20)`.
