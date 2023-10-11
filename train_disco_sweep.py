import os
import sys, getopt, optparse
import pickle
import tensorflow as tf
import numpy as np

# sys.path.insert(0, 'utils/')
# sys.path.insert(0, 'model/')
from model.disco import disco
from utils.config import Config
from utils.utils import save_object, sample_gaussian, calc_catNLL, gen_data_plot, calc_mode, D_KL
from evaluate_disco import convert_to_majority_index
from sklearn.metrics import accuracy_score,classification_report,f1_score
import argparse
import pdb
import random
import wandb
from wandb_creds import wandb_creds 
os.environ["WANDB_API_KEY"] = wandb_creds()
"""
    Trains a prototype label distributional learning neural model (LDL-NM) which
    is an artificial neural network that jointly learns to model label distributions
    for ground truth labels, items, and annotators. The resultant model can be
    used to iteratively infer embeddings for annotators (or be used to
    conduct majority/modal voting across its memory of known annotators).

    Here is an example run `python3 train_disco_sweep.py --config ./config_files/disco_config.cfg --sweep_id rit_pl/jobq2_sweep/smd8rl2y --gpu_id 8`

    In order to run this, you initially need to create a Sweep on Weights and Biases. 
"""



def wandb_logging_dev(disco_model_params,epoch,agg_acc, KLi, dev_agg_acc, dev_KLi, f1_macro, dev_f1_macro, precision_macro, dev_precision_macro, recall_macro, dev_recall_macro):
    wandb_log = {
        "train KL": KLi,
        "train F1": f1_macro,
        "train Accuracy": agg_acc,
        "train precision": precision_macro,
        "train recall": recall_macro,
        "dev KL": dev_KLi,
        "dev F1": dev_f1_macro,
        "dev Accuracy": dev_agg_acc,
        "dev precision": dev_precision_macro,
        "dev recall": dev_recall_macro,
        "epoch": epoch,
        "dataset": disco_model_params['dataset']
        # 'model_params': disco_model_params
        }

    # logging accuracy
    wandb.log(wandb_log)   


def wandb_logging_train(disco_model_params,epoch,agg_acc, KLi, f1_macro, precision_macro, recall_macro):
    wandb_log = {
        "train KL": KLi,
        "train F1": f1_macro,
        "train Accuracy": agg_acc,
        "train precision": precision_macro,
        "train recall": recall_macro,
        "epoch": epoch,
        "dataset": disco_model_params['dataset']
        # 'model_params': disco_model_params
        }

    # logging accuracy
    wandb.log(wandb_log)   

def split(design_mat, n_valid=10):  # a simple design matrix splitting function if needed
    valid_mat = design_mat[0:n_valid, :]
    train_mat = design_mat[n_valid:design_mat.shape[0], :]
    return valid_mat, train_mat


def calc_stats(model, Xi_, Yi_, Ya_, Y_, A_, I_, batch_size, agg_type="mode",
               n_subset=1000, eval_aggreg=True):
    """
        Calculates fixed-point statistics, i.e., accuracy and cost
    """
    drop_p = model.drop_p + 0
    model.drop_p = 0.0  # turn off drop-out
    Xi = Xi_
    Yi = Yi_
    Ya = Ya_
    Y = Y_
    A = A_
    I = I_
    if n_subset > 0:
        ptrs = np.random.permutation(Y.shape[0])[0:n_subset]
        Xi = Xi_[ptrs, :]
        Yi = Yi_[ptrs, :]
        Ya = Ya_[ptrs, :]
        Y = Y_[ptrs, :]
        A = A_[ptrs, :]
        I = I_[ptrs, :]

    ptrs = np.arange(Y.shape[0])
    ptr_s = 0
    ptr_e = batch_size
    mark = 0
    L = 0.0
    KLi = 0.0
    KLa = 0.0
    agg_KL = 0.0
    acc = 0.0
    agg_acc = 0.0  # aggregated accuracy
    Ns = 0.0
    while ptr_s < len(ptrs):
        if ptr_e > len(ptrs):
            ptr_e = len(ptrs)
        ptr_indx = ptrs[ptr_s:ptr_e]
        ptr_s += len(ptr_indx)
        ptr_e += len(ptr_indx)
        # sample without replacement the label distribution data
        i_s = I[ptr_indx, :]
        a_s = A[ptr_indx, :]
        xi_s = tf.cast(Xi[ptr_indx, :], dtype=tf.float32)
        y_s = tf.cast(Y[ptr_indx, :], dtype=tf.float32)
        y_ind = tf.cast(tf.argmax(tf.cast(y_s, dtype=tf.float32), 1), dtype=tf.int32)
        yi_s = tf.cast(Yi[ptr_indx, :], dtype=tf.float32)
        ya_s = tf.cast(Ya[ptr_indx, :], dtype=tf.float32)

        z = model.encode(xi_s, a_s)
        pY, _ = model.decode_y(z)
        pYi, _ = model.decode_yi(z)
        pYa, _ = model.decode_ya(z)
        # Ly = calc_catNLL(target=y_s,prob=pY,keep_batch=True)
        # L += tf.reduce_sum(Ly)
        Ns = y_s.shape[0]
        L_t, Ly_t, KLi_t, KLa_t = model.calc_loss(y_s, yi_s, ya_s, pY, pYi,
                                                  pYa)  # compute cost (loss over entire design matrices)
        L += L_t * Ns
        KLi += KLi_t * Ns
        KLa += KLa_t * Ns

  
        # compute accuracy of predictions
        y_pred = tf.cast(tf.argmax(pY, 1), dtype=tf.int32)
        comp = tf.cast(tf.equal(y_pred, y_ind), dtype=tf.float32)
        acc += tf.reduce_sum(comp)

        # compute aggregated accuracy across internally known annotators
        sub_acc = 0.0

        if eval_aggreg is True:
            for s in range(xi_s.shape[0]):
                xs = tf.expand_dims(xi_s[s, :], axis=0)
                ys = tf.expand_dims(y_s[s, :], axis=0)
                ys_ind = tf.cast(tf.argmax(tf.cast(ys, dtype=tf.float32), 1), dtype=tf.int32)
                py, _ = model.decode_y_ensemble(xs)
                y_label_preds = tf.reduce_mean(py, axis=0, keepdims=True)
                agg_KL +=  D_KL(ys,y_label_preds) #* Ns
                if agg_type == "mode":
                    yhat_set = tf.argmax(py, axis=1).numpy().tolist()
                    y_mode, y_freq = calc_mode(yhat_set)  # compute mode of predictions
                    comp = tf.cast(tf.equal(y_mode, ys_ind), dtype=tf.float32)
                    sub_acc += tf.reduce_sum(comp)
                else:  # == "expectation"
                    y_mean = tf.reduce_mean(py, axis=0, keepdims=True)
                    y_pred = tf.cast(tf.argmax(y_mean, 1), dtype=tf.int32)
                    comp = tf.cast(tf.equal(y_pred, ys_ind), dtype=tf.float32)
                    sub_acc += tf.reduce_sum(comp)

        agg_acc += sub_acc

    acc = acc / (Y.shape[0] * 1.0)
    L = L / (Y.shape[0] * 1.0)
    KLi = KLi / (Y.shape[0] * 1.0)
    KLa = KLa / (Y.shape[0] * 1.0)
    agg_KL = agg_KL / (Y.shape[0] * 1.0)
    agg_acc = agg_acc / (Y.shape[0] * 1.0)
    model.drop_p = drop_p  # turn dropout back on

    # classification report using y_pred and y_ind
    # y_pred = convert_to_majority_index(y_pred)
    # y_test = convert_to_majority_index(y_ind)
    y_test = y_ind
    f1_macro = f1_score(y_test, y_pred, average='macro')

    f1_micro = f1_score(y_test, y_pred, average='micro')

    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    results = classification_report(y_test, y_pred, digits=3, output_dict=True)
    precision_macro = results['macro avg']['precision']
    precision_weighted = results['weighted avg']['precision']

    recall_macro = results['macro avg']['recall']
    recall_weighted = results['weighted avg']['recall']

    return acc, L, KLi, KLa, agg_acc, f1_macro, f1_micro, f1_weighted, precision_macro, precision_weighted, recall_macro, recall_weighted, agg_KL


def get_config_file(options):
    # Collect arguments from argv
    cfg_fname = None
    use_gpu = False
    gpu_id = -1
    for opt, arg in options:
        if opt in ("--cfg_fname"):
            cfg_fname = arg.strip()
        elif opt in ("--gpu_id"):
            gpu_id = int(arg.strip())
            use_gpu = True
    mid = gpu_id
    if use_gpu:
        print(" > Using GPU ID {0}".format(mid))
        os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(mid)
        gpu_tag = '/GPU:0'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        gpu_tag = '/CPU:0'

    return cfg_fname, gpu_tag


def get_params(cfg_fname):
    args = Config(cfg_fname)
    params = {
        "xi_fname": args.getArg("xi_fname"),
        "yi_fname": args.getArg("yi_fname"),
        "ya_fname": args.getArg("ya_fname"),
        "i_fname": args.getArg("i_fname"),
        "a_fname": args.getArg("a_fname"),
        "y_fname": args.getArg("y_fname"),
        "dev_xi_fname": args.getArg("dev_xi_fname"),
        "dev_yi_fname": args.getArg("dev_yi_fname"),
        "dev_ya_fname": args.getArg("dev_ya_fname"),
        "dev_i_fname": args.getArg("dev_i_fname"),
        "dev_a_fname": args.getArg("dev_a_fname"),
        "dev_y_fname": args.getArg("dev_y_fname"),
        "out_dir": args.getArg("out_dir"),
    }

    simulation_params = {
        "n_epoch": int(args.getArg("n_epoch")),  # 100 #200 #500 #200 #50
        "batch_size": int(args.getArg("batch_size")),  # 100 #256 #128 #64 #256 #128
        "save_every": int(args.getArg("save_every")),  # 10
        "eval_every": int(args.getArg("eval_every"))
    }

    disco_model_params = {
        "act_fx": args.getArg("act_fx"),
        "weight_init_scheme": args.getArg("weight_init_scheme"),
        "lat_dim": int(args.getArg("lat_dim")),  # 100 #64 #32
        "opt_type": args.getArg("opt_type"),
        "learning_rate": float(args.getArg("learning_rate")),
        # "max_param_norm": float(args.getArg("max_param_norm")),
        "update_radius": float(args.getArg("update_radius")),
        "lat_fusion_type": args.getArg("lat_fusion_type"),  # "concat" "sum"
        "lat_i_dim": int(args.getArg("lat_i_dim")),  # 100
        "lat_a_dim": int(args.getArg("lat_a_dim")),  # 50
        "drop_p": float(args.getArg("drop_p")),
        "gamma_i": float(args.getArg("gamma_i")),
        "gamma_a": float(args.getArg("gamma_a"))
    }

    return params, simulation_params, disco_model_params


def read_data(params):
    data = {}
    data["Xi"] = np.load(params["xi_fname"], allow_pickle=True)
    data["Y"] = np.load(params["y_fname"], allow_pickle=True)
    gen_data_plot(data["Xi"], data["Y"], use_tsne=False)
    data["Yi"] = np.load(params["yi_fname"])
    data["Ya"] = np.load(params["ya_fname"])
    data["I"] = np.load(params["i_fname"])
    data["A"] = np.load(params["a_fname"])
    data["dev_Xi"] = None
    data["dev_Yi"] = None
    data["dev_Ya"] = None
    data["dev_Y"] = None
    data["dev_I"] = None
    data["dev_A"] = None
    if params["dev_y_fname"] is not None:
        data["dev_Xi"] = np.load(params["dev_xi_fname"])
        data["dev_Yi"] = np.load(params["dev_yi_fname"])
        data["dev_Ya"] = np.load(params["dev_ya_fname"])
        data["dev_Y"] = np.load(params["dev_y_fname"])
        data["dev_I"] = np.load(params["dev_i_fname"])
        data["dev_A"] = np.load(params["dev_a_fname"])
    # automatically count num of total items, total annotators, and get design matrix dimensions
    data["n_i"] = np.max(data["I"]) + 1  # 2000 # number items
    data["n_a"] = np.max(data["A"]) + 1  # 50 # number annotators
    data["yi_dim"] = data["Yi"].shape[1]
    data["ya_dim"] = data["Ya"].shape[1]
    data["y_dim"] = data["Y"].shape[1]
    data["n_xi"] = data["Xi"].shape[1]

    print("Xi.shape = ", data["Xi"].shape)
    print("Yi.shape = ", data["Yi"].shape)
    print("Y.shape = ", data["Y"].shape)
    print("Ya.shape = ", data["Ya"].shape)
    print("I.shape = ", data["I"].shape)
    print("A.shape = ", data["A"].shape)

    return data


def train_disco(data, simulation_params, disco_model_params, params):
    model = disco(xi_dim=data["n_xi"], yi_dim=data["yi_dim"], ya_dim=data["ya_dim"], y_dim=data["y_dim"],
                  a_dim=data["n_a"],
                  lat_dim=disco_model_params["lat_dim"], act_fx=disco_model_params["act_fx"],
                  init_type=disco_model_params["weight_init_scheme"],
                  lat_i_dim=disco_model_params["lat_i_dim"], lat_a_dim=disco_model_params["lat_a_dim"],
                  lat_fusion_type=disco_model_params["lat_fusion_type"],
                  drop_p=disco_model_params["drop_p"], gamma_i=disco_model_params["gamma_i"],
                  gamma_a=disco_model_params["gamma_a"])
    model.set_opt(disco_model_params["opt_type"], disco_model_params["learning_rate"])

    # Z = model.encode(Xi, A)
    # gen_data_plot(Z, Y, use_tsne=False, fname="latents")

    ################################################################################
    # fit model to the design matrices
    ################################################################################

    # wandb_initialize(disco_model_params,simulation_params["n_epoch"])
    acc, L, KLi, KLa, agg_acc, f1_macro, f1_micro, f1_weighted, precision_macro, precision_weighted, recall_macro, recall_weighted,train_agg_KL = calc_stats(model, data["Xi"], data["Yi"], data["Ya"], data["Y"], data["A"], data["I"],
                                           simulation_params["batch_size"])
    if data["dev_Y"] is not None:

        dev_acc, dev_L, dev_KLi, dev_KLa, dev_agg_acc, dev_f1_macro, dev_f1_micro, dev_f1_weighted, dev_precision_macro, dev_precision_weighted, dev_recall_macro, dev_recall_weighted,dev_agg_KL = calc_stats(model, data["dev_Xi"], data["dev_Yi"],
                                                                   data["dev_Ya"], data["dev_Y"], data["dev_A"],
                                                                   data["dev_I"], simulation_params["batch_size"])

        print(
            " {0}: Fit.Acc = {1} E.Acc = {2} L = {3} | Dev.Acc = {4} E.Acc = {5}  KL = {6} ".format(-1, acc, agg_acc,
                                                                                                    dev_acc, dev_L,
                                                                                                    dev_agg_acc,
                                                                                                    dev_agg_KL))
    else:
        print(
            " {0}: Fit.Acc = {1}  E.Acc = {2} L = {3}  KLi = {4}  KLa = {5}".format(-1, acc, agg_acc, L, KLi, KLa))
    for e in range(simulation_params["n_epoch"]):
        ptrs = np.random.permutation(data["Y"].shape[0])
        ptr_s = 0
        ptr_e = simulation_params["batch_size"]
        mark = 0
        L = 0.0  # epoch loss
        Ns = 0.0
        while ptr_s < len(ptrs):
            if ptr_e > len(ptrs):
                ptr_e = len(ptrs)
            ptr_indx = ptrs[ptr_s:ptr_e]
            ptr_s += len(ptr_indx)
            ptr_e += len(ptr_indx)
            # sample without replacement the label distribution data
            i_s = data["I"][ptr_indx, :]
            a_s = data["A"][ptr_indx, :]
            y_s = tf.cast(data["Y"][ptr_indx, :], dtype=tf.float32)
            xi_s = tf.cast(data["Xi"][ptr_indx, :], dtype=tf.float32)
            yi_s = tf.cast(data["Yi"][ptr_indx, :], dtype=tf.float32)
            ya_s = tf.cast(data["Ya"][ptr_indx, :], dtype=tf.float32)
            mark += 1

            # update model parameters and track approximate training loss
            L_t = model.update(xi_s, a_s, yi_s, ya_s, y_s, disco_model_params["update_radius"])
            L = (L_t * y_s.shape[0]) + L
            Ns += y_s.shape[0]
            print("\r{0}: L = {1}  ({2} samples seen)".format(e, (L / Ns), Ns), end="")
        print()
        if e % simulation_params["eval_every"] == 0:
            acc, L, KLi, KLa, agg_acc, f1_macro, f1_micro, f1_weighted, precision_macro, precision_weighted, recall_macro, recall_weighted, train_agg_KL = calc_stats(model, data["Xi"], data["Yi"], data["Ya"], data["Y"], data["A"],
                                                   data["I"], simulation_params["batch_size"])
            if data["dev_Y"] is not None:
                dev_acc, dev_L, dev_KLi, dev_KLa, dev_agg_acc, dev_f1_macro, dev_f1_micro, dev_f1_weighted, dev_precision_macro, dev_precision_weighted, dev_recall_macro, dev_recall_weighted, dev_agg_KL = calc_stats(model, data["dev_Xi"], data["dev_Yi"],
                                                                           data["dev_Ya"], data["dev_Y"], data["dev_A"],
                                                                           data["dev_I"],
                                                                           simulation_params["batch_size"])
                wandb_logging_dev(disco_model_params,e,agg_acc, train_agg_KL, dev_agg_acc, dev_agg_KL, f1_macro, dev_f1_macro, precision_macro, dev_precision_macro, recall_macro, dev_recall_macro)
                print(" {0}: Fit.Acc = {1} E.Acc = {2} L = {3} | Dev.Acc = {4} E.Acc = {5}  KL = {6} ".format(e, acc,
                                                                                                              agg_acc,
                                                                                                              dev_acc,
                                                                                                              dev_L,
                                                                                                              dev_agg_acc,
                                                                                                              dev_agg_KL))
            else:
                wandb_logging_train(disco_model_params,e,agg_acc, train_agg_KL, f1_macro, precision_macro, recall_macro)
                print(
                    " {0}: Fit.Acc = {1}  E.Acc = {2} L = {3}  KLi = {4}  KLa = {5}".format(e, acc, agg_acc, L, KLi,
                                                                                            KLa))
        if e % simulation_params["save_every"] == 0:  # save a checkpoint model
            save_object(model, "{0}trained_model.disco".format(params["out_dir"]))

    ################################################################################
    # save final model to disk
    ################################################################################
    save_object(model, "{0}trained_model.disco".format(params["out_dir"]))
    if data["dev_Y"] is not None:
        wandb_logging_dev(disco_model_params,e,agg_acc, train_agg_KL, dev_agg_acc, dev_agg_KL, f1_macro, dev_f1_macro, precision_macro, dev_precision_macro, recall_macro, dev_recall_macro)
    else:
        wandb_logging_train(disco_model_params,e,agg_acc, train_agg_KL, f1_macro, precision_macro, recall_macro)
    # wandb_logging(disco_model_params,e,agg_acc, KLi, dev_agg_acc, dev_KLi, f1_macro, dev_f1_macro, precision_macro, dev_precision_macro, recall_macro, dev_recall_macro)
    # write_to_wandb(disco_model_params,agg_acc, KLi, dev_agg_acc, dev_KLi, f1_macro, dev_f1_macro, precision_macro, dev_precision_macro, recall_macro, dev_recall_macro)
    #return agg_acc, KLi, dev_agg_acc, dev_KLi, f1_macro, dev_f1_macro, precision_macro, dev_precision_macro, recall_macro, dev_recall_macro


def read_wandb_sweep_id(sweep_id,params, simulation_params, gpu_tag,run_count,disco_model_params):
    data = read_data(params)
    def train(config=None):
        with wandb.init(config=config):
            disco_model_params = wandb.config
            with tf.device(gpu_tag):
                train_disco(data, simulation_params, disco_model_params, params)

    wandb.agent(sweep_id, train, count=run_count)


def main():
    ################################################################################
    # read in configuration file and extract necessary variables/constants
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Config file")
    parser.add_argument("--sweep_id", help="Sweep ID from WANDB")
    parser.add_argument("--gpu_id", help="GPU id",default=-1)
    parser.add_argument("--run_count", help="Run Count",default=10)
    # parser.add_argument("--project_name", help="Name of the project")
    args = parser.parse_args()
    cfg_fname = args.config
    sweep_id = args.sweep_id
    # folder = args.folder
    gpu_id = int(args.gpu_id)
    run_count = int(args.run_count)
    if gpu_id>-1:
        print(" > Using GPU ID {0}".format(gpu_id))
        os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(gpu_id)
        gpu_tag = '/GPU:0'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        gpu_tag = '/CPU:0'
    # project_name = args.project_name

    # options, remainder = getopt.getopt(sys.argv[1:], '', ["cfg_fname=", "gpu_id="])

    # cfg_fname, gpu_tag = get_config_file(config_file)
    # gpu_tag = 8
    params, simulation_params, disco_model_params = get_params(cfg_fname)
    read_wandb_sweep_id(sweep_id,params, simulation_params, gpu_tag,run_count,disco_model_params)

    # return agg_acc, KLi, dev_agg_acc, dev_KLi, f1_macro, dev_f1_macro, precision_macro, dev_precision_macro, recall_macro, dev_recall_macro


if __name__ == '__main__':
    main()