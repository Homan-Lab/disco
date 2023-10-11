import os
import sys, getopt
import tensorflow as tf
import numpy as np
import pandas as pd
from utils.utils import load_object
from collections import Counter
from sklearn.metrics import accuracy_score,classification_report,f1_score
import math 
from tqdm import tqdm
import json
import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt 


import wandb
from wandb_creds import wandb_creds 
os.environ["WANDB_API_KEY"] = wandb_creds()
wandb_entity = "disco_exps"

"""
    Trains a prototype label distributional learner (LDL) which is a simple
    neural model that jointly learns to model label distributions for
    ground truth labels, items, and annotators. The resultant model also
    acquires embeddings for items as well as for annotators.

    Example use:
    $ python evaluate_ldlnm.py --data_dir="../data/" --model_fname="../data/trained_model.ldlnm"
    $ python evaluate_ldlnm.py --data_dir=./experimental_data/jobQ1_BOTH/ --model_fname=./experimental_data/jobQ1_BOTH/trained_model.ldlnm --split_name=dev --empirical_fname=./datasets/jobQ123_BOTH/processed/jobQ1_BOTH/jobQ1_BOTH_dev_AIL.csv

    @author Alexander G. Ororbia
"""


def convert_to_majority_array(labels):
    output = []
    for label_set in labels:
        label_set = np.array(label_set)
        zero_ary = np.zeros(len(label_set))
        max_index = np.argmax(label_set)
        zero_ary[max_index] = 1
        output.append(zero_ary)
    return output

def convert_to_majority_index(labels):
    output = []
    for label_set in labels:
        label_set = np.array(label_set)
        max_index = np.argmax(label_set)
        output.append(max_index)
    return output

def KLdivergence(P, Q):
    # from Q to P
    # https://datascience.stackexchange.com/a/26318/30372
    """
    Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0.
    """
    epsilon = 0.00001

    P = P + epsilon
    Q = Q + epsilon

    return np.sum(P * np.log(P/Q))

def KL_PMI_empirical2pred(empirical_pcts, prediction_proba):

    KLsum = []


    for pair in zip(empirical_pcts, prediction_proba):

        empirical_pct = pair[0]
        prediction_pct = np.asarray(pair[1])
        
        # KL = entropy(empirical_pct, prediction_pct)
        # from prediction_pct to empirical_pct
        KL = KLdivergence(empirical_pct, prediction_pct)
        if (math.isnan(KL)):
            KL = 0.0

        KLsum.append(KL)


    KL = np.mean(KLsum)

    print('KL divergence: ', KL)

    return KL

def generate_confusion_matrix(labels_test,labels_preds,folder_path):
    label_dict = [x for x in range(len(labels_test[0]))]
    y_test = convert_to_majority_index(labels_test)
    y_pred = convert_to_majority_index(labels_preds)
    cm = confusion_matrix(y_test, y_pred, labels=label_dict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=label_dict)
    disp.plot(include_values=True,cmap=plt.cm.Blues)
    # fig, ax = plt.subplots(figsize=(20,20))
    # disp.plot(ax=ax,include_values=True,cmap=plt.cm.Blues)
    plt.title("DisCo - CM ")
    plt.savefig(folder_path)
    plt.close()

def measure_accuracy(y_test,y_pred):
    total_items = len(y_test)
    label_choices = len(y_test[0])
    matched = 0
    y_test = convert_to_majority_array(y_test)
    y_pred = convert_to_majority_array(y_pred)

    for y_test_item,y_pred_item in zip(y_test,y_pred):
        matched += accuracy_score(y_test_item, y_pred_item)
    accuracy = float(matched/total_items)
    return accuracy

def measure_f1(y_test,y_pred):

    y_test = convert_to_majority_index(y_test)
    y_pred = convert_to_majority_index(y_pred)
    precision = {}
    recall = {}
    f1_macro = f1_score(y_test, y_pred, average='macro')

    f1_micro = f1_score(y_test, y_pred, average='micro')

    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    results = classification_report(y_test, y_pred, digits=3,output_dict=True)
    precision['macro'] = results['macro avg']['precision']
    precision['weighted'] = results['weighted avg']['precision']

    recall['macro'] = results['macro avg']['recall']
    recall['weighted'] = results['weighted avg']['recall']
    return f1_macro,f1_micro,f1_weighted,precision,recall

def write_model_logs_to_json(MODEL_LOG_DIR, results_dict, output_name):
    with open(MODEL_LOG_DIR +"/"+ output_name + ".json", "a") as fp:
        json.dump(results_dict, fp, sort_keys=True, indent=4,default=str)

def write_results_to_wandb(wandb_name,results,model_type,dataset):

    wandb.init(project=wandb_name, entity=wandb_entity,name=dataset)
    wandb.config = {
        "model": model_type,
        "dataset": dataset
    }
    results_to_write = results
    results_to_write["dataset"] = dataset
    wandb.log(results_to_write)

def main():
    options, remainder = getopt.getopt(sys.argv[1:], '', ["data_dir=","model_fname=","gpu_id=","split_name=","dataset_name=","wandb_name=","empirical_fname="])
    # Collect arguments from argv
    batch_size = 100
    data_dir = None
    model_fname = None
    dataset_name = None
    wandb_name = None
    use_gpu = False
    gpu_id = -1
    for opt, arg in options:
        if opt in ("--data_dir"):
            data_dir = arg.strip()
        elif opt in ("--model_fname"):
            model_fname = arg.strip()
        elif opt in ("--gpu_id"):
            gpu_id = int(arg.strip())
            use_gpu = True
        elif opt in ("--split_name"):
            split_name = arg.strip()
        elif opt in ("--dataset_name"):
            dataset_name = arg.strip()
        elif opt in ("--wandb_name"):
            wandb_name = arg.strip()
        elif opt in ("--empirical_fname"):
            empirical_fname = arg.strip()
    mid = gpu_id
    if use_gpu:
        print(" > Using GPU ID {0}".format(mid))
        os.environ["CUDA_VISIBLE_DEVICES"]="{0}".format(mid)
        gpu_tag = '/GPU:0'
        #tf.config.experimental.set_memory_growth(gpu, True)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"
        gpu_tag = '/CPU:0'
    
    ################################################################################
    # load in design matrices
    ################################################################################
    Xi = np.load("{0}Xi_{1}.npy".format(data_dir,split_name), allow_pickle=True)
    Yi = np.load("{0}Yi_{1}.npy".format(data_dir,split_name), allow_pickle=True)
    Ya = np.load("{0}Ya_{1}.npy".format(data_dir,split_name), allow_pickle=True)
    Y = np.load("{0}Y_{1}.npy".format(data_dir,split_name), allow_pickle=True)
    I = np.load("{0}I_{1}.npy".format(data_dir,split_name), allow_pickle=True)
    A = np.load("{0}A_{1}.npy".format(data_dir,split_name), allow_pickle=True)
    raw_item_annotator_label = pd.read_csv(empirical_fname,names=['item','annotator','label','message']) #np.load("{0}A_{1}.npy".format(data_dir,split_name))
    unique_dataitems = pd.unique(raw_item_annotator_label['item'])
    
    n_i = np.max(I) + 1 # 2000 # number items
    n_a = np.max(A) + 1 #50 # number annotators
    yi_dim = Yi.shape[1]
    ya_dim = Ya.shape[1]
    y_dim = Y.shape[1]

    print("Y.shape = ",Y.shape)
    print("Yi.shape = ",Yi.shape)
    print("Ya.shape = ",Ya.shape)
    print("I.shape = ",I.shape)
    print("A.shape = ",A.shape)
    empirical_labels = []
    predicitions = []
    data_to_write = []
    for unique_dataitem in tqdm(unique_dataitems):
        item_index = raw_item_annotator_label.loc[raw_item_annotator_label['item'] == unique_dataitem].index[0] #there is a chance of multiple rows returned.
        items = raw_item_annotator_label.loc[raw_item_annotator_label['item'] == unique_dataitem].head(1)
        message = items['message'].values[0]
        empirical_label = Yi[item_index]
        empirical_labels.append(empirical_label)
        ################################################################################
        # set up label distributional learning model
        ################################################################################
        with tf.device(gpu_tag):
            eps = 1e-7
            model = load_object(model_fname)
            model.drop_p = 0.0
            # check model's accuracy on the dataset it was fit on
            # y_ind = tf.cast(tf.argmax(tf.cast(Y,dtype=tf.float32),1),dtype=tf.int32)
            # acc, L, _, _ = calc_stats(model, Xi, Yi, Ya, Y, A, I, batch_size)
            # print(" Acc = {0}  L = {1}".format(acc,L))

            ############################################################################
            # choose particular sample form the dataset to see how to use model at test time
            ############################################################################
            s_ptr = item_index #int( tf.argmax(comp) )
            x_s = tf.cast(np.expand_dims(Xi[s_ptr,:],axis=0),dtype=tf.float32)
            y_s = tf.cast(np.expand_dims(Y[s_ptr,:],axis=0),dtype=tf.float32)
            y_lab = int(tf.argmax(y_s,axis=1))
            yi_s = tf.cast(np.expand_dims(Yi[s_ptr,:],axis=0),dtype=tf.float32)
            ya_s = tf.cast(np.expand_dims(Ya[s_ptr,:],axis=0),dtype=tf.float32)
            i_s = I[s_ptr,:]
            a_s = A[s_ptr,:]

            py, _ = model.decode_y_ensemble(x_s)
            yhat_set = tf.argmax(py,axis=1).numpy().tolist()
            
            predicted_label = Counter(yhat_set)
            total_labels = sum(predicted_label.values())
            predicted_label_dist = [predicted_label[x]/total_labels for x in range(len(empirical_label))]

            if len(predicted_label_dist) == len(empirical_label):
                predicitions.append(predicted_label_dist)
                row_to_write = {}
                row_to_write['message'] = message
                row_to_write['message_id'] = item_index
                row_to_write['labels'] = predicted_label_dist
                data_to_write.append(row_to_write)
            else:
                print("Label class mismatch")
                sys.exit()

    results = {}
    print("Size of empirical label set {0}*{1} | Shape of predicted label set {2}*{3}".format(len(empirical_labels),len(empirical_labels[0]),len(predicitions),len(predicitions[0])))
    results['KL'] = KL_PMI_empirical2pred(empirical_labels,predicitions)
    results['accuracy'] = measure_accuracy(empirical_labels,predicitions)
    generate_confusion_matrix(empirical_labels,predicitions,data_dir+"/cmatrix.pdf")
    results['f1_macro'],results['f1_micro'],results['f1_weighted'],results['precision'],results['recall'] = measure_f1(empirical_labels,predicitions)
    results['timestamp'] = datetime.datetime.now()
    print("KL {0} | Accuracy {1} | F1 macro {2} | F1 micro {3} | F1 weighted {4} | Precision {5} | Recall {6}".format(results['KL'],results['accuracy'],results["f1_macro"], results['f1_micro'],results['f1_weighted'],results['precision'],results['recall']))
    write_results_to_wandb(wandb_name,results,"DisCo",dataset_name)
    pd.DataFrame(data_to_write).to_excel(data_dir+"/preds.xlsx")


if __name__== "__main__":
    main()
