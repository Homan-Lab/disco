import os
import sys, getopt, optparse
import pickle
import numpy as np
import tensorflow as tf
# sys.path.insert(0, 'utils/')
from utils.utils import scale_feat, gen_data_plot
import pandas as pd
import pdb


"""
    Transforms label dist files into a set of design matrices for easier
    handling in the training and analysis scripts.
    Example use:
    $ python gen_disco_dataset.py --inp_dir=../ --out_dir=../data/ --annotator_item_fname=annotator_items.txt
                                --item_lab_fname=item_labels.txt --annotator_lab_fname=annotator_labels.txt

    @author DisCo Authors
"""

def create_folder(folderpath):
    # Check whether the specified path exists or not
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)


def strToVec(str, delim=" "):
    vec_s = str.split(delim)
    vec = []
    for vs in vec_s:
        try:
            vec.append(int(vs))
        except:
            continue
            # pdb.set_trace()
    vec = np.expand_dims(np.asarray(vec),axis=0)
    return vec

def fileToMap(fname):
    print(" > Reading file: ",fname)
    map = {}
    fd = open(fname, 'r')
    count = 0
    line = fd.readline()
    while line:
        count += 1
        tok = line.split(",")
        if len(tok) > 1:
            idx = int(tok[0])
            vec_s = tok[1].replace('[', '').replace(']', '')
            map.update({idx: vec_s})
        line = fd.readline()
        print("\r {0} lines read".format(count),end="")
    print()
    return map

'''
out_dir = "/experimental_data/"
inp_dir = "/datasets/"
annotator_item_fname = "{0}{1}".format(inp_dir,"annotator_items.txt")
item_lab_fname = "{0}{1}".format(inp_dir,"item_labels.txt")
annotator_lab_fname = "{0}{1}".format(inp_dir,"annotator_labels.txt")
'''

################################################################################
# read in configuration file and extract necessary variables/constants
options, remainder = getopt.getopt(sys.argv[1:], '', ["inp_dir=","out_dir=","annotator_item_fname=","item_lab_fname=","annotator_lab_fname=","embeddings=","split_name="])
# Collect arguments from argv
inp_dir = None
out_dir = None
annotator_item_fname = None
item_lab_fname = None
annotator_lab_fname = None
for opt, arg in options:
    if opt in ("--inp_dir"):
        inp_dir = arg.strip()
    elif opt in ("--out_dir"):
        out_dir = arg.strip()
    elif opt in ("--annotator_item_fname"):
        annotator_item_fname = arg.strip()
    elif opt in ("--item_lab_fname"):
        item_lab_fname = arg.strip()
    elif opt in ("--annotator_lab_fname"):
        annotator_lab_fname = arg.strip()
    elif opt in ("--embeddings"):
        embeddings_fname = arg.strip()
    elif opt in ("--split_name"):
        split_name = arg.strip()
################################################################################

#data_map = fileToMap(annotator_item_fname)
yi_map = fileToMap("{0}{1}".format(inp_dir,item_lab_fname))

ya_map = fileToMap("{0}{1}".format(inp_dir,annotator_lab_fname))
embeddings = np.load("{0}{1}".format(inp_dir,embeddings_fname),allow_pickle=True)

embeddings = pd.DataFrame(embeddings)
embeddings.columns = ['data_i','embedding']
Yi = None # item label design matrix
Ya = None # annotator label design matrix
Y = None # label design matrix
X = None
Ii = [] # item id matrix
Ai = [] # annotator id matrix

print(" > Reading data file: ","{0}{1}".format(inp_dir,annotator_item_fname))
fd = open("{0}{1}".format(inp_dir,annotator_item_fname), 'r')
count = 0
line = fd.readline()


while line: #item,annotator,label
    count += 1
    tok = line.split(",")
    dl_row = []
    if len(tok) > 1:
        item_id = int(tok[0])
        annot_id = int(tok[1])
        embed = embeddings.loc[embeddings['data_i'] == item_id]
        embed = embed['embedding'].values

        embed = np.asarray(embed[0])
        embed = embed.astype('float32')
        embed = np.expand_dims(np.asarray(embed),axis=0)

        y_lab = strToVec(tok[2].replace('[', '').replace(']', ''))
        yi_lab = strToVec(yi_map.get(item_id))
        ya_lab = strToVec(ya_map.get(annot_id))
        #print("Y_lab",y_lab)
        #print("Yi_lab",yi_lab)
        #print("Ya_lab",ya_lab)
        # normalize yi and ya to probability dist vectors
        yi_lab = yi_lab / (np.sum(yi_lab) + 1e-8)
        ya_lab = ya_lab / (np.sum(ya_lab) + 1e-8)
        #print("Yi_lab",yi_lab)
        #print("Ya_lab",ya_lab)

        if count > 1:
            #print("Y",Y)
            #print("Y_lab",y_lab)
            Y = np.concatenate((Y,y_lab),axis=0)
            #pdb.set_trace()
            #print("Yi",Yi)
            #print("Yi_lab",yi_lab)
            Yi = np.concatenate((Yi,yi_lab),axis=0)
            #pdb.set_trace()
            #print("Ya",Ya)
            #print("Ya_lab",ya_lab)
            Ya = np.concatenate((Ya,ya_lab),axis=0)
            #pdb.set_trace()
            X = np.concatenate((X,embed),axis=0)
        else:
            Y = y_lab
            Yi = yi_lab
            Ya = ya_lab
            X = embed
        Ii.append(item_id)
        Ai.append(annot_id)


    print("\r {0} data lines read".format(count),end="")
    line = fd.readline()
Ilist = Ii
Ii = np.expand_dims(np.asarray(Ii),axis=1)
n_i = np.max(Ii) + 1
nC = Y.shape[1]
Ai = np.expand_dims(np.asarray(Ai),axis=1)
print(n_i)
Yn = tf.cast(Y,dtype=tf.float32)


Xi = X
# save design matrices to disk
print("Xi.shape = ",Xi.shape)
print("Yi.shape = ",Yi.shape)
print("Ya.shape = ",Ya.shape)
print("Y.shape = ",Y.shape)
print("Ii.shape = ",Ii.shape)
print("Ai.shape = ",Ai.shape)

# save design matrices to disk for easier loading in subsequent Python scripts
print(" >> Saving distribution design matrices to dir: {0}".format(out_dir))
create_folder(out_dir)
np.save("{0}Xi_{1}.npy".format(out_dir, split_name), Xi)
np.save("{0}Yi_{1}.npy".format(out_dir, split_name), Yi)
np.save("{0}Ya_{1}.npy".format(out_dir, split_name), Ya)
np.save("{0}Y_{1}.npy".format(out_dir, split_name), Y)
np.save("{0}I_{1}.npy".format(out_dir, split_name), Ii)
np.save("{0}A_{1}.npy".format(out_dir, split_name), Ai)

gen_data_plot(Xi, Yn, use_tsne=False,fname=split_name,out_dir=out_dir)
