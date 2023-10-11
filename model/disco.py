import tensorflow as tf
import sys
# sys.path.insert(0, '../utils/')
sys.path.append('../utils')
from utils.utils import softmax, init_weights, calc_catNLL, calc_gaussian_KL, calc_gaussian_KL_simple, D_KL, mse, calc_mode, drop_out,l1_l2_norm_calculation
import numpy as np
import copy

class DISCO:
    """
        The proposed DisCo model. See paper for details.

        @author DisCo Authors
    """
    def __init__(self, xi_dim, yi_dim, ya_dim, y_dim, a_dim, lat_i_dim=20, lat_a_dim=30,
                 lat_dim=10, act_fx="softsign", init_type="gaussian", name="disco",
                 i_dim=-1, lat_fusion_type="sum", drop_p=0.0, gamma_i=1.0, gamma_a=1.0,l1_norm=0.0,l2_norm=0.0):
        self.name = name
        self.seed = 69
        self.gamma_i = gamma_i #1.0
        self.gamma_a = gamma_a #1.0
        self.lat_fusion_type = lat_fusion_type
        self.i_dim = i_dim
        self.a_dim = a_dim
        self.y_dim = y_dim
        self.xi_dim = xi_dim
        self.yi_dim = yi_dim
        self.ya_dim = ya_dim
        self.lat_dim = lat_dim
        self.lat_i_dim = lat_i_dim
        self.lat_a_dim = lat_a_dim
        self.drop_p = drop_p #0.5
        self.l1_norm = l1_norm
        self.l2_norm = l2_norm

        if self.lat_fusion_type != "concat":
            self.lat_a_dim = self.lat_i_dim
            print(" > Setting lat_a.dim equal to lat_i.dim (dim = {0})".format(self.lat_i_dim))

        self.act_fx = act_fx
        self.fx = None
        if act_fx == "tanh":
            self.fx = tf.nn.tanh
        elif act_fx == "sigmoid":
            self.fx = tf.nn.sigmoid
        elif act_fx == "relu":
            self.fx = tf.nn.relu
        elif act_fx == "relu6":
            self.fx = tf.nn.relu6
        elif act_fx == "lrelu":
            self.fx = tf.nn.leaky_relu
        elif act_fx == "elu":
            self.fx = tf.nn.elu
        elif act_fx == "identity":
            self.fx = tf.identity
        else:
            print(" > Choosing base DisCo activation function - softsign(.)")
            self.fx = tf.nn.softsign # hidden layer activation function
        self.fx_y = softmax
        self.fx_yi = softmax
        self.fx_ya = softmax

        stddev = 0.05 # 0.025
        self.theta_y = []

        self.Wi = init_weights(init_type, [self.xi_dim,self.lat_i_dim], self.seed, stddev=stddev)
        self.Wi = tf.Variable(self.Wi, name="Wi")
        self.theta_y.append(self.Wi)

        self.Wa = init_weights(init_type, [self.a_dim,self.lat_a_dim], self.seed, stddev=stddev)
        self.Wa = tf.Variable(self.Wa, name="Wa")
        self.theta_y.append(self.Wa)

        bot_dim = self.lat_i_dim
        if self.lat_fusion_type == "concat":
            bot_dim = self.lat_i_dim + self.lat_a_dim

        self.Wp = init_weights(init_type, [bot_dim,self.lat_dim], self.seed, stddev=stddev)
        self.Wp = tf.Variable(self.Wp, name="Wp")
        self.theta_y.append(self.Wp)

        self.We = None
        #if collapse_We is False:
        self.We = init_weights(init_type, [self.lat_dim,self.lat_dim], self.seed, stddev=stddev)
        self.We = tf.Variable(self.We, name="We")
        self.theta_y.append(self.We)

        self.Wy = init_weights(init_type, [self.lat_dim,self.y_dim], self.seed, stddev=stddev)
        self.Wy = tf.Variable(self.Wy, name="Wy")
        self.theta_y.append(self.Wy)

        self.Wyi = init_weights(init_type, [self.lat_dim,self.yi_dim], self.seed, stddev=stddev)
        self.Wyi = tf.Variable(self.Wyi, name="Wyi")
        self.theta_y.append(self.Wyi)

        self.Wya = init_weights(init_type, [self.lat_dim,self.ya_dim], self.seed, stddev=stddev)
        self.Wya = tf.Variable(self.Wya, name="Wya")
        self.theta_y.append(self.Wya)

        self.z_i = tf.Variable(tf.zeros([1,self.lat_dim]), name="z_i")
        self.z_a = tf.Variable(tf.zeros([1,self.lat_dim]), name="z_a")

        self.eta_v = 0.002
        self.moment_v = 0.9
        adam_eps = 1e-7 #1e-8  1e-6
        self.y_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.eta_v,beta1=0.9, beta2=0.999, epsilon=adam_eps)

    def set_opt(self, opt_type, eta_v, moment_v=0.9):
        adam_eps = 1e-7
        self.eta_v = eta_v
        self.moment_v = moment_v
        if opt_type == "adam":
            self.y_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.eta_v,beta1=0.9, beta2=0.999, epsilon=adam_eps)
        elif opt_type == "rmsprop":
            self.y_opt = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.eta_v,decay=0.9, momentum=self.moment_v, epsilon=1e-6)
        else:
            self.y_opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.eta_v)

    def calc_loss(self, y, yi, ya, y_prob, yi_prob, ya_prob):
        Ly = calc_catNLL(target=y,prob=y_prob,keep_batch=True) #Ly = D_KL(y_prob, y)
        Ly = tf.reduce_mean(Ly) #Ly = tf.reduce_sum(Ly)
        Lyi = D_KL(yi, yi_prob) * self.gamma_i
        Lya = D_KL(ya, ya_prob) * self.gamma_a
        l1 = 0.0
        l2 = 0.0
        #NS = # of rows in the mini batch (rows in Y)
        mini_bath_size = y.shape[0] * 1.0
        if self.l1_norm > 0:
            l1 = l1_l2_norm_calculation(self.theta_y,1,mini_bath_size) * self.l1_norm 
        if self.l2_norm > 0:
            l2 = l1_l2_norm_calculation(self.theta_y,2,mini_bath_size) * self.l2_norm 
        L_t = Ly + Lyi + Lya + l1 + l2
        return L_t, Ly, Lyi, Lya

    def encode_i(self, xi):
        """
            Calculates projection/embedding of item feature vector x_i
        """
        z_enc = tf.matmul(xi, self.Wi)
        return z_enc

    def encode_a(self, a):
        """
            Calculates projection/embedding of annotator a
        """
        av = a
        z_enc = tf.squeeze( tf.nn.embedding_lookup(self.Wa, av) )
        if len(z_enc.shape) < 2:
            z_enc = tf.expand_dims(z_enc,axis=0)
        return z_enc

    def encode(self, xi, a):
        z = None
        if self.lat_fusion_type == "concat":
            z = self.fx(tf.concat([self.encode_i(xi), self.encode_a(a)],axis=1))
        else:
            z = self.fx(self.encode_i(xi) + self.encode_a(a))
        z = self.transform(z)
        return z

    def transform(self,z):
        z_e = z
        z_p = self.fx(tf.matmul(z, self.Wp))
        if self.drop_p > 0.0:
            z_p, _ = drop_out(z_p, rate=self.drop_p)
        z_e = self.fx(tf.matmul(z_p, self.We) + z_p)
        if self.drop_p > 0.0:
            z_e, _ = drop_out(z_e, rate=self.drop_p)
        return z_e

    def decode_yi(self, z):
        y_logits = tf.matmul(z, self.Wyi)
        y_dec = self.fx_yi(y_logits)
        return y_dec, y_logits

    def decode_ya(self, z):
        y_logits = tf.matmul(z, self.Wya)
        y_dec = self.fx_ya(y_logits)
        return y_dec, y_logits

    def decode_y(self, z):
        y_logits = tf.matmul(z, self.Wy)
        y_dec = self.fx_y(y_logits)
        return y_dec, y_logits

    def update(self, xi, a, yi, ya, y, update_radius=-1.):
        """
            Updates model parameters given data batch (i, a, yi, ya, y)
        """
        batch_size = yi.shape[0]

        # run the model under gradient-tape's awareness
        with tf.GradientTape(persistent=True) as tape:
            z = self.encode(xi, a)
            yi_prob, yi_logits = self.decode_yi(z)
            ya_prob, ya_logits = self.decode_ya(z)
            y_prob, y_logits = self.decode_y(z)

            Ly = calc_catNLL(target=y,prob=y_prob,keep_batch=True)
            Ly = tf.reduce_mean(Ly)

            Lyi = D_KL(yi, yi_prob) * self.gamma_i
            Lya = D_KL(ya, ya_prob) * self.gamma_a
            
            l1 = 0.0
            l2 = 0.0
            mini_bath_size = y.shape[0] * 1.0
            if self.l1_norm  > 0:
                l1 = l1_l2_norm_calculation(self.theta_y,1,mini_bath_size) * self.l1_norm
            if self.l2_norm  > 0:
                l2 = l1_l2_norm_calculation(self.theta_y,2,mini_bath_size) * self.l2_norm
            L_t = Ly + Lyi + Lya + l1 + l2
            # L_t = Ly + Lya + Lyi
        # get gradient w.r.t. parameters
        delta_y = tape.gradient(L_t, self.theta_y)
        # apply optional gradient clipping
        if update_radius > 0.0:
            for p in range(len(delta_y)):
                pdelta = delta_y[p]
                pdelta = tf.clip_by_value(pdelta, -update_radius, update_radius)
                delta_y[p] = pdelta
        # update parameters given derivatives
        self.y_opt.apply_gradients(zip(delta_y, self.theta_y))
        return L_t

    def decode_y_ensemble(self, xi):
        """
            Computes the label distribution given only an item feature vector
            (and model's knowledge of all known annotators).
        """
        drop_p = self.drop_p + 0
        self.drop_p = 0.0 # turn off dropout

        z_i = self.encode_i(xi)
        z_a = self.Wa + 0 # gather all known annotators
        z = None
        if self.lat_fusion_type == "concat":
            tiled_z_i = tf.zeros([z_a.shape[0],z_i.shape[1]]) + z_i # smear z_i across row dim of z_a (ensure same shapes)
            z = self.fx(tf.concat([tiled_z_i, z_a],axis=1))
        else:
            z = self.fx(z_a + z_i)
        z = self.transform(z)
        y_prob, y_logits = self.decode_y(z)

        self.drop_p = drop_p # turn dropout back on
        return y_prob, y_logits

    def infer_a(self, xi, yi, K, beta, gamma=0.0, is_verbose=False):
        """
            Infer an annotator embedding given only an item feature and label
            distribution vector pair.
        """
        print("WARNING: DO NOT USE THIS! NOT DEBUGGED FOR CONCAT AT THE MOMENT!")
        best_L = None
        batch_size = yi.shape[0]
        z_eps = 0.0 #0.001
        if "elu" in self.act_fx:
            z_eps = 0.001
        # Step 1: encode xi
        z_i = self.encode_i(xi)
        self.z_a = tf.Variable(tf.zeros([batch_size,self.lat_dim]) + z_eps, name="z_a")
        # Step 2: find za given xi, yi
        for k in range(K):
            with tf.GradientTape(persistent=True) as tape:
                z = self.fx(z_i + self.z_a)
                z = self.transform(z)
                yi_prob, yi_logits = self.decode_yi(z)
                Lyi = D_KL(yi, yi_prob) * self.gamma_i
                Lyi = tf.reduce_sum(Lyi)
                if is_verbose is True:
                    print("k({0}) KL(p(yi)||yi) = {1}".format(k, Lyi))
            # check early halting criterion
            if best_L is not None:
                if Lyi < best_L:
                    best_L = Lyi
                else:
                    break # early stop at this point
            else:
                best_L = Lyi
            d_z_a = tape.gradient(Lyi, self.z_a) # get KL gradient w.r.t. z_a
            self.z_a.assign( self.z_a - d_z_a * beta - self.z_a * gamma) # update latent z_a
        z_a = self.z_a
        return z_a

    def clear(self):
        self.z_i = tf.Variable(tf.zeros([1,self.lat_dim]), name="z_i")
        self.z_a = tf.Variable(tf.zeros([1,self.lat_dim]), name="z_a")
