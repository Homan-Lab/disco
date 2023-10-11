"""
Utilities function file

@author: Alex Ororbia
"""
import tensorflow as tf
import numpy as np
import pickle
import sys
seed = 69
#tf.random.set_random_seed(seed=seed)

tf.random.set_seed(seed=seed)
np.random.seed(seed)

def save_object(model, fname):
    fd = open(fname, 'wb')
    pickle.dump(model, fd)
    fd.close()

def load_object(fname):
    fd = open(fname, 'rb')
    model = pickle.load( fd )
    fd.close()
    return model

def scale_feat(x, a=-1.0, b=1.0):
    max_x = tf.reduce_max(x,axis=1,keepdims=True)
    min_x = tf.reduce_min(x,axis=1,keepdims=True)
    x_prime = a + ( ( (x - min_x) * (b - a) )/(max_x - min_x) )
    return tf.cast(x_prime, dtype=tf.float32)

def calc_mode(value_list):
    if len(value_list) == 0:
        return -1, 0
    freq_map = {}
    for i in range(len(value_list)):
        v_i = value_list[i]
        cnt = freq_map.get(v_i)
        if cnt != None:
            cnt = cnt + 1
            freq_map.update({v_i : cnt})
        else:
            freq_map.update({v_i : 1})
    most_freq_v = None
    max_cnt = 0
    for v, cnt in freq_map.items():
        if cnt > max_cnt:
            max_cnt = cnt
            most_freq_v = v
    return most_freq_v, max_cnt

def D_KL(px, qx, keep_batch=False):
    '''
    General KL divergence between probability dist p(x) and q(x), i.e., KL(p||q)
    -> q(x) is the approximating communication channel/distribution
    -> p(x) is the target channel/distribution (we wish to compress)
    <br>
    Notes that this function was derived from:  https://arxiv.org/pdf/1404.2000.pdf
    @author Alexander Ororbia
    '''
    eps = 1e-6
    px_ = tf.clip_by_value(px, eps, 1.0-eps)
    log_px = tf.math.log(px_)
    qx_ = tf.clip_by_value(qx, eps, 1.0-eps)
    log_qx = tf.math.log(qx_)

    term1 = tf.reduce_sum(-(px_ * log_px),axis=1,keepdims=True)
    term2 = tf.reduce_sum(px_ * log_qx,axis=1,keepdims=True)
    KL = -(term1 + term2)
    loss = KL
    if not keep_batch:
        loss = tf.math.reduce_mean(KL) #,axis=0)
    return loss

def D_KL_(qx, px, keep_batch=False):
    '''
    General KL divergence between probability dist q(x) and p(x), i.e., KL(q||p)
    @author Alexander Ororbia
    '''
    eps = 1e-6
    qx_ = tf.clip_by_value(qx, eps, 1.0-eps)
    px_ = tf.clip_by_value(px, eps, 1.0-eps)
    log_qx = tf.math.log(qx_)
    log_px = tf.math.log(px_)
    term1 = tf.reduce_sum(qx_ * log_qx,axis=1,keepdims=True)
    term2 = tf.reduce_sum(qx_ * log_px,axis=1,keepdims=True)
    KL = term1 - term2
    loss = KL
    if not keep_batch:
        loss = tf.math.reduce_mean(KL) #,axis=0)
    return loss

def mse(x_true, x_pred, keep_batch=False):
    '''
    Mean Squared Error
    @author Alexander Ororbia
    '''
    diff = x_pred - x_true
    se = diff * diff # 0.5 # squared error
    # NLL = -( -se )
    if not keep_batch:
        mse = tf.math.reduce_mean(se)
    else:
        mse = tf.reduce_sum(se, axis=-1)
        mse = tf.expand_dims(mse,axis=1)
    return mse

def drop_out(input, rate=0.0, seed=69):
    """
        Custom drop-out function -- returns output as well as binary mask
        -> scale the values of the output by 1/(1-rate) which allows us to just
           set rate to 0 at test time with no further changes needed to compute the
           expectation of the activation output
        @author Alexander G. Ororbia
    """
    mask = tf.math.less_equal( tf.random.uniform(shape=(input.shape[0],input.shape[1]), minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed),(1.0 - rate))
    mask = tf.cast(mask, tf.float32) * (1.0 / (1.0 - rate))
    output = input * mask
    return output, mask
    
def softmax(x, tau=0.0):
    """
        Softmax function with overflow control built in directly. Contains optional
        temperature parameter to control sharpness (tau > 1 softens probs, < 1 sharpens --> 0 yields point-mass)
    """
    if tau > 0.0:
        x = x / tau
    max_x = tf.expand_dims( tf.reduce_max(x, axis=1), axis=1)
    exp_x = tf.exp(tf.subtract(x, max_x))
    return exp_x / tf.expand_dims( tf.reduce_sum(exp_x, axis=1), axis=1)

def calc_catNLL(target, prob, keep_batch=False):
    """
        Calculates the (negative) Categorical log likelihood under a provided set of probabilities
        for a target (one-hot) label encoding.
    """
    eps = 1e-7
    py = tf.clip_by_value(prob, eps, 1.0-eps)
    Ly = -tf.reduce_sum(tf.math.log(py) * target, axis=1, keepdims=True)
    if keep_batch is False:
        return tf.reduce_mean(Ly)
    return Ly

def sample_gaussian(shape, mu=0.0, sig=1.0):
    """
        Samples a multivariate Gaussian assuming a diagonal covariance
    """
    eps = tf.random.normal(shape, mean=mu, stddev=sig, seed=seed)
    return eps * sig + mu

def calc_gaussian_KL(mu1, sigSqr1, log_var1, mu2, sigSqr2, log_var2):
    """
        Calculates Kullback-Leibler divergence (KL-D) between two multivariate
        Gaussians strictly assuming each has a diagonal covariance (vector variances).
    """
    eps = 1e-7
    term1 = log_var2 - log_var1 #tf.math.log(sig2) - tf.math.log(sig1)
    diff = (mu1 - mu2)
    term2 = (sigSqr1 + (diff ** 2))/(sigSqr2 * 2 + eps)
    kl = term1 + term2 - 0.5
    return tf.reduce_sum(kl,axis=1,keepdims=True)

def calc_gaussian_KL_simple(mu, log_sigma_sqr):
    return -0.5 * tf.reduce_sum(1 + log_sigma_sqr - (mu * mu) - tf.math.exp(log_sigma_sqr), axis=1)

def init_weights(init_type, shape, seed, stddev=1.0):
    if init_type == "he_uniform":
        initializer = tf.compat.v1.keras.initializers.he_uniform()
        params = initializer(shape) #, seed=seed )
    elif init_type == "he_normal":
        initializer = tf.compat.v1.keras.initializers.he_normal()
        params = initializer(shape) #, seed=seed )
    elif init_type == "classic_glorot":
        N = (shape[0] + shape[1]) * 1.0
        bound = 4.0 * np.sqrt(6.0/N)
        params = tf.random.uniform(shape, minval=-bound, maxval=bound, seed=seed)
    elif init_type == "glorot_normal":
        initializer = tf.compat.v1.keras.initializers.glorot_normal()
        params = initializer(shape) #, seed=seed )
    elif init_type == "glorot_uniform":
        initializer = tf.compat.v1.keras.initializers.glorot_uniform()
        params = initializer(shape) #, seed=seed )
    elif init_type == "orthogonal":
        initializer = tf.compat.v1.keras.initializers.orthogonal(gain=stddev)
        params = initializer(shape)
    elif init_type == "truncated_normal":
        params = tf.random.truncated_normal(shape, stddev=stddev, seed=seed)
    elif init_type == "normal":
        params = tf.random.normal(shape, stddev=stddev, seed=seed)
    else: # alex_uniform
        k = 1.0 / (shape[0] * 1.0) # 1/in_features
        bound = np.sqrt(k)
        params = tf.random.uniform(shape, minval=-bound, maxval=bound, seed=seed)

    return params

def l1_l2_norm_calculation(theta_y,norm_type,mini_bath_size):
    """
        Normalizes based on L1 and L2 operations. 
    """
    norm_value = 0.0
    for var in theta_y:
        w_norm = tf.norm(var,ord=norm_type)
        # if norm_type == 1:
        #     w_norm = tf.reduce_sum(tf.math.abs(var))
        # else:
        #     w_norm = tf.reduce_sum(var*var)
        norm_value += (w_norm*w_norm)
    norm_value = norm_value * 1/mini_bath_size
    return norm_value

################################################################################
# Functions for computing empirical KL divergence between two data samples
################################################################################
def ecdf(x):
    x = np.sort(x)
    u, c = np.unique(x, return_counts=True)
    n = len(x)
    y = (np.cumsum(c) - 0.5)/n
    def interpolate_(x_):
        yinterp = np.interp(x_, u, y, left=0.0, right=1.0)
        return yinterp
    return interpolate_

def cumulative_kl(x,y,fraction=0.5):
    dx = np.diff(np.sort(np.unique(x)))
    dy = np.diff(np.sort(np.unique(y)))
    ex = np.min(dx)
    ey = np.min(dy)
    e = np.min([ex,ey])*fraction
    n = len(x)
    P = ecdf(x)
    Q = ecdf(y)
    KL = (1./n)*np.sum(np.log((P(x) - P(x-e))/(Q(x) - Q(x-e))))
    return KL

def gen_data_plot(Xn, Yn, use_tsne=False, fname="Xi", out_dir=""):
    z_top = Xn
    y_ind = tf.cast(tf.argmax(tf.cast(Yn,dtype=tf.float32),1),dtype=tf.int32).numpy()
    import matplotlib #.pyplot as plt
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap = plt.cm.jet
    from sklearn.decomposition import IncrementalPCA
    print(" > Projecting latents via iPCA...")
    if use_tsne is True:
        n_comp = 10 #16 #50
        if z_top.shape[1] < n_comp:
            n_comp = z_top.shape[1] - 2 #z_top.shape[1]-2
            n_comp = max(2, n_comp)
        ipca = IncrementalPCA(n_components=n_comp, batch_size=50)
        ipca.fit(z_top)
        z_2D = ipca.transform(z_top)
        print(" > Finishing project via t-SNE...")
        from sklearn.manifold import TSNE
        z_2D = TSNE(n_components=2,perplexity=30).fit_transform(z_2D)
        #z_2D.shape
    else:
        ipca = IncrementalPCA(n_components=2, batch_size=50)
        ipca.fit(z_top)
        z_2D = ipca.transform(z_top)

    print(" > Plotting 2D encodings...")
    plt.figure(figsize=(8, 6))
    plt.scatter(z_2D[:, 0], z_2D[:, 1], c=y_ind, cmap=cmap)
    plt.colorbar()
    plt.grid()
    plt.savefig("{0}{1}.pdf".format(out_dir,fname))
    plt.clf()
