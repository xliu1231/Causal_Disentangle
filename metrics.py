import numpy as np
import numpy.random as npr
from scipy.spatial.distance import cdist

from vae import reparametrize

def calculate_modified_irs(data, model, dim_per_z=1, diff_quantile=0.99):
    
    mu, logvar = model.encode(data)
    latents = reparametrize(mu, logvar)
    l = latents.shape[1]
    num_concepts = l // dim_per_z
    
    # split by concepts
    
    latents = latents.detach().numpy()
    max_deviations = np.linalg.norm(latents - latents.mean(axis=0), axis=0)
    if dim_per_z > 1:
        max_deviations = max_deviations.reshape([-1, dim_per_z]).max(axis=-1)
    avg_deviations = np.zeros([num_concepts])
    
    for i in range(num_concepts):
        e_zi = np.mean(latents[:,i * dim_per_z:(i+1) * dim_per_z], axis=0)
        diffs = np.linalg.norm(latents[:,i * dim_per_z:(i+1) * dim_per_z] - e_zi)
        max_diff = np.max(diffs)
        avg_deviations[i] = max_diff.sum() / dim_per_z
        
    normalized_deviations = avg_deviations / max_deviations
    irs_matrix = 1.0 - normalized_deviations
    disentanglement_scores = irs_matrix.max()
    
    return disentanglement_scores



def calculate_uc(data, model, dim_per_z=1, diff_quantile=0.99):
    
    # UC score 
    
    mu, logvar = model.encode(data)
    latents = reparametrize(mu, logvar)
    l = latents.shape[1]
    num_concepts = l // dim_per_z
    
    latents = latents.detach().numpy()
    max_deviations = np.max(np.abs(latents - latents.mean(axis=0)), axis=0)
    cum_deviations = np.zeros([l, num_concepts])
    
    for i in range(num_concepts):
        e_loc = np.mean(latents, axis=0)
        diffs = np.abs(latents - e_loc)
        max_diffs = np.percentile(diffs, q=diff_quantile * 100, axis=0)
        cum_deviations[:, i] = max_diffs
        
    # Normalize value of each latent dimension with its maximal deviation.
    normalized_deviations = cum_deviations / max_deviations[:, np.newaxis]
    irs_matrix = 1.0 - normalized_deviations

    indices = irs_matrix.argmax(axis=0)
    sets = [set() for _ in range(num_concepts)]
    latent_set = set({i for i in range(l)})
    for i in range(num_concepts):
        temp = irs_matrix[:, i]
        indices = np.argpartition(temp, -1)[::-dim_per_z][:dim_per_z]
        for j in indices:
            sets[i].add(j)
    un_norm_score = 0
    
    for i in range(num_concepts-1):
        for j in range(i+1, num_concepts):
            un_norm_score += len(sets[i].intersection(sets[j]))/float(len(sets[i].union(sets[j])))
    norm_score = un_norm_score/(num_concepts*(num_concepts-1)/2.)
    uc = 1 - norm_score
    return uc


# Code source
#
# https://github.com/yixinwang/representation-causal-public/blob/master/disentanglement_expms/sec3-4-ioss_vae/src/disentangle_measure.py
#
#
def IOSS(data, model, metric = "euclidean", n_draws=10000, robust_k_prop = 1e-2):
    
    mu, _ = model.encode(data)
    mu = mu.detach().numpy()

    stdmu = (mu-np.min(mu,axis=0)) / (np.max(mu,axis=0) - np.min(mu,axis=0))

    maxs = np.max(stdmu, axis=0)
    mins = np.min(stdmu, axis=0)
    smps = (np.column_stack([npr.rand(n_draws) * (maxs[i]-mins[i]) + mins[i] 
                             for i in range(stdmu.shape[1])]))
    min_dist = np.min(cdist(smps, stdmu, metric=metric), axis=1)
    
    ortho = np.max(min_dist,axis=0)

    return ortho