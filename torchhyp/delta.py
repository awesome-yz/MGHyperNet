# ----------------------------------------------------------------
# Modified from 'hyperbolic-image-embeddings'
# Reference: https://github.com/leymir/hyperbolic-image-embeddings
# ----------------------------------------------------------------
import torch
import torch.nn as nn
import torchvision
from scipy.spatial import distance_matrix
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def delta_hyp(dismat):
    """
    computes delta hyperbolicity value from distance matrix
    """

    p = 0
    row = dismat[p, :][np.newaxis, :]
    col = dismat[:, p][:, np.newaxis]
    XY_p = 0.5 * (row + col - dismat)

    maxmin = np.max(np.minimum(XY_p[:, :, None], XY_p[None, :, :]), axis=1)
    return np.max(maxmin - XY_p)


def batched_delta_hyp(X, n_tries=10, batch_size=1500):
    vals = []
    for i in tqdm(range(n_tries)):
        idx = np.random.choice(len(X), batch_size)
        X_batch = X[idx]
        distmat = distance_matrix(X_batch, X_batch)
        diam = np.max(distmat)
        delta_rel = delta_hyp(distmat) / diam
        vals.append(delta_rel)
    return np.mean(vals), np.std(vals)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B = x.shape[0]
        #x.view(B, -1)
        return x.reshape(B, -1)


def get_delta(loader):
    """
    computes delta value for image data by extracting features using VGG network;
    input -- data loader for images

    altered method, original method at the link above
    """
    import pdb; pdb.set_trace()
    all_features = []
    for i, (input, text) in enumerate(loader):
        with torch.no_grad():
            
            # vis_batch, _, _ = batch
            # vis_batch = vis_batch.to(device)
            # all_features.append(vis_batch.detach().cpu().numpy())
            all_features.append(input)

    
    all_features = np.concatenate(all_features)
    idx = np.random.choice(len(all_features), 1500)
    all_features_small = all_features[idx]

    # mean of 10crop and flatten features
    flatten = Flatten()
    all_features_small = flatten(all_features_small.mean(1))
    
    d_rel_mean, d_rel_std = batched_delta_hyp(all_features_small)

    return d_rel_mean, d_rel_std
