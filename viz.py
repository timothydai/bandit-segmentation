import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.util import img_as_float
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch

from deep import *

def save_img_mask_pair(img, pred, mask, save_path):
    pred = pred.reshape(mask.shape)

    accuracy = (pred == mask).sum() / mask.shape[0] / mask.shape[1]
    f, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img)
    ax[1].imshow(pred)
    ax[2].imshow(mask)
    ax[1].set_title(f'Accuracy: {accuracy}')
    plt.tight_layout()
    plt.savefig(save_path)


def save_tcnb_graph(model, save_path):
    np.random.seed(0)
    if torch.backends.mps.is_available():
        print('USING MPS')
        device = torch.device('mps')
    elif torch.cuda.is_available():
        print('USING CUDA')
        device = torch.device('cuda')
    else:
        print('USING CPU')
        device = torch.device('cpu')
    model = model.to(device)
    
    with open('dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    path, img, mask = dataset[151]
    # path, img, mask = dataset[713]

    img = img_as_float(img)
    img = torch.from_numpy(img).permute(2, 0, 1)  # C, H, W
    img = img.float().to(device)
    
    mask = torch.from_numpy(mask)  # H, W

    embeddings = model(img.unsqueeze(0))
    embeddings = embeddings.squeeze(0).flatten(1, -1)
    y = mask.flatten()

    pos = embeddings[..., y==1]
    neg = embeddings[..., y==0]
    
    num_pts_per_class = 500
    random_is = np.random.choice(min(pos.shape[-1], neg.shape[-1]), num_pts_per_class)

    pos = pos.permute(1, 0).cpu().detach().numpy()
    neg = neg.permute(1, 0).cpu().detach().numpy()
    to_plot = np.concatenate([pos[random_is], neg[random_is]], axis=0)

    pca = PCA(n_components=8)
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, metric='cosine', n_iter=15000)
    # to_plot = pca.fit_transform(to_plot)
    tsne_results = tsne.fit_transform(to_plot)

    colors = plt.get_cmap('viridis')(np.linspace(0, 1, 10))
    plt.figure(figsize=(4, 2.5))
    plt.scatter(x=tsne_results[:num_pts_per_class, 0], y=tsne_results[:num_pts_per_class, 1], marker='x', color=colors[0], label='Foreground')
    plt.scatter(x=tsne_results[num_pts_per_class:, 0], y=tsne_results[num_pts_per_class:, 1], marker='x', color=colors[8], label='Background')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


def visualize_all_experiments_in(results_dir):
    for filename in glob.glob(f'{results_dir}/*'):
        if not filename.endswith('pkl'):
            continue
        with open(filename, 'rb') as f:
            img, pred, mask = pickle.load(f)
        save_img_mask_pair(img, pred, mask, (pred == mask).sum() / (img.shape[0] * img.shape[1]), f'{filename}_viz.png')
        
        pred = pred.reshape(-1)
        mask = mask.reshape(-1)
        print(
            (pred[:int(0.7 * len(pred))] == mask[:int(0.7 * len(pred))]).sum() / (int(0.7 * len(pred))), 
            (pred[int(0.7 * len(pred)):] == mask[int(0.7 * len(pred)):]).sum() / (int(0.3 * len(pred)))
        )

if __name__ == '__main__':
    model = BigModelUpsample()
    save_tcnb_graph(model, 'contrastive_save/base')

    model.load_state_dict(torch.load('./contrastive_save/contrastive_weights_best.pt', map_location=torch.device('cpu')))
    save_tcnb_graph(model, 'contrastive_save/trained')
