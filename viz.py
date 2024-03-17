import pickle

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.util import img_as_float
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch

def save_img_mask_pair(img, pred, mask, accuracy, save_path):
    f, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img)
    ax[1].imshow(pred)
    ax[2].imshow(mask)
    ax[1].set_title(f'Accuracy: {accuracy}')
    plt.tight_layout()
    plt.savefig(save_path)


def save_tcnb_graph(model, save_path, epoch):
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
    with open('dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    path, img, mask = dataset[4]

    img = img_as_float(img)
    img = torch.from_numpy(img).permute(2, 0, 1)  # C, H, W
    img = img.float().to(device)
    
    mask = torch.from_numpy(mask)  # H, W

    embeddings = model(img)
    embeddings = embeddings.flatten(1, -1)
    y = mask.flatten()

    pos = embeddings[..., y==1]
    neg = embeddings[..., y==0]
    
    num_pts_per_class = 1500
    random_is = np.random.choice(min(pos.shape[-1], neg.shape[-1]), num_pts_per_class)

    # pca = PCA(n_components=8)
    pos = pos.permute(1, 0).cpu().detach().numpy()
    neg = neg.permute(1, 0).cpu().detach().numpy()
    to_plot = np.concatenate([pos[random_is], neg[random_is]], axis=0)

    tsne = TSNE(n_components=2, verbose=1, perplexity=50)
    tsne_results = tsne.fit_transform(to_plot)

    colors = plt.get_cmap('viridis')(np.linspace(0, 1, 10))
    plt.figure(figsize=(6, 4))
    plt.scatter(x=tsne_results[:num_pts_per_class, 0], y=tsne_results[:num_pts_per_class, 1], marker='x', color=colors[0], label='Foreground')
    plt.scatter(x=tsne_results[num_pts_per_class:, 0], y=tsne_results[num_pts_per_class:, 1], marker='x', color=colors[8], label='Background')
    plt.legend()
    plt.title(f'Epoch {epoch + 1}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
