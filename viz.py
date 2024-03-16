import pickle

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.util import img_as_float
from sklearn.manifold import TSNE
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
    
    plt.rcParams.update({
        "text.usetex": True,
    })

    img = img_as_float(dataset[4][1])
    img = torch.from_numpy(img).permute(2, 0, 1)  # C, H, W
    img = img.float().to(device)
    embeddings = model(img)
    embeddings = embeddings.permute(1, 2, 0)  # H, W, C
    embeddings = embeddings.view(embeddings.shape[0] * embeddings.shape[1], -1)  # H * W, C
    embeddings = embeddings.cpu().detach().numpy()
    
    random_is = np.random.choice(embeddings.shape[0], 3000)
    print(random_is)

    tsne = TSNE(n_components=2, verbose=1)
    tsne_results = tsne.fit_transform(embeddings[random_is])
    labels = dataset[4][2].reshape(-1)[random_is]

    tsne_pos = tsne_results[labels==1]
    tsne_neg = tsne_results[labels==0]

    colors = plt.get_cmap('viridis')(np.linspace(0, 1, 10))
    plt.figure(figsize=(6, 4))
    plt.scatter(x=tsne_pos[:, 0], y=tsne_pos[:, 1], marker='x', color=colors[0], label='Foreground')
    plt.scatter(x=tsne_neg[:, 0], y=tsne_neg[:, 1], marker='x', color=colors[8], label='Background')
    plt.legend()
    plt.title(f'Epoch {epoch + 1}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)