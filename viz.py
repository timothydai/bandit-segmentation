import matplotlib.pyplot as plt

def save_img_mask_pair(img, pred, mask, accuracy, save_path):
    f, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img)
    ax[1].imshow(pred)
    ax[2].imshow(mask)
    ax[1].set_title(f'Accuracy: {accuracy}')
    plt.tight_layout()
    plt.savefig(save_path)