import os

import numpy as np
import pickle
from tqdm import tqdm

from skimage import transform
from scipy.spatial.distance import squareform, pdist, cdist
from sklearn.linear_model import LogisticRegression, SGDClassifier


def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find cdist (imported from scipy.spatial.distance) and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        ### YOUR CODE HERE
        old_assignments = assignments.copy()

        assignments = np.argmin(cdist(centers, features), axis=0)
        for i in range(k):
            centers[i] = features[assignments == i].mean(axis=0)

        if np.all(old_assignments == assignments):
            break
        ### END YOUR CODE

    return assignments


def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    ### YOUR CODE HERE
    pred_fore = (mask == 1)
    gt_fore = (mask_gt == 1)
    pred_back = (mask == 0)
    gt_back = (mask_gt == 0)

    tp = np.logical_and(pred_fore, gt_fore).sum()
    tn = np.logical_and(pred_back, gt_back).sum()
    p = (mask == 1).sum()
    n = (mask == 0).sum()

    accuracy = (tp + tn) / (p + n)
    ### END YOUR CODE

    return accuracy


def compute_segmentation(img, k,
        clustering_fn=kmeans_fast,
        feature_fn=None,
        scale=0):
    """ Compute a segmentation for an image.

    First a feature vector is extracted from each pixel of an image. Next a
    clustering algorithm is applied to the set of all feature vectors. Two
    pixels are assigned to the same segment if and only if their feature
    vectors are assigned to the same cluster.

    Args:
        img - An array of shape (H, W, C) to segment.
        k - The number of segments into which the image should be split.
        clustering_fn - The method to use for clustering. The function should
            take an array of N points and an integer value k as input and
            output an array of N assignments.
        feature_fn - A function used to extract features from the image.
        scale - (OPTIONAL) parameter giving the scale to which the image
            should be in the range 0 < scale <= 1. Setting this argument to a
            smaller value will increase the speed of the clustering algorithm
            but will cause computed segments to be blockier. This setting is
            usually not necessary for kmeans clustering, but when using HAC
            clustering this parameter will probably need to be set to a value
            less than 1.
    """

    assert scale <= 1 and scale >= 0, \
        'Scale should be in the range between 0 and 1'

    H, W, C = img.shape

    if scale > 0:
        # Scale down the image for faster computation.
        img = transform.rescale(img, scale, channel_axis=2)

    features = feature_fn(img)
    assignments = clustering_fn(features, k)
    segments = assignments.reshape((img.shape[:2]))

    if scale > 0:
        # Resize segmentation back to the image's original size
        segments = transform.resize(segments, (H, W), preserve_range=True)

        # Resizing results in non-interger values of pixels.
        # Round pixel values to the closest interger
        segments = np.rint(segments).astype(int)

    return segments


def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments.
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy


def logistic_reg(img, features, gt_mask, shuffle_pixels=False):
    gt_mask = gt_mask.reshape(-1)

    examples = np.arange(len(gt_mask))
    if shuffle_pixels:
        np.random.shuffle(examples)
    
    lr = LogisticRegression()

    split_start = int(len(examples) * 0.7)

    train = examples[:int(len(examples) * 0.7)]
    test = examples[int(len(examples) * 0.7):]

    lr.fit(X=features[train], y=gt_mask[train])
    
    pred = lr.predict(X=features)
    # test_acc = lr.score(X=features[test], y=gt_mask[test])
    return pred, examples


def sgd_classifier(img, features, gt_mask, shuffle_pixels=False):
    gt_mask = gt_mask.reshape(-1)

    examples = np.arange(len(gt_mask))
    if shuffle_pixels:
        np.random.shuffle(examples)

    sgd = SGDClassifier()

    pred = np.zeros(gt_mask.shape)
    split_start = int(len(pred) * 0.7)
    pbar = tqdm(examples)
    for i, pixel_index in enumerate(pbar):
        pixel_feature = features[pixel_index:pixel_index+1]
        pixel_class = gt_mask[pixel_index:pixel_index+1]
        if i < split_start:
            sgd.partial_fit(X=pixel_feature, y=pixel_class, classes=[0, 1])
        pred[pixel_index] = sgd.predict(X=pixel_feature)

    return pred, examples  # all predictions, pixel order.


def linucb_lite(img, features, gt_mask, d, shuffle_pixels=False):
    total_correct = 0
    total_fore = 0
    total_back = 0
    total_count = 0

    foreground_A = np.identity(d)
    foreground_b = np.zeros(d)
    background_A = np.identity(d)
    background_b = np.zeros(d)

    alpha = 0.1
    gt_mask = gt_mask.reshape(-1)

    examples = list(zip(list(range(len(gt_mask))), features, gt_mask))
    if shuffle_pixels:
        np.random.shuffle(examples)
    pbar = tqdm(examples)
    
    pred = np.zeros(gt_mask.shape)
    split_start = int(len(pred) * 0.7)
    
    for i, (pixel_index, pixel_feature, pixel_class) in enumerate(pbar):
        probs = []
        for A, b in [(foreground_A, foreground_b), (background_A, background_b)]:
            inv_A = np.linalg.inv(A)
            theta = inv_A @ b
            p = theta.T @ pixel_feature + alpha * np.sqrt(pixel_feature.T @ inv_A @ pixel_feature)
            probs.append(p)
        pulled_arm = np.array(probs).argmax()
        pred[pixel_index] = pulled_arm
        reward = 1 if pulled_arm == pixel_class else 0
        if pulled_arm == 0:
            if i < split_start:
                foreground_A = foreground_A + np.outer(pixel_feature, pixel_feature)
                foreground_b = foreground_b + reward * pixel_feature
            total_fore += 1
        else:
            if i < split_start:
                background_A = background_A + np.outer(pixel_feature, pixel_feature)
                background_b = background_b + reward * pixel_feature
            total_back += 1

        total_correct += reward
        total_count += 1
        pbar.set_postfix_str(f'Accuracy: {total_correct / total_count}, Total fore: {total_fore}, Total back: {total_back}')
    return pred, [x[0] for x in examples]


def linucb(
        imgs,
        features,
        gt_masks,
        save_dir,
        foreground_A,
        foreground_b,
        background_A,
        background_b,
        train=True,
        shuffle_pixels=False,
        internal_split=False,
    ):
    total_correct = 0
    total_fore = 0
    total_back = 0
    total_count = 0

    alpha = 0.1
    preds = []
    for i, (img, feature, gt_mask) in enumerate(zip(imgs, features, gt_masks)):
        gt_mask = gt_mask.reshape(-1)

        examples = list(zip(list(range(len(gt_mask))), feature, gt_mask))
        if shuffle_pixels:
            np.random.shuffle(examples)
        pbar = tqdm(examples)
        
        pred = np.zeros(gt_mask.shape)
        split_start = int(len(pred) * 0.7)
        for j, (pixel_index, pixel_feature, pixel_class) in enumerate(pbar):
            probs = []
            for A, b in [(foreground_A, foreground_b), (background_A, background_b)]:
                inv_A = np.linalg.inv(A)
                theta = inv_A @ b
                p = theta.T @ pixel_feature + alpha * np.sqrt(pixel_feature.T @ inv_A @ pixel_feature)
                probs.append(p)
            pulled_arm = np.array(probs).argmax()
            pred[pixel_index] = pulled_arm
            reward = 1 if pulled_arm == pixel_class else 0
            if pulled_arm == 0:
                if train and (not internal_split or j < split_start):
                    foreground_A = foreground_A + np.outer(pixel_feature, pixel_feature)
                    foreground_b = foreground_b + reward * pixel_feature
                total_fore += 1
            else:
                if train and (not internal_split or j < split_start):
                    background_A = background_A + np.outer(pixel_feature, pixel_feature)
                    background_b = background_b + reward * pixel_feature
                total_back += 1

            total_correct += reward
            total_count += 1
            pbar.set_postfix_str(f'Accuracy: {total_correct / total_count}, Total fore: {total_fore}, Total back: {total_back}')
        
        pred = pred.reshape(img.shape[:2])
        gt_mask = gt_mask.reshape(img.shape[:2])

        with open(os.path.join(save_dir, f'linucb_{i}.pkl'), 'wb') as f:
            pickle.dump([img, pred, gt_mask], f)

        preds.append(pred)
    return preds, foreground_A, foreground_b, background_A, background_b


def naive(
    img,
    gt_mask,
):
    pred = np.ones(gt_mask.shape)
    return pred
