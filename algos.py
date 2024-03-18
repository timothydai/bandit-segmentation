import os

import numpy as np
import pickle
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression, SGDClassifier

def logistic_reg(img, features, gt_mask, shuffle_pixels=False):
    gt_mask = gt_mask.reshape(-1)
    examples = np.arange(len(gt_mask))
    if shuffle_pixels:
        np.random.shuffle(examples)
    train = examples[:int(len(examples) * 0.7)]
    test = examples[int(len(examples) * 0.7):]

    lr = LogisticRegression()
    lr.fit(X=features[train], y=gt_mask[train])
    
    pred = lr.predict(X=features)
    test_acc = lr.score(X=features[test], y=gt_mask[test])
    return pred, test_acc


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
