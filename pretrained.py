import pickle

import torch
import torch.nn.functional as F
from torchvision.io.image import read_image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image


# This is equivalent to the second argument in the pickled dataset, except the channel dimension is the FIRST dimension (pickled as third dimension).
img = read_image("/Users/timothydai/Documents/cs 131/bandit-segmentation/000000012448.jpg")

with open('dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)
    print(len(dataset))
dataset = dataset[:5]

# Step 1: Initialize model with the best available weights
weights = FCN_ResNet50_Weights.DEFAULT
# weights = ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights)
# model = resnet50(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# print(model.backbone(batch).shape)
# assert False

# Step 4: Use the model and visualize the prediction
prediction = model.backbone(batch)["out"]
out = F.interpolate(prediction, size=img.shape[-2:], mode="bilinear", align_corners=False)
print(out.shape)
# normalized_masks = prediction.softmax(dim=1)
# class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
# mask = normalized_masks[0, class_to_idx["person"]]
# to_pil_image(mask).show()
