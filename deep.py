import pickle
from tqdm import tqdm

import numpy as np
from skimage.util import img_as_float
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import ResNet50_Weights


if torch.backends.mps.is_available():
    print('USING MPS')
    device = torch.device('mps')
elif torch.cuda.is_available():
    print('USING CUDA')
    device = torch.device('cuda')
else:
    print('USING CPU')
    device = torch.device('cpu')


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        with open('dataset.pkl', 'rb') as f:
            dataset = pickle.load(f)
        self.dataset = dataset[300:]
        assert len(self.dataset) == 1431
        # self.preprocess = ResNet50_Weights.DEFAULT.transforms()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        path, img, mask = example
        img = img_as_float(img)
        try:
            img = torch.from_numpy(img).permute(2, 0, 1)  # C, H, W
        except:
            print(path)
            assert False
        mask = torch.from_numpy(mask)  # H, W
        return img.float(), mask


class ContrastiveLoss(nn.Module):
    def forward(self, x, y):
        # sample_size = 64
        # for class_label in [0, 1]:
        #     P = x[:, y[0]==class_label]
        #     P_i = P[:, np.random.choice(len(P), size=sample_size+1)]
        #     N = x[:, y[0]==(1 - class_label)]
        #     N_i = N[:, np.random.choice(len(N), size=sample_size)]
        x = x.view(-1, x.shape[-2] * x.shape[-1])
        y = y.view(-1, y.shape[-2] * y.shape[-1])

        pos = x[:, y[0]==1]
        num_half_pos = len(pos[0]) // 2
        pos1 = pos[:, :num_half_pos].T
        pos2 = pos[:, num_half_pos:num_half_pos * 2].T
        pos_sim = pos1.unsqueeze(1) @ pos2.unsqueeze(-1)

        neg = x[:, y[0]==0]
        num_half_neg = len(neg[0]) // 2
        neg1 = neg[:, :num_half_neg].T
        neg2 = neg[:, num_half_neg:num_half_neg * 2].T
        neg_sim = neg1.unsqueeze(1) @ neg2.unsqueeze(-1)

        num_half_min = min(num_half_pos, num_half_neg)
        pos1 = pos[:, :num_half_min].T
        pos2 = pos[:, num_half_min:num_half_min * 2].T
        neg1 = neg[:, :num_half_min].T
        neg2 = neg[:, num_half_min:num_half_min * 2].T
        pos_neg_sim1 = pos1.unsqueeze(1) @ neg1.unsqueeze(-1)
        pos_neg_sim2 = pos2.unsqueeze(1) @ neg2.unsqueeze(-1)

        loss = (
            -pos_sim.mean()
            - neg_sim.mean()
            + torch.log(torch.exp(pos_neg_sim1).mean())
            + torch.log(torch.exp(pos_neg_sim2).mean())
        )
        # print(loss)
        return loss


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3)
        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        return x

dataset = Dataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
model = Model().to(device)
loss_fn = ContrastiveLoss()  # nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)

for epoch in range(3):
    print(f'STARTING EPOCH {epoch+1}')
    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'contrastive_weights.pt')