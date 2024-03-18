import pickle
from tqdm import tqdm

import numpy as np
from skimage.util import img_as_float
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, split='train'):
        with open('dataset_new.pkl', 'rb') as f:
            dataset = pickle.load(f)
        self.dataset = dataset[5000:]
        # Remove problematic images (unknown reason)
        self.dataset = list(filter(lambda x: x[0] != '000000363942.jpg', self.dataset))
        if split == 'train':
            self.dataset = self.dataset[:int(len(self.dataset) * 0.7)]
        elif split == 'val':
            self.dataset = self.dataset[int(len(self.dataset) * 0.7):]
        # self.preprocess = ResNet50_Weights.DEFAULT.transforms()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        path, img, mask = example
        img = img_as_float(img)
        img = torch.from_numpy(img).permute(2, 0, 1)  # C, H, W
        mask = torch.from_numpy(mask)  # H, W
        return (path, img.float()), mask


class ContrastiveLoss(nn.Module):
    def forward(self, x, y):
        x = x.flatten(2, -1)
        y = y.flatten()

        pos = x[..., y==1]
        num_half_pos = pos.shape[-1] // 2
        pos1 = pos[..., :num_half_pos].permute(0, 2, 1).unsqueeze(2)
        pos2 = pos[..., num_half_pos:num_half_pos * 2].permute(0, 2, 1).unsqueeze(3)
        pos_sim = pos1 @ pos2

        neg = x[..., y==0]
        num_half_neg = neg.shape[-1] // 2
        neg1 = neg[..., :num_half_neg].permute(0, 2, 1).unsqueeze(2)
        neg2 = neg[..., num_half_neg:num_half_neg * 2].permute(0, 2, 1).unsqueeze(3)
        neg_sim = neg1 @ neg2

        num_half_min = min(num_half_pos, num_half_neg)
        pos1 = pos[..., :num_half_min].permute(0, 2, 1).unsqueeze(2)
        pos2 = pos[..., num_half_min:num_half_min * 2].permute(0, 2, 1).unsqueeze(2)
        neg1 = neg[..., :num_half_min].permute(0, 2, 1).unsqueeze(3)
        neg2 = neg[..., num_half_min:num_half_min * 2].permute(0, 2, 1).unsqueeze(3)

        pos_neg_sim1 = pos1 @ neg1
        pos_neg_sim2 = pos2 @ neg2

        loss = (
            -pos_sim.mean()
            - neg_sim.mean()
            + torch.log(torch.exp(pos_neg_sim1).mean())
            + torch.log(torch.exp(pos_neg_sim2).mean())
        )
        return loss


class BigModelUpsample(nn.Module):
    def __init__(self):
        super(BigModelUpsample, self).__init__()
        self.bigmodel = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = self.bigmodel(x)
        x = F.interpolate(x, size)
        return x


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

if __name__ == '__main__':
    train_dataset = Dataset(split='train')
    val_dataset = Dataset(split='val')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)

    model = BigModelUpsample().to(device)
    loss_fn = ContrastiveLoss()  # nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_val_loss = np.inf
    batch_size = 4
    for epoch in range(30):
        train_loss = 0
        print(f'STARTING EPOCH {epoch+1}')
        train_pbar = tqdm(train_dataloader)
        val_pbar = tqdm(val_dataloader)
        model.train()
        optimizer.zero_grad()
        for i, (inputs, labels) in enumerate(train_pbar):
            name = inputs[0]
            inputs = inputs[1]
            inputs = inputs.to(device)
            labels = labels.to(device)
            #optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            if loss.isnan():
                print(name)
                assert False
            (loss / batch_size).backward()
            train_loss += loss.detach().item()
            if i != 0 and i % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
            train_pbar.set_postfix_str(f'Train loss: {loss.detach().item()}')
        print(f'EPOCH TRAINING LOSS {train_loss / len(train_dataset)},')
        #torch.save(model.state_dict(), f'contrastive_save/contrastive_weights_epoch_{epoch}.pt')

        val_loss = 0
        with torch.no_grad():
            model.eval()
            for i, (inputs, labels) in enumerate(val_pbar):
                name = inputs[0]
                inputs = inputs[1]
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.detach().item()
        print(f'EPOCH EVAL LOSS {val_loss / len(val_dataset)},')
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), f'contrastive_save/contrastive_weights_best.pt')
            best_val_loss = val_loss
    torch.save(model.state_dict(), f'contrastive_save/contrastive_weights_last.pt')
