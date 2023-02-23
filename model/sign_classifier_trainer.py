
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import pandas as pd
import os
import numpy as np
import cv2
from sign_classifier import SignClassifierModel
from torch.optim import SGD, Adam
import  torch.optim.lr_scheduler as lr_scheduler
from torch.nn import CrossEntropyLoss, Module
import torch
import logging


class SignDataset(Dataset):
    def __init__(
            self,
            root_dir,
            images_dir,
            label_file_name,
            num_classes=10,
            transforms=None
            ) -> None:
        super(SignDataset, self).__init__()
        labels_file = pd.read_csv(os.path.join(root_dir, label_file_name))
        self.labels = labels_file.label.to_numpy()
        self.labels = np.eye(num_classes)[self.labels]
        self.images_names = labels_file.image_name.to_numpy(dtype=str)
        self.image_dir = os.path.join(root_dir, images_dir)
        self.transforms = None

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images_names[index])
        label = self.labels[index]
        image = np.asarray(cv2.imread(img_path).transpose((2, 0, 1)), dtype=np.float32)
        if self.transforms is None:
            pass
        else:
            pass
        return image, label


class Trainer:
    def __init__(self,
                 model: Module,
                 train_loader,
                 val_loader,
                 optimzer,
                 scheduler,
                 criterion: CrossEntropyLoss
                 ) -> None:
        self.model = model
        self.loss = criterion
        self.optimizer = optimzer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler

    def train_epoch(self):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        for i, (inputs, targets) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, targets)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item() * inputs.size(0)

            # Calculate accuracy
            with torch.no_grad():
                total += targets.size(0)
                predicted_classes = torch.argmax(outputs, dim=1)
                true_classes = torch.argmax(targets, dim=1)
                correct += (predicted_classes == true_classes).float().sum()
                # correct += ((outputs > 0.5).float().argmax(1) == targets.argmax(1)).sum()

        # self.scheduler.step()
        epoch_train_loss = train_loss / len(self.train_loader.dataset)
        epoch_train_acc = 100 * (correct / total)
        return epoch_train_loss, epoch_train_acc

    def val_epoch(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.val_loader):
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

                # Calculate accuracy
                total += targets.size(0)
                predicted_classes = torch.argmax(outputs, dim=1)
                true_classes = torch.argmax(targets, dim=1)
                correct += (predicted_classes == true_classes).float().sum()

            epoch_val_loss = val_loss / len(self.val_loader.dataset)
            epoch_val_acc = 100 * (correct / total)

        return epoch_val_loss, epoch_val_acc

    def train(self, n_epochs):
        train_losses = []
        train_acc = []
        val_losses = []
        val_acc = []
        for i in range(n_epochs):
            epoch_train_loss, epoch_train_acc = self.train_epoch()
            epoch_val_loss, epoch_val_acc = self.val_epoch()
            train_losses.append(epoch_train_loss)
            train_acc.append(epoch_train_acc)
            val_losses.append(epoch_val_loss)
            val_acc.append(epoch_val_acc)
            logging.info(
                f"epoch {i + 1}/{n_epochs}:train loss = {epoch_train_loss},train acc = {epoch_train_acc},val loss = {epoch_val_loss},val_acc = {epoch_val_acc}")
        return train_losses, train_acc, val_losses, val_acc

class Evaluator:
    def __init__(self, model, test_loader, criterion) -> None:
        self.model = model
        self.test_loader = test_loader 
        self.loss = criterion

    def evaluate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.test_loader):
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

                # Calculate accuracy
                total += targets.size(0)
                predicted_classes = torch.argmax(outputs, dim=1)
                true_classes = torch.argmax(targets, dim=1)
                correct += (predicted_classes == true_classes).float().sum()

            epoch_val_loss = val_loss / len(self.test_loader.dataset)
            epoch_val_acc = 100 * (correct / total)

        return epoch_val_loss, epoch_val_acc

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    root_dir = "datasets/signs"
    train_data = SignDataset(root_dir, "train", "train.csv")
    val_data = SignDataset(root_dir, "val", "val.csv")
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=True)
    model = SignClassifierModel()
    # adam
    # optimizer = SGD(net.parameters(), lr=1e-5, momentum=0.9,
    #                  nesterov=False)
    optimizer = Adam(model.parameters(), lr=1e-5)
    # define a scheduler to reduce the learning rate every 10 epochs by a factor of 0.1
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = None
    loss = CrossEntropyLoss()
    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler,loss)
    trainer.train(100)

    test_data = SignDataset(root_dir, "test", "test.csv")
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
    evaluator = Evaluator(model, test_loader, loss)
    test_loss, test_acc = evaluator.evaluate()
    print(f"test_loss = {test_loss}, test_acc = {test_acc}")
    torch.save(model.state_dict(),"model1.pth")

