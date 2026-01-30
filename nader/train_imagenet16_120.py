import os, sys, hashlib
import argparse
import time
import random
from datetime import datetime
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math
import pdb
import json

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from ModelFactory.register import Registers, import_all_modules_for_register2
from train_utils.log import Log
from train_utils.config import get_device, get_num_workers, print_environment

# Get device from config
device = get_device()

def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    else:
        return check_md5(fpath, md5)

class ImageNet16(torch.utils.data.Dataset):
    # http://image-net.org/download-images
    # A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets
    # https://arxiv.org/pdf/1707.08819.pdf

    train_list = [
        ["train_data_batch_1", "27846dcaa50de8e21a7d1a35f30f0e91"],
        ["train_data_batch_2", "c7254a054e0e795c69120a5727050e3f"],
        ["train_data_batch_3", "4333d3df2e5ffb114b05d2ffc19b1e87"],
        ["train_data_batch_4", "1620cdf193304f4a92677b695d70d10f"],
        ["train_data_batch_5", "348b3c2fdbb3940c4e9e834affd3b18d"],
        ["train_data_batch_6", "6e765307c242a1b3d7d5ef9139b48945"],
        ["train_data_batch_7", "564926d8cbf8fc4818ba23d2faac7564"],
        ["train_data_batch_8", "f4755871f718ccb653440b9dd0ebac66"],
        ["train_data_batch_9", "bb6dd660c38c58552125b1a92f86b5d4"],
        ["train_data_batch_10", "8f03f34ac4b42271a294f91bf480f29b"],
    ]
    valid_list = [
        ["val_data", "3410e3017fdaefba8d5073aaa65e4bd6"],
    ]

    def __init__(self, root, train, transform, use_num_of_class_only=None):
        self.root = root
        self.transform = transform
        self.train = train  # training set or valid set
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.valid_list
        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for i, (file_name, checksum) in enumerate(downloaded_list):
            file_path = os.path.join(self.root, file_name)
            with open(file_path, "rb") as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                self.targets.extend(entry["labels"])
        self.data = np.vstack(self.data).reshape(-1, 3, 16, 16)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        if use_num_of_class_only is not None:
            assert (
                isinstance(use_num_of_class_only, int)
                and use_num_of_class_only > 0
                and use_num_of_class_only < 1000
            ), "invalid use_num_of_class_only : {:}".format(use_num_of_class_only)
            new_data, new_targets = [], []
            for I, L in zip(self.data, self.targets):
                if 1 <= L <= use_num_of_class_only:
                    new_data.append(I)
                    new_targets.append(L)
            self.data = new_data
            self.targets = new_targets

    def __repr__(self):
        return "{name}({num} images, {classes} classes)".format(
            name=self.__class__.__name__,
            num=len(self.data),
            classes=len(set(self.targets)),
        )

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index] - 1

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in self.train_list + self.valid_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, filename)
            if not check_integrity(fpath, md5):
                return False
        return True
    
def get_training_dataloader(mean, std, batch_size=16, shuffle=True):
    num_workers = get_num_workers()
    train_transform = transforms.Compose([
        transforms.RandomCrop(16, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_data = ImageNet16('datasets/ImageNet16', True, train_transform, 120)
    assert len(train_data) == 151700
    loader = DataLoader(train_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return loader

def get_val_dataloader(mean, std, path='nader/train_utils/imagenet-16-120-test-split.txt', batch_size=16, shuffle=True):
    num_workers = get_num_workers()
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_data = ImageNet16('datasets/ImageNet16', False, test_transform, 120)
    assert len(test_data) == 6000
    with open(path,'r') as f:
        ds = json.load(f)
        ids = [int(x) for x in ds['xvalid'][1]]
    loader = DataLoader(test_data, sampler=torch.utils.data.sampler.SubsetRandomSampler(ids), num_workers=num_workers, batch_size=batch_size)
    return loader
    
def get_test_dataloader(mean, std, batch_size=16, shuffle=True):
    num_workers = get_num_workers()
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_data = ImageNet16('datasets/ImageNet16', False, test_transform, 120)
    assert len(test_data) == 6000
    loader = DataLoader(test_data, shuffle=False, num_workers=num_workers, batch_size=batch_size)
    return loader


def train_one_epoch(net, loader, optimizer, loss_function, train_scheduler, epoch, log, scaler, use_amp=True, epoch_timeout=600):
    start = time.time()
    net.train()
    timeout_triggered = False
    
    for batch_index, (images, labels) in enumerate(loader):
        # Check timeout at the start of each batch
        elapsed = time.time() - start
        if elapsed > epoch_timeout:
            print(f'[Timeout] Epoch {epoch} exceeded {epoch_timeout}s at batch {batch_index}. Stopping...')
            timeout_triggered = True
            break
            
        labels = labels.to(device)
        images = images.to(device)
        optimizer.zero_grad()
        
        # Mixed Precision Training with autocast
        if use_amp and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = net(images)
                loss = loss_function(outputs, labels)
            # Scale loss and backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # FP32 fallback for CPU or when AMP disabled
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
        
        n_iter = (epoch - 1) * len(loader) + batch_index + 1
        if (batch_index+1)%50==0:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(loader.dataset)
            ))
        if (batch_index+1)%10==0:
            log.update('train_loss', loss.item(), n_iter)
    
    finish = time.time()
    elapsed = finish - start
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, elapsed))
    return elapsed, timeout_triggered  # Return elapsed time and timeout flag

@torch.no_grad()
def eval_training(net, loader, loss_function, tb=True):
    start = time.time()
    net.eval()
    test_loss = 0.0
    correct_1 = 0.0
    correct_5 = 0.0
    total = 0.0
    for (images, labels) in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        
        # Top-k 계산
        _, pred = outputs.topk(5, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        
        correct_1 += correct[:1].reshape(-1).float().sum(0, keepdim=True)
        correct_5 += correct[:5].reshape(-1).float().sum(0, keepdim=True)
        
        total += len(outputs)

    finish = time.time()
    
    acc1 = correct_1.item() / total
    acc5 = correct_5.item() / total
    
    res = {
        'loss': test_loss / len(loader),
        'acc': acc1,
        'acc5': acc5,
        'time': finish - start
    }
    return res

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(train_loader, test_loader, typ='val'):
    net = Registers.model[args.model_name](num_classes=120)
    net = net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Mixed Precision Training - GradScaler 초기화
    scaler = torch.cuda.amp.GradScaler(enabled=(args.use_amp and device.type == 'cuda'))
    if args.use_amp and device.type == 'cuda':
        print("[AMP] Mixed Precision Training (FP16) enabled")
    else:
        print("[AMP] Mixed Precision disabled, using FP32")

    best_epoch, best_acc, best_val_acc = -1, 0.0, 0.0
    epochs_without_improvement = 0  # For early stopping
    early_stop_reason = None
    
    for epoch in range(1, args.epochs + 1):
        epoch_time, timeout_triggered = train_one_epoch(net, train_loader, optimizer, loss_function, train_scheduler, epoch, log, scaler, args.use_amp, args.epoch_timeout)
        
        # Early stopping condition 4: Epoch timeout (triggered inside train_one_epoch)
        if timeout_triggered:
            print(f"[Early Stop] Epoch {epoch} timeout triggered inside training loop.")
            early_stop_reason = 'epoch_timeout'
            break
        
        train_scheduler.step(epoch)
        res = eval_training(net, test_loader, loss_function)
        current_acc = res['acc'] * 100  # Convert to percentage
        
        print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Top-5: {:.4f}, Time consumed:{:.2f}s'.format(
            epoch,
            res['loss'],
            res['acc'],
            res['acc5'],
            res['time']
        ))
        log.update(f'{typ}_acc', epoch, current_acc, res['acc5']*100)
        
        # Clear GPU cache periodically to prevent memory fragmentation
        if epoch % 10 == 0:
            torch.cuda.empty_cache()
        
        # Check for improvement
        if best_acc < res['acc']:
            weights_path = os.path.join(log_dir, 'best.pth')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = res['acc']
            best_epoch = epoch
            epochs_without_improvement = 0  # Reset counter on improvement
        else:
            epochs_without_improvement += 1
        
        print(f"{typ} best epoch {best_epoch}: acc{best_acc:.4f} (no improvement for {epochs_without_improvement} epochs)")
        sys.stdout.flush()  # Force flush to prevent I/O buffer issues
        
        # Early stopping condition 1: NaN loss
        if math.isnan(res['loss']):
            print(f"[Early Stop] NaN loss detected at epoch {epoch}. Stopping training.")
            early_stop_reason = 'nan_loss'
            break
        
        # Early stopping condition 2: Minimum accuracy threshold not met
        if epoch == args.min_acc_epoch and current_acc < args.min_acc_threshold:
            print(f"[Early Stop] Accuracy {current_acc:.2f}% < {args.min_acc_threshold}% at epoch {epoch}. Model appears untrainable.")
            early_stop_reason = 'min_acc_not_met'
            break
        
        # Early stopping condition 3: Patience exceeded
        if epochs_without_improvement >= args.patience:
            print(f"[Early Stop] No improvement for {args.patience} epochs. Best acc: {best_acc:.4f} at epoch {best_epoch}")
            early_stop_reason = 'patience_exceeded'
            break
    
    if early_stop_reason:
        print(f"Training stopped early due to: {early_stop_reason}")
    else:
        print(f"Training completed all {args.epochs} epochs.")
    
    return best_epoch, best_acc


if __name__ == '__main__':
    set_seed(888)
    print_environment()  # Print config settings

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=False)
    parser.add_argument('-b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('--code-dir', default='ModelFactory/codes/nas-bench-201', type=str)
    parser.add_argument('--output', type=str, default='output/nas-bench-201-imagenet16-120')
    parser.add_argument('--tag', default='1', help='tag of experiment')
    # Early stopping arguments
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--min-acc-epoch', type=int, default=50, help='Epoch to check minimum accuracy threshold')
    parser.add_argument('--min-acc-threshold', type=float, default=5.0, help='Minimum accuracy (%) required by min-acc-epoch')
    parser.add_argument('--epoch-timeout', type=int, default=600, help='Max seconds per epoch before timeout (default: 600s = 10min)')
    # Mixed Precision Training
    parser.add_argument('--use-amp', action='store_true', default=True, help='Enable FP16 Mixed Precision Training (default: True)')
    parser.add_argument('--no-amp', dest='use_amp', action='store_false', help='Disable Mixed Precision, use FP32')
    args = parser.parse_args()

    import_all_modules_for_register2(args.code_dir)

    log_dir = os.path.join(args.output, args.model_name, args.tag)
    status_path = os.path.join(log_dir, 'train_status.txt')
    
    # 로그 폴더 없으면 만들기
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log = Log(log_dir)

    # data preprocessing:
    mean = [x / 255 for x in [122.68, 116.66, 104.01]]
    std = [x / 255 for x in [63.22, 61.26, 65.09]]
    
    # DataLoaders - num_workers from config
    training_loader = get_training_dataloader(
        mean,
        std,
        batch_size=args.b,
    )
    val_loader = get_val_dataloader(
        mean,
        std,
        batch_size=args.b,
    )
    test_loader = get_test_dataloader(
        mean,
        std,
        batch_size=args.b,
    )
    
    # 실제 학습 실행 (Val set으로 학습)
    print("--- Start Training (Val) ---")
    best_val = train(training_loader, val_loader, 'val')
    
    # Test set은 학습 없이 best 모델로 평가만 수행
    print("--- Evaluating on Test Set ---")
    best_weights_path = os.path.join(log_dir, 'best.pth')
    if os.path.exists(best_weights_path):
        # Load best model and evaluate on test set
        net = Registers.model[args.model_name](num_classes=120)
        net = net.to(device)
        net.load_state_dict(torch.load(best_weights_path, map_location=device))
        loss_function = nn.CrossEntropyLoss()
        
        test_res = eval_training(net, test_loader, loss_function)
        test_acc = test_res['acc'] * 100
        test_acc5 = test_res['acc5'] * 100
        
        print(f"Test set evaluation: Accuracy: {test_acc:.2f}%, Top-5: {test_acc5:.2f}%")
        log.update('test_acc', best_val[0], test_acc, test_acc5)
        
        best_test = (best_val[0], test_res['acc'])
    else:
        print("Warning: best.pth not found. Skipping test evaluation.")
        best_test = (-1, -1)

    print(f"Val best epoch {best_val[0]}: acc {best_val[1]:.4f}")
    print(f"Test acc (using best val model): {best_test[1]:.4f}")

    # 정상 종료 시에만 success 기록
    with open(status_path, 'w') as f:
        f.write('success')
