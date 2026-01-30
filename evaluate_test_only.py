# =============================================================================
# Test-Only Evaluation Script
# best.pth 모델로 test set에 대해서만 평가
# =============================================================================

import os
import sys
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# NADER 경로 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nader'))

from ModelFactory.register import Registers, import_all_modules_for_register2
from train_utils.config import get_device, get_num_workers

device = get_device()


def get_test_loader(dataset, batch_size=256):
    """Test set만 로드"""
    num_workers = get_num_workers()
    
    if dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        num_classes = 10
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_data = torchvision.datasets.CIFAR10(
            'datasets', train=False, transform=test_transform, download=True
        )
        
    elif dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
        num_classes = 100
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_data = torchvision.datasets.CIFAR100(
            'datasets', train=False, transform=test_transform, download=True
        )
        
    elif dataset == 'imagenet16-120':
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22, 61.26, 65.09]]
        num_classes = 120
        
        from PIL import Image
        import pickle
        import numpy as np
        
        # ImageNet16 custom dataset
        class ImageNet16Test(torch.utils.data.Dataset):
            def __init__(self, root, transform):
                self.root = root
                self.transform = transform
                
                file_path = os.path.join(root, 'val_data')
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    self.data = entry['data']
                    self.targets = entry['labels']
                
                self.data = np.array(self.data).reshape(-1, 3, 16, 16)
                self.data = self.data.transpose((0, 2, 3, 1))
                
                # Filter to 120 classes
                new_data, new_targets = [], []
                for img, label in zip(self.data, self.targets):
                    if 1 <= label <= 120:
                        new_data.append(img)
                        new_targets.append(label)
                self.data = new_data
                self.targets = new_targets
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, index):
                img, target = self.data[index], self.targets[index] - 1
                img = Image.fromarray(img)
                if self.transform:
                    img = self.transform(img)
                return img, target
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_data = ImageNet16Test('datasets/ImageNet16', test_transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    test_loader = DataLoader(
        test_data, 
        shuffle=False, 
        num_workers=num_workers, 
        batch_size=batch_size
    )
    
    return test_loader, num_classes


@torch.no_grad()
def evaluate(net, loader):
    """모델 평가"""
    net.eval()
    
    correct_1 = 0.0
    correct_5 = 0.0
    total = 0.0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = net(images)
        
        # Top-k 계산
        _, pred = outputs.topk(5, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        
        correct_1 += correct[:1].reshape(-1).float().sum(0, keepdim=True)
        correct_5 += correct[:5].reshape(-1).float().sum(0, keepdim=True)
        
        total += len(outputs)
    
    acc1 = correct_1.item() / total * 100
    acc5 = correct_5.item() / total * 100
    
    return acc1, acc5


def main():
    parser = argparse.ArgumentParser(description='Evaluate best.pth on test set only')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to best.pth file')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Model name (same as training)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet16-120'],
                        help='Dataset to evaluate on')
    parser.add_argument('--code-dir', type=str, default='nader/ModelFactory/codes/nas-bench-201',
                        help='Directory containing model code')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for evaluation')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file to save results (optional)')
    
    args = parser.parse_args()
    
    # 모델 코드 import
    print(f"Loading model codes from: {args.code_dir}")
    import_all_modules_for_register2(args.code_dir)
    
    # 데이터셋별 클래스 수
    dataset_classes = {
        'cifar10': 10,
        'cifar100': 100,
        'imagenet16-120': 120
    }
    num_classes = dataset_classes[args.dataset]
    
    # 모델 로드
    print(f"Loading model: {args.model_name}")
    net = Registers.model[args.model_name](num_classes=num_classes)
    net = net.to(device)
    
    # 체크포인트 로드
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(checkpoint)
    
    # 테스트 데이터 로드
    print(f"Loading {args.dataset} test set...")
    test_loader, _ = get_test_loader(args.dataset, args.batch_size)
    
    # 평가
    print("Evaluating...")
    test_acc, test_acc5 = evaluate(net, test_loader)
    
    # 결과 출력
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")
    print("-" * 60)
    print(f"Test Accuracy (Top-1): {test_acc:.2f}%")
    print(f"Test Accuracy (Top-5): {test_acc5:.2f}%")
    print("=" * 60)
    
    # 결과 파일로 저장 (선택)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(f"model_name: {args.model_name}\n")
            f.write(f"dataset: {args.dataset}\n")
            f.write(f"checkpoint: {args.checkpoint}\n")
            f.write(f"test_acc: {test_acc:.2f}\n")
            f.write(f"test_acc5: {test_acc5:.2f}\n")
        print(f"Results saved to: {args.output}")
    
    return test_acc, test_acc5


if __name__ == '__main__':
    main()
