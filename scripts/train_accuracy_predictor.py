#!/usr/bin/env python3
"""
Train quantized accuracy predictor for custom hardware

This script trains a neural network to predict the accuracy of quantized
OFA sub-networks on custom NPU hardware.
"""

import argparse
import os
import logging
import json
import random
from typing import List, Dict, Tuple, Any
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision import transforms, datasets

from ofa.model_zoo import ofa_net
from ofa.custom_hardware import (
    ONNXConverter,
    QuantizedAccuracyPredictor,
    evaluate_quantized_model
)
from ofa.custom_hardware.utils import setup_logging, load_config, save_results
from ofa.nas.accuracy_predictor.arch_encoder import MobileNetV3ArchEncoder


class AccuracyDataset(torch.utils.data.Dataset):
    """Dataset for training accuracy predictor"""
    
    def __init__(self, arch_features: List[np.ndarray], accuracies: List[float]):
        self.arch_features = torch.tensor(np.array(arch_features), dtype=torch.float32)
        self.accuracies = torch.tensor(accuracies, dtype=torch.float32)
    
    def __len__(self):
        return len(self.arch_features)
    
    def __getitem__(self, idx):
        return self.arch_features[idx], self.accuracies[idx]


def sample_subnet_configs(ofa_network, num_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Sample random subnet configurations"""
    random.seed(seed)
    np.random.seed(seed)
    
    configs = []
    
    # Get the architecture space from the OFA network
    if hasattr(ofa_network, 'ks_list'):
        ks_list = ofa_network.ks_list
    else:
        ks_list = [3, 5, 7]  # Default kernel sizes
    
    if hasattr(ofa_network, 'expand_ratio_list'):  
        e_list = ofa_network.expand_ratio_list
    else:
        e_list = [3, 4, 6]  # Default expansion ratios
    
    if hasattr(ofa_network, 'depth_list'):
        d_list = ofa_network.depth_list  
    else:
        d_list = [2, 3, 4]  # Default depths
    
    if hasattr(ofa_network, 'width_mult_list'):
        w_list = ofa_network.width_mult_list
    else:
        w_list = [1.0]  # Default width multiplier
    
    # Sample configurations
    for _ in range(num_samples):
        config = {
            'ks': [random.choice(ks_list) for _ in range(20)],  # 20 blocks in MobileNetV3
            'e': [random.choice(e_list) for _ in range(20)],
            'd': [random.choice(d_list) for _ in range(5)],  # 5 stages
            'w': random.choice(w_list)
        }
        configs.append(config)
    
    return configs


def evaluate_subnet_accuracy(subnet, data_loader: torch.utils.data.DataLoader, 
                            device: str = 'cuda', max_batches: int = None) -> float:
    """Evaluate subnet accuracy on validation data"""
    subnet.eval()
    subnet = subnet.to(device)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            if max_batches and i >= max_batches:
                break
                
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = subnet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100.0 * correct / total


def create_calibration_loader(dataset_path: str, batch_size: int = 32, 
                             num_samples: int = 100, image_size: int = 224) -> torch.utils.data.DataLoader:
    """Create a small calibration dataset for quantization"""
    transform = transforms.Compose([
        transforms.Resize(int(image_size / 0.875)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(
        os.path.join(dataset_path, 'val'),
        transform=transform
    )
    
    # Create a subset for calibration
    indices = torch.randperm(len(dataset))[:num_samples]
    subset = torch.utils.data.Subset(dataset, indices)
    
    return torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )


def main():
    parser = argparse.ArgumentParser(description='Train quantized accuracy predictor')
    
    # Required arguments
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to ImageNet dataset')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save trained predictor')
    
    # Model configuration
    parser.add_argument('--ofa_network', type=str, default='ofa_mbv3_d234_e346_k357_w1.0',
                       help='OFA network name')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    
    # Training data generation
    parser.add_argument('--num_samples', type=int, default=2000,
                       help='Number of subnet samples to generate')
    parser.add_argument('--eval_batches', type=int, default=50,
                       help='Number of batches for accuracy evaluation')
    parser.add_argument('--calibration_samples', type=int, default=100,
                       help='Number of samples for quantization calibration')
    
    # Hardware configuration
    parser.add_argument('--config', type=str, default=None,
                       help='Hardware configuration YAML file')
    parser.add_argument('--provider', type=str, default='CPUExecutionProvider',
                       help='ONNX Runtime execution provider')
    parser.add_argument('--quantization', action='store_true',
                       help='Enable quantization during training data generation')
    
    # Training parameters
    parser.add_argument('--hidden_size', type=int, default=400,
                       help='Hidden size for predictor network')
    parser.add_argument('--n_layers', type=int, default=3,
                       help='Number of layers in predictor network')
    parser.add_argument('--epochs', type=int, default=300,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Training split ratio')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Training device')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--save_data', type=str, default=None,
                       help='Path to save generated training data')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(level=logging.INFO if args.verbose else logging.WARNING)
    logger = logging.getLogger(__name__)
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Load configuration
    config = load_config(args.config) if args.config else {}
    config.setdefault('hardware', {})
    config['hardware']['provider'] = args.provider
    config.setdefault('quantization', {})
    config['quantization']['enabled'] = args.quantization
    
    # Check device
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load OFA network
    logger.info(f"Loading OFA network: {args.ofa_network}")
    ofa_network = ofa_net(args.ofa_network, pretrained=True)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    val_transform = transforms.Compose([
        transforms.Resize(int(args.image_size / 0.875)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder(
        os.path.join(args.dataset_path, 'val'),
        transform=val_transform
    )
    
    eval_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Create calibration loader if quantization is enabled
    calibration_loader = None
    if args.quantization:
        calibration_loader = create_calibration_loader(
            args.dataset_path,
            num_samples=args.calibration_samples,
            image_size=args.image_size
        )
    
    # Initialize ONNX converter
    converter = ONNXConverter(
        quantization_enabled=args.quantization,
        config=config.get('quantization', {}),
        calibration_loader=calibration_loader
    )
    
    # Sample subnet configurations
    logger.info(f"Sampling {args.num_samples} subnet configurations...")
    subnet_configs = sample_subnet_configs(ofa_network, args.num_samples, args.seed)
    
    # Generate training data
    logger.info("Generating training data...")
    arch_features = []
    accuracies = []
    
    # Initialize architecture encoder
    arch_encoder = MobileNetV3ArchEncoder()
    
    input_shape = (1, 3, args.image_size, args.image_size)
    
    for i, config in enumerate(tqdm(subnet_configs, desc="Evaluating subnets")):
        try:
            # Set subnet configuration
            if hasattr(ofa_network, 'set_active_subnet'):
                ofa_network.set_active_subnet(**config)
            
            # Get subnet
            subnet = ofa_network.get_active_subnet() if hasattr(ofa_network, 'get_active_subnet') else ofa_network
            
            if args.quantization:
                # Convert to ONNX and evaluate quantized accuracy
                onnx_model_path = converter.convert_subnet(
                    subnet, config, input_shape,
                    output_path=f"temp_subnet_{i}.onnx"
                )
                
                onnx_results = evaluate_quantized_model(
                    onnx_model_path,
                    eval_loader,
                    provider=args.provider,
                    provider_options=config.get('hardware', {}).get('provider_options', {}),
                    image_size=args.image_size,
                    max_batches=args.eval_batches
                )
                
                accuracy = onnx_results['accuracy']['top1_accuracy']
                
                # Clean up temp file
                if os.path.exists(onnx_model_path):
                    os.remove(onnx_model_path)
            else:
                # Evaluate PyTorch model
                accuracy = evaluate_subnet_accuracy(
                    subnet, eval_loader, device, args.eval_batches
                )
            
            # Encode architecture
            arch_feature = arch_encoder.arch2feature(config)
            
            arch_features.append(arch_feature)
            accuracies.append(accuracy)
            
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(subnet_configs)} subnets, "
                          f"avg accuracy: {np.mean(accuracies):.2f}%")
        
        except Exception as e:
            logger.warning(f"Failed to evaluate subnet {i}: {str(e)}")
            continue
    
    logger.info(f"Generated {len(arch_features)} training samples")
    
    # Save training data if requested
    if args.save_data:
        training_data = {
            'arch_features': [feat.tolist() for feat in arch_features],
            'accuracies': accuracies,
            'subnet_configs': subnet_configs[:len(accuracies)]
        }
        with open(args.save_data, 'w') as f:
            json.dump(training_data, f, indent=2)
        logger.info(f"Training data saved to {args.save_data}")
    
    # Split data
    n_train = int(len(arch_features) * args.train_split)
    indices = np.random.permutation(len(arch_features))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_features = [arch_features[i] for i in train_indices]
    train_accuracies = [accuracies[i] for i in train_indices]
    val_features = [arch_features[i] for i in val_indices]
    val_accuracies = [accuracies[i] for i in val_indices]
    
    # Create datasets
    train_dataset = AccuracyDataset(train_features, train_accuracies)
    val_dataset = AccuracyDataset(val_features, val_accuracies)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    # Initialize predictor
    logger.info("Initializing accuracy predictor...")
    predictor = QuantizedAccuracyPredictor(
        arch_encoder=arch_encoder,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        device=device
    )
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        # Training
        predictor.train()
        train_loss = 0.0
        
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            predictions = predictor(features)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        predictor.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                
                predictions = predictor(features)
                loss = criterion(predictions, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'state_dict': predictor.state_dict(),
                'arch_encoder': arch_encoder,
                'config': {
                    'hidden_size': args.hidden_size,
                    'n_layers': args.n_layers,
                    'quantization_enabled': args.quantization
                },
                'training_stats': {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'num_samples': len(arch_features)
                }
            }, args.output_path)
        
        if epoch % 50 == 0 or epoch == args.epochs - 1:
            logger.info(f"Epoch {epoch}/{args.epochs}: "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to {args.output_path}")
    
    # Final evaluation
    logger.info("Running final evaluation...")
    predictor.eval()
    
    with torch.no_grad():
        train_features_tensor = torch.tensor(np.array(train_features), dtype=torch.float32).to(device)
        train_preds = predictor(train_features_tensor).cpu().numpy()
        train_targets = np.array(train_accuracies)
        
        val_features_tensor = torch.tensor(np.array(val_features), dtype=torch.float32).to(device)
        val_preds = predictor(val_features_tensor).cpu().numpy()
        val_targets = np.array(val_accuracies)
    
    # Calculate metrics
    train_mse = np.mean((train_preds - train_targets) ** 2)
    train_mae = np.mean(np.abs(train_preds - train_targets))
    val_mse = np.mean((val_preds - val_targets) ** 2)
    val_mae = np.mean(np.abs(val_preds - val_targets))
    
    results = {
        'training_config': vars(args),
        'final_metrics': {
            'train_mse': float(train_mse),
            'train_mae': float(train_mae),
            'val_mse': float(val_mse),
            'val_mae': float(val_mae)
        },
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    }
    
    # Save training results
    results_path = args.output_path.replace('.pth', '_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Training samples: {len(train_features)}")
    print(f"Validation samples: {len(val_features)}")
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Training MAE: {train_mae:.4f}")
    print(f"Validation MSE: {val_mse:.4f}")
    print(f"Validation MAE: {val_mae:.4f}")
    print(f"Quantization enabled: {args.quantization}")
    print(f"Model saved to: {args.output_path}")
    print("="*50)


if __name__ == '__main__':
    main()
