#!/usr/bin/env python3
"""
Training Script for Quantized Accuracy and Latency Predictors

This script trains neural network predictors using the collected data
for use in Neural Architecture Search (NAS).
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SubnetDataset(Dataset):
    """Dataset for subnet configurations and their metrics."""
    
    def __init__(self, data: List[Dict[str, Any]], target_metric: str = 'accuracy'):
        self.data = data
        self.target_metric = target_metric
        self.features = []
        self.targets = []
        
        # Extract features and targets
        for item in data:
            if item[target_metric] is not None and not np.isnan(item[target_metric]):
                feature = self._extract_features(item['config'])
                self.features.append(feature)
                self.targets.append(item[target_metric])
        
        self.features = np.array(self.features)
        self.targets = np.array(self.targets)
        
        # Normalize features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
        
        logger.info(f"Dataset loaded: {len(self.features)} samples, {self.features.shape[1]} features")
        logger.info(f"Target metric: {target_metric}")
        logger.info(f"Target range: [{np.min(self.targets):.4f}, {np.max(self.targets):.4f}]")
    
    def _extract_features(self, config: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from subnet configuration."""
        features = []
        
        # Kernel size features (20 blocks)
        ks_features = []
        for ks in config['ks']:
            ks_one_hot = [0, 0, 0]  # [3, 5, 7]
            if ks == 3:
                ks_one_hot[0] = 1
            elif ks == 5:
                ks_one_hot[1] = 1
            elif ks == 7:
                ks_one_hot[2] = 1
            ks_features.extend(ks_one_hot)
        features.extend(ks_features)
        
        # Expand ratio features (20 blocks)
        e_features = []
        for e in config['e']:
            e_one_hot = [0, 0, 0]  # [3, 4, 6]
            if e == 3:
                e_one_hot[0] = 1
            elif e == 4:
                e_one_hot[1] = 1
            elif e == 6:
                e_one_hot[2] = 1
            e_features.extend(e_one_hot)
        features.extend(e_features)
        
        # Depth features (5 stages)
        d_features = []
        for d in config['d']:
            d_one_hot = [0, 0, 0]  # [2, 3, 4]
            if d == 2:
                d_one_hot[0] = 1
            elif d == 3:
                d_one_hot[1] = 1
            elif d == 4:
                d_one_hot[2] = 1
            d_features.extend(d_one_hot)
        features.extend(d_features)
        
        # Resolution features
        r = config['r']
        r_normalized = (r - 160) / (224 - 160)  # Normalize to [0, 1]
        features.append(r_normalized)
        
        # Additional aggregate features
        features.append(np.mean(config['ks']))  # Average kernel size
        features.append(np.mean(config['e']))   # Average expand ratio
        features.append(np.mean(config['d']))   # Average depth
        features.append(np.sum(config['d']))    # Total depth
        
        return np.array(features)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor([self.targets[idx]])


class PredictorNet(nn.Module):
    """Neural network predictor for accuracy or latency."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128]):
        super(PredictorNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class PredictorTrainer:
    """Trainer for accuracy and latency predictors."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load data
        with open(dataset_path, 'r') as f:
            self.raw_data = json.load(f)
        
        logger.info(f"Loaded {len(self.raw_data)} data points")
    
    def train_predictor(self, metric: str, test_size: float = 0.2, 
                       epochs: int = 100, batch_size: int = 32,
                       learning_rate: float = 0.001) -> Tuple[PredictorNet, Dict[str, Any]]:
        """Train a predictor for the specified metric."""
        
        logger.info(f"Training {metric} predictor")
        
        # Create dataset
        dataset = SubnetDataset(self.raw_data, target_metric=metric)
        
        # Split data
        train_idx, test_idx = train_test_split(
            range(len(dataset)), test_size=test_size, random_state=42
        )
        
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        input_dim = dataset.features.shape[1]
        model = PredictorNet(input_dim).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_features, batch_targets in test_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    val_loss += loss.item()
            
            val_loss /= len(test_loader)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Evaluate final model
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(self.device)
                outputs = model(batch_features)
                predictions.extend(outputs.cpu().numpy().flatten())
                actuals.extend(batch_targets.cpu().numpy().flatten())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        mae = np.mean(np.abs(actuals - predictions))
        
        results = {
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'predictions': predictions,
            'actuals': actuals,
            'scaler': dataset.scaler
        }
        
        logger.info(f"{metric} predictor training complete:")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  R2: {r2:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        
        return model, results
    
    def save_model(self, model: PredictorNet, results: Dict[str, Any], 
                   metric: str, save_dir: str = "./models"):
        """Save trained model and results."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, f"{metric}_predictor.pth")
        torch.save(model.state_dict(), model_path)
        
        # Save scaler
        scaler_path = os.path.join(save_dir, f"{metric}_scaler.pkl")
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(results['scaler'], f)
        
        # Save results
        results_path = os.path.join(save_dir, f"{metric}_results.json")
        results_to_save = {
            'mse': float(results['mse']),
            'r2': float(results['r2']),
            'mae': float(results['mae'])
        }
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        logger.info(f"Results saved to {results_path}")
    
    def plot_results(self, results: Dict[str, Any], metric: str, save_dir: str = "./plots"):
        """Plot training results."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot training curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(results['train_losses'], label='Training Loss')
        plt.plot(results['val_losses'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{metric} Predictor Training Curves')
        plt.legend()
        plt.grid(True)
        
        # Plot predictions vs actuals
        plt.subplot(1, 2, 2)
        plt.scatter(results['actuals'], results['predictions'], alpha=0.5)
        plt.plot([np.min(results['actuals']), np.max(results['actuals'])],
                [np.min(results['actuals']), np.max(results['actuals'])], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{metric} Predictor: Actual vs Predicted')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{metric}_predictor_results.png'))
        plt.close()
        
        logger.info(f"Results plot saved to {save_dir}/{metric}_predictor_results.png")


def main():
    """Main function to train predictors."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train quantized accuracy and latency predictors")
    parser.add_argument(
        "--dataset", 
        default="./data/collected/complete_dataset.json",
        help="Path to collected dataset"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Batch size for training"
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = PredictorTrainer(args.dataset)
    
    # Train accuracy predictor
    accuracy_model, accuracy_results = trainer.train_predictor(
        'accuracy', epochs=args.epochs, batch_size=args.batch_size
    )
    trainer.save_model(accuracy_model, accuracy_results, 'accuracy')
    trainer.plot_results(accuracy_results, 'accuracy')
    
    # Train latency predictor
    latency_model, latency_results = trainer.train_predictor(
        'latency', epochs=args.epochs, batch_size=args.batch_size
    )
    trainer.save_model(latency_model, latency_results, 'latency')
    trainer.plot_results(latency_results, 'latency')
    
    print("Training complete!")
    print("Models saved in ./models/")
    print("Plots saved in ./plots/")


if __name__ == "__main__":
    main()
