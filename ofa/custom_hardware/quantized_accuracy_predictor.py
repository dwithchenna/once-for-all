"""
Quantized Accuracy Predictor for Custom NPU Hardware

This module provides accuracy prediction for quantized OFA sub-networks,
accounting for quantization-specific accuracy degradation.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import pickle

from ofa.utils import download_url
from .utils import load_hardware_config

logger = logging.getLogger(__name__)


class QuantizedAccuracyPredictor:
    """
    Predicts accuracy of quantized OFA sub-networks.
    
    This predictor is trained on a dataset of quantized model configurations
    and their actual accuracies, accounting for quantization degradation.
    """
    
    def __init__(
        self,
        quantization_scheme: str = 'int8',
        hardware_name: Optional[str] = None,
        pretrained: bool = True,
        model_path: Optional[str] = None,
        device: str = 'cuda:0'
    ):
        """
        Initialize the quantized accuracy predictor.
        
        Args:
            quantization_scheme: Quantization scheme ('int8', 'fp16', etc.)
            hardware_name: Target hardware name for hardware-specific predictions
            pretrained: Whether to load pretrained model
            model_path: Custom model path (overrides pretrained)
            device: Device for inference
        """
        self.quantization_scheme = quantization_scheme
        self.hardware_name = hardware_name
        self.device = device
        
        # Feature extraction configurations
        self.ks_map = self._construct_maps([3, 5, 7])
        self.ex_map = self._construct_maps([3, 4, 6])
        self.dp_map = self._construct_maps([2, 3, 4])
        
        # Build the predictor model
        self.model = self._build_model()
        
        if model_path:
            self._load_model(model_path)
        elif pretrained:
            self._load_pretrained_model()
            
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _construct_maps(self, keys: List[int]) -> Dict[int, int]:
        """Construct one-hot encoding maps for discrete values."""
        return {k: i for i, k in enumerate(sorted(set(keys)))}
    
    def _build_model(self) -> nn.Module:
        """
        Build the accuracy predictor neural network.
        
        Enhanced to account for quantization effects and hardware specifics.
        """
        # Base feature dimension (similar to original OFA predictor)
        base_dim = 128
        
        # Additional features for quantization
        quant_features = 16  # Features for quantization scheme, bit-width, etc.
        
        # Hardware-specific features
        hw_features = 8 if self.hardware_name else 0
        
        total_input_dim = base_dim + quant_features + hw_features
        
        model = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        return model
    
    def _load_pretrained_model(self):
        """Load pretrained quantized accuracy predictor."""
        try:
            # Try to load hardware and quantization specific model first
            if self.hardware_name:
                model_name = f"quantized_acc_predictor_{self.hardware_name}_{self.quantization_scheme}.pth"
                url = f"https://raw.githubusercontent.com/han-cai/files/master/ofa/quantized_predictors/{model_name}"
            else:
                model_name = f"quantized_acc_predictor_{self.quantization_scheme}.pth"
                url = f"https://raw.githubusercontent.com/han-cai/files/master/ofa/quantized_predictors/{model_name}"
            
            try:
                model_path = download_url(url)
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                logger.info(f"Loaded pretrained quantized accuracy predictor: {model_name}")
                return
            except:
                logger.warning(f"Hardware-specific model not found: {model_name}")
            
            # Fall back to general quantized predictor
            fallback_name = f"quantized_acc_predictor_{self.quantization_scheme}.pth"
            fallback_url = f"https://raw.githubusercontent.com/han-cai/files/master/ofa/quantized_predictors/{fallback_name}"
            
            try:
                model_path = download_url(fallback_url)
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                logger.info(f"Loaded fallback quantized accuracy predictor: {fallback_name}")
                return
            except:
                logger.warning(f"Quantized predictor not found: {fallback_name}")
            
            # Use original OFA predictor as last resort (with warning)
            original_url = "https://raw.githubusercontent.com/han-cai/files/master/ofa/acc_predictor.pth"
            model_path = download_url(original_url)
            
            # Load with adjusted dimensions
            state_dict = torch.load(model_path, map_location='cpu')
            self._adapt_pretrained_weights(state_dict)
            logger.warning("Using adapted original OFA predictor (may be less accurate for quantized models)")
            
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            logger.info("Using randomly initialized model")
    
    def _load_model(self, model_path: str):
        """Load custom model from path."""
        try:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            logger.info(f"Loaded custom model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load custom model: {e}")
    
    def _adapt_pretrained_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Adapt pretrained weights to new model architecture."""
        # Get current model state dict
        current_state = self.model.state_dict()
        
        # Adapt first layer if dimensions don't match
        first_layer_key = '0.weight'
        if first_layer_key in state_dict and first_layer_key in current_state:
            pretrained_weight = state_dict[first_layer_key]
            current_weight = current_state[first_layer_key]
            
            if pretrained_weight.shape != current_weight.shape:
                # Pad or truncate the pretrained weights
                min_in_features = min(pretrained_weight.shape[1], current_weight.shape[1])
                current_weight[:, :min_in_features] = pretrained_weight[:, :min_in_features]
                state_dict[first_layer_key] = current_weight
        
        # Load adapted weights
        self.model.load_state_dict(state_dict, strict=False)
    
    def _extract_features(self, subnet_config: Dict[str, Any]) -> torch.Tensor:
        """
        Extract features from subnet configuration.
        
        Includes base architectural features plus quantization-specific features.
        """
        features = []
        
        # Base architectural features (similar to original OFA)
        ks = subnet_config.get('ks', [7] * 20)
        e = subnet_config.get('e', [6] * 20) 
        d = subnet_config.get('d', [4] * 5)
        r = subnet_config.get('r', [224])
        
        # One-hot encode kernel sizes
        ks_features = np.zeros((20, len(self.ks_map)))
        for i, k in enumerate(ks[:20]):
            if k in self.ks_map:
                ks_features[i, self.ks_map[k]] = 1
        features.append(ks_features.flatten())
        
        # One-hot encode expand ratios
        e_features = np.zeros((20, len(self.ex_map)))
        for i, expand in enumerate(e[:20]):
            if expand in self.ex_map:
                e_features[i, self.ex_map[expand]] = 1
        features.append(e_features.flatten())
        
        # One-hot encode depths
        d_features = np.zeros((5, len(self.dp_map)))
        for i, depth in enumerate(d[:5]):
            if depth in self.dp_map:
                d_features[i, self.dp_map[depth]] = 1
        features.append(d_features.flatten())
        
        # Resolution features
        r_features = np.array([r[0] / 224.0])  # Normalized resolution
        features.append(r_features)
        
        # Quantization-specific features
        quant_features = self._extract_quantization_features()
        features.append(quant_features)
        
        # Hardware-specific features
        if self.hardware_name:
            hw_features = self._extract_hardware_features()
            features.append(hw_features)
        
        # Concatenate all features
        feature_vector = np.concatenate(features)
        return torch.tensor(feature_vector, dtype=torch.float32)
    
    def _extract_quantization_features(self) -> np.ndarray:
        """Extract quantization-specific features."""
        features = np.zeros(16)
        
        # Quantization scheme encoding
        if self.quantization_scheme == 'int8':
            features[0] = 1.0
            features[1] = 8.0 / 32.0  # Normalized bit width
        elif self.quantization_scheme == 'fp16':
            features[2] = 1.0
            features[1] = 16.0 / 32.0
        elif self.quantization_scheme == 'int4':
            features[3] = 1.0
            features[1] = 4.0 / 32.0
        else:  # fp32 or unknown
            features[4] = 1.0
            features[1] = 32.0 / 32.0
        
        # Additional quantization parameters
        features[5] = 1.0 if 'per_channel' in self.quantization_scheme else 0.0
        features[6] = 1.0 if 'static' in self.quantization_scheme else 0.0
        features[7] = 1.0 if 'dynamic' in self.quantization_scheme else 0.0
        
        return features
    
    def _extract_hardware_features(self) -> np.ndarray:
        """Extract hardware-specific features."""
        features = np.zeros(8)
        
        # Simple hardware encoding (can be extended)
        if self.hardware_name:
            # Hash hardware name to features
            hw_hash = hash(self.hardware_name) % 256
            for i in range(8):
                features[i] = ((hw_hash >> i) & 1)
        
        return features
    
    @torch.no_grad()
    def predict_accuracy(self, subnet_configs: List[Dict[str, Any]]) -> List[float]:
        """
        Predict accuracy for a list of subnet configurations.
        
        Args:
            subnet_configs: List of subnet configuration dictionaries
            
        Returns:
            List of predicted accuracies (0-100%)
        """
        if not subnet_configs:
            return []
        
        # Extract features for all configurations
        features_list = []
        for config in subnet_configs:
            features = self._extract_features(config)
            features_list.append(features)
        
        # Stack features into batch
        batch_features = torch.stack(features_list).to(self.device)
        
        # Predict
        predictions = self.model(batch_features)
        
        # Convert to accuracy percentages
        accuracies = [pred.item() * 100.0 for pred in predictions]
        
        return accuracies
    
    def predict_single(self, subnet_config: Dict[str, Any]) -> float:
        """
        Predict accuracy for a single subnet configuration.
        
        Args:
            subnet_config: Subnet configuration dictionary
            
        Returns:
            Predicted accuracy (0-100%)
        """
        return self.predict_accuracy([subnet_config])[0]
    
    def predict_accuracy_with_uncertainty(
        self, 
        subnet_configs: List[Dict[str, Any]],
        num_samples: int = 10
    ) -> List[Tuple[float, float]]:
        """
        Predict accuracy with uncertainty estimation using Monte Carlo dropout.
        
        Args:
            subnet_configs: List of subnet configuration dictionaries
            num_samples: Number of MC samples for uncertainty estimation
            
        Returns:
            List of (mean_accuracy, std_accuracy) tuples
        """
        if not subnet_configs:
            return []
        
        # Enable dropout for uncertainty estimation
        self.model.train()
        
        all_predictions = []
        for _ in range(num_samples):
            predictions = self.predict_accuracy(subnet_configs)
            all_predictions.append(predictions)
        
        # Compute statistics
        all_predictions = np.array(all_predictions)  # (num_samples, num_configs)
        mean_accuracies = np.mean(all_predictions, axis=0)
        std_accuracies = np.std(all_predictions, axis=0)
        
        # Return to eval mode
        self.model.eval()
        
        return list(zip(mean_accuracies, std_accuracies))
    
    def get_quantization_impact(self, subnet_config: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimate quantization impact by comparing with FP32 baseline.
        
        Args:
            subnet_config: Subnet configuration
            
        Returns:
            Dictionary with quantization impact metrics
        """
        # Predict with current quantization
        current_accuracy = self.predict_single(subnet_config)
        
        # Temporarily switch to FP32 and predict
        original_scheme = self.quantization_scheme
        self.quantization_scheme = 'fp32'
        fp32_accuracy = self.predict_single(subnet_config)
        self.quantization_scheme = original_scheme
        
        # Calculate impact
        accuracy_drop = fp32_accuracy - current_accuracy
        relative_drop = (accuracy_drop / fp32_accuracy) * 100 if fp32_accuracy > 0 else 0
        
        return {
            'fp32_accuracy': fp32_accuracy,
            'quantized_accuracy': current_accuracy,
            'accuracy_drop': accuracy_drop,
            'relative_drop_percent': relative_drop
        }
    
    def save_model(self, save_path: str):
        """Save the trained model."""
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")
    
    def benchmark_predictor(
        self, 
        test_configs: List[Dict[str, Any]], 
        true_accuracies: List[float]
    ) -> Dict[str, float]:
        """
        Benchmark predictor performance against ground truth.
        
        Args:
            test_configs: List of test configurations
            true_accuracies: List of true accuracies
            
        Returns:
            Performance metrics dictionary
        """
        predicted_accuracies = self.predict_accuracy(test_configs)
        
        # Calculate metrics
        true_acc = np.array(true_accuracies)
        pred_acc = np.array(predicted_accuracies)
        
        mae = np.mean(np.abs(true_acc - pred_acc))
        mse = np.mean((true_acc - pred_acc) ** 2)
        rmse = np.sqrt(mse)
        
        # Correlation
        correlation = np.corrcoef(true_acc, pred_acc)[0, 1]
        
        # Spearman rank correlation
        from scipy.stats import spearmanr
        spearman_corr, _ = spearmanr(true_acc, pred_acc)
        
        return {
            'mae': mae,
            'mse': mse, 
            'rmse': rmse,
            'correlation': correlation,
            'spearman_correlation': spearman_corr,
            'num_samples': len(test_configs)
        }
