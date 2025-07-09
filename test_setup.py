#!/usr/bin/env python3
"""
Test script to verify the setup for data collection.
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
        print(f"  Version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import onnx
        print("✓ ONNX imported successfully")
        print(f"  Version: {onnx.__version__}")
    except ImportError as e:
        print(f"✗ ONNX import failed: {e}")
        return False
    
    try:
        import onnxruntime as ort
        print("✓ ONNX Runtime imported successfully")
        print(f"  Version: {ort.__version__}")
        print(f"  Available providers: {ort.get_available_providers()}")
    except ImportError as e:
        print(f"✗ ONNX Runtime import failed: {e}")
        return False
    
    try:
        import yaml
        print("✓ PyYAML imported successfully")
    except ImportError as e:
        print(f"✗ PyYAML import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
        print(f"  Version: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration file loading."""
    print("\nTesting configuration...")
    
    config_path = "./configs/stx_npu_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"✗ Configuration file not found: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✓ Configuration loaded successfully")
        print(f"  Hardware: {config['hardware']['name']}")
        print(f"  Provider: {config['hardware']['provider']}")
        print(f"  Quantization enabled: {config['quantization']['enabled']}")
        print(f"  Quantization mode: {config['quantization']['mode']}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False

def test_directories():
    """Test that required directories exist or can be created."""
    print("\nTesting directories...")
    
    directories = [
        "./models",
        "./results", 
        "./data/calibration",
        "./data/collected",
        "./data/onnx_models",
        "./temp",
        "./plots"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Directory ready: {directory}")
        except Exception as e:
            print(f"✗ Directory creation failed: {directory} - {e}")
            return False
    
    return True

def test_ofa_model():
    """Test OFA model loading."""
    print("\nTesting OFA model...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from ofa.model_zoo import ofa_net
        
        # Try to load a simple OFA model
        print("  Loading OFA MobileNetV3...")
        ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.0', pretrained=True)
        
        print("✓ OFA model loaded successfully")
        print(f"  Model type: {type(ofa_network)}")
        
        # Test subnet sampling
        print("  Testing subnet sampling...")
        ofa_network.sample_active_subnet()
        subnet = ofa_network.get_active_subnet(preserve_weight=True)
        print(f"✓ Subnet sampling successful")
        print(f"  Subnet type: {type(subnet)}")
        
        return True
    except Exception as e:
        print(f"✗ OFA model test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("OFA Custom Hardware Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_directories,
        test_ofa_model
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✓ All tests passed! You're ready to collect data.")
    else:
        print("✗ Some tests failed. Please fix the issues before proceeding.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
