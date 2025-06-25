#!/usr/bin/env python3
"""
Integration test for OFA custom hardware components

This script tests that all custom hardware integration components
are properly installed and can be imported/used together.
"""

import sys
import traceback
from pathlib import Path


def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        # Core OFA imports
        from ofa.model_zoo import ofa_net
        print("✓ OFA model zoo import successful")
        
        from ofa.utils import AverageMeter, accuracy
        print("✓ OFA utils import successful")
        
        # Architecture encoder
        from ofa.nas.accuracy_predictor.arch_encoder import MobileNetArchEncoder
        print("✓ Architecture encoder import successful")
        
        # Custom hardware imports
        from ofa.custom_hardware import (
            QuantizedLatencyPredictor,
            QuantizedAccuracyPredictor,
            ONNXConverter,
            CustomHardwareEvolutionFinder
        )
        print("✓ Custom hardware module imports successful")
        
        from ofa.custom_hardware.utils import load_config, setup_logging
        print("✓ Custom hardware utils import successful")
        
        # Optional ONNX imports
        try:
            import onnx
            import onnxruntime as ort
            print("✓ ONNX and ONNX Runtime imports successful")
        except ImportError as e:
            print(f"⚠ ONNX imports failed: {e}")
            print("  Please install: pip install onnx onnxruntime")
            return False
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_ofa_network():
    """Test OFA network loading"""
    print("\nTesting OFA network loading...")
    
    try:
        # Load a small OFA network
        ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.0', pretrained=True)
        print("✓ OFA network loaded successfully")
        
        # Test subnet configuration
        subnet_config = {
            'ks': [3, 3, 3, 5, 5, 5, 3, 3, 5, 5, 5, 3, 3, 5, 7, 7, 7, 7, 7, 7],
            'e': [3, 3, 3, 4, 4, 4, 3, 3, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            'd': [2, 3, 4, 4, 4],
            'w': 1.0
        }
        
        if hasattr(ofa_network, 'set_active_subnet'):
            ofa_network.set_active_subnet(**subnet_config)
            print("✓ Subnet configuration successful")
        else:
            print("⚠ OFA network doesn't support subnet configuration")
        
        return True
        
    except Exception as e:
        print(f"✗ OFA network test failed: {e}")
        traceback.print_exc()
        return False


def test_custom_hardware_components():
    """Test custom hardware component initialization"""
    print("\nTesting custom hardware components...")
    
    try:
        # Test ONNX converter
        from ofa.custom_hardware import ONNXConverter
        converter = ONNXConverter(quantization_enabled=False)
        print("✓ ONNX converter initialized")
        
        # Test latency predictor
        from ofa.custom_hardware import QuantizedLatencyPredictor
        latency_predictor = QuantizedLatencyPredictor(
            provider="CPUExecutionProvider",
            provider_options={}
        )
        print("✓ Latency predictor initialized")
        
        # Test architecture encoder
        from ofa.nas.accuracy_predictor.arch_encoder import MobileNetArchEncoder
        arch_encoder = MobileNetArchEncoder()
        print("✓ Architecture encoder initialized")
        
        # Test accuracy predictor
        from ofa.custom_hardware import QuantizedAccuracyPredictor
        accuracy_predictor = QuantizedAccuracyPredictor(
            arch_encoder=arch_encoder,
            hidden_size=400,
            n_layers=3,
            device='cpu'
        )
        print("✓ Accuracy predictor initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Custom hardware component test failed: {e}")
        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration loading"""
    print("\nTesting configuration loading...")
    
    try:
        from ofa.custom_hardware.utils import load_config
        
        # Test with default config if it exists
        config_path = Path("configs/npu_config.yaml")
        if config_path.exists():
            config = load_config(str(config_path))
            print(f"✓ Configuration loaded from {config_path}")
            print(f"  Hardware: {config.get('hardware', {}).get('name', 'Unknown')}")
        else:
            print(f"⚠ Default config not found at {config_path}")
            print("  This is normal if you haven't created a config file yet")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_script_availability():
    """Test that required scripts are available"""
    print("\nTesting script availability...")
    
    scripts = [
        "scripts/build_latency_lut.py",
        "scripts/search_custom_hardware.py", 
        "scripts/evaluate_custom_hardware.py",
        "scripts/train_accuracy_predictor.py",
        "examples/custom_hardware_example.py"
    ]
    
    all_available = True
    for script in scripts:
        script_path = Path(script)
        if script_path.exists():
            print(f"✓ {script}")
        else:
            print(f"✗ {script} - Not found")
            all_available = False
    
    return all_available


def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("INTEGRATION TEST COMPLETED")
    print("="*60)
    
    print("\nNext steps to use custom hardware integration:")
    print("\n1. Install additional dependencies (if not already done):")
    print("   pip install onnx onnxruntime onnxruntime-extensions onnxoptimizer onnxsim pyyaml")
    
    print("\n2. Create a hardware configuration file:")
    print("   cp configs/npu_config.yaml configs/my_hardware.yaml")
    print("   # Edit my_hardware.yaml for your specific hardware")
    
    print("\n3. Run the complete workflow:")
    print("   python examples/custom_hardware_example.py \\")
    print("       --dataset_path /path/to/imagenet \\")
    print("       --config configs/my_hardware.yaml \\")
    print("       --quick_demo")
    
    print("\n4. Or run individual components:")
    print("   - Build latency table: python scripts/build_latency_lut.py")
    print("   - Train accuracy predictor: python scripts/train_accuracy_predictor.py")
    print("   - Search optimal subnets: python scripts/search_custom_hardware.py")
    print("   - Evaluate results: python scripts/evaluate_custom_hardware.py")
    
    print("\n5. For detailed documentation:")
    print("   - See docs/custom_hardware_integration.md")
    print("   - See scripts/README.md")


def main():
    """Run all integration tests"""
    print("OFA Custom Hardware Integration Test")
    print("="*50)
    
    tests = [
        ("Import Test", test_imports),
        ("OFA Network Test", test_ofa_network),
        ("Custom Components Test", test_custom_hardware_components),
        ("Configuration Test", test_config_loading),
        ("Script Availability Test", test_script_availability)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("✓ All tests passed! Custom hardware integration is ready to use.")
    else:
        print("⚠ Some tests failed. Please check the errors above.")
    
    print_next_steps()


if __name__ == '__main__':
    main()
