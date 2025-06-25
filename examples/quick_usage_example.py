#!/usr/bin/env python3
"""
Quick usage example for OFA custom hardware integration

This demonstrates how to use the custom hardware integration
programmatically without the command-line scripts.
"""

import torch
from ofa.model_zoo import ofa_net
from ofa.custom_hardware import (
    ONNXConverter,
    QuantizedLatencyPredictor,
    QuantizedAccuracyPredictor,
    CustomHardwareEvolutionFinder
)


def main():
    print("OFA Custom Hardware Integration - Quick Example")
    print("="*50)
    
    # 1. Load OFA network
    print("Loading OFA network...")
    ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.0', pretrained=True)
    
    # 2. Define a sample subnet configuration
    subnet_config = {
        'ks': [3, 3, 3, 5, 5, 5, 3, 3, 5, 5, 5, 3, 3, 5, 7, 7, 7, 7, 7, 7],
        'e': [3, 3, 3, 4, 4, 4, 3, 3, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        'd': [2, 3, 4, 4, 4],
        'w': 1.0
    }
    
    print(f"Subnet config: {subnet_config}")
    
    # 3. Set and get subnet
    if hasattr(ofa_network, 'set_active_subnet'):
        ofa_network.set_active_subnet(**subnet_config)
    
    subnet = ofa_network.get_active_subnet() if hasattr(ofa_network, 'get_active_subnet') else ofa_network
    
    # 4. Convert to ONNX with quantization
    print("\nConverting to quantized ONNX...")
    converter = ONNXConverter(quantization_enabled=True)
    
    input_shape = (1, 3, 224, 224)
    try:
        onnx_model_path = converter.convert_subnet(
            subnet, 
            subnet_config, 
            input_shape,
            output_path="example_model.onnx"
        )
        print(f"ONNX model saved to: {onnx_model_path}")
    except Exception as e:
        print(f"ONNX conversion failed: {e}")
        return
    
    # 5. Predict latency (requires actual measurement on target hardware)
    print("\nSetting up latency predictor...")
    latency_predictor = QuantizedLatencyPredictor(
        provider="CPUExecutionProvider",  # Change to your NPU provider
        provider_options={}
    )
    
    try:
        # This would normally load a pre-built lookup table
        # For demo, we'll just show the interface
        print("Latency predictor initialized (would need lookup table for real prediction)")
    except Exception as e:
        print(f"Latency predictor setup failed: {e}")
    
    # 6. Predict accuracy (requires trained predictor)
    print("\nSetting up accuracy predictor...")
    try:
        # This would normally load a trained model
        # For demo, we'll just show the interface
        print("Accuracy predictor initialized (would need trained model for real prediction)")
    except Exception as e:
        print(f"Accuracy predictor setup failed: {e}")
    
    # 7. Example of evolutionary search setup
    print("\nEvolutionary search example...")
    print("To run actual search, you would need:")
    print("1. Trained accuracy predictor")
    print("2. Built latency lookup table")
    print("3. Target hardware configuration")
    
    example_search_code = '''
    # Example search (once predictors are ready):
    finder = CustomHardwareEvolutionFinder(
        efficiency_predictor=latency_predictor,
        accuracy_predictor=accuracy_predictor
    )
    
    best_arch, best_efficiency = finder.run_evolution_search(
        constraint=10.0,  # 10ms latency constraint
        verbose=True
    )
    '''
    print(example_search_code)
    
    print("\n" + "="*50)
    print("Example completed!")
    print("\nNext steps:")
    print("1. Build latency lookup table: python scripts/build_latency_lut.py")
    print("2. Train accuracy predictor: python scripts/train_accuracy_predictor.py")
    print("3. Run search: python scripts/search_custom_hardware.py")
    print("4. Or use the complete workflow: python examples/custom_hardware_example.py")


if __name__ == '__main__':
    main()
