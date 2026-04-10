import torch
from inference import DRInference
import os

print('Loading inference...')
try:
    infer = DRInference('dr_hybrid_model.pth', device='cpu')
    print('DRInference initialized.')
    result = infer.predict('test_eye.png')
    
    print("\n--- Model Prediction:", result['prediction'], "---")
    print("Predicted Class Index:", result['class_idx'])
    for cls, prob in result['probabilities'].items():
        print(f"  {cls}: {prob:.4f}")
        
except Exception as e:
    print(f"FAILED TO RUN: {e}")
