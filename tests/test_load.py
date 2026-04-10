import torch
import sys
import os
sys.path.append(os.path.abspath('.'))
from models.hybrid_model import HybridDRClassifier

model = HybridDRClassifier()
try:
    model.load_state_dict(torch.load('dr_hybrid_model.pth', map_location='cpu'))
    print('LOAD SUCCESS')
except Exception as e:
    print(f'LOAD FAILED: {e}')
