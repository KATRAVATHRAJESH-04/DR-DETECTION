import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Args:
            model: The HybridDRClassifier model
            target_layer: The specific layer to compute Grad-CAM for (e.g., model.cnn[4])
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register a forward hook to capture the activations and their gradients
        self.target_layer.register_forward_hook(self.save_activation)
        
    def save_activation(self, module, input, output):
        self.activations = output

    def __call__(self, x_tensor, class_idx=None):
        """
        Compute Grad-CAM heatmap for a given input tensor.
        """
        self.model.eval()
        
        x_tensor.requires_grad_(True)
        logits, _ = self.model(x_tensor)
        
        # Now that we've done the forward pass, retain_grad is safe
        self.activations.retain_grad()
        
        if class_idx is None:
            class_idx = logits.argmax(dim=-1).item()
            
        score = logits[0, class_idx]
        
        self.model.zero_grad()
        score.backward()
        
        # Retrieve the gradients from our hooked activations
        self.gradients = self.activations.grad
        
        # Global Average Pooling of gradients to get importance weights
        weights = torch.mean(self.gradients, dim=(2, 3))[0] # Shape: [Channels]
        activations = self.activations[0] # Shape: [Channels, H, W]
        
        # Linear combination of activations and weights
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU to keep only features that have a positive influence
        cam = F.relu(cam)
        cam = cam.cpu().detach().numpy()
        
        # Resize CAM to the size of the original input image
        cam = cv2.resize(cam, (x_tensor.shape[3], x_tensor.shape[2]))
        
        # Min-max scaling to [0, 1]
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        return cam

def overlay_heatmap(img_pil, heatmap, max_alpha=0.6, colormap=cv2.COLORMAP_JET):
    """
    Overlays the generated heatmap directly onto the PIL image intelligently.
    Uses dynamic transparency so healthy areas remain completely visible.
    """
    img_np = np.array(img_pil)
    
    # Ensure background image is RGB
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    # Create colored heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    heatmap_colored = cv2.resize(heatmap_colored, (img_np.shape[1], img_np.shape[0]))
    
    # Create an intelligent alpha mask based on the heatmap intensity!
    # "Cold" regions will be completely transparent (showing the raw eye).
    # "Hot" regions will become opaque, up to max_alpha.
    heatmap_mask = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    # Apply a tiny threshold to completely hide noise
    heatmap_mask[heatmap_mask < 0.1] = 0.0
    
    mask_3d = (heatmap_mask * max_alpha)[..., np.newaxis]
    
    # Blend the images using the dynamic mask
    overlay = (img_np * (1 - mask_3d) + heatmap_colored * mask_3d).astype(np.uint8)
    
    return Image.fromarray(overlay)
