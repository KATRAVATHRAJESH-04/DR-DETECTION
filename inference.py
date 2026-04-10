import torch
from models.hybrid_model import HybridDRClassifier
from utils.preprocessing import DRPreprocessor
from utils.gradcam import GradCAM, overlay_heatmap
from PIL import Image
import os

class DRInference:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.is_multiclass = "odir" in model_path.lower()
        num_classes = 8 if self.is_multiclass else 5
        self.model = HybridDRClassifier(num_classes=num_classes)
        
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Warning: Model weights at {model_path} are incompatible. Using random weights. Error: {e}")
        else:
            print(f"Warning: Model not found at {model_path}. Using initial random weights for testing.")
            
        self.model.to(self.device)
        self.model.eval()
        
        self.preprocessor = DRPreprocessor(target_size=(224, 224), is_training=False)
        self.grad_cam = GradCAM(self.model, target_layer=self.model.cnn[-2])
        
        if self.is_multiclass:
            self.class_names = {
                0: "Normal", 1: "Diabetic Retinopathy", 2: "Glaucoma", 
                3: "Cataract", 4: "AMD", 5: "Hypertension", 6: "Myopia", 7: "Other"
            }
        else:
            self.class_names = {
                0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"
            }

    def predict(self, image_path_or_pil):
        """
        Runs full inference on an image and returns probability distribution.
        """
        # 1. Preprocess
        tensor, enhanced_pil, quality = self.preprocessor.preprocess(image_path_or_pil)
        tensor = tensor.to(self.device)
        
        # 2. Inference
        with torch.no_grad():
            logits, _ = self.model(tensor, tensor)
            if self.is_multiclass:
                probs = torch.sigmoid(logits) # Multi-label requires sigmoid
            else:
                probs = torch.nn.functional.softmax(logits, dim=1)
            
        prob_values = probs[0].cpu().numpy()
        pred_class = int(torch.argmax(probs, dim=1).item())
        
        num_c = 8 if self.is_multiclass else 5
        
        return {
            "prediction": self.class_names[pred_class],
            "class_idx": pred_class,
            "probabilities": {self.class_names[i]: float(prob_values[i]) for i in range(num_c)},
            "enhanced_image": enhanced_pil,
            "tensor": tensor,
            "quality": quality,
        }

    def generate_heatmap(self, tensor, class_idx=None):
        """
        Computes Grad-CAM for the given tensor and class.
        Needs requires_grad=True on the tensor.
        """
        tensor.requires_grad_(True)
        heatmap = self.grad_cam(tensor, class_idx=class_idx)
        return heatmap

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to retinal image for inference")
    parser.add_argument("--model", type=str, default="dr_hybrid_model.pth", help="Path to the trained PyTorch weights")
    args = parser.parse_args()

    print(f"Loading model and running inference on {args.image}...")
    infer = DRInference(model_path=args.model)
    result = infer.predict(args.image)
    
    print(f"\n--- Model Prediction: {result['prediction']} ---")
    print(f"Predicted Class Index: {result['class_idx']}")
    for cls, prob in result['probabilities'].items():
        print(f"  {cls}: {prob:.4f}")
        
    print("\nGenerating Grad-CAM heatmap...")
    heatmap = infer.generate_heatmap(result['tensor'], result['class_idx'])
    heatmap_img = overlay_heatmap(result['enhanced_image'], heatmap)
    
    heatmap_path = "gradcam_output.png"
    heatmap_img.save(heatmap_path)
    print(f"Saved Grad-CAM explainability heatmap to {heatmap_path}!")
