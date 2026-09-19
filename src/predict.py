"""
Standalone Inference Pipeline for New Knee Radiographs.
Loads trained ResNet-101 and DenseNet-201 checkpoints, passes outputs
through the Mamdani Fuzzy Fusion Engine, and outputs diagnostic predictions.

Usage:
  # Single Image:
  python src/predict.py --image_path "path/to/knee_xray.png"

  # Entire Directory of New Images:
  python src/predict.py --image_dir "path/to/new_images/" --output_csv "predictions.csv"
"""

import os
import argparse
from typing import Dict, Union, List, Tuple
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms

from models import build_resnet101, build_densenet201
from fuzzy_engine import FuzzyFusionEngine

class OsteoporosisPredictor:
    """
    Production-ready predictor combining ResNet-101, DenseNet-201, and Fuzzy Logic Fusion.
    """
    def __init__(
        self,
        resnet_weights: str = "./checkpoints/best_resnet101.pth",
        densenet_weights: str = "./checkpoints/best_densenet201.pth",
        threshold: float = 0.30,
        device: Optional[str] = None
    ):
        self.threshold = threshold
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Initializing Osteoporosis Predictor on: {self.device}")

        # Preprocessing matching ImageNet transfer learning standards
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load ResNet-101
        self.resnet = build_resnet101(num_classes=2, pretrained=False)
        if os.path.exists(resnet_weights):
            ckpt_res = torch.load(resnet_weights, map_location=self.device)
            state_dict = ckpt_res["model_state_dict"] if "model_state_dict" in ckpt_res else ckpt_res
            self.resnet.load_state_dict(state_dict)
            print(f"  --> Loaded ResNet-101 weights: {resnet_weights}")
        else:
            print(f"  [!] Warning: ResNet weights not found at {resnet_weights}. Using uninitialized head.")
        self.resnet = self.resnet.to(self.device).eval()

        # Load DenseNet-201
        self.densenet = build_densenet201(num_classes=2, pretrained=False)
        if os.path.exists(densenet_weights):
            ckpt_den = torch.load(densenet_weights, map_location=self.device)
            state_dict = ckpt_den["model_state_dict"] if "model_state_dict" in ckpt_den else ckpt_den
            self.densenet.load_state_dict(state_dict)
            print(f"  --> Loaded DenseNet-201 weights: {densenet_weights}")
        else:
            print(f"  [!] Warning: DenseNet weights not found at {densenet_weights}. Using uninitialized head.")
        self.densenet = self.densenet.to(self.device).eval()

        # Fuzzy Logic Engine
        self.fuzzy_engine = FuzzyFusionEngine()

    @torch.no_grad()
    def predict_single(self, image_path: str) -> Dict[str, Union[str, float]]:
        """
        Runs inference on a single knee X-ray image file.
        """
        assert os.path.exists(image_path), f"File not found: {image_path}"
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Get backbone logits and softmax probabilities
        out_res = self.resnet(tensor)
        out_den = self.densenet(tensor)

        p_res = float(torch.softmax(out_res, dim=1)[0, 1].cpu().item())
        p_den = float(torch.softmax(out_den, dim=1)[0, 1].cpu().item())

        # Fuzzy inference
        p_fused = float(self.fuzzy_engine.predict_proba(np.array([p_res]), np.array([p_den]))[0])

        # Diagnostic decision
        decision = "Osteoporosis" if p_fused >= self.threshold else "Normal"
        confidence_level = "High" if p_fused >= 0.70 or p_fused <= 0.20 else "Moderate"

        return {
            "image_path": image_path,
            "decision": decision,
            "fuzzy_fused_probability": round(p_fused, 4),
            "resnet101_probability": round(p_res, 4),
            "densenet201_probability": round(p_den, 4),
            "decision_threshold": self.threshold,
            "confidence_level": confidence_level
        }

    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Runs inference on a list of image paths."""
        return [self.predict_single(p) for p in image_paths if os.path.isfile(p)]


def main():
    parser = argparse.ArgumentParser(description="Predict Osteoporosis on New Knee X-Rays")
    parser.add_argument("--image_path", type=str, help="Path to a single knee X-ray image")
    parser.add_argument("--image_dir", type=str, help="Path to a directory of knee X-rays")
    parser.add_argument("--resnet_weights", type=str, default="./checkpoints/best_resnet101.pth")
    parser.add_argument("--densenet_weights", type=str, default="./checkpoints/best_densenet201.pth")
    parser.add_argument("--threshold", type=float, default=0.30, help="Diagnostic decision threshold tau (default 0.30)")
    parser.add_argument("--output_csv", type=str, default=None, help="Optional path to save batch predictions CSV")
    args = parser.parse_args()

    predictor = OsteoporosisPredictor(
        resnet_weights=args.resnet_weights,
        densenet_weights=args.densenet_weights,
        threshold=args.threshold
    )

    if args.image_path:
        res = predictor.predict_single(args.image_path)
        print("\n" + "=" * 60)
        print("DIAGNOSTIC INFERENCE REPORT")
        print("=" * 60)
        print(f"Target Image:           {res['image_path']}")
        print(f"Final Clinical Decision: >> {res['decision'].upper()} <<")
        print(f"Fuzzy Confidence (P):   {res['fuzzy_fused_probability']*100:.2f}%")
        print(f"Confidence Level:       {res['confidence_level']}")
        print("-" * 60)
        print(f"ResNet-101 Score:       {res['resnet101_probability']*100:.2f}%")
        print(f"DenseNet-201 Score:     {res['densenet201_probability']*100:.2f}%")
        print(f"Operating Threshold:    tau = {res['decision_threshold']}")
        print("=" * 60)

    elif args.image_dir:
        import csv
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
        files = [os.path.join(args.image_dir, f) for f in sorted(os.listdir(args.image_dir))
                 if os.path.splitext(f.lower())[1] in valid_exts]
        print(f"\nProcessing {len(files)} images from: {args.image_dir}...")
        results = predictor.predict_batch(files)

        for r in results[:10]:
            print(f"[{r['decision']:12s}] (P_fused={r['fuzzy_fused_probability']:.3f}) -> {os.path.basename(r['image_path'])}")
        if len(results) > 10:
            print(f"... and {len(results)-10} more images.")

        if args.output_csv:
            with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                writer.writeheader()
                writer.writerows(results)
            print(f"\nSaved batch predictions to: {args.output_csv}")
    else:
        print("Please provide --image_path or --image_dir. Run with -h for help.")

if __name__ == "__main__":
    main()
