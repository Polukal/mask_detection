import torch
import argparse
import os
from config import Config
from models import get_model
from train import train
from evaluate import evaluate_model
from webcam_demo import WebcamDemo
from dataset import get_dataloader


def main():
    parser = argparse.ArgumentParser(description='Face Mask Detection')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'demo'],
                        help='Mode to run the program in')
    parser.add_argument('--model_type', type=str, default='yolo', choices=['ssd', 'yolo'],
                        help='Type of model to use')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to saved model for evaluation or demo')

    args = parser.parse_args()

    # Create necessary directories
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(Config.LOGS_PATH, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = get_model(args.model_type)

    if args.mode == 'train':
        print("Starting training...")
        trained_model = train(args.model_type)
        print("Training completed!")

    elif args.mode == 'evaluate':
        if args.model_path is None:
            raise ValueError("Model path must be provided for evaluation")

        print("Loading model for evaluation...")
        model.load_state_dict(torch.load(args.model_path))
        model = model.to(device)

        test_loader = get_dataloader(Config.DATASET_PATH, Config.BATCH_SIZE, train=False)
        metrics = evaluate_model(model, test_loader, device)

        print("\nEvaluation Results:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"mAP: {metrics['mAP']:.4f}")
        print(f"Mean IoU: {metrics['IoU']:.4f}")

    elif args.mode == 'demo':
        if args.model_path is None:
            raise ValueError("Model path must be provided for demo")

        print("Loading model for demo...")
        model.load_state_dict(torch.load(args.model_path))
        model = model.to(device)

        print("Starting webcam demo...")
        demo = WebcamDemo(model, device)
        demo.run()


if __name__ == '__main__':
    main()