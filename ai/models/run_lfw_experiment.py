#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_training(data_dir: str, checkpoint_dir: str, config: dict):
    """Run training script"""
    cmd = [
        sys.executable, "ai/models/train_lfw.py",
        "--data_dir", data_dir,
        "--batch_size", str(config.get('batch_size', 32)),
        "--num_epochs", str(config.get('num_epochs', 50)),
        "--learning_rate", str(config.get('learning_rate', 0.001)),
        "--embedding_dim", str(config.get('embedding_dim', 512)),
        "--loss_type", config.get('loss_type', 'contrastive'),
        "--margin", str(config.get('margin', 1.0)),
        "--checkpoint_dir", checkpoint_dir,
        "--num_workers", str(config.get('num_workers', 4))
    ]
    
    if config.get('pretrained'):
        cmd.extend(["--pretrained", config['pretrained']])
    
    logger.info(f"Running training command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Training completed successfully")
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        logger.error(e.stderr)
        return False

def run_testing(data_dir: str, model_path: str, config_path: str, output_dir: str, protocol: str = "10fold"):
    """Run testing script"""
    cmd = [
        sys.executable, "ai/models/test_lfw.py",
        "--model_path", model_path,
        "--config_path", config_path,
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--protocol", protocol
    ]
    
    logger.info(f"Running testing command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Testing completed successfully")
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Testing failed: {e}")
        logger.error(e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description='Run complete LFW experiment')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to LFW data directory')
    parser.add_argument('--experiment_name', type=str, default='lfw_experiment', help='Experiment name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--loss_type', type=str, default='contrastive', choices=['contrastive', 'triplet'])
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for loss function')
    parser.add_argument('--pretrained', type=str, default=None, choices=[None, 'vggface2', 'casia-webface'])
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--skip_training', action='store_true', help='Skip training and only run testing')
    parser.add_argument('--protocol', type=str, default='10fold', choices=['standard', '10fold'])
    
    args = parser.parse_args()
    
    checkpoint_dir = f"checkpoints/{args.experiment_name}"
    output_dir = f"results/{args.experiment_name}"
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    config = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'embedding_dim': args.embedding_dim,
        'loss_type': args.loss_type,
        'margin': args.margin,
        'pretrained': args.pretrained,
        'num_workers': args.num_workers
    }
    
    logger.info(f"Starting LFW experiment: {args.experiment_name}")
    logger.info(f"Configuration: {config}")
    
    if not args.skip_training:
        logger.info("=== Starting Training ===")
        training_success = run_training(args.data_dir, checkpoint_dir, config)
        
        if not training_success:
            logger.error("Training failed. Exiting.")
            sys.exit(1)
    else:
        logger.info("Skipping training as requested")
    
    logger.info("=== Starting Testing ===")
    
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    config_path = os.path.join(checkpoint_dir, "config.json")
    
    if not os.path.exists(best_model_path):
        logger.error(f"Best model not found at {best_model_path}")
        sys.exit(1)
    
    testing_success = run_testing(
        args.data_dir, 
        best_model_path, 
        config_path, 
        output_dir, 
        args.protocol
    )
    
    if testing_success:
        logger.info(f"Experiment completed successfully!")
        logger.info(f"Results saved in: {output_dir}")
        logger.info(f"Model checkpoints saved in: {checkpoint_dir}")
    else:
        logger.error("Testing failed")
        sys.exit(1)

if __name__ == '__main__':
    main() 