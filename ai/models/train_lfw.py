import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import logging
import argparse
from datetime import datetime
import json
from typing import Dict, Tuple, List
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from sklearn.metrics import accuracy_score

from ai.models.facenet_model import InceptionResnetV1, ModelConfig
from ai.models.lfw_dataset import LFWIdentityDataset, LFWPairDataset
from ai.models.face_detection import SsdResNetDetector

def setup_logging(log_dir: str = None):
    """Setup logging to both console and file"""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir is provided)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        print(f"Logs will be saved to: {log_file}")
    
    return logging.getLogger(__name__)

# Initialize logger (will be properly set up in main())
logger = logging.getLogger(__name__)

class TripletLoss(nn.Module):
    """Triplet loss for face recognition following FaceNet"""
    
    def __init__(self, alpha: float = 0.2):
        super(TripletLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        
        loss = torch.clamp(pos_dist - neg_dist + self.alpha, min=0.0)
        return torch.mean(loss)

def sample_people(dataset: LFWIdentityDataset, people_per_batch: int, images_per_person: int):
    """Sample people and images following FaceNet strategy"""
    
    # Group samples by identity
    identity_to_samples = defaultdict(list)
    for idx, (img_path, label, person_name) in enumerate(dataset.samples):
        identity_to_samples[person_name].append(idx)
    
    # Filter identities with enough images
    valid_identities = [identity for identity, samples in identity_to_samples.items() 
                       if len(samples) >= 2]
    
    # Sample people
    num_people = min(people_per_batch, len(valid_identities))
    sampled_identities = np.random.choice(valid_identities, size=num_people, replace=False)
    
    sampled_indices = []
    num_per_class = []
    
    for identity in sampled_identities:
        available_samples = identity_to_samples[identity]
        num_samples = min(images_per_person, len(available_samples))
        
        selected_samples = np.random.choice(available_samples, size=num_samples, replace=False)
        sampled_indices.extend(selected_samples)
        num_per_class.append(num_samples)
    
    return sampled_indices, num_per_class

def select_triplets(embeddings: torch.Tensor, num_per_class: List[int], alpha: float = 0.2):
    """Select triplets based on embeddings following FaceNet semi-hard mining"""
    
    triplets = []
    emb_start_idx = 0
    
    for i, nrof_images in enumerate(num_per_class):
        for j in range(1, nrof_images):
            a_idx = emb_start_idx + j - 1
            
            # Calculate distances from anchor to all other embeddings
            anchor_emb = embeddings[a_idx].unsqueeze(0)
            all_dists_sq = torch.sum((embeddings - anchor_emb) ** 2, dim=1)
            
            for pair in range(j, nrof_images):
                p_idx = emb_start_idx + pair
                pos_dist_sq = all_dists_sq[p_idx]
                
                # Exclude same identity from negative selection
                neg_dists_sq = all_dists_sq.clone()
                neg_dists_sq[emb_start_idx:emb_start_idx + nrof_images] = float('inf')
                
                # Semi-hard negative mining: neg_dist - pos_dist < alpha
                valid_negatives = torch.where(neg_dists_sq - pos_dist_sq < alpha)[0]
                
                if len(valid_negatives) > 0:
                    # Randomly select one negative
                    n_idx = valid_negatives[torch.randint(len(valid_negatives), (1,))].item()
                    triplets.append((a_idx, p_idx, n_idx))
        
        emb_start_idx += nrof_images
    
    return triplets

class LFWTripletTrainer:
    """Trainer for InceptionResNetV1 with triplet loss on LFW dataset"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        
        self._setup_model()
        self._setup_training()
    
    def _setup_model(self):
        """Initialize the InceptionResNetV1 model"""
        self.model = InceptionResnetV1(
            pretrained=self.config.get('pretrained', None),
            classify=False,
            dropout_prob=self.config.get('dropout_prob', 0.6),
            device=self.device
        )
        
        self.model = self.model.to(self.device)
        print(f"Model loaded with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def _setup_training(self):
        """Setup optimizer, scheduler, and loss function"""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.get('lr_step_size', 15),
            gamma=self.config.get('lr_gamma', 0.1)
        )
        
        self.criterion = TripletLoss(alpha=self.config.get('alpha', 0.2))
    
    def evaluate_pairs(self, pairs_file: str, dataset_name: str) -> float:
        """Evaluate model on pairs dataset"""
        # Extract just the filename from the full path
        pairs_filename = os.path.basename(pairs_file)
        
        dataset = LFWPairDataset(
            data_dir=self.config['data_dir'],
            pairs_file=pairs_filename,
            target_size=(160, 160),
            face_detector=SsdResNetDetector(confidence_threshold=0.5),
            normalization="Facenet2018"
        )
        
        # Check if dataset is empty
        if len(dataset) == 0:
            print(f"Warning: {dataset_name} dataset is empty, skipping evaluation")
            return 0.0
        
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        embeddings1 = []
        embeddings2 = []
        labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating", leave=False):
                img1 = batch['img1'].to(self.device)
                img2 = batch['img2'].to(self.device)
                batch_labels = batch['label'].cpu().numpy()
                
                emb1 = self.model(img1).cpu().numpy()
                emb2 = self.model(img2).cpu().numpy()
                
                embeddings1.append(emb1)
                embeddings2.append(emb2)
                labels.append(batch_labels)
        
        # Check if we have any valid samples
        if not embeddings1:
            print(f"Warning: No valid samples found in {dataset_name} dataset")
            return 0.0
        
        embeddings1 = np.vstack(embeddings1)
        embeddings2 = np.vstack(embeddings2)
        labels = np.concatenate(labels)
        
        # Compute distances
        distances = np.linalg.norm(embeddings1 - embeddings2, axis=1)
        
        # Find best threshold
        thresholds = np.linspace(distances.min(), distances.max(), 1000)
        best_accuracy = 0.0
        
        for threshold in thresholds:
            predictions = (distances < threshold).astype(int)
            accuracy = accuracy_score(labels, predictions)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
        
        return best_accuracy
    
    def train_epoch(self, dataset: LFWIdentityDataset) -> float:
        """Train for one epoch following FaceNet strategy"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Simple progress bar for batches in epoch
        pbar = tqdm(range(self.config['epoch_size']), desc="Processing batches", leave=False)
        
        for batch_idx in pbar:
            # Sample people and images
            sampled_indices, num_per_class = sample_people(
                dataset, 
                self.config['people_per_batch'], 
                self.config['images_per_person']
            )
            
            if len(sampled_indices) == 0:
                continue
            
            # Load sampled images
            images = []
            for idx in sampled_indices:
                sample = dataset[idx]
                images.append(sample['image'])
            
            images = torch.stack(images).to(self.device)
            
            # Forward pass to get embeddings
            with torch.no_grad():
                embeddings = self.model(images)
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Select triplets based on embeddings
            triplets = select_triplets(embeddings, num_per_class, self.config.get('alpha', 0.2))
            
            if len(triplets) == 0:
                continue
            
            # Train on selected triplets
            triplet_indices = torch.tensor(triplets).to(self.device)
            triplet_images = images[triplet_indices.flatten()].view(-1, 3, *images.shape[1:])
            
            # Process triplets in batches
            batch_size = self.config['batch_size']
            num_triplet_batches = (len(triplets) + batch_size - 1) // batch_size
            
            batch_losses = []
            
            for i in range(num_triplet_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(triplets))
                
                batch_triplet_images = triplet_images[start_idx:end_idx]
                batch_triplet_images = batch_triplet_images.view(-1, *images.shape[1:])
                
                self.optimizer.zero_grad()
                
                # Get embeddings for triplet batch
                batch_embeddings = self.model(batch_triplet_images)
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                
                # Reshape to get anchor, positive, negative
                batch_embeddings = batch_embeddings.view(-1, 3, self.config['embedding_dim'])
                anchors = batch_embeddings[:, 0, :]
                positives = batch_embeddings[:, 1, :]
                negatives = batch_embeddings[:, 2, :]
                
                # Calculate loss
                loss = self.criterion(anchors, positives, negatives)
                loss.backward()
                self.optimizer.step()
                
                batch_losses.append(loss.item())
            
            if batch_losses:
                current_loss = np.mean(batch_losses)
                epoch_loss += current_loss
                num_batches += 1
        
        pbar.close()
        return epoch_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self, dataset: LFWIdentityDataset):
        """Main training loop with evaluation"""
        print("Starting triplet loss training...")
        
        # Prepare pairs files
        train_pairs = os.path.join(self.config['data_dir'], 'pairsDevTrain.txt')
        val_pairs = os.path.join(self.config['data_dir'], 'pairsDevTest.txt')
        test_pairs = os.path.join(self.config['data_dir'], 'pairs.txt')
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Training
            train_loss = self.train_epoch(dataset)
            self.train_losses.append(train_loss)
            self.scheduler.step()
            
            # Evaluation
            if os.path.exists(train_pairs):
                train_acc = self.evaluate_pairs(train_pairs, "Train")
                self.train_accuracies.append(train_acc)
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            else:
                print(f"Train Loss: {train_loss:.4f}")
            
            # Validation evaluation
            if os.path.exists(val_pairs):
                val_acc = self.evaluate_pairs(val_pairs, "Validation")
                self.val_accuracies.append(val_acc)
                print(f"Validation Acc: {val_acc:.4f}")
                
                # Save best model based on validation accuracy
                if val_acc > self.best_val_accuracy:
                    self.best_val_accuracy = val_acc
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"New best validation accuracy: {val_acc:.4f}")
            
            # Test evaluation (configurable frequency)
            if (epoch + 1) % self.config.get('test_every', 10) == 0 and os.path.exists(test_pairs):
                test_acc = self.evaluate_pairs(test_pairs, "Test")
                print(f"*********** Test Acc: {test_acc:.4f}")
            
            # Regular checkpoint saving (configurable frequency)
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print(f"\nTraining completed. Best validation accuracy: {self.best_val_accuracy:.4f}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        checkpoint_dir = self.config['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if is_best:
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"Best model saved to {checkpoint_path}")
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curve
        axes[0].plot(self.train_losses)
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)
        
        # Accuracy curves
        if self.train_accuracies:
            axes[1].plot(self.train_accuracies, label='Train Accuracy')
        if self.val_accuracies:
            axes[1].plot(self.val_accuracies, label='Validation Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['checkpoint_dir'], 'training_curves.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train InceptionResNetV1 with Triplet Loss on LFW dataset')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to LFW data directory')
    parser.add_argument('--batch_size', type=int, default=90, help='Batch size for triplet training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--alpha', type=float, default=0.2, help='Triplet loss margin')
    parser.add_argument('--people_per_batch', type=int, default=45, help='Number of people per batch')
    parser.add_argument('--images_per_person', type=int, default=40, help='Number of images per person')
    parser.add_argument('--epoch_size', type=int, default=1000, help='Number of batches per epoch')
    parser.add_argument('--pretrained', type=str, default=None, choices=[None, 'vggface2', 'casia-webface'])
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_triplet', help='Checkpoint directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--test_every', type=int, default=10, help='Number of epochs between test evaluations')
    parser.add_argument('--save_every', type=int, default=10, help='Number of epochs between regular checkpoint saves')
    
    args = parser.parse_args()
    
    config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'embedding_dim': args.embedding_dim,
        'alpha': args.alpha,
        'people_per_batch': args.people_per_batch,
        'images_per_person': args.images_per_person,
        'epoch_size': args.epoch_size,
        'pretrained': args.pretrained,
        'checkpoint_dir': args.checkpoint_dir,
        'num_workers': args.num_workers,
        'weight_decay': 1e-4,
        'lr_step_size': 15,
        'lr_gamma': 0.1,
        'dropout_prob': 0.6,
        'test_every': args.test_every,
        'save_every': args.save_every
    }
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    with open(os.path.join(config['checkpoint_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create dataset for triplet training
    dataset = LFWIdentityDataset(
        data_dir=config['data_dir'],
        min_images_per_person=2,
        target_size=(160, 160),
        face_detector=SsdResNetDetector(confidence_threshold=0.5),
        normalization="Facenet2018"
    )
    
    trainer = LFWTripletTrainer(config)
    trainer.train(dataset)
    trainer.plot_training_curves()
    
    print("Triplet loss training completed successfully!")

if __name__ == '__main__':
    main() 