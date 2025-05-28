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

from ai.models.facenet_model import InceptionResnetV1, ModelConfig
from ai.models.lfw_dataset import LFWIdentityDataset
from ai.models.face_detection import SsdResNetDetector

logging.basicConfig(level=logging.INFO)
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
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        self.train_losses = []
        self.best_loss = float('inf')
        
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
        logger.info(f"Model loaded with {sum(p.numel() for p in self.model.parameters())} parameters")
    
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
    
    def train_epoch(self, dataset: LFWIdentityDataset) -> float:
        """Train for one epoch following FaceNet strategy"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx in range(self.config['epoch_size']):
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
            print(f'Running forward pass on {len(images)} sampled images...', end='')
            start_time = time.time()
            
            with torch.no_grad():
                embeddings = self.model(images)
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            print(f' {time.time() - start_time:.3f}s')
            
            # Select triplets based on embeddings
            print('Selecting suitable triplets for training')
            start_time = time.time()
            
            triplets = select_triplets(embeddings, num_per_class, self.config.get('alpha', 0.2))
            
            selection_time = time.time() - start_time
            print(f'Selected {len(triplets)} triplets: time={selection_time:.3f}s')
            
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
                
                print(f'Epoch: [{batch_idx+1}/{self.config["epoch_size"]}][{i+1}/{num_triplet_batches}]\t'
                      f'Loss {loss.item():.3f}')
            
            if batch_losses:
                epoch_loss += np.mean(batch_losses)
                num_batches += 1
        
        return epoch_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self, dataset: LFWIdentityDataset):
        """Main training loop"""
        logger.info("Starting triplet loss training...")
        
        for epoch in range(self.config['num_epochs']):
            logger.info(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
            
            train_loss = self.train_epoch(dataset)
            
            self.train_losses.append(train_loss)
            self.scheduler.step()
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self.save_checkpoint(epoch, is_best=True)
            
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        logger.info(f"Training completed. Best loss: {self.best_loss:.4f}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
            'train_losses': self.train_losses
        }
        
        checkpoint_dir = self.config['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if is_best:
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Best model saved to {checkpoint_path}")
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
    
    def plot_training_curves(self):
        """Plot training curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses)
        plt.title('Triplet Loss Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
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
        'save_every': 10
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
    
    logger.info("Triplet loss training completed successfully!")

if __name__ == '__main__':
    main() 