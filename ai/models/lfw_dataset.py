import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
import logging
from PIL import Image
import random

from ai.models.face_detection import SsdResNetDetector
from ai.utils.preprocessing import normalize_input, resize_image

logger = logging.getLogger(__name__)

class LFWPairDataset(Dataset):
    """LFW dataset for face verification task"""
    
    def __init__(self, 
                 data_dir: str,
                 pairs_file: Optional[str] = None,
                 transform=None,
                 target_size: Tuple[int, int] = (160, 160),
                 face_detector: Optional[SsdResNetDetector] = None,
                 normalization: str = "Facenet2018"):
        """
        Initialize LFW dataset
        
        Args:
            data_dir: Path to lfw-funneled directory
            pairs_file: Path to pairs.txt file (optional, can be set later)
            transform: Optional transforms
            target_size: Target image size for model input
            face_detector: Face detector instance
            normalization: Normalization method
        """
        self.data_dir = data_dir
        self.pairs_file = pairs_file
        self.transform = transform
        self.target_size = target_size
        self.normalization = normalization
        
        if face_detector is None:
            self.face_detector = SsdResNetDetector(confidence_threshold=0.5)
        else:
            self.face_detector = face_detector
            
        if self.pairs_file is not None:
            self.pairs = self._load_pairs()
            logger.info(f"Loaded {len(self.pairs)} pairs from {pairs_file}")
        else:
            self.pairs = []
            logger.info("Dataset initialized without pairs file - pairs can be set manually")
    
    def _load_pairs(self) -> List[Tuple]:
        """Load pairs from pairs file"""
        pairs = []
        pairs_path = os.path.join(self.data_dir, self.pairs_file)
        
        if not os.path.exists(pairs_path):
            print(f"Warning: Pairs file not found: {pairs_path}")
            return pairs
        
        try:
            with open(pairs_path, 'r') as f:
                lines = f.readlines()
            
            # Skip header line if present
            start_idx = 1 if lines and not lines[0].strip().split()[0].isdigit() else 0
            
            for line_num, line in enumerate(lines[start_idx:], start_idx + 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                
                if len(parts) == 3:
                    # Same person pair: name img1_idx img2_idx
                    name = parts[0]
                    img1_idx = int(parts[1])
                    img2_idx = int(parts[2])
                    pairs.append((name, img1_idx, name, img2_idx, 1))
                    
                elif len(parts) == 4:
                    # Different person pair: name1 img1_idx name2 img2_idx
                    name1 = parts[0]
                    img1_idx = int(parts[1])
                    name2 = parts[2]
                    img2_idx = int(parts[3])
                    pairs.append((name1, img1_idx, name2, img2_idx, 0))
                else:
                    print(f"Warning: Invalid line format at line {line_num}: {line}")
                    continue
        
        except Exception as e:
            print(f"Error loading pairs file {pairs_path}: {e}")
            return []
        
        print(f"Loaded {len(pairs)} pairs from {pairs_path}")
        return pairs
    
    def _load_and_preprocess_image(self, person_name: str, img_idx: int) -> Optional[np.ndarray]:
        """Load and preprocess image for a person"""
        img_path = os.path.join(self.data_dir, "lfw_funneled", person_name, f"{person_name}_{img_idx:04d}.jpg")
        
        try:
            if not os.path.exists(img_path):
                return None
                
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            faces = self.face_detector.detect_faces(img)
            
            if not faces:
                img_resized = resize_image(img, self.target_size)
            else:
                largest_face = max(faces, key=lambda f: f.w * f.h)
                face_img = img[largest_face.y:largest_face.y + largest_face.h,
                              largest_face.x:largest_face.x + largest_face.w]
                img_resized = resize_image(face_img, self.target_size)
            
            img_normalized = normalize_input(img_resized, self.normalization)
            
            return img_normalized.squeeze(0)
            
        except Exception as e:
            return None
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a pair of images and label"""
        max_retries = 10
        attempts = 0
        
        while attempts < max_retries:
            current_idx = (idx + attempts) % len(self.pairs)
            name1, img1_idx, name2, img2_idx, label = self.pairs[current_idx]
            
            img1 = self._load_and_preprocess_image(name1, img1_idx)
            img2 = self._load_and_preprocess_image(name2, img2_idx)
            
            if img1 is not None and img2 is not None:
                if self.transform:
                    img1 = self.transform(img1)
                    img2 = self.transform(img2)
                else:
                    img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
                    img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
                
                return {
                    'img1': img1,
                    'img2': img2,
                    'label': torch.tensor(label, dtype=torch.float32),
                    'names': (name1, name2),
                    'indices': (img1_idx, img2_idx)
                }
            
            attempts += 1
        
        # If all retries failed, return a dummy sample with zeros
        print(f"Warning: Failed to load valid pair after {max_retries} attempts, returning dummy sample")
        dummy_img = torch.zeros(3, self.target_size[0], self.target_size[1])
        return {
            'img1': dummy_img,
            'img2': dummy_img,
            'label': torch.tensor(0.0, dtype=torch.float32),
            'names': ("dummy", "dummy"),
            'indices': (0, 0)
        }

class LFWIdentityDataset(Dataset):
    """LFW dataset for identity classification training"""
    
    def __init__(self,
                 data_dir: str,
                 min_images_per_person: int = 2,
                 transform=None,
                 target_size: Tuple[int, int] = (160, 160),
                 face_detector: Optional[SsdResNetDetector] = None,
                 normalization: str = "Facenet2018"):
        """
        Initialize LFW identity dataset
        
        Args:
            data_dir: Path to lfw-funneled directory
            min_images_per_person: Minimum images per person to include
            transform: Optional transforms
            target_size: Target image size
            face_detector: Face detector instance
            normalization: Normalization method
        """
        self.data_dir = data_dir
        self.min_images_per_person = min_images_per_person
        self.transform = transform
        self.target_size = target_size
        self.normalization = normalization
        
        if face_detector is None:
            self.face_detector = SsdResNetDetector(confidence_threshold=0.5)
        else:
            self.face_detector = face_detector
        
        self.identity_to_idx, self.samples = self._build_dataset()
        self.num_classes = len(self.identity_to_idx)
        
        logger.info(f"Loaded {len(self.samples)} samples from {self.num_classes} identities")
    
    def _build_dataset(self) -> Tuple[Dict[str, int], List[Tuple]]:
        """Build dataset from directory structure"""
        lfw_dir = os.path.join(self.data_dir, "lfw_funneled")
        identity_to_idx = {}
        samples = []
        
        print(f"Path exists: {os.path.exists(lfw_dir)}")
        print(f"lfw_dir: {lfw_dir}")
        
        if not os.path.exists(lfw_dir):
            logger.error(f"LFW directory not found: {lfw_dir}")
            return identity_to_idx, samples
            
        for person_name in os.listdir(lfw_dir):
            person_dir = os.path.join(lfw_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
            
            images = [f for f in os.listdir(person_dir) if f.endswith('.jpg')]
            
            if len(images) >= self.min_images_per_person:
                if person_name not in identity_to_idx:
                    identity_to_idx[person_name] = len(identity_to_idx)
                
                for img_file in images:
                    img_path = os.path.join(person_dir, img_file)
                    samples.append((img_path, identity_to_idx[person_name], person_name))
        
        return identity_to_idx, samples
    
    def _load_and_preprocess_image(self, img_path: str) -> Optional[np.ndarray]:
        """Load and preprocess a single image"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            faces = self.face_detector.detect_faces(img)
            
            if not faces:
                img_resized = resize_image(img, self.target_size)
            else:
                largest_face = max(faces, key=lambda f: f.w * f.h)
                face_img = img[largest_face.y:largest_face.y + largest_face.h,
                              largest_face.x:largest_face.x + largest_face.w]
                img_resized = resize_image(face_img, self.target_size)
            
            img_normalized = normalize_input(img_resized, self.normalization)
            
            return img_normalized.squeeze(0)
            
        except Exception as e:
            return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get an image and its identity label"""
        max_retries = 10
        attempts = 0
        
        while attempts < max_retries:
            current_idx = (idx + attempts) % len(self.samples)
            img_path, label, person_name = self.samples[current_idx]
            
            img = self._load_and_preprocess_image(img_path)
            
            if img is not None:
                if self.transform:
                    img = self.transform(img)
                else:
                    img = torch.from_numpy(img).permute(2, 0, 1).float()
                
                return {
                    'image': img,
                    'label': torch.tensor(label, dtype=torch.long),
                    'person_name': person_name,
                    'img_path': img_path
                }
            
            attempts += 1
        
        # If all retries failed, return a dummy sample with zeros
        print(f"Warning: Failed to load valid image after {max_retries} attempts, returning dummy sample")
        dummy_img = torch.zeros(3, self.target_size[0], self.target_size[1])
        return {
            'image': dummy_img,
            'label': torch.tensor(0, dtype=torch.long),
            'person_name': "dummy",
            'img_path': "dummy_path"
        }

def create_lfw_dataloaders(data_dir: str,
                          pairs_train_file: str,
                          pairs_test_file: str,
                          batch_size: int = 32,
                          num_workers: int = 4,
                          target_size: Tuple[int, int] = (160, 160)) -> Tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders for LFW verification task"""
    
    face_detector = SsdResNetDetector(confidence_threshold=0.5)
    
    train_dataset = LFWPairDataset(
        data_dir=data_dir,
        pairs_file=pairs_train_file,
        target_size=target_size,
        face_detector=face_detector,
        normalization="Facenet2018"
    )
    
    test_dataset = LFWPairDataset(
        data_dir=data_dir,
        pairs_file=pairs_test_file,
        target_size=target_size,
        face_detector=face_detector,
        normalization="Facenet2018"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader 