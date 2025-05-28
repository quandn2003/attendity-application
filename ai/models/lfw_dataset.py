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
        
        with open(self.pairs_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines[0].strip().split()) == 2:
            num_folds, pairs_per_fold = map(int, lines[0].strip().split())
            start_idx = 1
        else:
            start_idx = 1
            
        for line in lines[start_idx:]:
            parts = line.strip().split()
            if len(parts) == 3:
                name, img1_idx, img2_idx = parts
                pairs.append((name, int(img1_idx), name, int(img2_idx), 1))
            elif len(parts) == 4:
                name1, img1_idx, name2, img2_idx = parts
                pairs.append((name1, int(img1_idx), name2, int(img2_idx), 0))
        
        return pairs
    
    def _load_and_preprocess_image(self, person_name: str, img_idx: int) -> Optional[np.ndarray]:
        """Load and preprocess a single image"""
        img_name = f"{person_name}_{img_idx:04d}.jpg"
        img_path = os.path.join(self.data_dir, "lfw-funneled", "lfw_funneled", person_name, img_name)
        
        if not os.path.exists(img_path):
            logger.warning(f"Image not found: {img_path}")
            return None
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Failed to load image: {img_path}")
                return None
            
            faces = self.face_detector.detect_faces(img)
            
            if not faces:
                logger.warning(f"No face detected in: {img_path}")
                img_resized = resize_image(img, self.target_size)
            else:
                largest_face = max(faces, key=lambda f: f.w * f.h)
                face_img = img[largest_face.y:largest_face.y + largest_face.h,
                              largest_face.x:largest_face.x + largest_face.w]
                img_resized = resize_image(face_img, self.target_size)
            
            img_normalized = normalize_input(img_resized, self.normalization)
            
            return img_normalized.squeeze(0)
            
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a pair of images and label"""
        name1, img1_idx, name2, img2_idx, label = self.pairs[idx]
        
        img1 = self._load_and_preprocess_image(name1, img1_idx)
        img2 = self._load_and_preprocess_image(name2, img2_idx)
        
        if img1 is None or img2 is None:
            return self.__getitem__((idx + 1) % len(self.pairs))
        
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
        lfw_dir = os.path.join(self.data_dir, "lfw-funneled", "lfw_funneled")
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
            logger.error(f"Error processing image {img_path}: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get an image and its identity label"""
        img_path, label, person_name = self.samples[idx]
        
        img = self._load_and_preprocess_image(img_path)
        
        if img is None:
            return self.__getitem__((idx + 1) % len(self.samples))
        
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