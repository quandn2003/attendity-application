import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration class for FaceNet model parameters"""
    embedding_dim: int = 512
    input_size: Tuple[int, int] = (160, 160)
    num_classes: int = None
    dropout_rate: float = 0.2
    cpu_threads: int = 4
    quantization: bool = True
    model_path: Optional[str] = None
    
    def __post_init__(self):
        torch.set_num_threads(self.cpu_threads)

class InceptionResNetV1(nn.Module):
    """
    Inception ResNet V1 model optimized for mobile CPU inference
    Based on the FaceNet architecture with mobile optimizations
    """
    
    def __init__(self, embedding_dim: int = 512, dropout_rate: float = 0.2):
        super(InceptionResNetV1, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Initial convolution layers
        self.conv2d_1a = nn.Conv2d(3, 32, 3, stride=2, bias=False)
        self.conv2d_1a_bn = nn.BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True)
        
        self.conv2d_2a = nn.Conv2d(32, 32, 3, bias=False)
        self.conv2d_2a_bn = nn.BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True)
        
        self.conv2d_2b = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.conv2d_2b_bn = nn.BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True)
        
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        
        self.conv2d_3b = nn.Conv2d(64, 80, 1, bias=False)
        self.conv2d_3b_bn = nn.BatchNorm2d(80, eps=0.001, momentum=0.1, affine=True)
        
        self.conv2d_4a = nn.Conv2d(80, 192, 3, bias=False)
        self.conv2d_4a_bn = nn.BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True)
        
        self.conv2d_4b = nn.Conv2d(192, 256, 3, stride=2, bias=False)
        self.conv2d_4b_bn = nn.BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True)
        
        # Inception blocks (simplified for mobile)
        self.repeat_1 = nn.ModuleList([
            Block35(scale=0.17) for _ in range(5)
        ])
        
        self.mixed_6a = Mixed_6a()
        
        self.repeat_2 = nn.ModuleList([
            Block17(scale=0.10) for _ in range(10)
        ])
        
        self.mixed_7a = Mixed_7a()
        
        self.repeat_3 = nn.ModuleList([
            Block8(scale=0.20) for _ in range(5)
        ])
        
        self.block8 = Block8(noReLU=True)
        
        # Final layers
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.last_linear = nn.Linear(1792, embedding_dim, bias=False)
        self.last_bn = nn.BatchNorm1d(embedding_dim, eps=0.001, momentum=0.1, affine=True)
        
    def forward(self, x):
        """Forward pass optimized for CPU inference"""
        # Initial convolutions
        x = F.relu(self.conv2d_1a_bn(self.conv2d_1a(x)))
        x = F.relu(self.conv2d_2a_bn(self.conv2d_2a(x)))
        x = F.relu(self.conv2d_2b_bn(self.conv2d_2b(x)))
        x = self.maxpool_3a(x)
        x = F.relu(self.conv2d_3b_bn(self.conv2d_3b(x)))
        x = F.relu(self.conv2d_4a_bn(self.conv2d_4a(x)))
        x = F.relu(self.conv2d_4b_bn(self.conv2d_4b(x)))
        
        # Inception blocks
        for block in self.repeat_1:
            x = block(x)
        x = self.mixed_6a(x)
        for block in self.repeat_2:
            x = block(x)
        x = self.mixed_7a(x)
        for block in self.repeat_3:
            x = block(x)
        x = self.block8(x)
        
        # Final layers
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = self.last_linear(x)
        x = self.last_bn(x)
        
        return F.normalize(x, p=2, dim=1)

class Block35(nn.Module):
    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        self.scale = scale
        
        self.branch0 = BasicConv2d(256, 32, 1)
        
        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, 1),
            BasicConv2d(32, 32, 3, padding=1)
        )
        
        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, 1),
            BasicConv2d(32, 32, 3, padding=1),
            BasicConv2d(32, 32, 3, padding=1)
        )
        
        self.conv2d = nn.Conv2d(96, 256, 1)
        
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = F.relu(out)
        return out

class Block17(nn.Module):
    def __init__(self, scale=1.0):
        super(Block17, self).__init__()
        self.scale = scale
        
        self.branch0 = BasicConv2d(896, 128, 1)
        
        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, 1),
            BasicConv2d(128, 128, (1, 7), padding=(0, 3)),
            BasicConv2d(128, 128, (7, 1), padding=(3, 0))
        )
        
        self.conv2d = nn.Conv2d(256, 896, 1)
        
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = F.relu(out)
        return out

class Block8(nn.Module):
    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()
        self.scale = scale
        self.noReLU = noReLU
        
        self.branch0 = BasicConv2d(1792, 192, 1)
        
        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, 1),
            BasicConv2d(192, 192, (1, 3), padding=(0, 1)),
            BasicConv2d(192, 192, (3, 1), padding=(1, 0))
        )
        
        self.conv2d = nn.Conv2d(384, 1792, 1)
        
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = F.relu(out)
        return out

class Mixed_6a(nn.Module):
    def __init__(self):
        super(Mixed_6a, self).__init__()
        
        self.branch0 = BasicConv2d(256, 384, 3, stride=2)
        
        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, 1),
            BasicConv2d(192, 192, 3, padding=1),
            BasicConv2d(192, 256, 3, stride=2)
        )
        
        self.branch2 = nn.MaxPool2d(3, stride=2)
        
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

class Mixed_7a(nn.Module):
    def __init__(self):
        super(Mixed_7a, self).__init__()
        
        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, 1),
            BasicConv2d(256, 384, 3, stride=2)
        )
        
        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, 1),
            BasicConv2d(256, 256, 3, stride=2)
        )
        
        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, 1),
            BasicConv2d(256, 256, 3, padding=1),
            BasicConv2d(256, 256, 3, stride=2)
        )
        
        self.branch3 = nn.MaxPool2d(3, stride=2)
        
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1, affine=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

class FaceNetModel:
    """
    FaceNet model wrapper optimized for mobile CPU deployment
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.transform = self._create_transform()
        self.device = torch.device('cpu')
        
    def _create_transform(self):
        """Create preprocessing transform for input images"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.config.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load the FaceNet model with CPU optimizations"""
        try:
            self.model = InceptionResNetV1(
                embedding_dim=self.config.embedding_dim,
                dropout_rate=self.config.dropout_rate
            )
            
            if model_path or self.config.model_path:
                path = model_path or self.config.model_path
                checkpoint = torch.load(path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
                logger.info(f"Loaded model from {path}")
            
            self.model.to(self.device)
            self.model.eval()
            
            if self.config.quantization:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                )
                logger.info("Applied dynamic quantization for CPU optimization")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract face embedding from preprocessed face image
        
        Args:
            face_image: Preprocessed face image as numpy array
            
        Returns:
            embedding: 512-dimensional face embedding
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess image
            if len(face_image.shape) == 3:
                face_tensor = self.transform(face_image).unsqueeze(0)
            else:
                raise ValueError("Input image must be 3D (H, W, C)")
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model(face_tensor)
                embedding = embedding.cpu().numpy().flatten()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            raise
    
    def extract_batch_embeddings(self, face_images: list) -> np.ndarray:
        """
        Extract embeddings from multiple face images
        
        Args:
            face_images: List of preprocessed face images
            
        Returns:
            embeddings: Array of embeddings with shape (N, embedding_dim)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess all images
            face_tensors = []
            for face_image in face_images:
                if len(face_image.shape) == 3:
                    face_tensor = self.transform(face_image)
                    face_tensors.append(face_tensor)
                else:
                    raise ValueError("All input images must be 3D (H, W, C)")
            
            batch_tensor = torch.stack(face_tensors)
            
            # Extract embeddings
            with torch.no_grad():
                embeddings = self.model(batch_tensor)
                embeddings = embeddings.cpu().numpy()
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error extracting batch embeddings: {e}")
            raise
    
    def save_model(self, save_path: str) -> None:
        """Save the current model state"""
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics"""
        if self.model is None:
            return {"status": "Model not loaded"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "embedding_dim": self.config.embedding_dim,
            "input_size": self.config.input_size,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "quantized": self.config.quantization,
            "device": str(self.device)
        } 