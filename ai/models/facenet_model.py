import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass
import logging
import requests
import hashlib
import tempfile
import shutil
from urllib.request import urlopen, Request

try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration class for FaceNet model parameters"""
    embedding_dim: int = 512
    input_size: Tuple[int, int] = (160, 160)
    pretrained: Optional[str] = None  # 'vggface2' or 'casia-webface'
    classify: bool = False
    num_classes: Optional[int] = None
    dropout_prob: float = 0.6
    cpu_threads: int = 4
    quantization: bool = True
    model_path: Optional[str] = None
    device: Optional[str] = None
    
    def __post_init__(self):
        torch.set_num_threads(self.cpu_threads)

def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    """Download object at the given URL to a local path."""
    file_size = None
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)

def get_torch_home():
    """Get torch cache directory"""
    torch_home = os.path.expanduser(
        os.getenv(
            'TORCH_HOME',
            os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')
        )
    )
    return torch_home

class BasicConv2d(nn.Module):
    """Basic convolution block with batch normalization and ReLU"""
    
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001,
            momentum=0.1,
            affine=True
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Block35(nn.Module):
    """Inception-ResNet-A block"""
    
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out

class Block17(nn.Module):
    """Inception-ResNet-B block"""
    
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(128, 128, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out

class Block8(nn.Module):
    """Inception-ResNet-C block"""
    
    def __init__(self, scale=1.0, noReLU=False):
        super().__init__()
        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2d(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))
        )

        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out

class Mixed_6a(nn.Module):
    """Reduction-A block"""
    
    def __init__(self):
        super().__init__()

        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

class Mixed_7a(nn.Module):
    """Reduction-B block"""
    
    def __init__(self):
        super().__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class InceptionResnetV1(nn.Module):
    """Inception Resnet V1 model with optional loading of pretrained weights.

    Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
    datasets. Pretrained state_dicts are automatically downloaded on model instantiation if
    requested and cached in the torch cache. Subsequent instantiations use the cache rather than
    redownloading.

    Keyword Arguments:
        pretrained {str} -- Optional pretraining dataset. Either 'vggface2' or 'casia-webface'.
            (default: {None})
        classify {bool} -- Whether the model should output classification probabilities or feature
            embeddings. (default: {False})
        num_classes {int} -- Number of output classes. If 'pretrained' is set and num_classes not
            equal to that used for the pretrained model, the final linear layer will be randomly
            initialized. (default: {None})
        dropout_prob {float} -- Dropout probability. (default: {0.6})
        device {str} -- Device to run model on. (default: {None})
    """
    def __init__(self, pretrained=None, classify=False, num_classes=None, dropout_prob=0.6, device=None):
        super().__init__()

        self.pretrained = pretrained
        self.classify = classify
        self.num_classes = num_classes

        if pretrained == 'vggface2':
            tmp_classes = 8631
        elif pretrained == 'casia-webface':
            tmp_classes = 10575
        elif pretrained is None and self.classify and self.num_classes is None:
            raise Exception('If "pretrained" is not specified and "classify" is True, "num_classes" must be specified')

        # Define layers
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
        self.repeat_1 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
        )
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.last_linear = nn.Linear(1792, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

        if pretrained is not None:
            self.logits = nn.Linear(512, tmp_classes)
            load_weights(self, pretrained)

        if self.classify and self.num_classes is not None:
            self.logits = nn.Linear(512, self.num_classes)

        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)

    def forward(self, x):
        """Calculate embeddings or logits given a batch of input image tensors.

        Arguments:
            x {torch.tensor} -- Batch of image tensors representing faces.

        Returns:
            torch.tensor -- Batch of embedding vectors or multinomial logits.
        """
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)
        if self.classify:
            x = self.logits(x)
        else:
            x = F.normalize(x, p=2, dim=1)
        return x

def load_weights(mdl, name):
    """Download pretrained state_dict and load into model.

    Arguments:
        mdl {torch.nn.Module} -- Pytorch model.
        name {str} -- Name of dataset that was used to generate pretrained state_dict.

    Raises:
        ValueError: If 'pretrained' not equal to 'vggface2' or 'casia-webface'.
    """
    if name == 'vggface2':
        path = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt'
    elif name == 'casia-webface':
        path = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt'
    else:
        raise ValueError('Pretrained models only exist for "vggface2" and "casia-webface"')

    model_dir = os.path.join(get_torch_home(), 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)

    cached_file = os.path.join(model_dir, os.path.basename(path))
    if not os.path.exists(cached_file):
        logger.info(f"Downloading pretrained model from {path}")
        download_url_to_file(path, cached_file)

    logger.info(f"Loading pretrained weights from {cached_file}")
    state_dict = torch.load(cached_file, map_location='cpu')
    mdl.load_state_dict(state_dict)

class FaceNetModel:
    """
    FaceNet model wrapper optimized for mobile CPU deployment
    Compatible with facenet-pytorch pretrained models
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.transform = self._create_transform()
        self.device = torch.device(config.device or 'cpu')
        
    def _create_transform(self):
        """Create preprocessing transform for input images"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.config.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load the FaceNet model with optional pretrained weights"""
        try:
            self.model = InceptionResnetV1(
                pretrained=self.config.pretrained,
                classify=self.config.classify,
                num_classes=self.config.num_classes,
                dropout_prob=self.config.dropout_prob,
                device=self.device
            )
            
            if model_path or self.config.model_path:
                path = model_path or self.config.model_path
                logger.info(f"Loading custom model from {path}")
                checkpoint = torch.load(path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                logger.info(f"Loaded custom model from {path}")
            
            self.model.to(self.device)
            self.model.eval()
            
            if self.config.quantization and self.device.type == 'cpu':
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
            face_image: Preprocessed face image as numpy array (H, W, C) in RGB format
            
        Returns:
            embedding: 512-dimensional face embedding
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            if len(face_image.shape) == 3:
                face_tensor = self.transform(face_image).unsqueeze(0).to(self.device)
            else:
                raise ValueError("Input image must be 3D (H, W, C)")
            
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
            face_tensors = []
            for face_image in face_images:
                if len(face_image.shape) == 3:
                    face_tensor = self.transform(face_image)
                    face_tensors.append(face_tensor)
                else:
                    raise ValueError("All input images must be 3D (H, W, C)")
            
            batch_tensor = torch.stack(face_tensors).to(self.device)
            
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
            "embedding_dim": 512,
            "input_size": self.config.input_size,
            "pretrained": self.config.pretrained,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "quantized": self.config.quantization,
            "device": str(self.device),
            "classify": self.config.classify,
            "num_classes": self.config.num_classes
        } 