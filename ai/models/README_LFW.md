# LFW Dataset Training and Testing Scripts

This directory contains scripts for training and testing InceptionResNetV1 models on the LFW (Labeled Faces in the Wild) dataset with face detection and Facenet2018 normalization preprocessing.

## Files Overview

- `lfw_dataset.py` - Dataset classes for loading and preprocessing LFW data
- `train_lfw.py` - Training script for InceptionResNetV1 model
- `test_lfw.py` - Testing/evaluation script with comprehensive metrics
- `run_lfw_experiment.py` - Convenience script to run complete experiments

## Prerequisites

Make sure you have the LFW dataset properly set up in the `data/` directory with the following structure:

```
data/
├── lfw-funneled/
│   └── lfw_funneled/
│       ├── Person_Name/
│       │   ├── Person_Name_0001.jpg
│       │   └── ...
│       └── ...
├── pairs.txt
├── pairsDevTrain.txt
└── pairsDevTest.txt
```

## Quick Start

### Run Complete Experiment

The easiest way to run a complete training and testing experiment:

```bash
python3 ai/models/run_lfw_experiment.py \
    --data_dir data \
    --experiment_name my_experiment \
    --num_epochs 20 \
    --batch_size 32 \
    --learning_rate 0.001
```

### Training Only

To train a model:

```bash
python3 ai/models/train_lfw.py \
    --data_dir data \
    --batch_size 32 \
    --num_epochs 50 \
    --learning_rate 0.001 \
    --embedding_dim 512 \
    --loss_type contrastive \
    --checkpoint_dir checkpoints/my_model
```

### Testing Only

To test a trained model:

```bash
python3 ai/models/test_lfw.py \
    --model_path checkpoints/my_model/best_model.pth \
    --data_dir data \
    --protocol 10fold \
    --output_dir results/my_model
```

## Detailed Usage

### Training Script (`train_lfw.py`)

**Arguments:**
- `--data_dir`: Path to LFW data directory (default: 'data')
- `--batch_size`: Batch size for training (default: 32)
- `--num_epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Learning rate (default: 0.001)
- `--embedding_dim`: Embedding dimension (default: 512)
- `--loss_type`: Loss function type ('contrastive' or 'triplet', default: 'contrastive')
- `--margin`: Margin for loss function (default: 1.0)
- `--pretrained`: Use pretrained weights ('vggface2' or 'casia-webface', default: None)
- `--checkpoint_dir`: Directory to save checkpoints (default: 'checkpoints')
- `--num_workers`: Number of data loader workers (default: 4)

**Features:**
- Automatic face detection using SSD ResNet detector
- Facenet2018 normalization preprocessing
- Contrastive loss for face verification
- Learning rate scheduling
- Best model checkpointing
- Training curve visualization

### Testing Script (`test_lfw.py`)

**Arguments:**
- `--model_path`: Path to trained model checkpoint (required)
- `--config_path`: Path to config file (optional)
- `--data_dir`: Path to LFW data directory (default: 'data')
- `--test_file`: Test pairs file (default: 'pairs.txt')
- `--output_dir`: Output directory for results (default: 'test_results')
- `--protocol`: Evaluation protocol ('standard' or '10fold', default: 'standard')

**Evaluation Metrics:**
- Accuracy with optimal threshold
- ROC curve and AUC
- Equal Error Rate (EER)
- Distance and similarity-based metrics
- 10-fold cross-validation support

### Dataset Classes (`lfw_dataset.py`)

**LFWPairDataset:**
- Loads image pairs for face verification
- Supports both positive (same person) and negative (different person) pairs
- Automatic face detection and preprocessing

**LFWIdentityDataset:**
- Loads images by identity for classification tasks
- Filters identities with minimum number of images

## Preprocessing Pipeline

1. **Face Detection**: Uses SSD ResNet detector to locate faces in images
2. **Face Extraction**: Extracts the largest detected face region
3. **Resizing**: Resizes face to target size (160x160) with padding
4. **Normalization**: Applies Facenet2018 normalization (pixel values / 127.5 - 1)

## Model Architecture

- **Base Model**: InceptionResNetV1 (Facenet architecture)
- **Input Size**: 160x160x3 RGB images
- **Output**: 512-dimensional embeddings
- **Loss Function**: Contrastive loss for face verification

## Results and Outputs

### Training Outputs
- Model checkpoints saved in `checkpoint_dir/`
- Training configuration saved as `config.json`
- Training curves plotted and saved
- Best model saved as `best_model.pth`

### Testing Outputs
- Comprehensive evaluation metrics in `results.json`
- ROC curves and distribution plots
- Detailed accuracy and threshold analysis

## Example Results Format

```json
{
  "num_pairs": 6000,
  "num_positive": 3000,
  "num_negative": 3000,
  "distance_metrics": {
    "best_threshold": 1.234,
    "best_accuracy": 0.8567,
    "auc": 0.9123,
    "eer": 0.1234
  },
  "similarity_metrics": {
    "best_threshold": 0.567,
    "best_accuracy": 0.8678,
    "auc": 0.9234,
    "eer": 0.1123
  }
}
```

## Performance Tips

1. **GPU Usage**: The scripts automatically detect and use GPU if available
2. **Batch Size**: Adjust based on available memory (32 works well for most setups)
3. **Workers**: Set `num_workers` based on CPU cores for faster data loading
4. **Face Detection**: Lower confidence threshold may detect more faces but with more false positives

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use fewer workers
2. **No Faces Detected**: Check image quality and face detector confidence threshold
3. **Slow Training**: Increase number of workers or use GPU
4. **Poor Results**: Try different learning rates or pretrained weights

### Dependencies

Make sure you have installed:
- torch
- torchvision
- opencv-python
- numpy
- matplotlib
- scikit-learn
- tqdm

## Advanced Usage

### Custom Preprocessing

You can modify the preprocessing pipeline by:
1. Adjusting face detector confidence threshold
2. Changing normalization method
3. Modifying target image size

### Custom Loss Functions

The training script supports:
- Contrastive loss (implemented)
- Triplet loss (framework ready, needs triplet sampling)

### Model Variants

You can experiment with:
- Different embedding dimensions
- Pretrained weights (VGGFace2, CASIA-WebFace)
- Different dropout rates

## Citation

If you use these scripts in your research, please cite the original LFW dataset paper and the FaceNet paper for the model architecture.