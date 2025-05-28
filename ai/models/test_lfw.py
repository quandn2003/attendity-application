import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F
import numpy as np
import logging
import argparse
import json
from typing import Dict, Tuple, List
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
import seaborn as sns

from ai.models.facenet_model import InceptionResnetV1
from ai.models.lfw_dataset import LFWPairDataset
from ai.models.face_detection import SsdResNetDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LFWEvaluator:
    """Evaluator for InceptionResNetV1 on LFW dataset"""
    
    def __init__(self, model_path: str, config_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.config = None
        
        self._load_model_and_config(model_path, config_path)
    
    def _load_model_and_config(self, model_path: str, config_path: str = None):
        """Load trained model and configuration"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        elif config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'embedding_dim': 512,
                'dropout_prob': 0.6,
                'pretrained': None
            }
        
        self.model = InceptionResnetV1(
            pretrained=self.config.get('pretrained', None),
            classify=False,
            dropout_prob=self.config.get('dropout_prob', 0.6),
            device=self.device
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        if 'best_accuracy' in checkpoint:
            logger.info(f"Model best training accuracy: {checkpoint['best_accuracy']:.4f}")
    
    def extract_embeddings(self, dataloader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract embeddings for all pairs in the dataset"""
        embeddings1 = []
        embeddings2 = []
        labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting embeddings"):
                img1 = batch['img1'].to(self.device)
                img2 = batch['img2'].to(self.device)
                batch_labels = batch['label'].cpu().numpy()
                
                emb1 = self.model(img1).cpu().numpy()
                emb2 = self.model(img2).cpu().numpy()
                
                embeddings1.append(emb1)
                embeddings2.append(emb2)
                labels.append(batch_labels)
        
        embeddings1 = np.vstack(embeddings1)
        embeddings2 = np.vstack(embeddings2)
        labels = np.concatenate(labels)
        
        return embeddings1, embeddings2, labels
    
    def compute_distances(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between embeddings"""
        distances = np.linalg.norm(embeddings1 - embeddings2, axis=1)
        return distances
    
    def compute_similarities(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Compute cosine similarities between embeddings"""
        embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        similarities = np.sum(embeddings1_norm * embeddings2_norm, axis=1)
        return similarities
    
    def find_best_threshold(self, distances: np.ndarray, labels: np.ndarray, 
                           metric: str = 'distance') -> Tuple[float, float]:
        """Find the best threshold for face verification"""
        if metric == 'distance':
            thresholds = np.linspace(distances.min(), distances.max(), 1000)
            predictions_func = lambda t: (distances < t).astype(int)
        else:
            thresholds = np.linspace(-1, 1, 1000)
            predictions_func = lambda t: (distances > t).astype(int)
        
        best_threshold = 0.0
        best_accuracy = 0.0
        
        for threshold in thresholds:
            predictions = predictions_func(threshold)
            accuracy = accuracy_score(labels, predictions)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        return best_threshold, best_accuracy
    
    def compute_roc_metrics(self, distances: np.ndarray, labels: np.ndarray, 
                           metric: str = 'distance') -> Dict:
        """Compute ROC curve and AUC"""
        if metric == 'distance':
            scores = -distances
        else:
            scores = distances
        
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        eer_threshold = thresholds[np.nanargmin(np.absolute((fpr - (1 - tpr))))]
        eer = fpr[np.nanargmin(np.absolute((fpr - (1 - tpr))))]
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc,
            'eer': eer,
            'eer_threshold': eer_threshold
        }
    
    def evaluate_lfw_protocol(self, data_dir: str, pairs_file: str) -> Dict:
        """Evaluate using LFW protocol"""
        dataset = LFWPairDataset(
            data_dir=data_dir,
            pairs_file=pairs_file,
            target_size=(160, 160),
            face_detector=SsdResNetDetector(confidence_threshold=0.5),
            normalization="Facenet2018"
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Evaluating on {len(dataset)} pairs")
        
        embeddings1, embeddings2, labels = self.extract_embeddings(dataloader)
        
        distances = self.compute_distances(embeddings1, embeddings2)
        similarities = self.compute_similarities(embeddings1, embeddings2)
        
        distance_threshold, distance_accuracy = self.find_best_threshold(distances, labels, 'distance')
        similarity_threshold, similarity_accuracy = self.find_best_threshold(similarities, labels, 'similarity')
        
        distance_roc = self.compute_roc_metrics(distances, labels, 'distance')
        similarity_roc = self.compute_roc_metrics(similarities, labels, 'similarity')
        
        results = {
            'num_pairs': len(labels),
            'num_positive': int(labels.sum()),
            'num_negative': int(len(labels) - labels.sum()),
            'distance_metrics': {
                'best_threshold': distance_threshold,
                'best_accuracy': distance_accuracy,
                'auc': distance_roc['auc'],
                'eer': distance_roc['eer']
            },
            'similarity_metrics': {
                'best_threshold': similarity_threshold,
                'best_accuracy': similarity_accuracy,
                'auc': similarity_roc['auc'],
                'eer': similarity_roc['eer']
            },
            'embeddings_stats': {
                'embedding_dim': embeddings1.shape[1],
                'mean_distance': float(distances.mean()),
                'std_distance': float(distances.std()),
                'mean_similarity': float(similarities.mean()),
                'std_similarity': float(similarities.std())
            }
        }
        
        return results, {
            'distances': distances,
            'similarities': similarities,
            'labels': labels,
            'distance_roc': distance_roc,
            'similarity_roc': similarity_roc
        }
    
    def evaluate_10_fold_protocol(self, data_dir: str) -> Dict:
        """Evaluate using 10-fold cross-validation protocol"""
        pairs_file = os.path.join(data_dir, 'pairs.txt')
        
        with open(pairs_file, 'r') as f:
            lines = f.readlines()
        
        num_folds, pairs_per_fold = map(int, lines[0].strip().split())
        
        fold_results = []
        all_distances = []
        all_similarities = []
        all_labels = []
        
        for fold in range(num_folds):
            logger.info(f"Evaluating fold {fold + 1}/{num_folds}")
            
            start_idx = 1 + fold * pairs_per_fold
            end_idx = start_idx + pairs_per_fold
            
            fold_pairs = []
            for line in lines[start_idx:end_idx]:
                parts = line.strip().split()
                if len(parts) == 3:
                    name, img1_idx, img2_idx = parts
                    fold_pairs.append((name, int(img1_idx), name, int(img2_idx), 1))
                elif len(parts) == 4:
                    name1, img1_idx, name2, img2_idx = parts
                    fold_pairs.append((name1, int(img1_idx), name2, int(img2_idx), 0))
            
            fold_dataset = LFWPairDataset(
                data_dir=data_dir,
                pairs_file=None,
                target_size=(160, 160),
                face_detector=SsdResNetDetector(confidence_threshold=0.5),
                normalization="Facenet2018"
            )
            fold_dataset.pairs = fold_pairs
            
            fold_dataloader = torch.utils.data.DataLoader(
                fold_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            embeddings1, embeddings2, labels = self.extract_embeddings(fold_dataloader)
            distances = self.compute_distances(embeddings1, embeddings2)
            similarities = self.compute_similarities(embeddings1, embeddings2)
            
            distance_threshold, distance_accuracy = self.find_best_threshold(distances, labels, 'distance')
            similarity_threshold, similarity_accuracy = self.find_best_threshold(similarities, labels, 'similarity')
            
            fold_result = {
                'fold': fold + 1,
                'distance_accuracy': distance_accuracy,
                'similarity_accuracy': similarity_accuracy,
                'distance_threshold': distance_threshold,
                'similarity_threshold': similarity_threshold
            }
            
            fold_results.append(fold_result)
            all_distances.extend(distances)
            all_similarities.extend(similarities)
            all_labels.extend(labels)
        
        all_distances = np.array(all_distances)
        all_similarities = np.array(all_similarities)
        all_labels = np.array(all_labels)
        
        distance_roc = self.compute_roc_metrics(all_distances, all_labels, 'distance')
        similarity_roc = self.compute_roc_metrics(all_similarities, all_labels, 'similarity')
        
        mean_distance_acc = np.mean([r['distance_accuracy'] for r in fold_results])
        std_distance_acc = np.std([r['distance_accuracy'] for r in fold_results])
        mean_similarity_acc = np.mean([r['similarity_accuracy'] for r in fold_results])
        std_similarity_acc = np.std([r['similarity_accuracy'] for r in fold_results])
        
        results = {
            'num_folds': num_folds,
            'fold_results': fold_results,
            'mean_distance_accuracy': mean_distance_acc,
            'std_distance_accuracy': std_distance_acc,
            'mean_similarity_accuracy': mean_similarity_acc,
            'std_similarity_accuracy': std_similarity_acc,
            'overall_distance_auc': distance_roc['auc'],
            'overall_similarity_auc': similarity_roc['auc'],
            'overall_distance_eer': distance_roc['eer'],
            'overall_similarity_eer': similarity_roc['eer']
        }
        
        return results
    
    def plot_results(self, results_data: Dict, save_dir: str):
        """Plot evaluation results"""
        os.makedirs(save_dir, exist_ok=True)
        
        distances = results_data['distances']
        similarities = results_data['similarities']
        labels = results_data['labels']
        distance_roc = results_data['distance_roc']
        similarity_roc = results_data['similarity_roc']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].hist(distances[labels == 1], bins=50, alpha=0.7, label='Same person', density=True)
        axes[0, 0].hist(distances[labels == 0], bins=50, alpha=0.7, label='Different person', density=True)
        axes[0, 0].set_xlabel('Distance')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Distance Distribution')
        axes[0, 0].legend()
        
        axes[0, 1].hist(similarities[labels == 1], bins=50, alpha=0.7, label='Same person', density=True)
        axes[0, 1].hist(similarities[labels == 0], bins=50, alpha=0.7, label='Different person', density=True)
        axes[0, 1].set_xlabel('Similarity')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Similarity Distribution')
        axes[0, 1].legend()
        
        axes[1, 0].plot(distance_roc['fpr'], distance_roc['tpr'], 
                       label=f'Distance ROC (AUC = {distance_roc["auc"]:.3f})')
        axes[1, 0].plot([0, 1], [0, 1], 'k--')
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curve (Distance)')
        axes[1, 0].legend()
        
        axes[1, 1].plot(similarity_roc['fpr'], similarity_roc['tpr'], 
                       label=f'Similarity ROC (AUC = {similarity_roc["auc"]:.3f})')
        axes[1, 1].plot([0, 1], [0, 1], 'k--')
        axes[1, 1].set_xlabel('False Positive Rate')
        axes[1, 1].set_ylabel('True Positive Rate')
        axes[1, 1].set_title('ROC Curve (Similarity)')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'evaluation_results.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Test InceptionResNetV1 on LFW dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, default=None, help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to LFW data directory')
    parser.add_argument('--test_file', type=str, default='pairs.txt', help='Test pairs file')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Output directory for results')
    parser.add_argument('--protocol', type=str, default='standard', choices=['standard', '10fold'], 
                       help='Evaluation protocol')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    evaluator = LFWEvaluator(args.model_path, args.config_path)
    
    if args.protocol == '10fold':
        logger.info("Running 10-fold cross-validation evaluation")
        results = evaluator.evaluate_10_fold_protocol(args.data_dir)
        
        logger.info("=== 10-Fold Cross-Validation Results ===")
        logger.info(f"Mean Distance Accuracy: {results['mean_distance_accuracy']:.4f} ± {results['std_distance_accuracy']:.4f}")
        logger.info(f"Mean Similarity Accuracy: {results['mean_similarity_accuracy']:.4f} ± {results['std_similarity_accuracy']:.4f}")
        logger.info(f"Overall Distance AUC: {results['overall_distance_auc']:.4f}")
        logger.info(f"Overall Similarity AUC: {results['overall_similarity_auc']:.4f}")
        logger.info(f"Overall Distance EER: {results['overall_distance_eer']:.4f}")
        logger.info(f"Overall Similarity EER: {results['overall_similarity_eer']:.4f}")
        
    else:
        logger.info("Running standard evaluation")
        test_pairs_file = os.path.join(args.data_dir, args.test_file)
        results, results_data = evaluator.evaluate_lfw_protocol(args.data_dir, test_pairs_file)
        
        logger.info("=== Evaluation Results ===")
        logger.info(f"Number of pairs: {results['num_pairs']}")
        logger.info(f"Positive pairs: {results['num_positive']}")
        logger.info(f"Negative pairs: {results['num_negative']}")
        logger.info(f"Distance - Best Accuracy: {results['distance_metrics']['best_accuracy']:.4f}")
        logger.info(f"Distance - AUC: {results['distance_metrics']['auc']:.4f}")
        logger.info(f"Distance - EER: {results['distance_metrics']['eer']:.4f}")
        logger.info(f"Similarity - Best Accuracy: {results['similarity_metrics']['best_accuracy']:.4f}")
        logger.info(f"Similarity - AUC: {results['similarity_metrics']['auc']:.4f}")
        logger.info(f"Similarity - EER: {results['similarity_metrics']['eer']:.4f}")
        
        evaluator.plot_results(results_data, args.output_dir)
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    logger.info(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    main() 