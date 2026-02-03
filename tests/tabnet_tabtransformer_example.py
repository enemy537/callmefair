import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Configure GPU memory growth for both frameworks before importing them
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations for better compatibility
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress TensorFlow info messages

# Import TensorFlow with GPU configuration
import tensorflow as tf
if tf.config.list_physical_devices('GPU'):
    try:
        # Allow memory growth to avoid allocating all GPU memory at once
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        print("TensorFlow GPU configured with memory growth")
    except RuntimeError as e:
        print(f"Error configuring TensorFlow GPU: {e}")

# Import PyTorch with GPU configuration
import torch
if torch.cuda.is_available():
    # Set PyTorch to use the same GPU as TensorFlow
    torch.backends.cudnn.benchmark = True
    print(f"PyTorch using CUDA device: {torch.cuda.get_device_name(0)}")

# Now import model implementations
from callmefair.util.models import TabTransformer
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

class TabNetWrapper:
    """Wrapper for TabNet model to match TabTransformer interface"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = {
            'n_d': 8, 'n_a': 8, 'n_steps': 3,
            'gamma': 1.3, 'n_independent': 2, 'n_shared': 2,
            'cat_emb_dim': 1, 'momentum': 0.3, 'optimizer_fn': torch.optim.Adam,
            'optimizer_params': {'lr': 2e-2}, 'scheduler_params': {'step_size': 10, 'gamma': 0.9},
            'scheduler_fn': torch.optim.lr_scheduler.StepLR, 'verbose': 0,
            'device_name': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        if config:
            self.config.update(config)
        
        self.model = None
        self.label_encoders = {}
        self.cat_cols = []
        self.num_cols = []
        self.target_col = None
    
    def _preprocess_data(self, df: pd.DataFrame, target: str, fit: bool = True):
        """Preprocess data for TabNet"""
        df = df.copy()
        
        # Store column info
        if fit:
            self.target_col = target
            self.cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            self.num_cols = [col for col in df.columns if col != target and col not in self.cat_cols]
        
        # Encode categorical columns
        for col in self.cat_cols:
            if col == target:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str).fillna('NA'))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le is not None:
                    df[col] = le.transform(df[col].astype(str).fillna('NA'))
        
        # Separate features and target
        if target in df.columns:
            X = df.drop(columns=[target])
            y = df[target].values
            return X, y
        return df, None
    
    def fit(self, df: pd.DataFrame, target: str, **kwargs):
        """Train the TabNet model"""
        # Preprocess data
        X, y = self._preprocess_data(df, target, fit=True)
        
        # Initialize model
        self.model = TabNetClassifier(**{k: v for k, v in self.config.items() 
                                      if k not in ['device_name']})
        
        # Train model
        self.model.fit(
            X.values, y,
            eval_set=[(X.values, y)],
            eval_metric=['auc'],
            max_epochs=100,
            patience=10,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,  # Disable multiprocessing to avoid issues
            drop_last=False,
            **kwargs
        )
        return self
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        X, _ = self._preprocess_data(df, self.target_col, fit=False)
        return self.model.predict_proba(X.values)[:, 1]  # Return probabilities for class 1


def compare_models(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):
    """Compare TabNet and TabTransformer on the same dataset"""
    # Split data
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Train and evaluate TabNet
    print("\n" + "="*50)
    print("Training TabNet...")
    print("="*50)
    start_time = time.time()
    
    tabnet = TabNetWrapper()
    tabnet.fit(train_df, target)
    
    # Evaluate
    y_pred_proba = tabnet.predict_proba(test_df)
    y_true = test_df[target].values
    auc = roc_auc_score(y_true, y_pred_proba)
    
    tabnet_time = time.time() - start_time
    print(f"TabNet training time: {tabnet_time:.2f}s")
    print(f"TabNet Test AUC: {auc:.4f}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Train and evaluate TabTransformer
    print("\n" + "="*50)
    print("Training TabTransformer...")
    print("="*50)
    start_time = time.time()
    
    tabtransformer = TabTransformer({
        'batch_size': 256,
        'epochs': 20,
        'learning_rate': 1e-3,
        'transformer_layers': 4,
        'transformer_heads': 4,
        'transformer_embedding_dim': 32,
        'ff_dim': 32,
        'dropout': 0.1,
        'mlp_units': [64, 32],
        'mlp_dropout': 0.2,
        'dense_activation': 'gelu',
        'verbose': 1
    }, target=target)
    
    # TabTransformer.fit expects only the data as positional argument;
    # the target column name is already stored in the instance.
    tabtransformer.fit(train_df)
    
    # Evaluate
    y_pred_proba = tabtransformer.predict_proba(test_df)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    tt_time = time.time() - start_time
    print(f"TabTransformer training time: {tt_time:.2f}s")
    print(f"TabTransformer Test AUC: {auc:.4f}")
    
    # Clear GPU memory
    if tf.config.list_physical_devices('GPU'):
        tf.keras.backend.clear_session()
    
    return {
        'tabnet': {'time': tabnet_time, 'auc': auc},
        'tabtransformer': {'time': tt_time, 'auc': auc}
    }

if __name__ == "__main__":
    # Example usage with a sample dataset
    from sklearn.datasets import make_classification
    
    # Create a sample dataset
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Convert to DataFrame with meaningful column names
    feature_cols = [f'feature_{i+1}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)
    df['target'] = y
    
    # Add some categorical features
    for i in range(3):
        df[f'cat_{i}'] = np.random.choice(['A', 'B', 'C', 'D'], size=len(df))
    
    # Run comparison
    results = compare_models(df, 'target')
    
    print("\n" + "="*50)
    print("Comparison Results:")
    print("-"*20)
    for model, metrics in results.items():
        print(f"{model.upper()}:")
        print(f"  - Training Time: {metrics['time']:.2f}s")
        print(f"  - Test AUC: {metrics['auc']:.4f}")
    print("="*50)
