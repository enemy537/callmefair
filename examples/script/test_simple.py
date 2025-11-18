#!/usr/bin/env python3
"""
Simple test to verify TabTransformer Dataset iteration fix.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Make local package available without installation
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Eager execution enabled (before import): {tf.executing_eagerly()}")
    
    from callmefair.util.models import TabTransformer
    
    print(f"Eager execution enabled (after import): {tf.executing_eagerly()}")
    print(f"Functions run eagerly: {tf.config.functions_run_eagerly()}")
    
    print("Testing TabTransformer with minimal synthetic data...")
    
    # Create very simple synthetic data
    np.random.seed(42)
    n_samples = 32  # Use power of 2 for batch size compatibility
    
    data = {
        'num_col': np.random.randn(n_samples).astype(np.float32),
        'cat_col': np.random.choice(['A', 'B'], n_samples),
    }
    
    X = pd.DataFrame(data)
    y = np.random.choice([0, 1], n_samples).astype(np.int32)
    
    print(f"Data shape: {X.shape}")
    print(f"Data types:\n{X.dtypes}")
    print(f"Target shape: {y.shape}")
    print(f"Target dtype: {y.dtype}")
    print(f"Target distribution: {np.bincount(y)}")
    print(f"Unique target values: {np.unique(y)}")
    
    # Test with minimal configuration
    model = TabTransformer(
        epochs=2,  # Just 2 epochs for testing
        batch_size=8,  # Smaller batch size for small dataset
        verbose=1,
    )
    
    print("\n--- Testing fit() ---")
    model.fit(X, y)
    print("✓ fit() completed successfully!")
    
    print("\n--- Testing predict() ---")
    predictions = model.predict(X[:5])
    print(f"✓ predict() completed successfully! Predictions: {predictions}")
    
    print("\n--- Testing predict_proba() ---")
    probabilities = model.predict_proba(X[:5])
    print(f"✓ predict_proba() completed successfully! Shape: {probabilities.shape}")
    print(f"Sample probabilities:\n{probabilities}")
    
    print("\n🎉 TabTransformer Dataset iteration fix is working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
