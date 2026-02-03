
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from callmefair.search._search_base import BaseSearch

class TestMultiGPU(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        self.df = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'sensitive': np.random.randint(0, 2, 100),
            'target': np.random.randint(0, 2, 100)
        })
        self.searcher = BaseSearch(self.df, 'target', n_threads=4)

    @patch('callmefair.search._search_base._get_gpu_count')
    @patch('callmefair.search._search_base._is_gpu_available')
    @patch('callmefair.search._search_base.TrainingOptimizer')
    def test_gpu_distribution(self, mock_optimizer_cls, mock_is_gpu, mock_get_gpu_count):
        # Mock 2 GPUs available
        mock_get_gpu_count.return_value = 2
        mock_is_gpu.return_value = True
        
        # Mock optimizer instance
        mock_optimizer = MagicMock()
        mock_optimizer_cls.return_value = mock_optimizer
        # Mock map to return empty list (we just want to check args)
        mock_optimizer.map.return_value = []

        # Run evaluation with a GPU-capable model
        # We need to mock wrapper_training or process_wrapper_training to avoid actual training
        # But we are checking what is passed to optimizer.map
        
        try:
            self.searcher.evaluate_attribute('sensitive', iterate=4, model_name='xgb')
        except Exception:
            # It might fail after map because we return empty list and it expects results
            # But we just want to check the call to map
            pass

        # Check if optimizer was initialized with correct backend
        # Should be 'process' or 'thread' depending on logic. 
        # With 2 GPUs and 4 threads, it should run parallel.
        
        # Check args passed to map
        # map(func, args_list, max_workers=...)
        call_args = mock_optimizer.map.call_args
        if call_args:
            func, args_list = call_args[0][0], call_args[0][1]
            # args_list should be a list of tuples
            # Each tuple should have gpu_id at the end
            
            print(f"Captured args list length: {len(args_list)}")
            gpu_ids = [arg[-1] for arg in args_list]
            print(f"Captured GPU IDs: {gpu_ids}")
            
            # We expect round robin: 0, 1, 0, 1
            expected_ids = [0, 1, 0, 1]
            self.assertEqual(gpu_ids, expected_ids)
        else:
            self.fail("optimizer.map was not called")

if __name__ == '__main__':
    unittest.main()
