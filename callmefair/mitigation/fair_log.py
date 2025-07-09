"""
Logging and Result Management for Bias Mitigation Experiments

This module provides comprehensive logging and result management capabilities for
bias mitigation experiments. It includes CSV logging functionality and result
aggregation tools for analyzing multiple experiments.

The module implements:
- CSV-based experiment logging with automatic file management
- Multiprocessing support for efficient result aggregation
- Automatic cleanup of processed files
- Comprehensive error handling and logging

Classes:
    csvLogger: CSV-based logger for experiment results

Functions:
    read_csv_file: Read a single CSV file with error handling
    aggregate_csv_files: Aggregate multiple CSV files using multiprocessing

Example:
    >>> from callmefair.mitigation.fair_log import csvLogger, aggregate_csv_files
    >>> 
    >>> # Create logger for experiment
    >>> logger = csvLogger('experiment_2024_01_15')
    >>> 
    >>> # Log experiment results
    >>> results = [
    >>>     {'model': 'RandomForest', 'BM': 'baseline', 'accuracy': 0.85},
    >>>     {'model': 'RandomForest', 'BM': 'reweighing', 'accuracy': 0.83}
    >>> ]
    >>> logger(results)
    >>> 
    >>> # Aggregate results from multiple experiments
    >>> aggregate_csv_files('./results/', './results/aggregated_results.csv')
"""

import pandas as pd
import os
import glob
from multiprocessing import Pool
import logging


class csvLogger:
    """
    CSV-based logger for experiment results.
    
    This class provides a simple interface for logging experiment results to CSV
    files. It automatically creates the output directory if it doesn't exist and
    appends results to the specified file.
    
    Attributes:
        count (int): Counter for logged entries
        filename (str): Name of the output CSV file (without extension)
        path (str): Directory path for storing CSV files
        
    Example:
        >>> logger = csvLogger('experiment_results', path='./results/')
        >>> 
        >>> # Log a single result
        >>> result = {'model': 'RandomForest', 'accuracy': 0.85}
        >>> logger([result])
        >>> 
        >>> # Log multiple results
        >>> results = [
        >>>     {'model': 'RandomForest', 'accuracy': 0.85},
        >>>     {'model': 'LogisticRegression', 'accuracy': 0.82}
        >>> ]
        >>> logger(results)
    """
    
    def __init__(self, filename: str, path: str = 'results'):
        """
        Initialize the CSV logger.
        
        Args:
            filename (str): Name of the output CSV file (without extension)
            path (str): Directory path for storing CSV files. Defaults to 'results'.
                
        Example:
            >>> logger = csvLogger('experiment_2024_01_15', path='./experiments/')
        """
        self.count = 1
        self.filename = filename
        self.path = path

        self.__check_path__()

    def __check_path__(self) -> None:
        """
        Check and create the output directory if it doesn't exist.
        
        This method ensures that the output directory exists before attempting
        to write CSV files. If the directory doesn't exist, it creates it.
        
        Example:
            >>> logger = csvLogger('test', path='./new_directory/')
            >>> # Directory './new_directory/' is automatically created
        """
        isExist = os.path.exists(self.path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.path)

    def __call__(self, named_dict: list[dict]) -> None:
        """
        Log experiment results to CSV file.
        
        This method takes a list of dictionaries (each representing one experiment
        result) and appends them to the CSV file. The method automatically handles
        DataFrame conversion and CSV writing.
        
        Args:
            named_dict (list[dict]): List of dictionaries containing experiment results.
                Each dictionary should have consistent keys across all entries.
                
        Example:
            >>> logger = csvLogger('experiment_results')
            >>> 
            >>> # Log single result
            >>> result = {'model': 'RandomForest', 'accuracy': 0.85, 'fairness': 0.92}
            >>> logger([result])
            >>> 
            >>> # Log multiple results
            >>> results = [
            >>>     {'model': 'RandomForest', 'accuracy': 0.85, 'fairness': 0.92},
            >>>     {'model': 'LogisticRegression', 'accuracy': 0.82, 'fairness': 0.89}
            >>> ]
            >>> logger(results)
        """
        df = pd.DataFrame(named_dict)
        df.to_csv(f"./{self.path}/{self.filename}.csv", mode='a')


def read_csv_file(file_path: str) -> pd.DataFrame:
    """
    Read a single CSV file and return its DataFrame.
    
    This function provides a robust way to read CSV files with comprehensive
    error handling. It's designed to work with the multiprocessing aggregation
    functionality.
    
    Args:
        file_path (str): Path to the CSV file to read
        
    Returns:
        pd.DataFrame: DataFrame containing the CSV data. Returns empty DataFrame
            if reading fails.
            
    Example:
        >>> df = read_csv_file('./results/experiment_1.csv')
        >>> print(f"Loaded {len(df)} rows from CSV file")
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {str(e)}")
        return pd.DataFrame()
    

def aggregate_csv_files(folder_path: str, output_file: str = 'aggregated_data.csv', 
                       num_processes: int = 10) -> None:
    """
    Aggregate multiple CSV files from a folder into a single CSV file using multiprocessing.
    
    This function efficiently combines multiple CSV files into a single file for
    analysis. It uses multiprocessing for improved performance on large datasets
    and includes comprehensive error handling and logging.
    
    The function:
    1. Finds all CSV files in the specified folder
    2. Reads them in parallel using multiprocessing
    3. Combines all DataFrames into a single DataFrame
    4. Saves the aggregated data to the output file
    5. Optionally deletes the original files after successful aggregation
    
    Args:
        folder_path (str): Path to the folder containing CSV files to aggregate
        output_file (str): Name of the output CSV file. Defaults to 'aggregated_data.csv'
        num_processes (int): Number of processes to use for parallel processing.
            Defaults to 10. Use None to use all available CPU cores.
            
    Raises:
        Exception: If aggregation fails due to file system or processing errors
        
    Example:
        >>> # Aggregate all CSV files in the results folder
        >>> aggregate_csv_files(
        >>>     folder_path='./results/',
        >>>     output_file='./results/aggregated_results.csv',
        >>>     num_processes=8
        >>> )
        >>> 
        >>> # Use all available CPU cores
        >>> aggregate_csv_files(
        >>>     folder_path='./experiments/',
        >>>     output_file='./experiments/all_results.csv',
        >>>     num_processes=None
        >>> )
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Get list of all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {folder_path}")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    # Create a pool of workers
    if num_processes is None:
        num_processes = os.cpu_count()
    
    try:
        # Read all CSV files in parallel
        with Pool(processes=num_processes) as pool:
            dataframes = pool.map(read_csv_file, csv_files)
        
        # Filter out empty DataFrames (from failed reads)
        dataframes = [df for df in dataframes if not df.empty]
        
        if not dataframes:
            logger.error("No valid data found in CSV files")
            return
        
        # Concatenate all DataFrames
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Remove duplicate rows
        # initial_rows = len(combined_df)
        # combined_df.drop_duplicates(inplace=True)
        # dropped_rows = initial_rows - len(combined_df)
        
        # Save the aggregated data
        combined_df.to_csv(output_file, index=False)
        
        logger.info(f"Successfully aggregated {len(csv_files)} files")
        # logger.info(f"Removed {dropped_rows} duplicate rows")
        logger.info(f"Final dataset has {len(combined_df)} rows")
        
        # Delete original CSV files
        for file_path in csv_files:
            try:
                os.remove(file_path)
                logger.debug(f"Deleted: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error during aggregation: {str(e)}")
        raise

if __name__ == '__main__':
    # Example usage
    folder_path = './results/'
    output_file = './results/aggregated_data.csv'
    aggregate_csv_files(folder_path, output_file)



