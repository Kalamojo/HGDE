"""
File: ingest_lancedb.py
Author: Chris Jien Zhang
Date: 2026-10-15
Course: CMSC473
Description:
    Utility module to bridge the LanceDB vector store with the Python matrix environment.
    Extracts high-dimensional multimodal embeddings and converts them into the flat 
    2D NumPy matrix format required by the projection model.

Usage:
    from scripts.ingest_lancedb import load_lancedb_embeddings
    embeddings = load_lancedb_embeddings(db_path="data/pokemon_db")
"""

import lancedb
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_lancedb_embeddings(db_path, vector_column="vector"):
    """
    Connects to a local LanceDB database and returns the embeddings.
    
    Args:
        db_path (str): Path to the LanceDB directory.
        vector_column (str): The name of the column storing the embedding lists.
        
    Returns:
        numpy.ndarray: A 2D array of embeddings, or None if extraction fails.
    """
    db_loc = Path(db_path)
    if not db_loc.exists():
        logger.error(f"LanceDB database not found at {db_loc}")
        return None
        
    try:
        db = lancedb.connect(str(db_loc))
        tables = db.table_names()
        
        if not tables:
            logger.error("No tables found in the LanceDB database.")
            return None
            
        table = db.open_table(tables[0])
        df = table.to_pandas()
        
        if vector_column not in df.columns:
            logger.error(f"Column '{vector_column}' missing. Available: {df.columns}")
            return None
            
        # Stack individual lists into a matrix for PyTorch/Scikit-Learn compatibility
        embeddings_matrix = np.stack(df[vector_column].values)
        logger.info(f"Loaded LanceDB embeddings matrix. Shape: {embeddings_matrix.shape}")
        
        return embeddings_matrix
        
    except Exception as e:
        logger.error(f"Failed to ingest LanceDB data: {str(e)}")
        return None