"""
Script 07 : Normalisation L2 des vecteurs
Normalise la longueur mathématique de chaque vecteur à 1
Input: Matrice TF-IDF
Output: Matrice TF-IDF normalisée
"""
import pandas as pd
import logging
from sklearn.preprocessing import normalize
import numpy as np

logger = logging.getLogger(__name__)


def normalize_vectors(X_tfidf, norm='l2'):
    """
    Normalise les vecteurs TF-IDF à une longueur de 1
    
    Parameters:
    -----------
    X_tfidf : scipy sparse matrix
        Matrice TF-IDF non normalisée
    norm : str
        Type de normalisation ('l2' ou 'l1')
    
    Returns:
    --------
    X_normalized : scipy sparse matrix
        Matrice normalisée
    """
    logger.info(f"Normalisation des vecteurs (norme {norm})...")
    
    X_normalized = normalize(X_tfidf, norm=norm, axis=1)
    
    logger.info(f"Matrice normalisée: {X_normalized.shape}")
    logger.info(f"Densité: {X_normalized.nnz / (X_normalized.shape[0] * X_normalized.shape[1]):.4f}")
    
    norms = np.array(X_normalized.multiply(X_normalized).sum(axis=1)).flatten()
    
    if norm == 'l2':
        norms = np.sqrt(norms)
    
    logger.info(f"Normes L2 - Min: {norms.min():.6f}, Max: {norms.max():.6f}, Moyenne: {norms.mean():.6f}")
    logger.info(f"Tous les vecteurs ont norme 1: {np.allclose(norms, 1.0)}")
    
    logger.info(f"Valeurs normalisées - Min: {X_normalized.min():.6f}, Max: {X_normalized.max():.6f}")
    
    return X_normalized
