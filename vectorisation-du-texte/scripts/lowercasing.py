"""
Script 02 : Lowercasing (Transformation en minuscules)
Input: Dataframe avec colonne 'texte_complet'
Output: Dataframe avec colonne 'texte_lowercased'
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def apply_lowercasing(df, apply_lowercasing=True):
    """
    Transforme tous les caractères en minuscules
    
    Parameters:
    -----------
    df : DataFrame
        Dataframe avec colonne 'texte_complet'
    apply_lowercasing : bool
        Si True, applique le lowercasing. Si False, garde le texte original
    
    Returns:
    --------
    df : DataFrame
        Dataframe avec colonne 'texte_lowercased'
    """
    df = df.copy()
    
    if apply_lowercasing:
        logger.info("Application du lowercasing...")
        df['texte_lowercased'] = df['texte_complet'].str.lower()
        logger.info("Lowercasing appliqué")
    else:
        logger.info("Lowercasing désactivé, copie du texte original...")
        df['texte_lowercased'] = df['texte_complet']
    
    logger.info(f"Exemple avant: {df['texte_complet'].iloc[0][:80]}")
    logger.info(f"Exemple après: {df['texte_lowercased'].iloc[0][:80]}")
    
    return df
