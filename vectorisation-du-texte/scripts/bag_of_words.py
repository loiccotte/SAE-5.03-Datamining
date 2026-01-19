"""
Script 05 : Bag of Words avec N-grammes
Input: Dataframe avec colonne 'texte_lemmatized'
Output: Dictionnaire des n-grammes et vecteur de comptage
"""
import pandas as pd
import logging
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)


def create_bag_of_words(df, ngram_range=(1, 1), min_df=2, max_df=0.8):
    """
    Crée un vecteur Bag of Words avec n-grammes
    
    Parameters:
    -----------
    df : DataFrame
        Dataframe avec colonne 'texte_lemmatized'
    ngram_range : tuple
        Range des n-grammes, ex: (1, 1) pour unigrammes, (1, 2) pour uni+bigrammes
    min_df : int
        Fréquence minimale d'un terme pour être inclus
    max_df : float
        Ratio maximal de documents où un terme peut apparaître
    
    Returns:
    --------
    X : scipy sparse matrix
        Matrice de comptage des n-grammes
    feature_names : list
        Noms des n-grammes (vocabulaire)
    vectorizer : CountVectorizer
        L'objet vectoriseur (pour futur usage)
    """
    logger.info(f"Création du Bag of Words avec n-grammes {ngram_range}...")
    
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        lowercase=False,
        token_pattern=r'\b\w+\b'
    )
    
    X = vectorizer.fit_transform(df['texte_lemmatized'])
    feature_names = vectorizer.get_feature_names_out().tolist()
    
    logger.info(f"Vocabulaire créé: {len(feature_names)} n-grammes uniques")
    logger.info(f"Matrice creuse: {X.shape}")
    logger.info(f"Densité: {X.nnz / (X.shape[0] * X.shape[1]):.4f}")
    
    term_frequency = X.sum(axis=0).A1
    top_indices = term_frequency.argsort()[-10:][::-1]
    logger.info("Top 10 n-grammes les plus courants:")
    for idx in top_indices:
        logger.info(f"  - {feature_names[idx]}: {term_frequency[idx]}")
    
    return X, feature_names, vectorizer
