"""
Script 06 : Pondération TF-IDF (Term Frequency - Inverse Document Frequency)
Input: Matrice Bag of Words
Output: Matrice TF-IDF
"""
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def apply_tfidf(df, ngram_range=(1, 1), min_df=2, max_df=0.8):
    """
    Applique la pondération TF-IDF au texte
    
    Parameters:
    -----------
    df : DataFrame
        Dataframe avec colonne 'texte_lemmatized'
    ngram_range : tuple
        Range des n-grammes
    min_df : int
        Fréquence minimale d'un terme
    max_df : float
        Ratio maximal de documents
    
    Returns:
    --------
    X_tfidf : scipy sparse matrix
        Matrice TF-IDF
    feature_names : list
        Noms des n-grammes
    tfidf_vectorizer : TfidfVectorizer
        L'objet vectoriseur
    """
    logger.info(f"Application de TF-IDF avec n-grammes {ngram_range}...")
    
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        lowercase=False,
        token_pattern=r'\b\w+\b',
        sublinear_tf=True
    )
    
    X_tfidf = tfidf_vectorizer.fit_transform(df['texte_lemmatized'])
    feature_names = tfidf_vectorizer.get_feature_names_out().tolist()
    
    logger.info(f"Vocabulaire TF-IDF créé: {len(feature_names)} n-grammes uniques")
    logger.info(f"Matrice TF-IDF: {X_tfidf.shape}")
    logger.info(f"Densité: {X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]):.4f}")
    
    mean_tfidf = X_tfidf.mean()
    max_tfidf = X_tfidf.max()
    
    logger.info(f"Valeurs TF-IDF - Min: 0.0, Max: {max_tfidf:.4f}, Moyenne: {mean_tfidf:.4f}")
    
    tfidf_scores = X_tfidf.mean(axis=0).A1
    top_indices = tfidf_scores.argsort()[-10:][::-1]
    logger.info("Top 10 n-grammes par score TF-IDF moyen:")
    for idx in top_indices:
        logger.info(f"  - {feature_names[idx]}: {tfidf_scores[idx]:.4f}")
    
    return X_tfidf, feature_names, tfidf_vectorizer
