"""
Script 06 : Pondération TF-IDF (Term Frequency - Inverse Document Frequency)
Input: Matrice Bag of Words
Output: Matrice TF-IDF
"""
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from scripts.bm25 import BM25Vectorizer

logger = logging.getLogger(__name__)


def apply_tfidf(df, ngram_range=(1, 1), min_df=2, max_df=0.8, vect_lib='tfidf'):
    """
    Applique la pondération TF-IDF ou BM25 au texte

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
    vect_lib : str
        Bibliothèque de vectorisation : 'tfidf' ou 'bm25'

    Returns:
    --------
    X_tfidf : scipy sparse matrix
        Matrice de pondération
    feature_names : list
        Noms des n-grammes
    tfidf_vectorizer : TfidfVectorizer or BM25Vectorizer
        L'objet vectoriseur
    """
    cv_kwargs = dict(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        lowercase=False,
        token_pattern=r'\b\w+\b',
    )

    if vect_lib == 'bm25':
        logger.info(f"Application de BM25 avec n-grammes {ngram_range}...")
        tfidf_vectorizer = BM25Vectorizer(**cv_kwargs)
    else:
        logger.info(f"Application de TF-IDF avec n-grammes {ngram_range}...")
        tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, **cv_kwargs)

    X_tfidf = tfidf_vectorizer.fit_transform(df['texte_lemmatized'])
    feature_names = tfidf_vectorizer.get_feature_names_out().tolist()

    logger.info(f"Vocabulaire créé: {len(feature_names)} n-grammes uniques")
    logger.info(f"Matrice: {X_tfidf.shape}")
    logger.info(f"Densité: {X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]):.4f}")

    mean_val = X_tfidf.mean()
    max_val = X_tfidf.max()
    logger.info(f"Valeurs - Min: 0.0, Max: {max_val:.4f}, Moyenne: {mean_val:.4f}")

    scores = X_tfidf.mean(axis=0).A1
    top_indices = scores.argsort()[-10:][::-1]
    logger.info("Top 10 n-grammes par score moyen:")
    for idx in top_indices:
        logger.info(f"  - {feature_names[idx]}: {scores[idx]:.4f}")

    return X_tfidf, feature_names, tfidf_vectorizer
