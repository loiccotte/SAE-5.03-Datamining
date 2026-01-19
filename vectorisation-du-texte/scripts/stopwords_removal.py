"""
Script 03 : Suppression des mots vides (Stopwords)
Input: Dataframe avec colonne 'texte_lowercased'
Output: Dataframe avec colonne 'texte_no_stopwords'
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def get_stopwords():
    """
    Importe et retourne les stopwords français NLTK
    """
    try:
        from nltk.corpus import stopwords
        stopwords.words('french')
        return set(stopwords.words('french'))
    except LookupError:
        logger.info("Téléchargement des ressources NLTK...")
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        return set(stopwords.words('french'))


def remove_stopwords(df, apply_stopwords=True):
    """
    Supprime les mots vides (stopwords) du texte
    
    Parameters:
    -----------
    df : DataFrame
        Dataframe avec colonne 'texte_lowercased'
    apply_stopwords : bool
        Si True, supprime les stopwords. Si False, garde le texte original
    
    Returns:
    --------
    df : DataFrame
        Dataframe avec colonne 'texte_no_stopwords'
    """
    df = df.copy()
    
    if apply_stopwords:
        logger.info("Suppression des stopwords français...")
        
        stopwords_fr = get_stopwords()
        logger.info(f"Nombre de stopwords chargés: {len(stopwords_fr)}")
        
        def remove_stop(text):
            words = text.split()
            words_filtered = [w for w in words if w not in stopwords_fr and len(w) > 1]
            return ' '.join(words_filtered)
        
        df['texte_no_stopwords'] = df['texte_lowercased'].apply(remove_stop)
        logger.info("Stopwords supprimés")
    else:
        logger.info("Suppression des stopwords désactivée, copie du texte original...")
        df['texte_no_stopwords'] = df['texte_lowercased']
    
    # Statistiques
    avg_words_before = df['texte_lowercased'].apply(lambda x: len(x.split())).mean()
    avg_words_after = df['texte_no_stopwords'].apply(lambda x: len(x.split())).mean()
    
    logger.info(f"Moyenne de mots avant: {avg_words_before:.2f}")
    logger.info(f"Moyenne de mots après: {avg_words_after:.2f}")
    logger.info(f"Exemple avant: {df['texte_lowercased'].iloc[0][:80]}")
    logger.info(f"Exemple après: {df['texte_no_stopwords'].iloc[0][:80]}")
    
    return df
