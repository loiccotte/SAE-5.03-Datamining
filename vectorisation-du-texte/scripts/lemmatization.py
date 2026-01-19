"""
Script 04 : Lemmatisation (Ramener les mots à leur forme canonique)
Input: Dataframe avec colonne 'texte_no_stopwords'
Output: Dataframe avec colonne 'texte_lemmatized'
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Variable globale pour le modèle spaCy
_nlp_model = None


def get_spacy_model():
    """
    Importe et retourne le modèle spaCy français
    Utilise lazy loading pour éviter les erreurs au démarrage
    """
    global _nlp_model
    
    if _nlp_model is None:
        try:
            import spacy
            _nlp_model = spacy.load('fr_core_news_sm')
            logger.info("Modèle spacy français chargé")
        except OSError:
            logger.warning("Modèle spacy français non trouvé. Téléchargement en cours...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "fr_core_news_sm"])
            import spacy
            _nlp_model = spacy.load('fr_core_news_sm')
            logger.info("Modèle spacy français téléchargé et chargé")
    
    return _nlp_model


def lemmatize_text(text):
    """
    Lemmatise un texte en utilisant spacy
    """
    nlp = get_spacy_model()
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_punct]
    return ' '.join(lemmas)


def apply_lemmatization(df, apply_lemmatization=True):
    """
    Applique la lemmatisation au texte
    
    Parameters:
    -----------
    df : DataFrame
        Dataframe avec colonne 'texte_no_stopwords'
    apply_lemmatization : bool
        Si True, applique la lemmatisation. Si False, garde le texte original
    
    Returns:
    --------
    df : DataFrame
        Dataframe avec colonne 'texte_lemmatized'
    """
    df = df.copy()
    
    if apply_lemmatization:
        logger.info("Application de la lemmatisation...")
        logger.info("Cette étape peut prendre du temps...")
        
        texts = df['texte_no_stopwords'].tolist()
        lemmatized_texts = []
        
        total = len(texts)
        for i, text in enumerate(texts):
            if i % 100 == 0:
                logger.info(f"Progression: {i}/{total}")
            
            lemmatized_texts.append(lemmatize_text(text))
        
        df['texte_lemmatized'] = lemmatized_texts
        logger.info("Lemmatisation appliquée")
    else:
        logger.info("Lemmatisation désactivée, copie du texte original...")
        df['texte_lemmatized'] = df['texte_no_stopwords']
    
    logger.info(f"Exemple avant: {df['texte_no_stopwords'].iloc[0][:80]}")
    logger.info(f"Exemple après: {df['texte_lemmatized'].iloc[0][:80]}")
    
    return df
