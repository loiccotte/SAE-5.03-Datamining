"""
Script 04 : Lemmatisation (Ramener les mots à leur forme canonique)
Input: Dataframe avec colonne 'texte_no_stopwords'
Output: Dataframe avec colonne 'texte_lemmatized'
"""
# torch doit être préchargé EN PREMIER : pandas/numpy chargent des DLLs BLAS qui
# entrent en conflit avec c10.dll (torch) si torch est importé après eux (WinError 1114).
try:
    import torch  # noqa: F401
except (ImportError, OSError):
    pass  # stanza non installé ou DLL manquante → on continue avec spaCy uniquement

import pandas as pd
import logging
import re

logger = logging.getLogger(__name__)

# Variable globale pour le modèle spaCy
_nlp_model = None

# Variable globale pour le modèle Stanza (lazy loading)
_stanza_model = None


def get_stanza_model():
    """
    Importe et retourne le modèle Stanza français (lazy loading)
    """
    global _stanza_model

    if _stanza_model is None:
        import stanza
        stanza.download('fr', verbose=False)
        _stanza_model = stanza.Pipeline('fr', processors='tokenize,lemma', verbose=False)
        logger.info("Modèle Stanza français chargé")

    return _stanza_model


def lemmatize_text_stanza(text):
    """
    Lemmatise un texte en utilisant Stanza.
    Exclut les tokens dont upos est DET, PRON ou AUX — même logique que spaCy.
    """
    nlp = get_stanza_model()
    doc = nlp(text)
    excluded_upos = {'DET', 'PRON', 'AUX'}
    return ' '.join(
        word.lemma
        for sent in doc.sentences
        for word in sent.words
        if word.upos not in excluded_upos
    )


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
    Exclut également les déterminants, pronoms et auxiliaires
    """
    nlp = get_spacy_model()
    doc = nlp(text)
    
    # Exclure ponctuation, déterminants (DET), pronoms (PRON) et auxiliaires (AUX)
    excluded_pos = {'DET', 'PRON', 'AUX'}
    
    # Liste de mots à exclure même si mal taggués par spaCy
    excluded_lemmas = {
        'le', 'la', 'les', 'un', 'une', 'des', 'ce', 'cette', 'ces',
        'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses',
        'notre', 'nos', 'votre', 'vos', 'leur', 'leurs',
        'du', 'au', 'aux', 'de', 'à',
        'être', 'avoir', 'aller',
        'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles',
        'me', 'te', 'se', 'lui', 'leur', 'y', 'en'
    }
    
    lemmas = []
    for token in doc:
        if token.is_punct:
            continue
        if token.pos_ in excluded_pos:
            continue
        if token.lemma_.lower() in excluded_lemmas:
            continue
        lemmas.append(token.lemma_)
    
    return ' '.join(lemmas)


def apply_lemmatization(df, apply_lemmatization=True, remove_stopwords_after=False,
                        lemma_lib='spacy'):
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
        logger.info(f"Application de la lemmatisation ({lemma_lib})...")
        logger.info("Cette étape peut prendre du temps...")

        if lemma_lib == 'stanza':
            lemmatize_fn = lemmatize_text_stanza
        else:
            lemmatize_fn = lemmatize_text

        texts = df['texte_no_stopwords'].tolist()
        lemmatized_texts = []

        total = len(texts)
        for i, text in enumerate(texts):
            if i % 100 == 0:
                logger.info(f"Progression: {i}/{total}")

            lemmatized_texts.append(lemmatize_fn(text))
        
        df['texte_lemmatized'] = lemmatized_texts
        
        if remove_stopwords_after:
            logger.info("Suppression des stopwords après lemmatisation...")
            try:
                from nltk.corpus import stopwords
                stopwords.words('french')
                stopwords_fr = set(stopwords.words('french'))
            except LookupError:
                logger.info("Téléchargement des ressources NLTK...")
                import nltk
                nltk.download('stopwords')
                from nltk.corpus import stopwords
                stopwords_fr = set(stopwords.words('french'))

            def remove_stopwords_lem(text):
                tokens = re.findall(r"\b\w+\b", text)
                tokens_filtered = [
                    tok for tok in tokens
                    if tok.lower() not in stopwords_fr and len(tok) > 1
                ]
                return ' '.join(tokens_filtered)

            df['texte_lemmatized'] = df['texte_lemmatized'].apply(remove_stopwords_lem)
            logger.info("Stopwords supprimés après lemmatisation")
        logger.info("Lemmatisation appliquée")
    else:
        logger.info("Lemmatisation désactivée, copie du texte original...")
        df['texte_lemmatized'] = df['texte_no_stopwords']
    
    logger.info(f"Exemple avant: {df['texte_no_stopwords'].iloc[0][:80]}")
    logger.info(f"Exemple après: {df['texte_lemmatized'].iloc[0][:80]}")
    
    return df
