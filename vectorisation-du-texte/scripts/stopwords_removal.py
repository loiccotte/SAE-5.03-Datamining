"""
Script 03 : Suppression des mots vides (Stopwords)
Input: Dataframe avec colonne 'texte_lowercased'
Output: Dataframe avec colonne 'texte_no_stopwords'
"""
import pandas as pd
import logging
import re

logger = logging.getLogger(__name__)


def get_stopwords(language='french'):
    """
    Importe et retourne les stopwords NLTK pour la langue donnée
    """
    try:
        from nltk.corpus import stopwords
        stopwords.words(language)
        return set(stopwords.words(language))
    except LookupError:
        logger.info("Téléchargement des ressources NLTK...")
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        return set(stopwords.words(language))


def remove_stopwords(df, text_col='texte_lowercased', output_col='texte_no_stopwords',
                     apply_stopwords=True, sw_mode='all', preserved_words=None,
                     language='french'):
    """
    Supprime les mots vides (stopwords) du texte

    Parameters:
    -----------
    df : DataFrame
        Dataframe avec colonne texte source
    text_col : str
        Nom de la colonne texte d'entrée (défaut: 'texte_lowercased')
    output_col : str
        Nom de la colonne texte de sortie (défaut: 'texte_no_stopwords')
    apply_stopwords : bool
        Rétro-compatibilité : si False, équivaut à sw_mode='none'
    sw_mode : str
        Mode de suppression : 'none' (S0), 'all' (S1), 'partial' (S2)
    preserved_words : set or None
        Mots à conserver en mode 'partial' (ex: marqueurs de négation/intensité)
    language : str
        Langue des stopwords NLTK (défaut: 'french')

    Returns:
    --------
    df : DataFrame
        Dataframe avec colonne texte_no_stopwords
    """
    df = df.copy()

    # Rétro-compatibilité : apply_stopwords=False → sw_mode='none'
    effective_mode = sw_mode if apply_stopwords else 'none'

    if effective_mode == 'none':
        logger.info("Suppression des stopwords désactivée (S0), copie du texte original...")
        df[output_col] = df[text_col]
    elif effective_mode == 'all':
        logger.info("Suppression totale des stopwords français (S1)...")
        stopwords_fr = get_stopwords(language)
        logger.info(f"Nombre de stopwords chargés: {len(stopwords_fr)}")

        def remove_stop_all(text):
            tokens = re.findall(r"\b\w+\b", text)
            tokens_filtered = [
                tok for tok in tokens
                if tok.lower() not in stopwords_fr and len(tok) > 1
            ]
            return ' '.join(tokens_filtered)

        df[output_col] = df[text_col].apply(remove_stop_all)
        logger.info("Stopwords supprimés (mode S1)")
    elif effective_mode == 'partial':
        logger.info("Suppression partielle des stopwords (S2) — préservation négation/intensité...")
        stopwords_fr = get_stopwords(language)
        preserved = set(preserved_words) if preserved_words else set()
        # Construire l'ensemble actif : NLTK stopwords MINUS mots préservés
        active_stopwords = stopwords_fr - {w.lower() for w in preserved}
        logger.info(f"Stopwords actifs: {len(active_stopwords)} (sur {len(stopwords_fr)} NLTK, "
                    f"{len(preserved)} préservés)")

        def remove_stop_partial(text):
            tokens = re.findall(r"\b\w+\b", text)
            tokens_filtered = [
                tok for tok in tokens
                if tok.lower() not in active_stopwords and len(tok) > 1
            ]
            return ' '.join(tokens_filtered)

        df[output_col] = df[text_col].apply(remove_stop_partial)
        logger.info("Stopwords supprimés (mode S2)")
    else:
        logger.warning(f"sw_mode inconnu '{effective_mode}', copie du texte original...")
        df[output_col] = df[text_col]

    # Statistiques
    avg_words_before = df[text_col].apply(lambda x: len(x.split())).mean()
    avg_words_after = df[output_col].apply(lambda x: len(x.split())).mean()

    logger.info(f"Moyenne de mots avant: {avg_words_before:.2f}")
    logger.info(f"Moyenne de mots après: {avg_words_after:.2f}")
    logger.info(f"Exemple avant: {df[text_col].iloc[0][:80]}")
    logger.info(f"Exemple après: {df[output_col].iloc[0][:80]}")

    return df
