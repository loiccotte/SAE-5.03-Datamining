"""
Configuration globale pour la pipeline de vectorisation
"""
import os
from pathlib import Path

# Répertoires
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent.parent
INPUT_FILE = DATA_DIR / "avis_annotés.csv"
OUTPUT_DIR = BASE_DIR / "output"
SCRIPTS_DIR = BASE_DIR / "scripts"

# Créer le répertoire output s'il n'existe pas
OUTPUT_DIR.mkdir(exist_ok=True)

# Paramètres de traitement
LOWERCASING_OPTIONS = [True, False]
STOPWORDS_OPTIONS = [True, False]
LEMMATIZATION_OPTIONS = [True, False]
NGRAM_OPTIONS = [1, 2, 3]

# Colonnes du dataset
TEXT_COLUMNS = ["corps", "titre"]  # Les colonnes textuelles à traiter
TARGET_COLUMN = "avis"  # La colonne cible (sentiment)

# Paramètres de preprocessing
MIN_DOC_FREQ = 2  # Fréquence minimale d'apparition d'un mot
MAX_DF_RATIO = 0.8  # Ratio max de documents contenant le mot

# Langue pour le traitement NLP
LANGUAGE = "french"

# Logging
LOG_FILE = OUTPUT_DIR / "pipeline.log"

# ─────────────────────────────────────────────────────────────
# Nouvelles dimensions — pipeline étendue (feature/library-comparison)
# ─────────────────────────────────────────────────────────────

# SW_MODE : mode de suppression des stopwords
#   'none'    → S0 : aucune suppression
#   'partial' → S2 : suppression partielle (neutres seulement, préserve négation/intensité)
#   'all'     → S1 : suppression totale (comportement historique)
SW_MODE_OPTIONS = ['none', 'partial', 'all']

# Bibliothèques de lemmatisation testées
LEMMA_LIB_OPTIONS = ['spacy', 'stanza']

# Bibliothèques de vectorisation testées
VECT_LIB_OPTIONS = ['tfidf', 'bm25']

# Mots préservés en mode S2 (modifiable librement ici)
# Ces mots portent du sens en analyse de sentiment et ne doivent PAS être supprimés.
SENTIMENT_PRESERVED_WORDS = {
    # Marqueurs de négation
    'ne', 'pas', 'plus', 'jamais', 'aucun', 'aucune',
    'sans', 'ni', 'non', 'guère', 'nullement', 'peu', 'rarement',
    # Marqueurs d'intensité
    'très', 'trop', 'assez', 'vraiment', 'totalement', 'absolument',
    'complètement', 'parfaitement', 'fort', 'bien',
    # Marqueurs restrictifs
    'seulement', 'juste', 'uniquement',
}
