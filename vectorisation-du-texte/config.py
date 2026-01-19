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
