"""
Script de test rapide pour valider l'installation et les imports
"""
import sys
from pathlib import Path

print("=" * 80)
print("TEST D'INSTALLATION - VECTORISATION DU TEXTE")
print("=" * 80)

# Test 1: Imports pandas, numpy, scipy
print("\n[1/5] Test des imports de base...")
try:
    import pandas as pd
    import numpy as np
    from scipy import sparse
    print("✓ pandas, numpy, scipy OK")
except ImportError as e:
    print(f"✗ Erreur: {e}")
    sys.exit(1)

# Test 2: Imports sklearn
print("[2/5] Test de scikit-learn...")
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.preprocessing import normalize
    print("✓ scikit-learn OK")
except ImportError as e:
    print(f"✗ Erreur: {e}")
    sys.exit(1)

# Test 3: Imports NLTK
print("[3/5] Test de NLTK...")
try:
    import nltk
    from nltk.corpus import stopwords
    stopwords.words('french')
    print("✓ NLTK OK")
except LookupError:
    print("⚠ Téléchargement de ressources NLTK...")
    nltk.download('stopwords')
    print("✓ NLTK OK")
except ImportError as e:
    print(f"✗ Erreur: {e}")
    sys.exit(1)

# Test 4: Imports spaCy
print("[4/5] Test de spaCy...")
try:
    import spacy
    try:
        nlp = spacy.load('fr_core_news_sm')
        print("✓ spaCy OK (modèle français)")
    except OSError:
        print("⚠ Modèle français non trouvé. Téléchargement...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "fr_core_news_sm"])
        nlp = spacy.load('fr_core_news_sm')
        print("✓ spaCy OK (modèle français téléchargé)")
except ImportError as e:
    print(f"✗ Erreur: {e}")
    sys.exit(1)

# Test 5: Fichiers de configuration
print("[5/5] Test des fichiers de configuration...")
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from config import INPUT_FILE, OUTPUT_DIR, NGRAM_OPTIONS
    from utils import get_config_name
    
    print(f"✓ Configuration OK")
    print(f"  - Fichier d'entrée: {INPUT_FILE}")
    print(f"  - Répertoire de sortie: {OUTPUT_DIR}")
    print(f"  - N-grammes: {NGRAM_OPTIONS}")
    
    # Test la génération de noms de config
    name = get_config_name(True, True, True, 1)
    print(f"  - Exemple de nom: {name}")
    
except Exception as e:
    print(f"✗ Erreur: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ TOUS LES TESTS RÉUSSIS!")
print("=" * 80)
print("\nVous pouvez maintenant lancer le pipeline:")
print("  python run_pipeline.py")
print("=" * 80)
