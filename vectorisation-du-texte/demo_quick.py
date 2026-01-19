"""
Script de démonstration rapide de la pipeline avec une petite portion de données
"""
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUT_DIR
from scripts.load_data import load_data
from scripts.lowercasing import apply_lowercasing
from scripts.stopwords_removal import remove_stopwords
from scripts.lemmatization import apply_lemmatization
from scripts.tfidf import apply_tfidf
from scripts.normalize import normalize_vectors

print("\n" + "=" * 80)
print("DÉMONSTRATION RAPIDE - Pipeline de vectorisation")
print("=" * 80)

# Charger les données
print("\n[1] Chargement des données...")
df = load_data()
print(f"✓ {len(df)} avis chargés")

# Prendre seulement 100 avis pour la démo
df = df.head(100)
print(f"✓ Limité à {len(df)} avis pour la démo")

# Lowercasing
print("\n[2] Lowercasing...")
df = apply_lowercasing(df, apply_lowercasing=True)
print("✓ Lowercasing appliqué")

# Stopwords
print("\n[3] Suppression des stopwords...")
df = remove_stopwords(df, apply_stopwords=True)
print("✓ Stopwords supprimés")

# Lemmatisation
print("\n[4] Lemmatisation...")
df = apply_lemmatization(df, apply_lemmatization=True)
print("✓ Lemmatisation appliquée")

# TF-IDF
print("\n[5] TF-IDF...")
X_tfidf, feature_names, vectorizer = apply_tfidf(df, ngram_range=(1, 1))
print(f"✓ Matrice TF-IDF créée: {X_tfidf.shape}")

# Normalisation
print("\n[6] Normalisation...")
X_normalized = normalize_vectors(X_tfidf, norm='l2')
print(f"✓ Vecteurs normalisés: {X_normalized.shape}")

print("\n" + "=" * 80)
print("✓ DÉMONSTRATION RÉUSSIE!")
print("=" * 80)
print(f"\nLe pipeline fonctionne correctement!")
print(f"- Données: {X_normalized.shape[0]} documents")
print(f"- Features: {X_normalized.shape[1]} n-grammes")
print(f"- Tous les fichiers ont été créés dans: {OUTPUT_DIR}")
print("\nPour lancer le pipeline complet (24 configurations):")
print("  python run_pipeline.py")
print("=" * 80 + "\n")
