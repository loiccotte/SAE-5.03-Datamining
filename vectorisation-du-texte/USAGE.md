"""
Utilisation rapide du pipeline de vectorisation
"""

# ============================================================================
# DÉMARRAGE RAPIDE
# ============================================================================

# 1. Se placer dans le répertoire
cd vectorisation-du-texte

# 2. Vérifier l'installation
python test_installation.py

# 3. Lancer le pipeline complet (24 configurations)
python run_pipeline.py

# ============================================================================
# RÉSULTATS
# ============================================================================

# Les résultats sont sauvegardés dans output/
# - config_L*_S*_LEM*_NG*_FINAL.pkl (24 fichiers)
# - status.json (suivi de progression)
# - pipeline.log (logs détaillés)

# ============================================================================
# CHARGER UN RÉSULTAT
# ============================================================================

import pickle

# Charger une configuration spécifique
with open('output/config_L1_S1_LEM1_NG1_FINAL.pkl', 'rb') as f:
    data = pickle.load(f)

# Accéder aux résultats
X_normalized = data['X_normalized']      # Matrice (n_samples, n_features)
features = data['feature_names']         # Vocabulaire
y = data['target']                       # Labels (positif/négatif)
config = data['config']                  # Configuration utilisée

# ============================================================================
# COMPRENDRE LE NOMMAGE
# ============================================================================

# config_L1_S1_LEM1_NG1
#         └─ L1   = Lowercasing activé (1) ou désactivé (0)
#            └─ S1   = Stopwords supprimés (1) ou non (0)
#               └─ LEM1 = Lemmatisation activée (1) ou non (0)
#                  └─ NG1  = N-grammes: 1=unigrammes, 2=uni+bigrammes, 3=uni+bi+trigrammes

# Exemples:
# config_L1_S1_LEM1_NG1 = Lowercasing + Stopwords + Lemmatisation + Unigrammes
# config_L0_S0_LEM0_NG3 = Aucun prétraitement + Uni/Bi/Trigrammes
# config_L1_S0_LEM1_NG2 = Lowercasing + Lemmatisation + Uni/Bigrammes

# ============================================================================
# COMBINAISONS RECOMMANDÉES
# ============================================================================

# Pour une classification simple:
#   config_L1_S1_LEM1_NG1  ✓ Complet, équilibré

# Pour une classification conservatrice:
#   config_L1_S1_LEM0_NG1  = Moins de perte d'information qu'avec lemmatisation

# Pour une vectorisation simple:
#   config_L1_S0_LEM0_NG1  = Juste lowercasing + unigrammes

# Pour plus de contexte:
#   config_L1_S1_LEM1_NG2  = Inclut les bigrammes (relations entre mots)

# ============================================================================
# REPRISE APRÈS INTERRUPTION
# ============================================================================

# Si le pipeline s'arrête prématurément, simplement relancer:
# python run_pipeline.py

# Le script détecte automatiquement les configurations déjà traitées
# et ne retraite que celles manquantes.

# ============================================================================
# STRUCTURE DU RÉPERTOIRE
# ============================================================================

vectorisation-du-texte/
├── scripts/
│   ├── 01_load_data.py              # Charger les données
│   ├── 02_lowercasing.py            # Minuscules
│   ├── 03_stopwords_removal.py      # Mots vides
│   ├── 04_lemmatization.py          # Lemmatisation
│   ├── 05_bag_of_words.py           # N-grammes
│   ├── 06_tfidf.py                  # TF-IDF
│   ├── 07_normalize.py              # Normalisation
│   └── import_*.py                  # Imports
├── output/
│   ├── config_*_FINAL.pkl           # 24 fichiers résultats
│   ├── status.json                  # Statut
│   └── pipeline.log                 # Logs
├── config.py                        # Configuration
├── utils.py                         # Utilitaires
├── run_pipeline.py                  # SCRIPT PRINCIPAL
├── test_installation.py             # Test d'installation
└── README.md                        # Documentation

# ============================================================================
# PARAMÈTRES CONFIGURABLES
# ============================================================================

# Modifier config.py pour:
# - MIN_DOC_FREQ: Fréquence minimale (défaut: 2)
# - MAX_DF_RATIO: Ratio maximal (défaut: 0.8)
# - LANGUAGE: Langue (défaut: "french")

# Exemple:
# from config import MIN_DOC_FREQ
# # Augmenter MIN_DOC_FREQ pour moins de features:
# # MIN_DOC_FREQ = 5

# ============================================================================
# PERFORMANCE
# ============================================================================

# Configuration sans lemmatisation (ex: config_L1_S1_LEM0_NG1):
#   ~30 secondes par configuration

# Configuration avec lemmatisation (ex: config_L1_S1_LEM1_NG1):
#   ~2-5 minutes par configuration (dépend du texte)

# Total pour 24 configurations:
#   ~30-60 minutes (si toutes avec lemmatisation)
#   Plus rapide sans lemmatisation

# Checkpoints intermédiaires permettent une reprise immédiate

# ============================================================================
# STATISTIQUES DES RÉSULTATS
# ============================================================================

# Exemple pour config_L1_S1_LEM1_NG1:
# - Nombre de documents: ~900
# - Nombre de features (unigrammes): ~800
# - Nombre de features (tous n-grammes): ~2000
# - Matrice: (900, 2000) = 1.8M valeurs
# - Densité: ~0.5% (très creuse)

# ============================================================================
