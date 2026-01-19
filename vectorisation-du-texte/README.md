# Vectorisation du Texte

## Vue d'ensemble

Cet espace contient tous les scripts nécessaires pour vectoriser les avis textuels en matrices numériques exploitables par les modèles de classification.

## Structure

```
vectorisation-du-texte/
├── scripts/
│   ├── 01_load_data.py              # Charger les données CSV
│   ├── 02_lowercasing.py            # Transformer en minuscules
│   ├── 03_stopwords_removal.py      # Supprimer les mots vides
│   ├── 04_lemmatization.py          # Lemmatiser
│   ├── 05_bag_of_words.py           # Créer n-grammes
│   ├── 06_tfidf.py                  # Appliquer TF-IDF
│   ├── 07_normalize.py              # Normaliser les vecteurs
│   └── import_*.py                  # Fichiers d'import helper
├── output/                          # Dossier de résultats
├── config.py                        # Configuration globale
├── utils.py                         # Utilitaires et checkpoint
├── run_pipeline.py                  # Script orchestrateur (MAIN)
└── README.md                        # Ce fichier
```

## Étapes de prétraitement

### 1. **Chargement des données** (`01_load_data.py`)
- Lit le CSV `avis_annotés.csv`
- Fusionne les colonnes "titre" et "corps"
- Supprime les lignes vides

### 2. **Lowercasing** (`02_lowercasing.py`)
- Transforme tout le texte en minuscules
- Option: activé/désactivé

### 3. **Suppression des stopwords** (`03_stopwords_removal.py`)
- Supprime les mots grammaticaux très courants
- Utilise la liste NLTK française
- Option: activé/désactivé

### 4. **Lemmatisation** (`04_lemmatization.py`)
- Ramène les mots à leur forme canonique (racine)
- Utilise spaCy avec le modèle français `fr_core_news_sm`
- Option: activé/désactivé
- ⚠️ Étape la plus lente

### 5. **Bag of Words + N-grammes** (`05_bag_of_words.py`)
- Crée le vocabulaire avec n-grammes
- N peut être 1 (unigrammes), 2 (bigrammes), 3 (trigrammes)
- Utilise `CountVectorizer` de scikit-learn

### 6. **TF-IDF** (`06_tfidf.py`)
- Applique la pondération TF-IDF
- Réduit l'importance des mots courants
- Utilise `TfidfVectorizer` de scikit-learn

### 7. **Normalisation** (`07_normalize.py`)
- Normalise les vecteurs à une longueur L2 = 1
- Rend les vecteurs comparables en distance

## Combinaisons

**Total de combinaisons possibles: 24**

```
Lowercasing:   Non (0) / Oui (1)                    = 2 options
Stopwords:     Non (0) / Oui (1)                    = 2 options
Lemmatisation: Non (0) / Oui (1)                    = 2 options
N-grammes:     1 / 2 / 3                            = 3 options

Total: 2 × 2 × 2 × 3 = 24 combinaisons
```

## Nommage des configurations

Format: `config_L{0|1}_S{0|1}_LEM{0|1}_NG{1|2|3}`

Exemples:
- `config_L1_S1_LEM1_NG1` : Lowercasing + Stopwords + Lemmatisation + Unigrammes
- `config_L0_S0_LEM0_NG3` : Aucun prétraitement + Trigrammes
- `config_L1_S1_LEM0_NG2` : Lowercasing + Stopwords + Uni/Bigrammes

## Utilisation

### Lancer le pipeline complet

```bash
# Depuis le répertoire vectorisation-du-texte
cd vectorisation-du-texte

# Activer l'environnement (si pas déjà fait)
..\..\.venv\Scripts\Activate.ps1

# Lancer le pipeline
python run_pipeline.py
```

### Résultat

- ✓ Toutes les 24 configurations sont traitées
- ✓ Chaque configuration crée un fichier `config_*_FINAL.pkl` dans `output/`
- ✓ Les checkpoints intermédiaires permettent la reprise en cas d'interruption
- ✓ Un fichier `status.json` suit la progression
- ✓ Logs sauvegardés dans `output/pipeline.log`

### Reprendre après une interruption

```bash
python run_pipeline.py
```

Le script détecte automatiquement les configurations complétées et reprend à partir de celles non traitées.

## Format des fichiers de sortie

### Fichier final (`config_*_FINAL.pkl`)

Contient un dictionnaire pickl\u00e9 avec:

```python
{
    'X_normalized': <scipy sparse matrix>,  # Matrice finale normalisée
    'feature_names': [list],                # Noms des n-grammes
    'tfidf_vectorizer': <TfidfVectorizer>,  # Objet vectoriseur
    'df': <DataFrame>,                      # Données originales traitées
    'target': <numpy array>,                # Labels (positif/négatif)
    'config': <dict>,                       # Configuration utilisée
    'shape': <tuple>,                       # Forme de la matrice (n_docs, n_features)
    'n_features': <int>,                    # Nombre de features
    'timestamp': <str>                      # Timestamp de création
}
```

### Charger un résultat

```python
import pickle

with open('output/config_L1_S1_LEM1_NG1_FINAL.pkl', 'rb') as f:
    result = pickle.load(f)

X = result['X_normalized']          # Matrice vectorisée
features = result['feature_names']  # Vocabulaire
y = result['target']                # Labels
```

## Configuration

Modifier [config.py](config.py) pour ajuster:

- `MIN_DOC_FREQ`: Fréquence minimale d'un mot (défaut: 2)
- `MAX_DF_RATIO`: Ratio maximal de documents (défaut: 0.8)
- `LANGUAGE`: Langue pour NLP (défaut: "french")

## Troubleshooting

### Erreur : "ModuleNotFoundError: No module named 'spacy'"

```bash
pip install spacy
python -m spacy download fr_core_news_sm
```

### Erreur : "ModuleNotFoundError: No module named 'nltk'"

```bash
pip install nltk
python -c "import nltk; nltk.download('stopwords')"
```

### Le script est très lent

La lemmatisation (étape 4) est la plus longue. C'est normal.
- Les configurations sans lemmatisation sont plus rapides
- Le checkpointing permet de reprendre sans relancer

### Erreur lors du chargement du CSV

Vérifier que le fichier `avis_annotés.csv` existe dans le répertoire parent.

## Résultats attendus

| Configuration | Unigrammes | Bigrammes | Trigrammes |
|---------------|-----------|-----------|-----------|
| Sans prétraitement | ~2000 features | ~5000 features | ~8000 features |
| Avec lowercasing + stopwords | ~1200 features | ~3000 features | ~5000 features |
| Complet (LC+SW+LEM) | ~900 features | ~2200 features | ~3500 features |

*Nombres approximatifs, varient selon les données*

## Prochaines étapes

Les fichiers `FINAL.pkl` peuvent être utilisés par:
- [../classification-supervisee/](../classification-supervisee/) pour la classification
- [../annotation-thematique/](../annotation-thematique/) pour l'extraction de thèmes
