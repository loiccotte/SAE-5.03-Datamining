# Tutoriel - Environnement et Scripts

## 📋 Table des Matières
1. [Structure du Projet](#structure-du-projet)
2. [Configuration de l'Environnement](#configuration-de-lenvironnement)
3. [Activation de l'Environnement](#activation-de-lenvironnement)
4. [Installation des Dépendances](#installation-des-dépendances)
5. [Exécuter des Scripts Python](#exécuter-des-scripts-python)
6. [Utiliser Jupyter Notebook](#utiliser-jupyter-notebook)
7. [Guide Pratique par Répertoire](#guide-pratique-par-répertoire)
8. [Troubleshooting](#troubleshooting)

---

## 📁 Structure du Projet

```
SAE-5.03-Datamining/
├── .venv/                          # Environnement virtuel Python
├── vectorisation-du-texte/         # Espace 1 : Vectorisation
├── classification-supervisee/      # Espace 2 : Classification
├── annotation-thematique/          # Espace 3 : Annotation thématique
├── avis_annotés.csv               # Données d'entrée
├── README.md                        # Contexte du projet
├── TUTORIEL.md                     # Ce fichier
└── requirements.txt                # Liste des dépendances Python
```

### Les 3 Espaces de Travail

| Dossier | Objectif |
|---------|----------|
| **vectorisation-du-texte/** | Préparation et transformation du texte en vecteurs (Bag of Words, TF-IDF) |
| **classification-supervisee/** | Modèles de classification (positif/négatif) |
| **annotation-thematique/** | Extraction et catégorisation automatique des thèmes |

---

## ⚙️ Configuration de l'Environnement

### Prérequis
- **Python 3.13+** (déjà installé)
- **pip** (gestionnaire de paquets Python)
- Environnement virtuel `.venv` (déjà créé)

### Fichier requirements.txt
Toutes les dépendances sont listées dans `requirements.txt` :
```
pandas
numpy
scipy
spacy
nltk
scikit-learn
matplotlib
seaborn
jupyter
```

---

## 🚀 Activation de l'Environnement

### Sous Windows (PowerShell)
```powershell
# Se placer dans le répertoire du projet
cd C:\Users\frank\Documents\SAE-5.03-Datamining

# Activer l'environnement virtuel
.\.venv\Scripts\Activate.ps1
```

Après activation, vous verrez `(.venv)` au début de la ligne de commande :
```
(.venv) PS C:\Users\frank\Documents\SAE-5.03-Datamining>
```

### Sous Windows (CMD)
```cmd
.\.venv\Scripts\activate.bat
```

### Sous macOS/Linux
```bash
source .venv/bin/activate
```

---

## 📦 Installation des Dépendances

### Installation initiale (déjà faite)
```powershell
pip install -r requirements.txt
```

### Réinstaller ou mettre à jour
```powershell
pip install --upgrade -r requirements.txt
```

### Vérifier les packages installés
```powershell
pip list
```

---

## 🐍 Exécuter des Scripts Python

### Structure générale
Tous les scripts Python peuvent être exécutés de n'importe quel répertoire si l'environnement `.venv` est activé.

### Exemple : Exécuter un script depuis le répertoire racine
```powershell
# Activation de l'environnement
.\.venv\Scripts\Activate.ps1

# Exécuter un script dans vectorisation-du-texte/
python vectorisation-du-texte\mon_script.py

# Exécuter un script dans classification-supervisee/
python classification-supervisee\mon_script.py

# Exécuter un script dans annotation-thematique/
python annotation-thematique\mon_script.py
```

### Exemple : Exécuter un script depuis son répertoire
```powershell
# Se placer dans le répertoire
cd vectorisation-du-texte

# Exécuter le script
python mon_script.py

# Revenir au répertoire racine
cd ..
```

### Passer des arguments à un script
```powershell
python vectorisation-du-texte\mon_script.py --input avis_annotés.csv --output resultats.csv
```

---

## 📓 Utiliser Jupyter Notebook

### Démarrer Jupyter
```powershell
# S'assurer que l'environnement est activé
.\.venv\Scripts\Activate.ps1

# Lancer Jupyter Notebook
jupyter notebook
```

Une fenêtre navigateur s'ouvrira automatiquement avec l'interface Jupyter.

### Créer un Notebook
1. Cliquer sur **New** → **Python 3**
2. Choisir l'emplacement et le nom du notebook
3. Vérifier que le noyau sélectionné est le bon (voir en haut à droite)

### Utiliser Jupyter dans chaque répertoire
```powershell
# Aller dans le répertoire de travail
cd vectorisation-du-texte

# Lancer Jupyter depuis ce répertoire
jupyter notebook

# Tous les fichiers créés seront dans ce répertoire
```

### Sauvegarder et exporter
- **Sauvegarde auto** : Ctrl+S (ou Cmd+S)
- **Export en Python** : File → Download as → Python
- **Export en PDF** : File → Download as → PDF

---

## 📖 Guide Pratique par Répertoire

### 🔤 Espace 1 : vectorisation-du-texte/

**Objectif** : Préparer et transformer le texte brut en vecteurs numériques

**Tâches typiques** :
1. Charger les données (`avis_annotés.csv`)
2. Nettoyer le texte (stopwords, lemmatisation, minuscules)
3. Appliquer TF-IDF
4. Exporter les vecteurs en format exploitable

**Exemple de workflow** :
```powershell
cd vectorisation-du-texte

# Créer un notebook
jupyter notebook

# Ou exécuter un script
python preprocess.py
```

**Fichiers d'entrée** : `../avis_annotés.csv`
**Fichiers de sortie** : vectores_tfidf.pkl, corpus_nettoyé.csv, etc.

---

### 🎯 Espace 2 : classification-supervisee/

**Objectif** : Construire des modèles de classification (positif/négatif)

**Tâches typiques** :
1. Charger les vecteurs depuis l'espace 1
2. Diviser en ensemble d'entraînement/test
3. Entraîner les modèles (Logistic Regression, Naive Bayes, SVM, etc.)
4. Évaluer les performances (accuracy, precision, recall, F1-score)
5. Créer une pipeline complète

**Exemple de workflow** :
```powershell
cd classification-supervisee

# Entraîner le modèle
python train_model.py

# Évaluer le modèle
python evaluate_model.py

# Tester sur de nouvelles données
python predict.py --text "C'est fantastique!"
```

**Fichiers d'entrée** : Vecteurs de l'espace 1
**Fichiers de sortie** : modele.pkl, metriques.json, predictions.csv

---

### 🏷️ Espace 3 : annotation-thematique/

**Objectif** : Extraire et catégoriser les thèmes des avis

**Tâches typiques** :
1. Utiliser le texte brut ou vectorisé
2. Appliquer des techniques NLP (NER, topic modeling, clustering)
3. Étiqueter automatiquement les thèmes
4. Générer des résumés thématiques

**Exemple de workflow** :
```powershell
cd annotation-thematique

# Identifier les thèmes
python extract_themes.py

# Générer un rapport thématique
python theme_report.py
```

**Fichiers d'entrée** : `../avis_annotés.csv`
**Fichiers de sortie** : themes.json, rapport_themes.html

---

## 🛠️ Troubleshooting

### Problème : "Python not found" ou "commande non reconnue"

**Solution** : Vérifier que l'environnement virtuel est activé
```powershell
# Vérifier si .venv est activé
Get-Alias python

# Si ce n'est pas bon, activer l'env
.\.venv\Scripts\Activate.ps1
```

---

### Problème : "ModuleNotFoundError: No module named 'pandas'"

**Solution** : Réinstaller les dépendances
```powershell
# Activer l'environnement
.\.venv\Scripts\Activate.ps1

# Réinstaller
pip install -r requirements.txt

# Vérifier
pip list | grep pandas
```

---

### Problème : Jupyter ne démarre pas

**Solution** : 
```powershell
# Vérifier l'installation
pip install --upgrade jupyter

# Relancer
jupyter notebook

# Alternative : Si le port 8888 est occupé
jupyter notebook --port 8889
```

---

### Problème : Les imports échouent depuis différents répertoires

**Solution** : Utiliser des chemins absolus ou relatifs corrects
```python
import sys
from pathlib import Path

# Ajouter le répertoire parent au chemin
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Importer depuis d'autres répertoires
from vectorisation_du_texte.utils import preprocess_text
```

---

## 💡 Bonnes Pratiques

### 1. **Toujours activer l'environnement avant de coder**
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. **Utiliser des notebooks pour l'exploration**
Jupyter est idéal pour tester du code et visualiser les résultats en temps réel.

### 3. **Convertir en scripts réutilisables**
Une fois le code validé dans un notebook, le transformer en script `.py` pour la production.

### 4. **Partager les données entre répertoires**
Placer les fichiers de sortie d'un espace dans le répertoire parent accessible à tous.

### 5. **Maintenir un journal des expériences**
Ajouter des commentaires et de la documentation pour tracer les modifications.

### 6. **Utiliser .gitignore**
Exclure les fichiers volumineux et les caches :
```
__pycache__/
*.pyc
.jupyter/
data/raw/
*.pkl
*.joblib
```

---

## 📊 Exemple de Workflow Complet

```powershell
# 1. Activation
.\.venv\Scripts\Activate.ps1

# 2. Vectorisation (espace 1)
cd vectorisation-du-texte
python preprocess.py
jupyter notebook  # Explorer les résultats
cd ..

# 3. Classification (espace 2)
cd classification-supervisee
python train_model.py
python evaluate_model.py
cd ..

# 4. Annotation thématique (espace 3)
cd annotation-thematique
python extract_themes.py
cd ..

# 5. Résultats finaux
# Vérifier tous les fichiers de sortie
ls vectorisation-du-texte/
ls classification-supervisee/
ls annotation-thematique/
```

---

## 📚 Ressources Utiles

- [Documentation pandas](https://pandas.pydata.org/docs/)
- [Documentation scikit-learn](https://scikit-learn.org/stable/documentation.html)
- [Documentation NLTK](https://www.nltk.org/)
- [Documentation spaCy](https://spacy.io/)
- [Documentation Jupyter](https://jupyter.org/documentation)

---

**Dernière mise à jour** : 19 janvier 2026
