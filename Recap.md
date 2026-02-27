# Récap — Optimisation Classification Sentiment (SAE-5.03)

> Branche : `feature/library-comparison` | Date : 2026-02-27
> Bibliothèques autorisées : `pandas`, `numpy`, `scipy`, `spacy`, `nltk`, `sklearn`, `matplotlib`, `seaborn`

---

## 1. Résultats — Classement global (10-fold CV stratifié, accuracy)

| Rang | Config | Accuracy | ±std | Autorisé |
|------|--------|----------|------|----------|
| **#1** | **VotingClassifier soft (3 pipelines, titre x3)** | **91.99%** | ±1.64% | ✓ |
| #2 | Titre x3 + FeatureUnion + C=20 | 91.94% | ±1.70% | ✓ |
| #3 | Titre x3 + FeatureUnion + C=17 | 91.94% | ±1.78% | ✓ |
| #4 | Titre x4 + FeatureUnion | 91.88% | ±1.79% | ✓ |
| #5 | Titre x3 + FeatureUnion + C=10 | 91.88% | ±1.78% | ✓ |
| — | Titre x2 + FeatureUnion | 91.67% | ±1.92% | ✓ |
| — | FeatureUnion mots+chars SANS lemma | 90.53% | ±1.56% | ✓ |
| — | TF-IDF SANS lemmatisation, C=10 | 90.48% | ±1.72% | ✓ |
| — | Baseline TF-IDF NG3 + lemma spaCy | 89.45% | ±2.01% | ✓ |
| — | CamemBERT embeddings + LR (80/20) | ~94.32% | — | ✗ non autorisé |
| — | Stanza + TF-IDF NG3 (meilleur PKL, 80/20) | 90.00% | — | ✗ non autorisé |

> Les scores du VotingClassifier et du Titre x3 C=20 sont **statistiquement équivalents** (différence < 1 std).
> **Config recommandée pour le livrable** : Titre x3 + FeatureUnion + C=20 (simple, reproductible, documentable).

---

## 2. Meilleure configuration recommandée

### Config finale : Titre x3 + FeatureUnion(word+char) + LR L2 C=20

```
Texte d'entrée : titre répété 3 fois + corps
Vectorisation  : FeatureUnion [
    TfidfVectorizer(word, ngram=(1,3), sublinear_tf, min_df=2, max_df=0.8, lowercase)
    TfidfVectorizer(char_wb, ngram=(3,5), sublinear_tf, min_df=3, max_df=0.9, lowercase)
]
Normalisation  : L2
Classifieur    : LogisticRegression(C=20, penalty='l2', solver='lbfgs', max_iter=1000)
Évaluation     : StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
Score          : 91.94% accuracy (±1.70%)
```

**Code :** [test_optimisations.py](test_optimisations.py) — configs [7] et C tuning section

---

## 3. Variantes testées et résultats détaillés

### 3.1 Phase 1 — Comparaison 108 configurations pipeline

**Code :** [vectorisation-du-texte/run_pipeline_extended.py](vectorisation-du-texte/run_pipeline_extended.py) | **Éval :** [eval_pipeline.py](eval_pipeline.py)

Dimensions explorées :
- **Lowercase** (L0/L1) : L1 +0.35% en moyenne
- **Stopwords** (S0=aucun / S1=tous / S2=partiel, garde négations) : S0 > S1 > S2 → hypothèse de la négation **non confirmée**
- **Lemmatisation** (spaCy / Stanza / aucune) : Stanza +0.84% vs spaCy — mais Stanza **non autorisé**
- **Vectorisation** (TF-IDF / BM25) : BM25 +0.25% vs TF-IDF
- **N-grams** (NG1/NG2/NG3) : NG3 meilleur

| Dimension | Valeur | Accuracy moy. |
|-----------|--------|--------------|
| Lowercase | L1 | 0.8788 |
| Lowercase | L0 | 0.8753 |
| SW_MODE | S0-none | 0.8800 |
| SW_MODE | S1-all | 0.8767 |
| SW_MODE | S2-partial | 0.8745 |
| Lemma | Stanza | 0.8848 |
| Lemma | spaCy | 0.8764 |
| Vect | BM25 | 0.8783 |
| Vect | TF-IDF | 0.8758 |

### 3.2 Phase 2 — Optimisations sur texte brut + FeatureUnion

**Code :** [test_optimisations.py](test_optimisations.py)

| # | Config | Accuracy | Enseignement |
|---|--------|----------|--------------|
| [1] | Baseline TF-IDF NG3 lemma spaCy, C=10 | 89.45% | référence |
| [2] | FeatureUnion mots NG3 + chars 3-5g (lemma) | 89.93% | chars +0.48% |
| [3] | TF-IDF SANS lemmatisation, C=10 | 90.48% | lemma nuit |
| [4] | FeatureUnion mots+chars SANS lemma | 90.53% | chars+nolemma meilleur |
| [5] | FeatureUnion lemma + C=50 | 89.56% | C=50 ne compense pas |
| [6] | Titre x2 + FeatureUnion | 91.67% | pondération titre +1.14% |
| [7] | Titre x3 + FeatureUnion | 91.88% | titre x3 meilleur |
| [8] | Word(1,4)+char(3,5) | 90.58% | NG4 marginal sans pondér. |
| [12] | Titre x3 + Word(1,4)+char(2,5) | 91.83% | pas mieux que [7] |
| [13] | Titre x4 + FeatureUnion | 91.88% | plateau |
| C=17 | Titre x3 + C=17 | 91.94% | dépasse C=10 |
| C=20 | **Titre x3 + C=20** | **91.94%** | **optimum** |
| C=50 | Titre x3 + C=50 | 91.34% | sur-régularise |
| [18] | Branches séparées titre/corps | 91.18% | moins bon que répétition |
| [19] | Branches séparées + transformer_weights | 88.58% | contre-productif |
| [20] | VotingClassifier soft (3 pipelines) | 91.99% | +0.05% vs C=20, plus complexe |

### 3.3 Fenêtre C optimale (titre x3, FeatureUnion standard)

| C | Accuracy |
|---|----------|
| 5 | 91.77% |
| 10 | 91.88% |
| 12 | 91.83% |
| 15 | 91.88% |
| **17** | **91.94%** |
| **20** | **91.94%** |
| 22 | 91.88% |
| 25 | 91.72% |
| 50 | 91.34% |

Plateau clair entre C=17 et C=22 → **C=20 recommandé**.

---

## 4. Résumé des enseignements clés

### Ce qui aide
- **Pondération du titre** (répéter dans le texte) : +1.35% vs sans pondération
  → Le titre ("N'achetez pas", "Parfait") est très discriminatif mais noyé dans le corps par TF-IDF
- **FeatureUnion word + char n-grams** : +0.48–0.56%
  → Les n-grams de caractères capturent morphologie, fautes d'orthographe, suffixes
- **Pas de lemmatisation** (spaCy `fr_core_news_sm`) : +1.03% vs avec
  → La lemmatisation fusionne des formes discriminatives ("adoré"→"adorer", "déçu"→"décevoir")
- **Lowercase** : +0.35% en moyenne
- **C=20 vs C=10** : +0.06% — léger ajustement de régularisation

### Ce qui ne fonctionne pas
- **Lemmatisation spaCy** : perd de l'information discriminative sur ce corpus d'avis
- **Branches FeatureUnion séparées** : le titre est trop court seul → représentation sparse insuffisante
- **transformer_weights** : amplification contre-productive des features sparse du titre
- **C > 25** : sur-apprentissage
- **Arbres (RF, AdaBoost, GradientBoosting)** : nécessitent une réduction SVD qui détruit l'information sparse → LR reste meilleur
- **Stanza** (non autorisé) : +0.84% vs spaCy, mais bibliothèque non autorisée
- **CamemBERT** (non autorisé) : 94.32%, mais bibliothèque non autorisée

---

## 5. Biais possibles et limites

### Biais de sélection des hyperparamètres
- Tous les hyperparamètres (C, n-grams, titre×N) ont été sélectionnés en observant les scores CV sur le **même dataset** utilisé pour l'évaluation finale. Il y a un risque de **surapprentissage sur les splits CV** — le vrai score sur des données réellement inédites sera probablement légèrement inférieur.

### Taille du corpus
- 1848 documents — corpus relativement petit. Les écart-types (~±1.5–2%) sont élevés : une différence de 0.5% entre deux configs est **statistiquement non significative**. Le classement peut varier entre exécutions (seed différente).

### Déséquilibre des classes
- 954 négatifs / 894 positifs (~52/48%) — faible déséquilibre, sans impact majeur, mais `StratifiedKFold` assure une distribution représentative à chaque fold.

### Biais de pondération du titre (répétition)
- Répéter le titre 3 fois modifie artificiellement les fréquences TF-IDF. Cette technique n'est pas une représentation fidèle des données réelles : un titre répété 3 fois n'existe pas dans un vrai avis. Sur de nouvelles données, l'effet peut être moins important si les titres sont absents ou très courts.

### Évaluation CV vs test réel
- `cross_val_score` évalue correctement (pas de fuite entraînement→test dans chaque fold). Cependant, l'optimisation des hyperparamètres a été faite manuellement par observation des scores CV successifs, ce qui constitue une forme d'optimisation indirecte sur les données.

### Lemmatisation — sens inverse attendu
- Intuitivement, la lemmatisation devrait aider. Son impact négatif ici est spécifique au corpus d'avis courts en français avec `fr_core_news_sm` (modèle léger). Un modèle spaCy plus grand (`fr_core_news_lg`) ou Stanza pourrait donner un résultat différent.

### Généralisabilité
- Le pipeline est optimisé pour des avis de bijouterie en ligne (court, sentimental, français). Les conclusions (pas de lemma, titre × 3, char n-grams) ne généralisent pas nécessairement à d'autres domaines ou langues.

---

## 6. Fichiers associés

| Fichier | Rôle |
|---------|------|
| [test_optimisations.py](test_optimisations.py) | Script de comparaison des 21+ configs d'optimisation |
| [eval_pipeline.py](eval_pipeline.py) | Évaluation des 108 configs pipeline (PKL) |
| [vectorisation-du-texte/run_pipeline_extended.py](vectorisation-du-texte/run_pipeline_extended.py) | Génération des 108 configs (2×3×[1+2]×3×2) |
| [vectorisation-du-texte/scripts/bm25.py](vectorisation-du-texte/scripts/bm25.py) | BM25Vectorizer sklearn-compatible |
| [vectorisation-du-texte/scripts/lemmatization.py](vectorisation-du-texte/scripts/lemmatization.py) | Lemmatisation spaCy + Stanza |
| [test_camemBERT/test_camembert.py](test_camemBERT/test_camembert.py) | Test CamemBERT (non autorisé, référence uniquement) |
| [Comparaison_Bibliotheques.ipynb](Comparaison_Bibliotheques.ipynb) | Notebook synthèse comparaison bibliothèques |
| [Anayse_Dataming_Toutes_Config.ipynb](Anayse_Dataming_Toutes_Config.ipynb) | Notebook analyse GridSearchCV + test score |

---

## 7. Prochaines étapes possibles

- [ ] Intégrer la config finale (Titre x3 + FeatureUnion + C=20) dans le notebook livrable
- [ ] Implémenter la Mission 2 (NMF + Random Forest pour importance thématique)
- [ ] Ajouter l'interface utilisateur (prédiction sur nouvel avis)
