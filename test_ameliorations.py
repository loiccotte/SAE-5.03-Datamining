"""
Script de test : differentes approches pour améliorer le score de classification.
Compare chaque approche via validation croisée 10-fold stratifiée.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    cross_validate, StratifiedKFold, GridSearchCV
)
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack, csr_matrix

# ── Chargement ───────────────────────────────────────────────────────────────
df = pd.read_csv('avis_annotés.csv')
y = df['avis'].values
cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = {}

def evaluate(name, pipeline, X):
    """Evalue un pipeline et affiche les résultats."""
    scores = cross_validate(
        pipeline, X, y, cv=cv10,
        scoring='accuracy', return_train_score=True, n_jobs=-1
    )
    train = scores['train_score'].mean()
    test = scores['test_score'].mean()
    std = scores['test_score'].std()
    gap = train - test
    results[name] = {'train': train, 'test': test, 'std': std, 'gap': gap}
    print(f"  Train: {train:.4f} | Test: {test:.4f} (+/- {std:.4f}) | Ecart: {gap:.4f}")


# ── Préparer les textes avec differentes repetitions du titre ────────────────
def make_texts(repeat=3):
    parts = [df['titre'].fillna('')] * repeat + [df['corps'].fillna('')]
    return parts[0].str.cat(parts[1:], sep=' ').str.strip().tolist()

texts_x3 = make_texts(3)

print("=" * 70)
print("COMPARAISON DES APPROCHES")
print("=" * 70)

# ── 0. Baseline actuel (C=1) ────────────────────────────────────────────────
print("\n[0] BASELINE — LogReg C=1, word(1,3)+char(3,5)")
pipeline_baseline = Pipeline([
    ('features', FeatureUnion([
        ('mots', TfidfVectorizer(
            analyzer='word', ngram_range=(1, 3),
            sublinear_tf=True, min_df=3, max_df=0.8
        )),
        ('caracteres', TfidfVectorizer(
            analyzer='char_wb', ngram_range=(3, 5),
            sublinear_tf=True, min_df=5, max_df=0.9
        )),
    ])),
    ('classifieur', LogisticRegression(C=1, max_iter=1000)),
])
evaluate("0. Baseline (C=1)", pipeline_baseline, texts_x3)


# ── 1. GridSearch sur C ─────────────────────────────────────────────────────
print("\n[1] GRIDSEARCH sur C (0.01 a 50)")
pipeline_grid = Pipeline([
    ('features', FeatureUnion([
        ('mots', TfidfVectorizer(
            analyzer='word', ngram_range=(1, 3),
            sublinear_tf=True, min_df=3, max_df=0.8
        )),
        ('caracteres', TfidfVectorizer(
            analyzer='char_wb', ngram_range=(3, 5),
            sublinear_tf=True, min_df=5, max_df=0.9
        )),
    ])),
    ('classifieur', LogisticRegression(max_iter=1000)),
])

grid = GridSearchCV(
    pipeline_grid,
    {'classifieur__C': [0.01, 0.05, 0.1, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 50]},
    cv=cv10, scoring='accuracy', n_jobs=-1
)
grid.fit(texts_x3, y)
best_C = grid.best_params_['classifieur__C']
print(f"  Meilleur C: {best_C}")
print(f"  Meilleur score CV: {grid.best_score_:.4f}")
results["1. GridSearch C"] = {
    'train': '-', 'test': grid.best_score_,
    'std': grid.cv_results_['std_test_score'][grid.best_index_],
    'gap': '-', 'best_C': best_C
}


# ── 2. LinearSVC ────────────────────────────────────────────────────────────
print("\n[2] LinearSVC (C=best)")
for c_val in [0.1, 0.5, 1.0, best_C]:
    pipeline_svc = Pipeline([
        ('features', FeatureUnion([
            ('mots', TfidfVectorizer(
                analyzer='word', ngram_range=(1, 3),
                sublinear_tf=True, min_df=3, max_df=0.8
            )),
            ('caracteres', TfidfVectorizer(
                analyzer='char_wb', ngram_range=(3, 5),
                sublinear_tf=True, min_df=5, max_df=0.9
            )),
        ])),
        ('classifieur', LinearSVC(C=c_val, max_iter=5000)),
    ])
    print(f"  LinearSVC C={c_val}:")
    evaluate(f"2. LinearSVC C={c_val}", pipeline_svc, texts_x3)


# ── 3. SGDClassifier (modified_huber) ────────────────────────────────────────
print("\n[3] SGDClassifier (modified_huber)")
pipeline_sgd = Pipeline([
    ('features', FeatureUnion([
        ('mots', TfidfVectorizer(
            analyzer='word', ngram_range=(1, 3),
            sublinear_tf=True, min_df=3, max_df=0.8
        )),
        ('caracteres', TfidfVectorizer(
            analyzer='char_wb', ngram_range=(3, 5),
            sublinear_tf=True, min_df=5, max_df=0.9
        )),
    ])),
    ('classifieur', SGDClassifier(
        loss='modified_huber', alpha=1e-4,
        max_iter=1000, random_state=42
    )),
])
evaluate("3. SGDClassifier", pipeline_sgd, texts_x3)


# ── 4. class_weight='balanced' ──────────────────────────────────────────────
print("\n[4] LogReg C=best + class_weight='balanced'")
pipeline_balanced = Pipeline([
    ('features', FeatureUnion([
        ('mots', TfidfVectorizer(
            analyzer='word', ngram_range=(1, 3),
            sublinear_tf=True, min_df=3, max_df=0.8
        )),
        ('caracteres', TfidfVectorizer(
            analyzer='char_wb', ngram_range=(3, 5),
            sublinear_tf=True, min_df=5, max_df=0.9
        )),
    ])),
    ('classifieur', LogisticRegression(
        C=best_C, max_iter=1000, class_weight='balanced'
    )),
])
evaluate("4. class_weight=balanced", pipeline_balanced, texts_x3)


# ── 5. Differentes repetitions du titre ──────────────────────────────────────
print("\n[5] Test de repetition du titre (x1 a x5)")
for rep in [1, 2, 3, 4, 5]:
    texts_rep = make_texts(rep)
    pipeline_rep = Pipeline([
        ('features', FeatureUnion([
            ('mots', TfidfVectorizer(
                analyzer='word', ngram_range=(1, 3),
                sublinear_tf=True, min_df=3, max_df=0.8
            )),
            ('caracteres', TfidfVectorizer(
                analyzer='char_wb', ngram_range=(3, 5),
                sublinear_tf=True, min_df=5, max_df=0.9
            )),
        ])),
        ('classifieur', LogisticRegression(C=best_C, max_iter=1000)),
    ])
    print(f"  Titre x{rep}:")
    evaluate(f"5. Titre x{rep}", pipeline_rep, texts_rep)


# ── 6. Features manuelles (longueur, ponctuation, majuscules) ────────────────
print("\n[6] Features manuelles + TF-IDF")

class ManualFeatures(BaseEstimator, TransformerMixin):
    """Extrait des features manuelles du texte."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for text in X:
            n_chars = len(text)
            n_words = len(text.split())
            n_excl = text.count('!')
            n_quest = text.count('?')
            n_upper = sum(1 for c in text if c.isupper())
            ratio_upper = n_upper / max(n_chars, 1)
            n_dots = text.count('...')
            feats.append([n_chars, n_words, n_excl, n_quest,
                          ratio_upper, n_dots])
        return csr_matrix(np.array(feats))

pipeline_manual = Pipeline([
    ('features', FeatureUnion([
        ('mots', TfidfVectorizer(
            analyzer='word', ngram_range=(1, 3),
            sublinear_tf=True, min_df=3, max_df=0.8
        )),
        ('caracteres', TfidfVectorizer(
            analyzer='char_wb', ngram_range=(3, 5),
            sublinear_tf=True, min_df=5, max_df=0.9
        )),
        ('manual', ManualFeatures()),
    ])),
    ('classifieur', LogisticRegression(C=best_C, max_iter=1000)),
])
evaluate("6. Features manuelles", pipeline_manual, texts_x3)


# ── 7. Variation des n-grammes ───────────────────────────────────────────────
print("\n[7] Variations n-grammes")

configs_ngram = [
    ("word(1,2)+char(3,5)", (1, 2), (3, 5)),
    ("word(1,3)+char(2,6)", (1, 3), (2, 6)),
    ("word(1,3)+char(3,4)", (1, 3), (3, 4)),
    ("word(1,2)+char(2,5)", (1, 2), (2, 5)),
]
for label, word_ng, char_ng in configs_ngram:
    pipeline_ng = Pipeline([
        ('features', FeatureUnion([
            ('mots', TfidfVectorizer(
                analyzer='word', ngram_range=word_ng,
                sublinear_tf=True, min_df=3, max_df=0.8
            )),
            ('caracteres', TfidfVectorizer(
                analyzer='char_wb', ngram_range=char_ng,
                sublinear_tf=True, min_df=5, max_df=0.9
            )),
        ])),
        ('classifieur', LogisticRegression(C=best_C, max_iter=1000)),
    ])
    print(f"  {label}:")
    evaluate(f"7. {label}", pipeline_ng, texts_x3)


# ── 8. GridSearch etendu (C + min_df + ngram) ────────────────────────────────
print("\n[8] GridSearch etendu (C + min_df + ngram_range)")
pipeline_full = Pipeline([
    ('features', FeatureUnion([
        ('mots', TfidfVectorizer(
            analyzer='word', sublinear_tf=True, max_df=0.8
        )),
        ('caracteres', TfidfVectorizer(
            analyzer='char_wb', sublinear_tf=True, max_df=0.9
        )),
    ])),
    ('classifieur', LogisticRegression(max_iter=1000)),
])

param_grid = {
    'features__mots__ngram_range': [(1, 2), (1, 3)],
    'features__mots__min_df': [2, 3, 5],
    'features__caracteres__ngram_range': [(3, 5), (2, 5), (3, 6)],
    'features__caracteres__min_df': [3, 5],
    'classifieur__C': [0.5, 1, 2, 3, 5],
}

grid_full = GridSearchCV(
    pipeline_full, param_grid,
    cv=cv10, scoring='accuracy', n_jobs=-1, verbose=0
)
grid_full.fit(texts_x3, y)
print(f"  Meilleurs params: {grid_full.best_params_}")
print(f"  Meilleur score CV:    {grid_full.best_score_:.4f}")
results["8. GridSearch etendu"] = {
    'train': '-', 'test': grid_full.best_score_,
    'std': grid_full.cv_results_['std_test_score'][grid_full.best_index_],
    'gap': '-', 'params': grid_full.best_params_
}


# ── Resume ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RÉSUMÉ — CLASSEMENT PAR SCORE TEST")
print("=" * 70)

summary = []
for name, r in results.items():
    test = r['test']
    std = r['std']
    summary.append((name, test, std))

summary.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'Rang':>4}  {'Approche':<40} {'Test':>8}  {'Std':>8}")
print("-" * 65)
for i, (name, test, std) in enumerate(summary):
    marker = " <-- BEST" if i == 0 else ""
    print(f"{i+1:>4}  {name:<40} {test:>8.4f}  {std:>8.4f}{marker}")

print("\n" + "=" * 70)
