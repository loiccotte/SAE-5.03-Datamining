"""Analyse d'overfitting de la meilleure config (Titre x3 + FeatureUnion + LR C=20)."""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    cross_validate, cross_val_score, StratifiedKFold,
    learning_curve, train_test_split
)
from sklearn.metrics import accuracy_score

# ── Données ──
df = pd.read_csv('avis_annotés.csv')
texts = (
    df['titre'].fillna('') + ' ' +
    df['titre'].fillna('') + ' ' +
    df['titre'].fillna('') + ' ' +
    df['corps'].fillna('')
).str.strip().tolist()
y = df['avis'].values
print(f'Corpus: {len(texts)} docs | Classes: {dict(zip(*np.unique(y, return_counts=True)))}')


def make_pipe(c_val=20):
    return Pipeline([
        ('features', FeatureUnion([
            ('word', TfidfVectorizer(
                analyzer='word', ngram_range=(1, 3),
                sublinear_tf=True, min_df=2, max_df=0.8, lowercase=True)),
            ('char', TfidfVectorizer(
                analyzer='char_wb', ngram_range=(3, 5),
                sublinear_tf=True, min_df=3, max_df=0.9, lowercase=True)),
        ])),
        ('clf', LogisticRegression(C=c_val, penalty='l2', solver='lbfgs', max_iter=1000)),
    ])


cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ══════════════════════════════════════════════════════════════════
print('\n' + '=' * 65)
print('TEST 1 : SCORE TRAIN vs SCORE TEST (cross_validate, 10 folds)')
print('=' * 65)
cv_results = cross_validate(make_pipe(), texts, y, cv=cv10,
                            scoring='accuracy', return_train_score=True)
train_scores = cv_results['train_score']
test_scores = cv_results['test_score']

print(f'Score TRAIN moyen : {train_scores.mean():.4f} (+/- {train_scores.std():.4f})')
print(f'Score TEST moyen  : {test_scores.mean():.4f} (+/- {test_scores.std():.4f})')
gap = train_scores.mean() - test_scores.mean()
print(f'Ecart (gap)       : {gap:.4f}')

if gap > 0.10:
    print('>>> ALERTE : gap > 10% => OVERFITTING SIGNIFICATIF')
elif gap > 0.05:
    print('>>> ATTENTION : gap entre 5-10% => overfitting modere')
else:
    print('>>> OK : gap < 5% => pas d\'overfitting majeur')

print(f'\n{"Fold":>5} {"Train":>8} {"Test":>8} {"Gap":>8}')
for i in range(10):
    g = train_scores[i] - test_scores[i]
    print(f'{i+1:>5} {train_scores[i]:>8.4f} {test_scores[i]:>8.4f} {g:>8.4f}')

# ══════════════════════════════════════════════════════════════════
print('\n' + '=' * 65)
print('TEST 2 : LEARNING CURVE (score en fonction de la taille du train)')
print('=' * 65)
train_sizes_abs, train_sc, test_sc = learning_curve(
    make_pipe(), texts, y, cv=cv10, scoring='accuracy',
    train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    n_jobs=-1
)
print(f'{"N_train":>8} {"Train":>10} {"Test":>10} {"Gap":>8}')
for i, sz in enumerate(train_sizes_abs):
    tr = train_sc[i].mean()
    te = test_sc[i].mean()
    print(f'{sz:>8} {tr:>10.4f} {te:>10.4f} {(tr - te):>8.4f}')

print('\nSi le gap se REDUIT quand N augmente => le modele apprend, pas de surapprentissage pur.')
print('Si le test score PLAFONNE => le corpus est le facteur limitant (pas le modele).')

# ══════════════════════════════════════════════════════════════════
print('\n' + '=' * 65)
print('TEST 3 : STABILITE ACROSS RANDOM SEEDS (10 seeds differents)')
print('=' * 65)
seed_scores = []
for seed in [0, 7, 13, 21, 42, 56, 73, 88, 99, 123]:
    cv_s = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    sc = cross_val_score(make_pipe(), texts, y, cv=cv_s, scoring='accuracy')
    seed_scores.append(sc.mean())
    print(f'  seed={seed:<4} => accuracy={sc.mean():.4f} (+/- {sc.std():.4f})')

print(f'\nMoyenne globale : {np.mean(seed_scores):.4f}')
print(f'Ecart-type      : {np.std(seed_scores):.4f}')
print(f'Min / Max       : {np.min(seed_scores):.4f} / {np.max(seed_scores):.4f}')

# ══════════════════════════════════════════════════════════════════
print('\n' + '=' * 65)
print('TEST 4 : HOLDOUT MULTIPLE (5 splits 80/20 avec seeds differents)')
print('=' * 65)
print(f'{"Seed":>6} {"Train":>8} {"Test":>8} {"Gap":>8}')
holdout_tests = []
for seed in [0, 21, 42, 77, 99]:
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, y, test_size=0.2, random_state=seed, stratify=y)
    p = make_pipe()
    p.fit(X_tr, y_tr)
    tr_acc = p.score(X_tr, y_tr)
    te_acc = p.score(X_te, y_te)
    holdout_tests.append(te_acc)
    print(f'{seed:>6} {tr_acc:>8.4f} {te_acc:>8.4f} {(tr_acc - te_acc):>8.4f}')

print(f'\nMoyenne test holdout : {np.mean(holdout_tests):.4f} (+/- {np.std(holdout_tests):.4f})')

# ══════════════════════════════════════════════════════════════════
print('\n' + '=' * 65)
print('TEST 5 : COURBE DE REGULARISATION (C variable, train vs test)')
print('=' * 65)
print('Un C trop grand => modele trop libre => overfitting')
print(f'{"C":>6} {"Train":>8} {"Test":>8} {"Gap":>8}')
for c_val in [0.01, 0.1, 0.5, 1, 5, 10, 20, 50, 100, 500]:
    cv_r = cross_validate(make_pipe(c_val), texts, y, cv=cv10,
                          scoring='accuracy', return_train_score=True)
    tr = cv_r['train_score'].mean()
    te = cv_r['test_score'].mean()
    marker = ' <-- CONFIG ACTUELLE' if c_val == 20 else ''
    print(f'{c_val:>6} {tr:>8.4f} {te:>8.4f} {(tr - te):>8.4f}{marker}')

print('\n' + '=' * 65)
print('CONCLUSION')
print('=' * 65)
