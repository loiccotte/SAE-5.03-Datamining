"""Test des approches d'optimisation du score test — bibliothèques autorisées uniquement."""
# torch DOIT être en premier (DLL Windows)
import torch  # noqa: F401

import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'vectorisation-du-texte')

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import VotingClassifier
import spacy

nlp = spacy.load('fr_core_news_sm')

# ── Données ───────────────────────────────────────────────────────────────────
df = pd.read_csv('avis_annotés.csv')
texts_raw = (df['titre'].fillna('') + ' ' + df['corps'].fillna('')).str.strip().tolist()
y = df['avis'].values
print(f'Corpus: {len(texts_raw)} docs | Classes: {dict(zip(*np.unique(y, return_counts=True)))}')

# ── Lemmatisation spaCy (identique au pipeline principal) ─────────────────────
excluded_pos    = {'DET', 'PRON', 'AUX'}
excluded_lemmas = {
    'le','la','les','un','une','des','ce','cette','ces','du','au','aux','de','a',
    'etre','avoir','aller','je','tu','il','elle','nous','vous','ils','elles',
    'me','te','se','lui','y','en','lui'
}
print('Lemmatisation spaCy...')
texts_lem = []
for doc in nlp.pipe(texts_raw, batch_size=64, disable=['ner', 'parser']):
    lemmas = [
        t.lemma_.lower() for t in doc
        if not t.is_punct
        and t.pos_ not in excluded_pos
        and t.lemma_.lower() not in excluded_lemmas
        and len(t.lemma_) > 1
    ]
    texts_lem.append(' '.join(lemmas))
print(f'  Exemple: "{texts_lem[0][:70]}"')

cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print('\n' + '=' * 65)
print('COMPARAISON DES APPROCHES (10-fold CV, accuracy)')
print('=' * 65)

resultats = []

# ── 1. Baseline : TF-IDF mots NG3, C=10, L2 (config actuelle) ───────────────
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True,
                               min_df=2, max_df=0.8, lowercase=False,
                               token_pattern=r'\b\w+\b')),
    ('norm',  Normalizer('l2')),
    ('clf',   LogisticRegression(C=10, penalty='l2', solver='lbfgs', max_iter=1000)),
])
s = cross_val_score(pipe, texts_lem, y, cv=cv10, scoring='accuracy', n_jobs=-1)
resultats.append(('Baseline TF-IDF mots NG3, C=10 L2', s.mean(), s.std()))
print(f'[1] {resultats[-1][0]:<45} {s.mean():.4f} ±{s.std():.4f}')

# ── 2. FeatureUnion : mots NG3 + caractères 3-5grams ─────────────────────────
pipe = Pipeline([
    ('feat', FeatureUnion([
        ('mots',  TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True,
                                   min_df=2, max_df=0.8, analyzer='word',
                                   lowercase=False, token_pattern=r'\b\w+\b')),
        ('chars', TfidfVectorizer(ngram_range=(3, 5), sublinear_tf=True,
                                   min_df=3, max_df=0.9, analyzer='char_wb')),
    ])),
    ('norm', Normalizer('l2')),
    ('clf',  LogisticRegression(C=10, penalty='l2', solver='lbfgs', max_iter=1000)),
])
s = cross_val_score(pipe, texts_lem, y, cv=cv10, scoring='accuracy', n_jobs=-1)
resultats.append(('FeatureUnion mots NG3 + chars 3-5g', s.mean(), s.std()))
print(f'[2] {resultats[-1][0]:<45} {s.mean():.4f} ±{s.std():.4f}')

# ── 3. TF-IDF texte brut lowercase (sans lemmatisation) ──────────────────────
texts_lc = [t.lower() for t in texts_raw]
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True,
                               min_df=2, max_df=0.8, lowercase=False,
                               token_pattern=r'\b\w+\b')),
    ('norm',  Normalizer('l2')),
    ('clf',   LogisticRegression(C=10, penalty='l2', solver='lbfgs', max_iter=1000)),
])
s = cross_val_score(pipe, texts_lc, y, cv=cv10, scoring='accuracy', n_jobs=-1)
resultats.append(('TF-IDF SANS lemmatisation, C=10 L2', s.mean(), s.std()))
print(f'[3] {resultats[-1][0]:<45} {s.mean():.4f} ±{s.std():.4f}')

# ── 4. FeatureUnion texte brut (mots + chars) ────────────────────────────────
pipe = Pipeline([
    ('feat', FeatureUnion([
        ('mots',  TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True,
                                   min_df=2, max_df=0.8, analyzer='word', lowercase=True,
                                   token_pattern=r'\b\w+\b')),
        ('chars', TfidfVectorizer(ngram_range=(3, 5), sublinear_tf=True,
                                   min_df=3, max_df=0.9, analyzer='char_wb', lowercase=True)),
    ])),
    ('norm', Normalizer('l2')),
    ('clf',  LogisticRegression(C=10, penalty='l2', solver='lbfgs', max_iter=1000)),
])
s = cross_val_score(pipe, texts_raw, y, cv=cv10, scoring='accuracy', n_jobs=-1)
resultats.append(('FeatureUnion mots+chars SANS lemma', s.mean(), s.std()))
print(f'[4] {resultats[-1][0]:<45} {s.mean():.4f} ±{s.std():.4f}')

# ── 5. FeatureUnion lemmatisé + C=50 ─────────────────────────────────────────
pipe = Pipeline([
    ('feat', FeatureUnion([
        ('mots',  TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True,
                                   min_df=2, max_df=0.8, analyzer='word',
                                   lowercase=False, token_pattern=r'\b\w+\b')),
        ('chars', TfidfVectorizer(ngram_range=(3, 5), sublinear_tf=True,
                                   min_df=3, max_df=0.9, analyzer='char_wb')),
    ])),
    ('norm', Normalizer('l2')),
    ('clf',  LogisticRegression(C=50, penalty='l2', solver='lbfgs', max_iter=1000)),
])
s = cross_val_score(pipe, texts_lem, y, cv=cv10, scoring='accuracy', n_jobs=-1)
resultats.append(('FeatureUnion lemma mots+chars, C=50', s.mean(), s.std()))
print(f'[5] {resultats[-1][0]:<45} {s.mean():.4f} ±{s.std():.4f}')

# ── 6. Pondération titre x2 + FeatureUnion sans lemma ────────────────────────
# Référence : config [4] = FeatureUnion mots+chars SANS lemma = ~90.53%
# Idée : le titre est plus court mais souvent plus informatif → le répéter
texts_titre2 = (df['titre'].fillna('') + ' ' + df['titre'].fillna('') + ' ' + df['corps'].fillna('')).str.strip().tolist()
pipe = Pipeline([
    ('feat', FeatureUnion([
        ('mots',  TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True,
                                   min_df=2, max_df=0.8, analyzer='word', lowercase=True,
                                   token_pattern=r'\b\w+\b')),
        ('chars', TfidfVectorizer(ngram_range=(3, 5), sublinear_tf=True,
                                   min_df=3, max_df=0.9, analyzer='char_wb', lowercase=True)),
    ])),
    ('norm', Normalizer('l2')),
    ('clf',  LogisticRegression(C=10, penalty='l2', solver='lbfgs', max_iter=1000)),
])
s = cross_val_score(pipe, texts_titre2, y, cv=cv10, scoring='accuracy', n_jobs=-1)
resultats.append(('FeatureUnion titre x2 + corps', s.mean(), s.std()))
print(f'[6] {resultats[-1][0]:<45} {s.mean():.4f} ±{s.std():.4f}')

# ── 7. Pondération titre x3 + FeatureUnion sans lemma ────────────────────────
texts_titre3 = (df['titre'].fillna('') + ' ' + df['titre'].fillna('') + ' ' + df['titre'].fillna('') + ' ' + df['corps'].fillna('')).str.strip().tolist()
pipe = Pipeline([
    ('feat', FeatureUnion([
        ('mots',  TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True,
                                   min_df=2, max_df=0.8, analyzer='word', lowercase=True,
                                   token_pattern=r'\b\w+\b')),
        ('chars', TfidfVectorizer(ngram_range=(3, 5), sublinear_tf=True,
                                   min_df=3, max_df=0.9, analyzer='char_wb', lowercase=True)),
    ])),
    ('norm', Normalizer('l2')),
    ('clf',  LogisticRegression(C=10, penalty='l2', solver='lbfgs', max_iter=1000)),
])
s = cross_val_score(pipe, texts_titre3, y, cv=cv10, scoring='accuracy', n_jobs=-1)
resultats.append(('FeatureUnion titre x3 + corps', s.mean(), s.std()))
print(f'[7] {resultats[-1][0]:<45} {s.mean():.4f} ±{s.std():.4f}')

print()
print('--- Fenêtres n-grams (référence [4] : word(1,3)+char(3,5)) ---')

# ── 8. Word NG(1,4) + char NG(3,5) ───────────────────────────────────────────
pipe = Pipeline([
    ('feat', FeatureUnion([
        ('mots',  TfidfVectorizer(ngram_range=(1, 4), sublinear_tf=True,
                                   min_df=2, max_df=0.8, analyzer='word', lowercase=True,
                                   token_pattern=r'\b\w+\b')),
        ('chars', TfidfVectorizer(ngram_range=(3, 5), sublinear_tf=True,
                                   min_df=3, max_df=0.9, analyzer='char_wb', lowercase=True)),
    ])),
    ('norm', Normalizer('l2')),
    ('clf',  LogisticRegression(C=10, penalty='l2', solver='lbfgs', max_iter=1000)),
])
s = cross_val_score(pipe, texts_raw, y, cv=cv10, scoring='accuracy', n_jobs=-1)
resultats.append(('Word(1,4)+char(3,5) sans lemma', s.mean(), s.std()))
print(f'[8] {resultats[-1][0]:<45} {s.mean():.4f} ±{s.std():.4f}')

# ── 9. Word NG(1,3) + char NG(2,5) ───────────────────────────────────────────
pipe = Pipeline([
    ('feat', FeatureUnion([
        ('mots',  TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True,
                                   min_df=2, max_df=0.8, analyzer='word', lowercase=True,
                                   token_pattern=r'\b\w+\b')),
        ('chars', TfidfVectorizer(ngram_range=(2, 5), sublinear_tf=True,
                                   min_df=3, max_df=0.9, analyzer='char_wb', lowercase=True)),
    ])),
    ('norm', Normalizer('l2')),
    ('clf',  LogisticRegression(C=10, penalty='l2', solver='lbfgs', max_iter=1000)),
])
s = cross_val_score(pipe, texts_raw, y, cv=cv10, scoring='accuracy', n_jobs=-1)
resultats.append(('Word(1,3)+char(2,5) sans lemma', s.mean(), s.std()))
print(f'[9] {resultats[-1][0]:<45} {s.mean():.4f} ±{s.std():.4f}')

# ── 10. Word NG(1,4) + char NG(2,5) ──────────────────────────────────────────
pipe = Pipeline([
    ('feat', FeatureUnion([
        ('mots',  TfidfVectorizer(ngram_range=(1, 4), sublinear_tf=True,
                                   min_df=2, max_df=0.8, analyzer='word', lowercase=True,
                                   token_pattern=r'\b\w+\b')),
        ('chars', TfidfVectorizer(ngram_range=(2, 5), sublinear_tf=True,
                                   min_df=3, max_df=0.9, analyzer='char_wb', lowercase=True)),
    ])),
    ('norm', Normalizer('l2')),
    ('clf',  LogisticRegression(C=10, penalty='l2', solver='lbfgs', max_iter=1000)),
])
s = cross_val_score(pipe, texts_raw, y, cv=cv10, scoring='accuracy', n_jobs=-1)
resultats.append(('Word(1,4)+char(2,5) sans lemma', s.mean(), s.std()))
print(f'[10] {resultats[-1][0]:<45} {s.mean():.4f} ±{s.std():.4f}')

# ── 11. Titre x2 + Word NG(1,4) + char NG(2,5) ───────────────────────────────
# Combinaison des deux optimisations : pondération titre + fenêtres élargies
pipe = Pipeline([
    ('feat', FeatureUnion([
        ('mots',  TfidfVectorizer(ngram_range=(1, 4), sublinear_tf=True,
                                   min_df=2, max_df=0.8, analyzer='word', lowercase=True,
                                   token_pattern=r'\b\w+\b')),
        ('chars', TfidfVectorizer(ngram_range=(2, 5), sublinear_tf=True,
                                   min_df=3, max_df=0.9, analyzer='char_wb', lowercase=True)),
    ])),
    ('norm', Normalizer('l2')),
    ('clf',  LogisticRegression(C=10, penalty='l2', solver='lbfgs', max_iter=1000)),
])
s = cross_val_score(pipe, texts_titre2, y, cv=cv10, scoring='accuracy', n_jobs=-1)
resultats.append(('Titre x2 + Word(1,4)+char(2,5)', s.mean(), s.std()))
print(f'[11] {resultats[-1][0]:<45} {s.mean():.4f} ±{s.std():.4f}')

print()
print('--- Combos avancés (titre x3/x4 + fenêtres + C tuning) ---')

# ── 12. Titre x3 + Word NG(1,4) + char NG(2,5) ───────────────────────────────
pipe = Pipeline([
    ('feat', FeatureUnion([
        ('mots',  TfidfVectorizer(ngram_range=(1, 4), sublinear_tf=True,
                                   min_df=2, max_df=0.8, analyzer='word', lowercase=True,
                                   token_pattern=r'\b\w+\b')),
        ('chars', TfidfVectorizer(ngram_range=(2, 5), sublinear_tf=True,
                                   min_df=3, max_df=0.9, analyzer='char_wb', lowercase=True)),
    ])),
    ('norm', Normalizer('l2')),
    ('clf',  LogisticRegression(C=10, penalty='l2', solver='lbfgs', max_iter=1000)),
])
s = cross_val_score(pipe, texts_titre3, y, cv=cv10, scoring='accuracy', n_jobs=-1)
resultats.append(('Titre x3 + Word(1,4)+char(2,5)', s.mean(), s.std()))
print(f'[12] {resultats[-1][0]:<45} {s.mean():.4f} ±{s.std():.4f}')

# ── 13. Titre x4 ─────────────────────────────────────────────────────────────
texts_titre4 = (df['titre'].fillna('') + ' ') * 4 + df['corps'].fillna('')
texts_titre4 = texts_titre4.str.strip().tolist()
pipe = Pipeline([
    ('feat', FeatureUnion([
        ('mots',  TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True,
                                   min_df=2, max_df=0.8, analyzer='word', lowercase=True,
                                   token_pattern=r'\b\w+\b')),
        ('chars', TfidfVectorizer(ngram_range=(3, 5), sublinear_tf=True,
                                   min_df=3, max_df=0.9, analyzer='char_wb', lowercase=True)),
    ])),
    ('norm', Normalizer('l2')),
    ('clf',  LogisticRegression(C=10, penalty='l2', solver='lbfgs', max_iter=1000)),
])
s = cross_val_score(pipe, texts_titre4, y, cv=cv10, scoring='accuracy', n_jobs=-1)
resultats.append(('Titre x4 + Word(1,3)+char(3,5)', s.mean(), s.std()))
print(f'[13] {resultats[-1][0]:<45} {s.mean():.4f} ±{s.std():.4f}')

# ── 14. Titre x4 + fenêtres élargies ─────────────────────────────────────────
pipe = Pipeline([
    ('feat', FeatureUnion([
        ('mots',  TfidfVectorizer(ngram_range=(1, 4), sublinear_tf=True,
                                   min_df=2, max_df=0.8, analyzer='word', lowercase=True,
                                   token_pattern=r'\b\w+\b')),
        ('chars', TfidfVectorizer(ngram_range=(2, 5), sublinear_tf=True,
                                   min_df=3, max_df=0.9, analyzer='char_wb', lowercase=True)),
    ])),
    ('norm', Normalizer('l2')),
    ('clf',  LogisticRegression(C=10, penalty='l2', solver='lbfgs', max_iter=1000)),
])
s = cross_val_score(pipe, texts_titre4, y, cv=cv10, scoring='accuracy', n_jobs=-1)
resultats.append(('Titre x4 + Word(1,4)+char(2,5)', s.mean(), s.std()))
print(f'[14] {resultats[-1][0]:<45} {s.mean():.4f} ±{s.std():.4f}')

# ── 15-17. C tuning sur meilleur candidat (titre x3, fenêtres de base) ───────
for C_val in [5, 20, 50]:
    pipe = Pipeline([
        ('feat', FeatureUnion([
            ('mots',  TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True,
                                       min_df=2, max_df=0.8, analyzer='word', lowercase=True,
                                       token_pattern=r'\b\w+\b')),
            ('chars', TfidfVectorizer(ngram_range=(3, 5), sublinear_tf=True,
                                       min_df=3, max_df=0.9, analyzer='char_wb', lowercase=True)),
        ])),
        ('norm', Normalizer('l2')),
        ('clf',  LogisticRegression(C=C_val, penalty='l2', solver='lbfgs', max_iter=1000)),
    ])
    s = cross_val_score(pipe, texts_titre3, y, cv=cv10, scoring='accuracy', n_jobs=-1)
    label = f'Titre x3 + C={C_val}'
    resultats.append((label, s.mean(), s.std()))
    print(f'     {label:<45} {s.mean():.4f} ±{s.std():.4f}')

print()
print('--- Branches séparées titre/corps (FunctionTransformer) ---')

# ── 18. FeatureUnion branches titre + corps séparées ─────────────────────────
# Idée : titre et corps ont des espaces de features INDÉPENDANTS
# → le modèle apprend des poids différents pour chaque champ
# (différent de la répétition qui mélange tout dans le même espace TF-IDF)
X_pairs = list(zip(df['titre'].fillna(''), df['corps'].fillna('')))
get_titre = FunctionTransformer(lambda x: [t[0] for t in x], validate=False)
get_corps = FunctionTransformer(lambda x: [t[1] for t in x], validate=False)

pipe = Pipeline([
    ('feat', FeatureUnion([
        ('titre_w', Pipeline([('sel', get_titre),
                               ('tfidf', TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True,
                                                          min_df=2, max_df=0.9, lowercase=True,
                                                          token_pattern=r'\b\w+\b'))])),
        ('corps_w', Pipeline([('sel', get_corps),
                               ('tfidf', TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True,
                                                          min_df=2, max_df=0.8, lowercase=True,
                                                          token_pattern=r'\b\w+\b'))])),
        ('titre_c', Pipeline([('sel', get_titre),
                               ('tfidf', TfidfVectorizer(ngram_range=(3, 5), sublinear_tf=True,
                                                          min_df=2, max_df=0.9, analyzer='char_wb',
                                                          lowercase=True))])),
        ('corps_c', Pipeline([('sel', get_corps),
                               ('tfidf', TfidfVectorizer(ngram_range=(3, 5), sublinear_tf=True,
                                                          min_df=3, max_df=0.8, analyzer='char_wb',
                                                          lowercase=True))])),
    ])),
    ('norm', Normalizer('l2')),
    ('clf',  LogisticRegression(C=20, penalty='l2', solver='lbfgs', max_iter=1000)),
])
s = cross_val_score(pipe, X_pairs, y, cv=cv10, scoring='accuracy', n_jobs=-1)
resultats.append(('Branches sep titre/corps C=20', s.mean(), s.std()))
print(f'[18] {resultats[-1][0]:<45} {s.mean():.4f} ±{s.std():.4f}')

# ── 19. Branches séparées + transformer_weights titre x2 ─────────────────────
# transformer_weights double le poids des features du titre
pipe = Pipeline([
    ('feat', FeatureUnion([
        ('titre_w', Pipeline([('sel', get_titre),
                               ('tfidf', TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True,
                                                          min_df=2, max_df=0.9, lowercase=True,
                                                          token_pattern=r'\b\w+\b'))])),
        ('corps_w', Pipeline([('sel', get_corps),
                               ('tfidf', TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True,
                                                          min_df=2, max_df=0.8, lowercase=True,
                                                          token_pattern=r'\b\w+\b'))])),
        ('titre_c', Pipeline([('sel', get_titre),
                               ('tfidf', TfidfVectorizer(ngram_range=(3, 5), sublinear_tf=True,
                                                          min_df=2, max_df=0.9, analyzer='char_wb',
                                                          lowercase=True))])),
        ('corps_c', Pipeline([('sel', get_corps),
                               ('tfidf', TfidfVectorizer(ngram_range=(3, 5), sublinear_tf=True,
                                                          min_df=3, max_df=0.8, analyzer='char_wb',
                                                          lowercase=True))])),
    ], transformer_weights={'titre_w': 2.0, 'corps_w': 1.0, 'titre_c': 2.0, 'corps_c': 1.0})),
    ('norm', Normalizer('l2')),
    ('clf',  LogisticRegression(C=20, penalty='l2', solver='lbfgs', max_iter=1000)),
])
s = cross_val_score(pipe, X_pairs, y, cv=cv10, scoring='accuracy', n_jobs=-1)
resultats.append(('Branches sep + tw titre x2 C=20', s.mean(), s.std()))
print(f'[19] {resultats[-1][0]:<45} {s.mean():.4f} ±{s.std():.4f}')

# ── 20. VotingClassifier soft vote (3 pipelines complémentaires) ──────────────
# Chaque classifieur voit le problème différemment ; le vote soft moyenne les proba
clf_a = Pipeline([  # titre x3, fenêtres standard
    ('feat', FeatureUnion([
        ('mots',  TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True,
                                   min_df=2, max_df=0.8, analyzer='word', lowercase=True,
                                   token_pattern=r'\b\w+\b')),
        ('chars', TfidfVectorizer(ngram_range=(3, 5), sublinear_tf=True,
                                   min_df=3, max_df=0.9, analyzer='char_wb', lowercase=True)),
    ])),
    ('norm', Normalizer('l2')),
    ('clf',  LogisticRegression(C=20, penalty='l2', solver='lbfgs', max_iter=1000)),
])
clf_b = Pipeline([  # titre x3, fenêtres élargies
    ('feat', FeatureUnion([
        ('mots',  TfidfVectorizer(ngram_range=(1, 4), sublinear_tf=True,
                                   min_df=2, max_df=0.8, analyzer='word', lowercase=True,
                                   token_pattern=r'\b\w+\b')),
        ('chars', TfidfVectorizer(ngram_range=(2, 5), sublinear_tf=True,
                                   min_df=3, max_df=0.9, analyzer='char_wb', lowercase=True)),
    ])),
    ('norm', Normalizer('l2')),
    ('clf',  LogisticRegression(C=15, penalty='l2', solver='lbfgs', max_iter=1000)),
])
clf_c = Pipeline([  # titre x4, fenêtres standard
    ('feat', FeatureUnion([
        ('mots',  TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True,
                                   min_df=2, max_df=0.8, analyzer='word', lowercase=True,
                                   token_pattern=r'\b\w+\b')),
        ('chars', TfidfVectorizer(ngram_range=(3, 5), sublinear_tf=True,
                                   min_df=3, max_df=0.9, analyzer='char_wb', lowercase=True)),
    ])),
    ('norm', Normalizer('l2')),
    ('clf',  LogisticRegression(C=10, penalty='l2', solver='lbfgs', max_iter=1000)),
])
voting = VotingClassifier([('a', clf_a), ('b', clf_b), ('c', clf_c)], voting='soft')
s = cross_val_score(voting, texts_titre3, y, cv=cv10, scoring='accuracy', n_jobs=-1)
resultats.append(('VotingClassifier soft (3 pipelines)', s.mean(), s.std()))
print(f'[20] {resultats[-1][0]:<45} {s.mean():.4f} ±{s.std():.4f}')

# ── 21. C fine-tuning entre 15-25 sur titre x3 ───────────────────────────────
print()
for C_val in [12, 15, 17, 22, 25]:
    pipe = Pipeline([
        ('feat', FeatureUnion([
            ('mots',  TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True,
                                       min_df=2, max_df=0.8, analyzer='word', lowercase=True,
                                       token_pattern=r'\b\w+\b')),
            ('chars', TfidfVectorizer(ngram_range=(3, 5), sublinear_tf=True,
                                       min_df=3, max_df=0.9, analyzer='char_wb', lowercase=True)),
        ])),
        ('norm', Normalizer('l2')),
        ('clf',  LogisticRegression(C=C_val, penalty='l2', solver='lbfgs', max_iter=1000)),
    ])
    s = cross_val_score(pipe, texts_titre3, y, cv=cv10, scoring='accuracy', n_jobs=-1)
    label = f'Titre x3 + C={C_val}'
    resultats.append((label, s.mean(), s.std()))
    print(f'     {label:<45} {s.mean():.4f} ±{s.std():.4f}')

# ── Résumé ────────────────────────────────────────────────────────────────────
print()
print('=' * 65)
print('CLASSEMENT COMPLET')
print('=' * 65)
for i, (nom, mean, std) in enumerate(sorted(resultats, key=lambda x: x[1], reverse=True), 1):
    print(f'  #{i:<2} {nom:<45} {mean:.4f} ±{std:.4f}')
print()
best = max(resultats, key=lambda x: x[1])
print(f'MEILLEUR : {best[0]}')
print(f'           {best[1]:.4f} ({best[1]*100:.2f}%)')
