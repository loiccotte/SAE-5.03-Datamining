"""
Test v2 : ameliorations sans overfitting sur la baseline C=1 + char(2,5) min_df=3.
Focus : preprocessing, regularisation, stopwords, max_df.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import re
import spacy
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin

nlp = spacy.load('fr_core_news_sm')
stopwords_fr = set(stopwords.words('french'))

df = pd.read_csv('avis_annotés.csv')
y = df['avis'].values
cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = {}

def evaluate(name, pipeline, X):
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


# ── Textes bruts (titre x3 + corps) ─────────────────────────────────────────
texts_raw = (
    df['titre'].fillna('') + ' ' +
    df['titre'].fillna('') + ' ' +
    df['titre'].fillna('') + ' ' +
    df['corps'].fillna('')
).str.strip().tolist()

print("=" * 70)
print("TESTS V2 — AMELIORATIONS SANS OVERFITTING")
print("=" * 70)


# ── 0. Nouvelle baseline (C=1, char(2,5), min_df=3) ─────────────────────────
print("\n[0] BASELINE — C=1, word(1,3)+char(2,5), min_df=3")
pipeline_base = Pipeline([
    ('features', FeatureUnion([
        ('mots', TfidfVectorizer(
            analyzer='word', ngram_range=(1, 3),
            sublinear_tf=True, min_df=3, max_df=0.8
        )),
        ('caracteres', TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2, 5),
            sublinear_tf=True, min_df=3, max_df=0.9
        )),
    ])),
    ('classifieur', LogisticRegression(C=1, max_iter=1000)),
])
evaluate("0. Baseline C=1 char(2,5)", pipeline_base, texts_raw)


# ── 1. Lemmatisation du texte avant TF-IDF ───────────────────────────────────
print("\n[1] LEMMATISATION (spaCy) avant TF-IDF")

excluded_pos = {'DET', 'PRON', 'AUX'}
excluded_lemmas = {
    'le', 'la', 'les', 'un', 'une', 'des', 'ce', 'cette', 'ces',
    'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses',
    'notre', 'nos', 'votre', 'vos', 'leur', 'leurs',
    'du', 'au', 'aux', 'de', 'a',
    'etre', 'avoir', 'aller',
    'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles',
    'me', 'te', 'se', 'lui', 'leur', 'y', 'en'
}

print("  Lemmatisation en cours...")
texts_lem = []
for doc in nlp.pipe(texts_raw, batch_size=64, disable=['ner', 'parser']):
    lemmas = [
        t.lemma_ for t in doc
        if not t.is_punct
        and t.pos_ not in excluded_pos
        and t.lemma_.lower() not in excluded_lemmas
    ]
    texts_lem.append(' '.join(lemmas))
print("  Done.")

evaluate("1a. Lemmatisation", pipeline_base, texts_lem)

# Lemmatisation + suppression stopwords
texts_lem_sw = []
for t in texts_lem:
    tokens = re.findall(r'\b\w+\b', t)
    texts_lem_sw.append(' '.join(tok for tok in tokens if tok.lower() not in stopwords_fr and len(tok) > 1))

evaluate("1b. Lemma + stopwords", pipeline_base, texts_lem_sw)


# ── 2. Stopwords seules (sans lemmatisation) ─────────────────────────────────
print("\n[2] STOPWORDS uniquement (sans lemmatisation)")
pipeline_sw = Pipeline([
    ('features', FeatureUnion([
        ('mots', TfidfVectorizer(
            analyzer='word', ngram_range=(1, 3),
            sublinear_tf=True, min_df=3, max_df=0.8,
            stop_words=list(stopwords_fr)
        )),
        ('caracteres', TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2, 5),
            sublinear_tf=True, min_df=3, max_df=0.9
        )),
    ])),
    ('classifieur', LogisticRegression(C=1, max_iter=1000)),
])
evaluate("2. Stopwords (TfidfVec)", pipeline_sw, texts_raw)


# ── 3. Regularisation L1 ────────────────────────────────────────────────────
print("\n[3] L1 (Lasso)")
for c in [0.5, 1, 2]:
    pipeline_l1 = Pipeline([
        ('features', FeatureUnion([
            ('mots', TfidfVectorizer(
                analyzer='word', ngram_range=(1, 3),
                sublinear_tf=True, min_df=3, max_df=0.8
            )),
            ('caracteres', TfidfVectorizer(
                analyzer='char_wb', ngram_range=(2, 5),
                sublinear_tf=True, min_df=3, max_df=0.9
            )),
        ])),
        ('classifieur', LogisticRegression(
            C=c, penalty='l1', solver='saga', max_iter=2000
        )),
    ])
    print(f"  L1 C={c}:")
    evaluate(f"3. L1 C={c}", pipeline_l1, texts_raw)


# ── 4. Elastic-Net ──────────────────────────────────────────────────────────
print("\n[4] ELASTIC-NET")
for c in [0.5, 1, 2]:
    for ratio in [0.3, 0.5, 0.7]:
        pipeline_en = Pipeline([
            ('features', FeatureUnion([
                ('mots', TfidfVectorizer(
                    analyzer='word', ngram_range=(1, 3),
                    sublinear_tf=True, min_df=3, max_df=0.8
                )),
                ('caracteres', TfidfVectorizer(
                    analyzer='char_wb', ngram_range=(2, 5),
                    sublinear_tf=True, min_df=3, max_df=0.9
                )),
            ])),
            ('classifieur', LogisticRegression(
                C=c, penalty='elasticnet', solver='saga',
                l1_ratio=ratio, max_iter=2000
            )),
        ])
        print(f"  EN C={c} ratio={ratio}:")
        evaluate(f"4. EN C={c} r={ratio}", pipeline_en, texts_raw)


# ── 5. max_df tuning ────────────────────────────────────────────────────────
print("\n[5] max_df tuning")
for word_mdf in [0.7, 0.8, 0.9]:
    for char_mdf in [0.8, 0.9, 0.95]:
        pipeline_mdf = Pipeline([
            ('features', FeatureUnion([
                ('mots', TfidfVectorizer(
                    analyzer='word', ngram_range=(1, 3),
                    sublinear_tf=True, min_df=3, max_df=word_mdf
                )),
                ('caracteres', TfidfVectorizer(
                    analyzer='char_wb', ngram_range=(2, 5),
                    sublinear_tf=True, min_df=3, max_df=char_mdf
                )),
            ])),
            ('classifieur', LogisticRegression(C=1, max_iter=1000)),
        ])
        print(f"  word_max_df={word_mdf} char_max_df={char_mdf}:")
        evaluate(f"5. mdf w={word_mdf} c={char_mdf}", pipeline_mdf, texts_raw)


# ── 6. Lemmatisation + meilleurs params ──────────────────────────────────────
print("\n[6] LEMMA + L1/EN")
for penalty, solver, c, ratio in [
    ('l1', 'saga', 1, None),
    ('l1', 'saga', 2, None),
    ('elasticnet', 'saga', 1, 0.5),
]:
    params = dict(C=c, penalty=penalty, solver=solver, max_iter=2000)
    if ratio:
        params['l1_ratio'] = ratio
    pipeline_lem = Pipeline([
        ('features', FeatureUnion([
            ('mots', TfidfVectorizer(
                analyzer='word', ngram_range=(1, 3),
                sublinear_tf=True, min_df=3, max_df=0.8
            )),
            ('caracteres', TfidfVectorizer(
                analyzer='char_wb', ngram_range=(2, 5),
                sublinear_tf=True, min_df=3, max_df=0.9
            )),
        ])),
        ('classifieur', LogisticRegression(**params)),
    ])
    label = f"{penalty} C={c}" + (f" r={ratio}" if ratio else "")
    print(f"  Lemma + {label}:")
    evaluate(f"6. Lemma+{label}", pipeline_lem, texts_lem)


# ── Resume ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CLASSEMENT PAR SCORE TEST (ecart < 7% uniquement)")
print("=" * 70)

summary = []
for name, r in results.items():
    if isinstance(r['gap'], str) or r['gap'] < 0.07:
        summary.append((name, r['test'], r['std'], r['train'], r['gap']))

summary.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'Rg':>3}  {'Approche':<35} {'Test':>7}  {'Std':>7}  {'Train':>7}  {'Ecart':>7}")
print("-" * 75)
for i, (name, test, std, train, gap) in enumerate(summary[:15]):
    marker = " <--" if i == 0 else ""
    gap_s = f"{gap:.4f}" if isinstance(gap, float) else gap
    print(f"{i+1:>3}  {name:<35} {test:>7.4f}  {std:>7.4f}  {train:>7.4f}  {gap_s:>7}{marker}")

print("\n" + "=" * 70)
