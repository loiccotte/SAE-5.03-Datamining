"""
Test CamemBERT — Comparaison avec le pipeline TF-IDF/BM25

Approche : CamemBERT comme extracteur de features (embeddings mean-pooling)
           + LogisticRegression (même classifieur que le pipeline principal)
           → Comparaison équitable : seule la représentation du texte change.

Modèle : camembert-base (Inria, ~440 Mo)
         Pas de fine-tuning ici — test rapide sur embeddings bruts.
"""
# torch DOIT être importé en premier (conflit DLL Windows c10.dll)
import torch  # noqa: E402

import sys
import os
import time
import pandas as pd
import numpy as np
from transformers import CamembertTokenizer, CamembertModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── Chemins ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, 'avis_annotés.csv')
BEST_TFIDF_SCORE = 0.9000   # config_L1_S0_LEM1_NG3_STANZA_TFIDF

# ── 1. Chargement des données ─────────────────────────────────────────────────
print("=" * 60)
print("TEST CamemBERT — Extracteur de features + LogisticRegression")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
# Fusion titre + corps (même logique que le pipeline principal)
if 'titre' in df.columns and 'corps' in df.columns:
    texts = (df['titre'].fillna('') + ' ' + df['corps'].fillna('')).str.strip().tolist()
elif 'texte' in df.columns:
    texts = df['texte'].fillna('').tolist()
else:
    texts = df.iloc[:, 0].fillna('').tolist()

labels = df['avis'].values
print(f"Corpus : {len(texts)} textes | Classes : {sorted(set(labels))}")
print(f"Longueur moyenne : {np.mean([len(t.split()) for t in texts]):.1f} mots")
print()

# ── 2. Chargement du modèle CamemBERT ────────────────────────────────────────
print("[1/4] Chargement de camembert-base...")
t0 = time.time()
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
model = CamembertModel.from_pretrained('camembert-base')
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"  Modèle chargé sur {device} en {time.time()-t0:.1f}s")
print()

# ── 3. Génération des embeddings (mean-pooling) ───────────────────────────────
print("[2/4] Génération des embeddings (mean-pooling last hidden state)...")
BATCH_SIZE = 16
MAX_LEN = 256   # suffisant pour ~25 mots en moyenne, économise de la RAM

embeddings = []
t0 = time.time()

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i + BATCH_SIZE]
    encoded = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors='pt',
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        output = model(**encoded)

    # Mean-pooling sur les tokens non-padding
    hidden = output.last_hidden_state           # (B, seq_len, 768)
    mask = encoded['attention_mask'].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    embeddings.append(pooled.cpu().numpy())

    if (i // BATCH_SIZE + 1) % 20 == 0:
        done = min(i + BATCH_SIZE, len(texts))
        elapsed = time.time() - t0
        print(f"  {done}/{len(texts)} textes — {elapsed:.1f}s")

X = np.vstack(embeddings)
print(f"  Shape embeddings : {X.shape}  ({time.time()-t0:.1f}s total)")
print()

# ── 4. Classification LogisticRegression ─────────────────────────────────────
print("[3/4] Entraînement LogisticRegression (C=1.0, même que pipeline)...")
X_tr, X_te, y_tr, y_te = train_test_split(X, labels, test_size=0.2, random_state=42)
clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
t0 = time.time()
clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_te)
print(f"  Entraînement terminé en {time.time()-t0:.1f}s")
print()

# ── 5. Résultats ──────────────────────────────────────────────────────────────
print("[4/4] Résultats")
print("=" * 60)
acc = accuracy_score(y_te, y_pred)
f1 = f1_score(y_te, y_pred, average='macro')

print(f"Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
print(f"F1-macro : {f1:.4f}  ({f1*100:.2f}%)")
print()
print("Comparaison :")
print(f"  TF-IDF + Stanza (meilleur pipeline) : {BEST_TFIDF_SCORE:.4f} (90.00%)")
print(f"  CamemBERT embeddings + LR            : {acc:.4f} ({acc*100:.2f}%)")
delta = acc - BEST_TFIDF_SCORE
sign = '+' if delta >= 0 else ''
print(f"  Delta                                : {sign}{delta*100:.2f} points")
print()
print("Rapport de classification :")
print(classification_report(y_te, y_pred))

# Sauvegarder résultats
results_path = os.path.join(os.path.dirname(__file__), 'resultats_camembert.csv')
pd.DataFrame({
    'modele': ['CamemBERT-base (embeddings + LR)', 'TF-IDF + Stanza NG3 + LR'],
    'accuracy': [acc, BEST_TFIDF_SCORE],
    'f1_macro': [f1, 0.8995],
}).to_csv(results_path, index=False)
print(f"Résultats sauvegardés : {results_path}")
