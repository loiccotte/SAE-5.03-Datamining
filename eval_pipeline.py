"""Script d'évaluation des 132 configurations pipeline."""
import sys
import glob
import pickle
import re
sys.path.insert(0, 'vectorisation-du-texte')  # pour scripts.bm25
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# On charge uniquement les nouveaux fichiers (108 configs étendues)
# Les 24 originaux (format ancien, numpy 2.x) sont exclus car incompatibles numpy 1.x
pkl_files_all = sorted(glob.glob('vectorisation-du-texte/output/*_FINAL.pkl'))
# Nouveaux fichiers : ont au moins 7 segments (contiennent LEMMA_LIB et VECT_LIB dans le nom)
pkl_files = [p for p in pkl_files_all if len(p.replace('\\', '/').split('/')[-1].replace('_FINAL.pkl', '').split('_')) >= 7]
skipped = len(pkl_files_all) - len(pkl_files)
print(f'{len(pkl_files_all)} fichiers totaux, {skipped} anciens ignorés (numpy 2.x incompatible)')
print(f'{len(pkl_files)} nouveaux fichiers à évaluer')

results = []
for path in pkl_files:
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        X = data['X_normalized']
        y = data['target']
        cfg = data.get('config', {})
        fname = path.replace('\\', '/').split('/')[-1].replace('_FINAL.pkl', '')
        name = cfg.get('name', fname)
        parts = name.split('_')
        sw_match = re.search(r'S(\d)', name)
        sw_code = int(sw_match.group(1)) if sw_match else -1
        sw_mode_label = {0: 'S0-none', 1: 'S1-all', 2: 'S2-partial'}.get(sw_code, 'unknown')
        if len(parts) >= 7:
            lemma_lib = parts[5]
            vect_lib = parts[6]
        else:
            lemma_lib = 'LEGACY'
            vect_lib = 'TFIDF'
        results.append({
            'name': name, 'sw_mode': sw_mode_label,
            'lemma_lib': lemma_lib, 'vect_lib': vect_lib,
            'X': X, 'y': y,
        })
    except Exception as e:
        print(f'SKIP {path}: {e}')

print(f'Chargées: {len(results)} configs')
print('Lancement évaluation...')

scores = []
for i, r in enumerate(results):
    X_tr, X_te, y_tr, y_te = train_test_split(r['X'], r['y'], test_size=0.2, random_state=42)
    clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average='macro')
    scores.append({
        'config': r['name'], 'sw_mode': r['sw_mode'],
        'lemma_lib': r['lemma_lib'], 'vect_lib': r['vect_lib'],
        'accuracy': acc, 'f1_macro': f1,
    })
    if (i + 1) % 20 == 0:
        print(f'  {i+1}/{len(results)} configs évaluées...')

df_scores = pd.DataFrame(scores).sort_values('accuracy', ascending=False)
df_scores.to_csv('resultats_evaluation.csv', index=False)
print('CSV sauvegardé.')
print()
print('=== TOP 15 CONFIGURATIONS ===')
print(df_scores[['config', 'sw_mode', 'lemma_lib', 'vect_lib', 'accuracy', 'f1_macro']].head(15).to_string(index=False))
print()
print('=== ACCURACY MOYENNE PAR SW_MODE ===')
print(df_scores.groupby('sw_mode')[['accuracy', 'f1_macro']].mean().to_string())
print()
print('=== ACCURACY MOYENNE PAR LEMMA_LIB (hors NONE/LEGACY) ===')
lem_df = df_scores[~df_scores.lemma_lib.isin(['NONE', 'LEGACY'])]
print(lem_df.groupby('lemma_lib')[['accuracy', 'f1_macro']].mean().to_string())
print()
print('=== ACCURACY MOYENNE PAR VECT_LIB ===')
vect_df = df_scores[df_scores.vect_lib.isin(['TFIDF', 'BM25'])]
print(vect_df.groupby('vect_lib')[['accuracy', 'f1_macro']].mean().to_string())
print()
best = df_scores.iloc[0]
print('=== MEILLEURE CONFIGURATION ===')
print(f'Config  : {best["config"]}')
print(f'Accuracy: {best["accuracy"]:.4f}  F1-macro: {best["f1_macro"]:.4f}')
print(f'SW_MODE={best["sw_mode"]}  LEMMA={best["lemma_lib"]}  VECT={best["vect_lib"]}')
print()
s0 = df_scores[df_scores.sw_mode == 'S0-none']['accuracy'].mean()
s1 = df_scores[df_scores.sw_mode == 'S1-all']['accuracy'].mean()
s2 = df_scores[df_scores.sw_mode == 'S2-partial']['accuracy'].mean()
print('=== HYPOTHESE NEGATION (S2 vs S1) ===')
print(f'S0 (aucune suppression)   : {s0:.4f}')
print(f'S1 (suppression totale)   : {s1:.4f}')
print(f'S2 (suppression partielle): {s2:.4f}')
if s2 > s1:
    print('-> S2 > S1 : conserver les marqueurs de négation AMELIORE les performances [CONFIRME]')
else:
    print('-> S2 <= S1 : hypothèse de la négation NON CONFIRMEE sur ce corpus')
