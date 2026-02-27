"""
Orchestrateur étendu — génère les 108 nouvelles configurations.

Nouvelles dimensions par rapport à run_pipeline.py :
  - sw_mode  : none (S0) / partial (S2) / all (S1)
  - lemma_lib: spacy / stanza  (si lemmatization=True)
  - vect_lib : tfidf / bm25

Total : 2(L) × 3(SW) × [2(LEM=0) + 2×2(LEM=1)] × 3(NG) × 2(VL) = 108 configurations

NE PAS TOUCHER : run_pipeline.py, les 24 pkl originaux, les notebooks existants.
"""

import sys
import pickle
import logging
from pathlib import Path
from datetime import datetime

# Ajouter le répertoire courant au path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    OUTPUT_DIR, TARGET_COLUMN,
    SW_MODE_OPTIONS, LEMMA_LIB_OPTIONS, VECT_LIB_OPTIONS,
    SENTIMENT_PRESERVED_WORDS,
)
from utils import get_config_name_extended, logger


def generate_extended_configs():
    """
    Génère les 108 combinaisons de configurations étendues.
    Quand lemmatization=False, lemma_lib est fixé à 'none'.
    """
    configs = []
    for lowercase in [True, False]:
        for sw_mode in SW_MODE_OPTIONS:           # none, partial, all
            for lemmatization in [False, True]:
                lemma_libs = LEMMA_LIB_OPTIONS if lemmatization else ['none']
                for lemma_lib in lemma_libs:
                    for ngram in [1, 2, 3]:
                        for vect_lib in VECT_LIB_OPTIONS:
                            name = get_config_name_extended(
                                lowercase, sw_mode, lemmatization, ngram, lemma_lib, vect_lib
                            )
                            configs.append({
                                'name': name,
                                'lowercase': lowercase,
                                'sw_mode': sw_mode,
                                'lemmatization': lemmatization,
                                'lemma_lib': lemma_lib,
                                'ngram': ngram,
                                'vect_lib': vect_lib,
                            })
    return configs


def run_single_config(config):
    """
    Exécute une configuration complète (7 étapes) et sauvegarde le pkl final.
    Retourne True si succès, False sinon.
    """
    from scripts.load_data import load_data
    from scripts.lowercasing import apply_lowercasing
    from scripts.stopwords_removal import remove_stopwords
    from scripts.lemmatization import apply_lemmatization
    from scripts.tfidf import apply_tfidf
    from scripts.normalize import normalize_vectors

    name = config['name']
    logger.info("=" * 80)
    logger.info(f"Traitement : {name}")
    logger.info(f"  LC={config['lowercase']}  SW={config['sw_mode']}  "
                f"LEM={config['lemmatization']}  LEMLIB={config['lemma_lib']}  "
                f"NG={config['ngram']}  VECT={config['vect_lib']}")
    logger.info("=" * 80)

    try:
        # Étape 1 : Chargement
        logger.info("[1/7] Chargement des données...")
        df = load_data()

        # Étape 2 : Lowercasing
        logger.info("[2/7] Lowercasing...")
        df = apply_lowercasing(df, apply_lowercasing=config['lowercase'])

        # Étape 3 : Stopwords (nouvelle signature étendue)
        logger.info("[3/7] Suppression des stopwords...")
        df = remove_stopwords(
            df,
            apply_stopwords=(config['sw_mode'] != 'none'),
            sw_mode=config['sw_mode'],
            preserved_words=SENTIMENT_PRESERVED_WORDS,
        )

        # Étape 4 : Lemmatisation
        logger.info("[4/7] Lemmatisation...")
        df = apply_lemmatization(
            df,
            apply_lemmatization=config['lemmatization'],
            remove_stopwords_after=(config['sw_mode'] != 'none'),
            lemma_lib=config['lemma_lib'] if config['lemmatization'] else 'spacy',
        )

        # Étape 5 : Vectorisation TF-IDF ou BM25
        logger.info("[5/7] Vectorisation...")
        ngram_range = (1, config['ngram'])
        X_tfidf, feature_names, vectorizer = apply_tfidf(
            df,
            ngram_range=ngram_range,
            vect_lib=config['vect_lib'],
        )

        # Étape 6 : Normalisation L2
        logger.info("[6/7] Normalisation...")
        X_normalized = normalize_vectors(X_tfidf, norm='l2')

        # Étape 7 : Sauvegarde du pkl final
        logger.info("[7/7] Sauvegarde...")
        final_output = {
            'X_normalized': X_normalized,
            'X_tfidf': X_tfidf,
            'feature_names': feature_names,
            'tfidf_vectorizer': vectorizer,   # BM25Vectorizer ou TfidfVectorizer
            'df': df,
            'target': df[TARGET_COLUMN].values,
            'config': {
                **config,
                'sw_mode': config['sw_mode'],
                'lemma_lib': config['lemma_lib'],
                'vect_lib': config['vect_lib'],
            },
            'shape': X_normalized.shape,
            'n_features': len(feature_names),
            'timestamp': datetime.now().isoformat(),
        }

        output_path = OUTPUT_DIR / f"{name}_FINAL.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(final_output, f)

        logger.info(f"✓ Sauvegardé : {output_path}")
        logger.info(f"  Shape={final_output['shape']}  Features={final_output['n_features']}")
        return True

    except Exception as exc:
        logger.error(f"✗ Erreur sur {name} : {exc}")
        logger.exception(exc)
        return False


def main():
    logger.info("=" * 80)
    logger.info("Pipeline étendue — feature/library-comparison")
    logger.info(f"Répertoire de sortie : {OUTPUT_DIR}")

    configs = generate_extended_configs()
    total = len(configs)
    logger.info(f"Configurations à calculer : {total}")

    pending = [c for c in configs if not (OUTPUT_DIR / f"{c['name']}_FINAL.pkl").exists()]
    skipped = total - len(pending)
    logger.info(f"Déjà calculées (skip) : {skipped}")
    logger.info(f"À traiter : {len(pending)}")

    successful = 0
    failed = 0

    for i, config in enumerate(pending, 1):
        logger.info(f"\n[{i}/{len(pending)}] {config['name']}")
        t0 = datetime.now()

        if run_single_config(config):
            successful += 1
        else:
            failed += 1

        elapsed = (datetime.now() - t0).total_seconds()
        logger.info(f"  Temps : {elapsed:.1f}s")

    logger.info("\n" + "=" * 80)
    logger.info(f"Pipeline terminée : {successful} réussies, {failed} échouées, {skipped} ignorées")
    logger.info("=" * 80)
    return successful, failed


if __name__ == '__main__':
    successful, failed = main()
    sys.exit(0 if failed == 0 else 1)
