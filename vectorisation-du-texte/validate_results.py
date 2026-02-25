"""
Script de validation des résultats de la pipeline de vectorisation.
Vérifie que chaque transformation a bien été appliquée selon la configuration.
"""
import pickle
import logging
from pathlib import Path
import re
from collections import Counter
import numpy as np

from config import OUTPUT_DIR
from utils import get_config_name

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_stopwords():
    """Charge la liste des stopwords français"""
    try:
        import nltk
    except ModuleNotFoundError:
        return None
    try:
        stopwords_list = set(nltk.corpus.stopwords.words('french'))
    except LookupError:
        nltk.download('stopwords')
        stopwords_list = set(nltk.corpus.stopwords.words('french'))
    return stopwords_list


def validate_lowercasing(feature_names, config):
    """Vérifie qu'il n'y a pas de majuscules si lowercasing est activé"""
    if not config['lowercase']:
        return True, "Lowercasing désactivé - pas de vérification"
    
    uppercase_features = [f for f in feature_names if any(c.isupper() for c in f)]
    
    if uppercase_features:
        return False, f"Majuscules détectées ({len(uppercase_features)} features): {uppercase_features[:5]}"
    
    return True, f"OK - Aucune majuscule détectée sur {len(feature_names)} features"


def validate_stopwords(feature_names, config):
    """Vérifie qu'il n'y a pas de stopwords si removal est activé"""
    if not config['stopwords']:
        return True, "Stopwords removal désactivé - pas de vérification"
    
    stopwords = load_stopwords()
    if stopwords is None:
        return True, "SKIP - NLTK manquant (validation stopwords ignorée)"
    
    # Pour les n-grammes, on vérifie chaque mot individuellement
    all_words = set()
    for feature in feature_names:
        words = feature.split()
        all_words.update(words)

    all_words_lower = {w.lower() for w in all_words}
    found_stopwords = all_words_lower.intersection(stopwords)
    
    if found_stopwords:
        # Compter combien de features contiennent ces stopwords
        affected_features = [
            f for f in feature_names
            if any(sw in [tok.lower() for tok in f.split()] for sw in found_stopwords)
        ]
        return False, f"Stopwords détectés: {list(found_stopwords)[:10]} dans {len(affected_features)} features"
    
    return True, f"OK - Aucun stopword détecté parmi {len(all_words)} mots uniques"


def validate_lemmatization(feature_names, config):
    """Vérifie des indices de lemmatisation (absence de déterminants courants)"""
    if not config['lemmatization']:
        return True, "Lemmatization désactivée - pas de vérification"
    
    # Déterminants et mots qui devraient disparaître avec la lemmatisation
    determinants_courants = {
        'le', 'la', 'les', 'un', 'une', 'des', 'ce', 'cette', 'ces',
        'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses',
        'notre', 'nos', 'votre', 'vos', 'leur', 'leurs', 'du', 'au', 'aux'
    }
    
    # Verbes conjugués typiques (ils devraient être à l'infinitif)
    verbes_conjugues = {
        'suis', 'es', 'est', 'sommes', 'êtes', 'sont',  # être
        'ai', 'as', 'a', 'avons', 'avez', 'ont',  # avoir
        'vais', 'vas', 'va', 'allons', 'allez', 'vont',  # aller
    }
    
    problematic_words = determinants_courants.union(verbes_conjugues)
    
    # Extraire tous les mots
    all_words = set()
    for feature in feature_names:
        words = feature.split()
        all_words.update(words)
    
    found_issues = all_words.intersection(problematic_words)
    
    if found_issues:
        affected_features = [f for f in feature_names if any(w in f.split() for w in found_issues)]
        # Tolérer un petit nombre de features problématiques (< 2% du total)
        error_rate = len(affected_features) / len(feature_names)
        if error_rate < 0.02:  # Moins de 2% d'erreur
            return True, f"OK - Quelques mots résiduels ({len(affected_features)} features, {error_rate*100:.2f}%): {list(found_issues)[:5]}"
        return False, f"Mots non lemmatisés détectés: {list(found_issues)[:10]} dans {len(affected_features)} features ({error_rate*100:.1f}%)"
    
    return True, f"OK - Aucun déterminant/verbe conjugué détecté parmi {len(all_words)} mots"


def validate_ngrams(feature_names, config):
    """Vérifie que les n-grammes correspondent à la configuration"""
    expected_ngram = config['ngram']
    
    # Compter la distribution des n-grammes
    ngram_counts = Counter()
    for feature in feature_names:
        n = len(feature.split())
        ngram_counts[n] += 1
    
    # Pour n-grammes = 1, on devrait avoir QUE des unigrammes
    # Pour n-grammes = 2, on devrait avoir des uni + bigrammes
    # Pour n-grammes = 3, on devrait avoir des uni + bi + trigrammes
    
    max_ngram_found = max(ngram_counts.keys()) if ngram_counts else 0
    
    if max_ngram_found > expected_ngram:
        return False, f"N-grammes trop longs détectés: max={max_ngram_found}, attendu={expected_ngram}. Distribution: {dict(ngram_counts)}"
    
    if expected_ngram > 1 and max_ngram_found < expected_ngram:
        # Vérifier qu'on a au moins quelques n-grammes de la taille attendue
        count_expected = ngram_counts.get(expected_ngram, 0)
        if count_expected == 0:
            return False, f"Aucun {expected_ngram}-gramme détecté. Distribution: {dict(ngram_counts)}"
    
    return True, f"OK - Distribution des n-grammes: {dict(ngram_counts)}"


def validate_final_file(filepath):
    """Valide un fichier FINAL.pkl"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Validation de: {filepath.name}")
    logger.info(f"{'='*80}")
    
    try:
        with open(filepath, 'rb') as f:
            result = pickle.load(f)
    except Exception as e:
        logger.error(f"Erreur lors du chargement: {e}")
        return False
    
    # Extraire les informations nécessaires
    config = result.get('config', {})
    feature_names = result.get('feature_names', [])
    
    if not feature_names:
        logger.error("Aucune feature trouvée dans le fichier")
        return False
    
    logger.info(f"Configuration: L={config.get('lowercase', '?')} S={config.get('stopwords', '?')} "
                f"LEM={config.get('lemmatization', '?')} NG={config.get('ngram', '?')}")
    logger.info(f"Nombre de features: {len(feature_names)}")
    
    # Validation des transformations
    all_valid = True
    
    # 1. Lowercasing
    valid, msg = validate_lowercasing(feature_names, config)
    status = "OK" if valid else "ERREUR"
    logger.info(f"[{status}] Lowercasing: {msg}")
    all_valid = all_valid and valid
    
    # 2. Stopwords
    valid, msg = validate_stopwords(feature_names, config)
    status = "OK" if valid else "ERREUR"
    logger.info(f"[{status}] Stopwords: {msg}")
    all_valid = all_valid and valid
    
    # 3. Lemmatization
    valid, msg = validate_lemmatization(feature_names, config)
    status = "OK" if valid else "ERREUR"
    logger.info(f"[{status}] Lemmatization: {msg}")
    all_valid = all_valid and valid
    
    # 4. N-grams
    valid, msg = validate_ngrams(feature_names, config)
    status = "OK" if valid else "ERREUR"
    logger.info(f"[{status}] N-grams: {msg}")
    all_valid = all_valid and valid
    
    # Afficher quelques exemples de features
    logger.info(f"\nExemples de features (10 premiers):")
    for i, feature in enumerate(feature_names[:10], 1):
        logger.info(f"  {i}. {feature}")
    
    return all_valid


def validate_all_configurations():
    """Valide tous les fichiers FINAL.pkl"""
    logger.info("Démarrage de la validation de tous les fichiers FINAL")
    logger.info(f"Répertoire: {OUTPUT_DIR}")
    
    final_files = sorted(OUTPUT_DIR.glob("*FINAL.pkl"))
    
    if not final_files:
        logger.error("Aucun fichier FINAL.pkl trouvé!")
        return
    
    logger.info(f"Fichiers trouvés: {len(final_files)}\n")
    
    results = {}
    for filepath in final_files:
        try:
            is_valid = validate_final_file(filepath)
            results[filepath.name] = is_valid
        except Exception as e:
            logger.error(f"Erreur lors de la validation de {filepath.name}: {e}")
            results[filepath.name] = False
    
    # Résumé
    logger.info(f"\n{'='*80}")
    logger.info("RÉSUMÉ DE LA VALIDATION")
    logger.info(f"{'='*80}")
    
    valid_count = sum(results.values())
    total_count = len(results)
    
    logger.info(f"Fichiers valides: {valid_count}/{total_count}")
    
    if valid_count < total_count:
        logger.info("\nFichiers avec problèmes:")
        for filename, is_valid in results.items():
            if not is_valid:
                logger.info(f"  - {filename}")
    else:
        logger.info("\nTous les fichiers sont valides!")
    
    return results


def validate_specific_config(lowercase, stopwords, lemmatization, ngram):
    """Valide une configuration spécifique"""
    config_name = get_config_name(lowercase, stopwords, lemmatization, ngram)
    filepath = OUTPUT_DIR / f"{config_name}_FINAL.pkl"
    
    if not filepath.exists():
        logger.error(f"❌ Fichier non trouvé: {filepath}")
        return False
    
    return validate_final_file(filepath)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Mode: validation d'une configuration spécifique
        # Usage: python validate_results.py 1 1 1 1
        if len(sys.argv) == 5:
            lowercase = bool(int(sys.argv[1]))
            stopwords = bool(int(sys.argv[2]))
            lemmatization = bool(int(sys.argv[3]))
            ngram = int(sys.argv[4])
            validate_specific_config(lowercase, stopwords, lemmatization, ngram)
        else:
            logger.error("Usage: python validate_results.py [lowercase] [stopwords] [lemmatization] [ngram]")
            logger.error("Exemple: python validate_results.py 1 1 1 1")
    else:
        # Mode: validation de tous les fichiers
        validate_all_configurations()
