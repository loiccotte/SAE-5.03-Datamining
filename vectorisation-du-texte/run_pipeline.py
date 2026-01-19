"""
Script Orchestrateur : Lance toutes les combinaisons de traitement
Gère les checkpoints pour reprendre en cas d'arrêt prématuré

Ce script lance chaque combinaison possible de prétraitement:
- Lowercasing: Oui/Non (2 options)
- Stopwords: Oui/Non (2 options)
- Lemmatisation: Oui/Non (2 options)
- N-grammes: 1, 2, 3 (3 options)

Total: 2 × 2 × 2 × 3 = 24 combinaisons

Chaque résultat est sauvegardé dans output/ avec un nom unique.
"""

import sys
import os
from pathlib import Path
import logging
import pickle
import json
from datetime import datetime

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    LOWERCASING_OPTIONS, STOPWORDS_OPTIONS, LEMMATIZATION_OPTIONS, 
    NGRAM_OPTIONS, OUTPUT_DIR, TARGET_COLUMN
)
from utils import (
    get_config_name, save_checkpoint, load_checkpoint, 
    get_completed_configs, log_config_complete, logger
)


def generate_all_configs():
    """
    Génère toutes les combinaisons de configurations possibles
    """
    configs = []
    for lowercase in LOWERCASING_OPTIONS:
        for stopwords in STOPWORDS_OPTIONS:
            for lemmatization in LEMMATIZATION_OPTIONS:
                for ngram in NGRAM_OPTIONS:
                    config_name = get_config_name(lowercase, stopwords, lemmatization, ngram)
                    configs.append({
                        'name': config_name,
                        'lowercase': lowercase,
                        'stopwords': stopwords,
                        'lemmatization': lemmatization,
                        'ngram': ngram
                    })
    return configs


def process_config(config):
    """
    Traite une configuration complète
    """
    # Import local pour éviter les problèmes de dépendances au démarrage
    from scripts.load_data import load_data
    from scripts.lowercasing import apply_lowercasing
    from scripts.stopwords_removal import remove_stopwords
    from scripts.lemmatization import apply_lemmatization
    from scripts.tfidf import apply_tfidf
    from scripts.normalize import normalize_vectors
    
    config_name = config['name']
    logger.info("=" * 80)
    logger.info(f"Traitement de la configuration: {config_name}")
    logger.info(f"Paramètres: LC={config['lowercase']}, SW={config['stopwords']}, "
                f"LEM={config['lemmatization']}, NG={config['ngram']}")
    logger.info("=" * 80)
    
    try:
        # Étape 1: Charger les données
        logger.info("[1/7] Chargement des données...")
        df = load_data()
        save_checkpoint(df, config_name, "step01_loaded")
        
        # Étape 2: Lowercasing
        logger.info("[2/7] Lowercasing...")
        df = apply_lowercasing(df, apply_lowercasing=config['lowercase'])
        save_checkpoint(df, config_name, "step02_lowercased")
        
        # Étape 3: Stopwords
        logger.info("[3/7] Suppression des stopwords...")
        df = remove_stopwords(df, apply_stopwords=config['stopwords'])
        save_checkpoint(df, config_name, "step03_no_stopwords")
        
        # Étape 4: Lemmatisation
        logger.info("[4/7] Lemmatisation...")
        df = apply_lemmatization(df, apply_lemmatization=config['lemmatization'])
        save_checkpoint(df, config_name, "step04_lemmatized")
        
        # Étape 5: Bag of Words / TF-IDF
        logger.info("[5/7] Création de la matrice TF-IDF...")
        
        # Déterminer le range des n-grammes
        ngram_range = (1, config['ngram'])
        
        X_tfidf, feature_names, tfidf_vectorizer = apply_tfidf(
            df, 
            ngram_range=ngram_range
        )
        
        # Sauvegarder la matrice et les métadonnées
        checkpoint_data = {
            'X_tfidf': X_tfidf,
            'feature_names': feature_names,
            'tfidf_vectorizer': tfidf_vectorizer,
            'df': df
        }
        save_checkpoint(checkpoint_data, config_name, "step05_tfidf")
        
        # Étape 6: Normalisation
        logger.info("[6/7] Normalisation des vecteurs...")
        X_normalized = normalize_vectors(X_tfidf, norm='l2')
        
        checkpoint_data['X_normalized'] = X_normalized
        save_checkpoint(checkpoint_data, config_name, "step06_normalized")
        
        # Étape 7: Sauvegarde du résultat final
        logger.info("[7/7] Sauvegarde du résultat final...")
        
        final_output = {
            'X_normalized': X_normalized,
            'feature_names': feature_names,
            'tfidf_vectorizer': tfidf_vectorizer,
            'df': df,
            'target': df[TARGET_COLUMN].values,
            'config': config,
            'shape': X_normalized.shape,
            'n_features': len(feature_names),
            'timestamp': datetime.now().isoformat()
        }
        
        final_file = OUTPUT_DIR / f"{config_name}_FINAL.pkl"
        with open(final_file, 'wb') as f:
            pickle.dump(final_output, f)
        
        logger.info(f"✓ Configuration complétée et sauvegardée: {final_file}")
        logger.info(f"  - Matrice: {final_output['shape']}")
        logger.info(f"  - Features: {final_output['n_features']}")
        
        # Marquer comme complétée
        log_config_complete(config_name, final_file)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Erreur lors du traitement de {config_name}: {str(e)}")
        logger.exception(e)
        return False


def print_summary(all_configs, completed):
    """
    Affiche un résumé du traitement
    """
    total = len(all_configs)
    completed_count = len(completed)
    pending = total - completed_count
    
    print("\n" + "=" * 80)
    print("RÉSUMÉ DU TRAITEMENT")
    print("=" * 80)
    print(f"Total de configurations: {total}")
    print(f"Configurations complétées: {completed_count}")
    print(f"Configurations en attente: {pending}")
    print(f"Pourcentage complété: {100 * completed_count / total:.1f}%")
    print("=" * 80 + "\n")
    
    # Détails des configurations
    completed_names = set(config['name'] for config in all_configs if config['name'] in completed)
    print("Configurations complétées:")
    for config in all_configs:
        if config['name'] in completed:
            status = "✓"
        else:
            status = "○"
        print(f"{status} {config['name']}")
    
    print("\n" + "=" * 80 + "\n")


def main():
    """
    Fonction principale
    """
    logger.info("Démarrage du pipeline de vectorisation")
    logger.info(f"Répertoire de sortie: {OUTPUT_DIR}")
    
    # Générer toutes les configurations
    all_configs = generate_all_configs()
    logger.info(f"Total de configurations à traiter: {len(all_configs)}")
    
    # Charger les configurations déjà complétées
    completed_configs = get_completed_configs()
    logger.info(f"Configurations déjà complétées: {len(completed_configs)}")
    
    if completed_configs:
        logger.info("Configurations existantes:")
        for config_name in sorted(completed_configs):
            logger.info(f"  - {config_name}")
    
    # Traiter chaque configuration
    successful = 0
    failed = 0
    
    for i, config in enumerate(all_configs, 1):
        logger.info(f"\nConfiguration {i}/{len(all_configs)}")
        
        # Vérifier si déjà complétée
        if config['name'] in completed_configs:
            logger.info(f"Configuration {config['name']} déjà complétée, passage...")
            continue
        
        # Traiter la configuration
        if process_config(config):
            successful += 1
        else:
            failed += 1
    
    # Afficher le résumé final
    print_summary(all_configs, get_completed_configs())
    
    logger.info(f"Pipeline terminée: {successful} réussies, {failed} échouées")
    
    return successful, failed


if __name__ == "__main__":
    try:
        successful, failed = main()
        exit_code = 0 if failed == 0 else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrompue par l'utilisateur")
        logger.info("Les checkpoints ont été sauvegardés. Relancer le script pour reprendre.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Erreur critique: {str(e)}")
        logger.exception(e)
        sys.exit(1)
