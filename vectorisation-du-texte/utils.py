"""
Utilitaires pour la pipeline de vectorisation
"""
import json
import logging
from pathlib import Path
from config import LOG_FILE, OUTPUT_DIR
import os

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def get_checkpoint_file(config_name):
    """
    Retourne le chemin du fichier de checkpoint pour une configuration donnée
    """
    return OUTPUT_DIR / f"checkpoint_{config_name}.pkl"


def get_output_file(config_name, step_name):
    """
    Retourne le chemin du fichier de sortie pour une étape donnée
    """
    return OUTPUT_DIR / f"{config_name}_{step_name}.pkl"


def get_config_name(lowercase, stopwords, lemmatization, ngram):
    """
    Génère un nom unique pour une configuration de traitement
    
    Format: config_L{0|1}_S{0|1}_LEM{0|1}_NG{1|2|3}
    """
    return (f"config_L{int(lowercase)}_S{int(stopwords)}_"
            f"LEM{int(lemmatization)}_NG{ngram}")


def save_checkpoint(data, config_name, step_name):
    """
    Sauvegarde un checkpoint des données
    """
    filepath = OUTPUT_DIR / f"{config_name}_{step_name}.pkl"
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Checkpoint sauvegardé: {filepath}")


def load_checkpoint(config_name, step_name):
    """
    Charge un checkpoint s'il existe
    """
    filepath = OUTPUT_DIR / f"{config_name}_{step_name}.pkl"
    if filepath.exists():
        import pickle
        with open(filepath, 'rb') as f:
            logger.info(f"Checkpoint chargé: {filepath}")
            return pickle.load(f)
    return None


def get_config_status():
    """
    Retourne le statut de traitement de toutes les configurations
    """
    status = {}
    for file in OUTPUT_DIR.glob("config_*.pkl"):
        parts = file.stem.replace("config_", "").split("_")
        config_key = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}"
        step = parts[-1] if len(parts) > 4 else "unknown"
        
        if config_key not in status:
            status[config_key] = {"completed_steps": []}
        status[config_key]["completed_steps"].append(step)
    
    return status


def log_config_complete(config_name, final_output_file):
    """
    Log qu'une configuration est terminée
    """
    status_file = OUTPUT_DIR / "status.json"
    
    status = {}
    if status_file.exists():
        with open(status_file, 'r') as f:
            status = json.load(f)
    
    status[config_name] = {
        "completed": True,
        "output_file": str(final_output_file)
    }
    
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)


def get_completed_configs():
    """
    Retourne la liste des configurations complètement traitées
    """
    status_file = OUTPUT_DIR / "status.json"
    if not status_file.exists():
        return set()
    
    with open(status_file, 'r') as f:
        status = json.load(f)
    
    return {k for k, v in status.items() if v.get("completed", False)}


logger.info("Utilitaires chargés avec succès")
