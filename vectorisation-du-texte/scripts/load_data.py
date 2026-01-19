"""
Script 01 : Charger les données brutes depuis le CSV
Output: Dataframe pandas avec les colonnes textuelles et la cible
"""
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_data(input_file=None):
    """
    Charge les données depuis le fichier CSV
    """
    if input_file is None:
        # Remonter deux niveaux depuis scripts/ vers la racine
        input_file = Path(__file__).parent.parent.parent / "avis_annotés.csv"
    
    logger.info(f"Chargement des données depuis {input_file}")
    
    # Charger le CSV
    df = pd.read_csv(input_file)
    
    logger.info(f"Données chargées: {len(df)} avis")
    logger.info(f"Colonnes: {list(df.columns)}")
    
    # Fusionner le titre et le corps pour le traitement
    df['texte_complet'] = df['titre'].fillna('') + ' ' + df['corps'].fillna('')
    df['texte_complet'] = df['texte_complet'].str.strip()
    
    logger.info(f"Texte fusionné créé (titre + corps)")
    logger.info(f"Exemple premier avis:\n{df['texte_complet'].iloc[0][:200]}...")
    
    # Supprimer les lignes avec texte vide
    initial_count = len(df)
    df = df[df['texte_complet'].str.len() > 0].reset_index(drop=True)
    logger.info(f"Lignes supprimées (texte vide): {initial_count - len(df)}")
    
    logger.info(f"Données finales: {len(df)} avis")
    
    return df


if __name__ == "__main__":
    df = load_data()
    print("\nAperçu des données:")
    print(df.head())
    print(f"\nFormes: {df.shape}")
    print(f"Distribution des sentiments:\n{df['avis'].value_counts()}")
