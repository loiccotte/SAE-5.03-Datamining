# SAE-5.03-Datamining

### 1. Le Contexte et l'Objectif

Vous travaillez pour une **bijouterie en ligne**. L'entreprise souhaite mieux comprendre ses clients en analysant les avis textuels qu'ils laissent sur le site.

Le service commercial a déjà fait une partie du travail "à la main" : ils ont classé un échantillon d'avis en deux catégories : **« positif »** et **« négatif »**.

Votre mission est d'automatiser ce processus pour le futur via deux systèmes distincts :

1. 
**Catégorisation automatique :** Prédire si un nouvel avis est positif ou négatif.


2. 
**Synthèse thématique :** Résumer automatiquement de quoi parlent les avis (les thèmes principaux).



---

### 2. Les Données à votre disposition

Vous disposez d'un fichier nommé `avis_annotés.csv`. D'après l'aperçu du fichier, il contient trois colonnes essentielles pour votre analyse :

* **corps** : Le texte principal du commentaire client (ex: "Contrefaçon du vulgaire plaqué or...").
* **titre** : Le résumé ou titre donné par le client (ex: "N'achetez pas").
* **avis** : L'étiquette (label) de sentiment : `négatif` ou `positif`. C'est votre variable cible pour la classification.



---

### 3. La Méthodologie Imposée (Le "Cœur" du sujet)

Le sujet est très directif sur les techniques à utiliser. Vous ne pouvez pas utiliser n'importe quel algorithme ; vous devez suivre une méthodologie de **Data Mining** précise.

#### A. Préparation des données (Preprocessing)

Avant de lancer des modèles, vous devez transformer le texte en chiffres :

* 
**Nettoyage :** Retrait des "mots vides" (stopwords), lemmatisation (mettre les mots à leur racine), et retrait de la casse (minuscules).


* 
**Vectorisation :** Utilisation de l'approche "Sac de mots" (Bag of Words) avec une pondération **TF-IDF** (pour donner moins de poids aux mots très courants et plus de poids aux mots rares et importants).



#### B. Mission 1 : La Classification Supervisée (Sentiment)

Pour prédire si un avis est positif ou négatif, vous devez :

* Utiliser une **Régression Logistique Régularisée**.


* Tester différentes régularisations : **L1, L2, Elastic-Net**.


* Sélectionner le meilleur modèle en optimisant les hyperparamètres.



#### C. Mission 2 : L'Annotation Thématique (Topic Modeling)

Pour découvrir les thèmes cachés dans les avis (ex: "Livraison", "Qualité", "Service Client"), vous devez :

* Utiliser une réduction de dimension par **Factorisation de Matrice Non Négative (NMF)**.


* Chercher les bons hyperparamètres (fonction de perte Frobenius ou Kullback-Leibler).


* Une fois les thèmes trouvés, vous devez mesurer leur importance dans les avis en utilisant une **Forêt Aléatoire (Random Forest)**.



---

### 4. Les Contraintes et le Rendu

* 
**Bibliothèques autorisées :** Uniquement les classiques de la data science Python : `pandas`, `numpy`, `scipy`, `spacy`, `nltk`, `sklearn`, `matplotlib`, `seaborn`.


* **Le Livrable :** Un **Notebook Jupyter** propre. Il ne doit pas être un brouillon de toutes vos tentatives, mais présenter le code final qui crée les deux systèmes et permet d'analyser un *nouvel avis* entré par l'utilisateur.


* **Évaluation :** Le notebook compte pour 1/3 de la note (travail de groupe). Les 2/3 restants sont une note individuelle basée sur un quizz de méthodologie.



---

### Résumé des étapes à suivre pour vous

1. **Exploration :** Charger le CSV, regarder la répartition positif/négatif, nettoyer le texte.
2. **Vectorisation :** Transformer le texte nettoyé en matrice TF-IDF.
3. **Modélisation 1 (Sentiments) :** Entraîner et optimiser la régression logistique.
4. **Modélisation 2 (Thèmes) :** Appliquer la NMF pour extraire les sujets.
5. **Interface :** Créer une petite fonction dans le notebook où l'on tape une phrase et le système répond : "Sentiment prédit : X" et "Thèmes principaux : Y".
