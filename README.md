# Article Recommender — Azure Functions & Streamlit

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Azure Functions](https://img.shields.io/badge/Azure%20Functions-API-0062AD?style=flat-square&logo=azurefunctions&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-data-150458?style=flat-square&logo=pandas&logoColor=white)
![Pytest](https://img.shields.io/badge/tests-pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)
![Status](https://img.shields.io/badge/status-MVP-2EA44F?style=flat-square)
[![CI](https://github.com/VicoD3X/reco-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/VicoD3X/reco-engine/actions/workflows/ci.yml)

MVP de recommandation d’articles combinant un moteur content-based, une API Azure Functions et une interface Streamlit de démonstration.

## Présentation du projet

Ce dépôt présente un système de recommandation d’articles de bout en bout. Le projet contient un moteur de recommandation basé sur les contenus consultés par les utilisateurs, une API HTTP déployable avec Azure Functions et une interface Streamlit permettant de tester les recommandations.

L’objectif n’est pas de présenter un produit production-ready, mais un MVP sérieux, lisible et suffisamment structuré pour montrer une démarche de Data Scientist / AI Engineer junior : préparation des données, logique de recommandation, exposition d’un service d’inférence et validation des comportements principaux.

## Objectif métier

Le besoin métier est de proposer des articles pertinents à un utilisateur à partir de son historique de clics. Lorsqu’un utilisateur possède déjà un historique, le système construit un profil utilisateur et recommande des articles proches de ses intérêts.

Le projet couvre aussi le cas cold-start : lorsqu’un utilisateur est inconnu ou ne possède pas d’historique exploitable, le moteur renvoie une sélection d’articles populaires afin de conserver une réponse utile.

## Architecture

```text
Interface Streamlit
        |
        v
API HTTP Azure Functions
        |
        v
Moteur de recommandation
        |
        v
Historique de clics + embeddings d’articles
```

La logique de recommandation est isolée côté API. L’interface Streamlit reste volontairement simple : elle sélectionne un utilisateur, appelle l’API déployée, affiche les recommandations et permet de comparer le résultat à une baseline de popularité.

Pour le déploiement, les assets nécessaires à l’inférence sont chargés depuis Azure Blob Storage.

## Approche de recommandation

Le moteur utilise une approche content-based :

- les articles sont représentés par des embeddings réduits en 50 dimensions via PCA ;
- le profil utilisateur est construit en moyennant les embeddings des articles déjà consultés ;
- les articles candidats sont classés par similarité cosinus avec le profil utilisateur ;
- les articles déjà cliqués sont exclus des recommandations finales ;
- un fallback par popularité est utilisé lorsqu’aucun historique utilisateur n’est disponible.

La réponse de l’API contient un champ `mode` permettant de distinguer une recommandation personnalisée (`content_based_pca50`) d’un fallback cold-start (`cold_start_popularity`).

## Structure du dépôt

```text
.
|-- app/                 # Interface Streamlit
|-- azure_function/      # API HTTP Azure Functions
|-- data/                # Données d’exemple et documentation des assets
|-- docs/                # Supports de présentation et validation locale
|-- notebooks/           # Notebook d’exploration
|-- src/                 # Moteur de recommandation local
|-- tests/               # Tests unitaires du moteur de recommandation
|-- requirements.txt     # Dépendances de l’application Streamlit
|-- requirements-dev.txt # Outils de développement
`-- README.md
```

Les supports de présentation sont rangés dans `docs/` afin de garder la racine du dépôt lisible.

## Installation locale

Créer puis activer un environnement virtuel :

```bash
py -m venv .venv
.venv\Scripts\activate
```

Installer les dépendances de l’interface Streamlit :

```bash
pip install -r requirements.txt
```

Définir l’URL de l’API Azure Functions :

```powershell
$env:RECOMMENDER_API_URL = "https://<function-app>.azurewebsites.net/api/recommend"
```

Lancer l’interface Streamlit :

```bash
streamlit run app/streamlit_app.py
```

Pour tester le moteur localement, installer aussi les dépendances de l’Azure Function :

```bash
pip install -r azure_function/requirements.txt
```

Le moteur local attend un fichier `data/articles_embeddings_pca50.joblib`. Ce fichier n’est pas versionné dans le dépôt principal : il doit être généré ou récupéré avant de lancer le script local.

```bash
py -m src.run_local_test
```

## Lancer les tests

Installer les dépendances nécessaires aux tests :

```bash
pip install -r requirements.txt -r azure_function/requirements.txt -r requirements-dev.txt
```

Lancer la suite de tests :

```bash
pytest
```

## Utilisation de l’API

Endpoint attendu :

```text
https://<function-app>.azurewebsites.net/api/recommend
```

Exemple d’appel :

```http
GET /api/recommend?user_id=0&k=5
```

Exemple de réponse :

```json
{
  "user_id": 0,
  "k": 5,
  "mode": "content_based_pca50",
  "recommendations": [157519, 157944, 159495, 156690, 162857]
}
```

Pour un utilisateur inconnu :

```http
GET /api/recommend?user_id=999999&k=5
```

Le comportement attendu est une réponse basée sur la popularité globale, avec le mode `cold_start_popularity`.

## Interface Streamlit

L’application Streamlit permet de :

- sélectionner un identifiant utilisateur disponible dans l’échantillon local ;
- choisir le nombre d’articles à recommander ;
- appeler l’API Azure Functions déployée ;
- afficher la réponse JSON brute si nécessaire ;
- comparer les recommandations personnalisées à une baseline de popularité.

L’interface dépend notamment de `data/articles_metadata.csv`, qui est bien présent dans le dépôt et contient les colonnes nécessaires à l’affichage des métadonnées d’articles.

## Résultats et validation

La validation du MVP couvre les comportements principaux :

- le moteur retourne le nombre de recommandations demandé lorsque suffisamment de candidats sont disponibles ;
- les utilisateurs connus reçoivent des recommandations personnalisées ;
- les articles déjà consultés sont exclus des recommandations ;
- les utilisateurs inconnus déclenchent le fallback par popularité ;
- l’API renvoie une réponse JSON compacte et exploitable par l’interface.

Des exemples de sorties locales sont disponibles dans `docs/local_validation.txt`.

Des tests unitaires simples sont présents dans `tests/test_recommender.py` pour vérifier les comportements essentiels du moteur sans dépendre des fichiers d’artefacts complets.

## Limites actuelles

- Le projet reste un MVP et ne couvre pas tous les besoins d’un système de recommandation en production.
- L’approche est content-based uniquement ; elle ne compare pas encore plusieurs familles d’algorithmes.
- Les embeddings d’articles complets ne sont pas versionnés dans le dépôt.
- L’API ne contient pas encore de couche d’authentification avancée.
- Le frontend Streamlit sert à la démonstration et non à un usage produit final.

## Améliorations possibles

- Ajouter une gestion d’erreur plus sobre côté API, avec détails conservés uniquement dans les logs.
- Comparer l’approche content-based à une approche collaborative ou hybride.
- Ajouter une évaluation quantitative des recommandations.
- Documenter la génération des embeddings PCA depuis les données sources.

## Contexte du projet

Ce projet est conservé comme MVP de recommandation applicatif. Il met l’accent sur la clarté du dépôt, la lisibilité du flux API/interface et la capacité à expliquer les choix techniques sans le présenter comme un système de recommandation industriel.
