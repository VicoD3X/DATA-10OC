# MyContent — MVP Recommandation (Projet 10 OpenClassrooms)

Ce projet implémente un MVP de recommandation d’articles :
- une Azure Function (serverless) qui retourne un top-k d’articles recommandés à partir d’un `user_id`
- une interface Streamlit qui consomme l’API et affiche les recommandations (avec une baseline popularité)

## Structure
- `azure_function/` : code de la Function (API HTTP)
- `app/` : application Streamlit
- `src/` : fonctions utilitaires / recommandation
- `data/` : échantillons de données (clicks + metadata)
- `notebooks/` : exploration et tests

## Lancer l’application Streamlit
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app/streamlit_app.py




API Azure Functions

URL :

https://p10oc-recommender.azurewebsites.net/api/recommend

Exemple :

GET ?user_id=0&k=5

Réponse (exemple) :
{"user_id":0,"k":5,"mode":"content_based_pca50","recommendations":[...]}




