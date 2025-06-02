import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# =============== CONFIGURATION ===============
ligues_populaires = {
    "Premier League (Angleterre)": "PL",
    "La Liga (Espagne)": "LL",
    "Serie A (Italie)": "SA",
    "Bundesliga (Allemagne)": "BL",
    "Ligue 1 (France)": "L1",
    "MLS (USA)": "MLS",
    "Eredivisie (Pays-Bas)": "ED",
    "Primeira Liga (Portugal)": "PRL",
    "Championship (Angleterre D2)": "CH",
    "Saudi Pro League": "SPL"
}

# =============== MOCK MATCH DATA (à remplacer par API réelle) ===============
mock_fixtures = {
    "Premier League (Angleterre)": [
        "Manchester City vs Arsenal - 2025-06-05",
        "Liverpool vs Chelsea - 2025-06-06"
    ],
    "La Liga (Espagne)": [
        "Real Madrid vs Barcelona - 2025-06-04",
        "Atletico Madrid vs Sevilla - 2025-06-07"
    ]
}

# =============== PAGE STREAMLIT ===============
st.set_page_config(page_title="Prédiction Pari Foot", layout="centered")
st.title("⚽ Prédiction de Résultat - Conseils de Pari")

st.markdown("Remplis les informations ci-dessous pour obtenir une estimation du résultat d’un match.")

# Clé API (simulée)
api_key = st.text_input("🔑 Ta clé API (optionnelle pour le moment)", type="password")

# Choix de la ligue
championnat = st.selectbox("📍 Choisis un championnat :", list(ligues_populaires.keys()))

# Simuler récupération des matchs
matchs = mock_fixtures.get(championnat, ["Match 1", "Match 2"])
match_selectionne = st.selectbox("⚽ Choisis un match :", matchs)

# Bouton d’analyse
if st.button("Analyser ce match"):

    st.subheader("📊 Analyse prédictive")

    # =============== SIMULATION DE DONNÉES ===============
    # Ces valeurs seront à récupérer via l'API Football plus tard
    match_features = pd.DataFrame([{
        "home_team_form": np.random.randint(8, 15),
        "away_team_form": np.random.randint(5, 12),
        "home_avg_goals_scored": np.round(np.random.uniform(1.5, 3.0), 2),
        "away_avg_goals_scored": np.round(np.random.uniform(0.5, 2.5), 2),
        "home_avg_goals_conceded": np.round(np.random.uniform(0.5, 1.5), 2),
        "away_avg_goals_conceded": np.round(np.random.uniform(1.0, 2.5), 2),
        "rank_diff": np.random.randint(1, 10)
    }])

    # =============== MODELE SIMULÉ (sera remplacé par vrai entraînement) ===============
    # Données fictives d'entraînement
    X_mock = pd.DataFrame([
        [12, 7, 2.1, 1.1, 0.9, 1.7, 3],
        [8, 10, 1.9, 1.6, 1.3, 1.1, -2],
        [14, 5, 2.5, 1.0, 0.8, 1.9, 6],
        [10, 10, 1.8, 1.8, 1.2, 1.2, 0],
    ], columns=match_features.columns)

    y_mock = [0, 2, 0, 1]  # 0 = domicile, 1 = nul, 2 = extérieur

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_mock, y_mock)

    # Prédiction
    proba = model.predict_proba(match_features)[0]
    classes = ["🏠 Victoire domicile", "🤝 Match nul", "🚶 Victoire extérieur"]

    for i, p in enumerate(proba):
        st.write(f"{classes[i]} : **{round(p*100, 1)}%**")

    # Recommandation
    index = np.argmax(proba)
    st.markdown(f"### 💬 **Recommandation : Parier sur \"{classes[index]}\"**")
