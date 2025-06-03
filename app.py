import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from ta.volatility import AverageTrueRange
from ta.trend import MACD
from sklearn.ensemble import RandomForestRegressor

# === CONFIG ===
API_KEY = "b3fce28a971a4e408afe6c459f72dcc5"
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD"]

st.set_page_config(page_title="Analyse Forex", layout="centered")
st.title("📊 Estimations sur 2h et 4h - Forex & Or")

# === Interface ===
pair = st.selectbox("Sélectionne une paire :", PAIRS)
date_input = st.date_input("Date de l'analyse", pd.Timestamp.today())

if st.button("Lancer l'analyse"):

    # === Requête API ===
    url = f"https://api.twelvedata.com/time_series?symbol={pair}&interval=1h&outputsize=500&apikey={API_KEY}"
    data = requests.get(url).json()

    if "values" not in data:
        st.error("❌ Erreur : données non disponibles.")
    else:
        # === Préparation ===
        df = pd.DataFrame(data["values"])
        df = df.rename(columns={
            "datetime": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close"
        })
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # === Indicateurs ===
        df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
        macd = MACD(close=df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()

        # === Cibles 2h ===
        df["high_2h"] = df["high"].shift(-2)
        df["low_2h"] = df["low"].shift(-2)
        df["delta_high_2h"] = df["high_2h"] - df["open"]
        df["delta_low_2h"] = df["low_2h"] - df["open"]

        # === Cibles 4h ===
        df["high_4h"] = df["high"].shift(-4)
        df["low_4h"] = df["low"].shift(-4)
        df["delta_high_4h"] = df["high_4h"] - df["open"]
        df["delta_low_4h"] = df["low_4h"] - df["open"]

        df.dropna(inplace=True)

        # === Modèle ML pour 2h ===
        features = df[["open", "atr", "macd", "macd_signal"]]
        X_train_2h = features[:-1]
        y_high_2h = df["delta_high_2h"][:-1]
        y_low_2h = df["delta_low_2h"][:-1]

        model_high_2h = RandomForestRegressor(n_estimators=100).fit(X_train_2h, y_high_2h)
        model_low_2h = RandomForestRegressor(n_estimators=100).fit(X_train_2h, y_low_2h)

        # === Modèle ML pour 4h ===
        X_train_4h = features[:-1]
        y_high_4h = df["delta_high_4h"][:-1]
        y_low_4h = df["delta_low_4h"][:-1]

        model_high_4h = RandomForestRegressor(n_estimators=100).fit(X_train_4h, y_high_4h)
        model_low_4h = RandomForestRegressor(n_estimators=100).fit(X_train_4h, y_low_4h)

        # === Prédiction sur le dernier point
        dernier = features.iloc[-1]
        open_now = df["open"].iloc[-1]

        # -- 2h
        pred_high_2h = model_high_2h.predict([dernier])[0]
        pred_low_2h = model_low_2h.predict([dernier])[0]
        prix_max_2h = round(open_now + pred_high_2h, 5)
        prix_min_2h = round(open_now + pred_low_2h, 5)

        # -- 4h
        pred_high_4h = model_high_4h.predict([dernier])[0]
        pred_low_4h = model_low_4h.predict([dernier])[0]
        prix_max_4h = round(open_now + pred_high_4h, 5)
        prix_min_4h = round(open_now + pred_low_4h, 5)

        # === Résultats ===
        st.success(f"💡 Prix d'ouverture actuel : {open_now}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Prévision 2 heures")
            st.metric("📈 Max (2h)", prix_max_2h)
            st.metric("📉 Min (2h)", prix_min_2h)

        with col2:
            st.subheader("📊 Prévision 4 heures")
            st.metric("📈 Max (4h)", prix_max_4h)
            st.metric("📉 Min (4h)", prix_min_4h)

        # === Alerte intelligente pour 2h ===
        alertes = []

        if (prix_max_2h - prix_min_2h) > 2 * df["atr"].mean():
            alertes.append("⚠️ Volatilité anormale prévue dans les 2 prochaines heures")

        if abs(df["macd"].iloc[-1] - df["macd_signal"].iloc[-1]) > 0.001:
            direction = "haussière 📈" if df["macd"].iloc[-1] > df["macd_signal"].iloc[-1] else "baissière 📉"
            alertes.append(f"Signal MACD détecté : tendance {direction} (2h)")

        if alertes:
            st.warning("📢 Alerte marché :")
            for alerte in alertes:
                st.write(alerte)
        else:
            st.info("✅ Aucun signal particulier détecté (sur 2h)")

        # === Graphique prévisionnel pour les 4 prochaines heures ===
        st.subheader("📉 Graphique prévisionnel - 4 prochaines heures")

        # Données des 4 dernières heures + point futur
        df_graph = df[["date", "open", "high", "low", "close"]].copy()
        df_graph = df_graph.iloc[-4:]

        # Point de prévision à H+4
        heure_future = df_graph["date"].iloc[-1] + pd.Timedelta(hours=4)
        ligne_prevue = {
            "date": heure_future,
            "open": open_now,
            "high": prix_max_4h,
            "low": prix_min_4h,
            "close": (prix_max_4h + prix_min_4h) / 2
        }
        df_graph = pd.concat([df_graph, pd.DataFrame([ligne_prevue])], ignore_index=True)

        # Affichage
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df_graph["date"], df_graph["open"], label="Open", marker='o')
        ax.plot(df_graph["date"], df_graph["high"], label="High", linestyle='--')
        ax.plot(df_graph["date"], df_graph["low"], label="Low", linestyle='--')
        ax.plot(df_graph["date"], df_graph["close"], label="Close", linestyle=':')

        ax.axvline(heure_future, color="gray", linestyle=":", label="Prévision")
        ax.set_title("Prévision sur 4 heures")
        ax.set_xlabel("Heure")
        ax.set_ylabel("Prix")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)
