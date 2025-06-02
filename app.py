import streamlit as st
import pandas as pd
import requests
from ta.volatility import AverageTrueRange
from ta.trend import MACD
from sklearn.ensemble import RandomForestRegressor

# === CONFIG ===
API_KEY = "b3fce28a971a4e408afe6c459f72dcc5"
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD"]

st.set_page_config(page_title="Analyse Forex", layout="centered")
st.title("📊 Estimation Journalière - Forex & Or")

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

        # === Targets ===
        df["delta_high"] = df["high"] - df["open"]
        df["delta_low"] = df["low"] - df["open"]
        df.dropna(inplace=True)

        # === Modèle ML ===
        features = df[["open", "atr", "macd", "macd_signal"]]
        X_train = features[:-1]
        y_high = df["delta_high"][:-1]
        y_low = df["delta_low"][:-1]

        model_high = RandomForestRegressor(n_estimators=100).fit(X_train, y_high)
        model_low = RandomForestRegressor(n_estimators=100).fit(X_train, y_low)

        # === Prédiction du dernier point
        dernier = features.iloc[-1]
        open_now = df["open"].iloc[-1]
        pred_high = model_high.predict([dernier])[0]
        pred_low = model_low.predict([dernier])[0]
        prix_max = round(open_now + pred_high, 5)
        prix_min = round(open_now + pred_low, 5)

        # === Résultat ===
        st.success(f"💡 Prix ouverture : {open_now}")
        st.metric("📈 Estimation max", prix_max)
        st.metric("📉 Estimation min", prix_min)

        # === Alerte intelligente ===
        alertes = []

        if (prix_max - prix_min) > 2 * df["atr"].mean():
            alertes.append("⚠️ Volatilité anormale prévue")

        if abs(df["macd"].iloc[-1] - df["macd_signal"].iloc[-1]) > 0.001:
            direction = "haussière 📈" if df["macd"].iloc[-1] > df["macd_signal"].iloc[-1] else "baissière 📉"
            alertes.append(f"Signal MACD détecté : tendance {direction}")

        if alertes:
            st.warning("📢 Alerte marché :")
            for alerte in alertes:
                st.write(alerte)
        else:
            st.info("✅ Aucun signal particulier détecté")
