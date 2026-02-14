import pandas as pd
import streamlit as st
import requests

st.set_page_config(page_title="Supply Dashboard", layout="wide")

st.title("Supply Dashboard — Forecast & Stock Risk")

api_url = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")

store_id = st.sidebar.number_input("store_id", min_value=1, value=1)
item_id = st.sidebar.text_input("item_id (family)", "AUTOMOTIVE")
horizon = st.sidebar.slider("horizon_days", 7, 60, 14)
on_hand = st.sidebar.number_input("on_hand", min_value=0.0, value=0.0)
lead_time = st.sidebar.slider("lead_time_days", 1, 60, 7)
service_level = st.sidebar.slider("service_level", 0.50, 0.999, 0.95)

if st.sidebar.button("Run forecast"):
    payload = {
        "store_id": int(store_id),
        "item_id": str(item_id),
        "horizon_days": int(horizon),
        "on_hand": float(on_hand),
        "lead_time_days": int(lead_time),
        "service_level": float(service_level)
    }
    r = requests.post(f"{api_url}/forecast", json=payload, timeout=30)
    if r.status_code != 200:
        st.error(r.text)
    else:
        data = r.json()
        df_fc = pd.DataFrame([{
            "date": p["date"],
            "yhat": p["yhat"],
            "lower": p["yhat_lower"],
            "upper": p["yhat_upper"],
        } for p in data["series"]])

        c1, c2, c3 = st.columns(3)
        c1.metric("Prob stockout", f"{data['stock']['prob_stockout']:.2%}")
        c2.metric("Reorder point", f"{data['stock']['reorder_point']:.1f}")
        c3.metric("Recommended order", f"{data['stock']['recommended_order_qty']:.1f}")

        st.subheader("Forecast")
        st.line_chart(df_fc.set_index("date")[["yhat", "lower", "upper"]])

        st.subheader("Details")
        st.dataframe(df_fc, use_container_width=True)
        st.json(data["stock"])
        st.json(data["meta"])