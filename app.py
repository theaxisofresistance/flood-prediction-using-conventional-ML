import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# =========================
# CONFIG
# =========================
DEFAULT_MODEL_PATH = "./model_data.pkl"

REGIONS = {
    "Jakarta Pusat": (-6.1862, 106.8063),
    "Jakarta Selatan": (-6.2615, 106.8106),
    "Jakarta Timur": (-6.2251, 106.9004),
    "Jakarta Utara": (-6.1335, 106.8821),
}

OWM_UNITS = "metric"
DEFAULT_FLOOD_THRESHOLD = 0.45


# =========================
# MODEL LOADER
# =========================
@st.cache_resource
def load_model_from_path(model_path: str):
    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "Failed to load model_data.pkl because a training dependency is missing. "
            "Add the package used to create the pickle to requirements.txt "
            f"(missing module: {err.name})."
        ) from err

    required_keys = [
        "le_region",
        "region_map",
        "feature_cols",
        "scaler",
        "flood_model",
        "best_params",
    ]

    missing = [key for key in required_keys if key not in model_data]
    if missing:
        raise KeyError(f"Missing key(s) in model_data.pkl: {missing}")

    return model_data


def load_model_from_uploaded_file(uploaded_file):
    try:
        model_data = pickle.load(uploaded_file)
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "Failed to load the uploaded model because a training dependency is missing. "
            "Add the package used to create the pickle to requirements.txt "
            f"(missing module: {err.name})."
        ) from err

    required_keys = [
        "le_region",
        "region_map",
        "feature_cols",
        "scaler",
        "flood_model",
        "best_params",
    ]

    missing = [key for key in required_keys if key not in model_data]
    if missing:
        raise KeyError(f"Missing key(s) in uploaded model: {missing}")

    return model_data


# =========================
# WEATHER FETCHING
# =========================
def fetch_owm_forecast(region_name: str, lat: float, lon: float, api_key: str):
    url = "https://api.openweathermap.org/data/2.5/forecast"

    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": OWM_UNITS,
        "cnt": 40,
    }

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()

    data = resp.json()
    data["_region_name"] = region_name

    return data


def parse_owm_response(data: dict):
    rows = []
    region_name = data.get("_region_name", "Unknown")

    for slot in data["list"]:
        rows.append(
            {
                "region_name": region_name,
                "dt_txt": slot["dt_txt"],
                "temp_min": slot["main"]["temp_min"],
                "temp_max": slot["main"]["temp_max"],
                "temp": slot["main"]["temp"],
                "humidity": slot["main"]["humidity"],
                "rain_3h": slot.get("rain", {}).get("3h", 0.0),
                "wind_speed": slot["wind"]["speed"],
                "wind_deg": slot["wind"]["deg"],
            }
        )

    df = pd.DataFrame(rows)
    df["dt_txt"] = pd.to_datetime(df["dt_txt"])

    return df


# =========================
# PREPROCESSING
# =========================
def circular_mean_deg(angles_deg):
    rad = np.deg2rad(angles_deg)
    return np.degrees(
        np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())
    ) % 360


def preprocess_inference(df: pd.DataFrame, le_region):
    df_3h = df.copy()
    df_3h["date"] = df_3h["dt_txt"].dt.normalize()

    df_daily = (
        df_3h.groupby(["region_name", "date"], sort=True)
        .agg(
            Tn=("temp_min", "min"),
            Tx=("temp_max", "max"),
            Tavg=("temp", "mean"),
            RH_avg=("humidity", "mean"),
            RR=("rain_3h", "sum"),
            ff_avg=("wind_speed", "mean"),
            ddd_x=("wind_deg", circular_mean_deg),
        )
        .reset_index()
    )

    df_daily = df_daily.sort_values(["region_name", "date"]).reset_index(drop=True)

    try:
        df_daily["region_id"] = le_region.transform(df_daily["region_name"].astype(str))
    except ValueError as err:
        raise ValueError(
            "Ada region yang tidak dikenal oleh LabelEncoder model. "
            "Pastikan region di Streamlit sama dengan data training."
        ) from err

    rad = np.deg2rad(df_daily["ddd_x"])
    df_daily["wind_direction_sin"] = np.sin(rad)
    df_daily["wind_direction_cos"] = np.cos(rad)
    df_daily.drop(columns=["ddd_x"], inplace=True)

    return df_daily


def feature_engineer_inference(df: pd.DataFrame):
    df = df.copy().sort_values(["region_id", "date"]).reset_index(drop=True)

    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    df.drop(columns=["month", "day_of_year"], inplace=True)

    grp = df.groupby("region_id")["RR"]

    df["rainfall_lag1d"] = grp.shift(1).fillna(0.0)
    df["rainfall_lag3d"] = grp.shift(3).fillna(0.0)
    df["rainfall_lag7d"] = grp.shift(7).fillna(0.0)

    df["rainfall_rolling3d_sum"] = grp.transform(
        lambda s: s.rolling(3, min_periods=1).sum()
    )
    df["rainfall_rolling7d_sum"] = grp.transform(
        lambda s: s.rolling(7, min_periods=1).sum()
    )
    df["rainfall_rolling14d_sum"] = grp.transform(
        lambda s: s.rolling(14, min_periods=1).sum()
    )
    df["rainfall_rolling3d_max"] = grp.transform(
        lambda s: s.rolling(3, min_periods=1).max()
    )
    df["rainfall_rolling7d_max"] = grp.transform(
        lambda s: s.rolling(7, min_periods=1).max()
    )
    df["rainfall_rolling7d_std"] = grp.transform(
        lambda s: s.rolling(7, min_periods=2).std().fillna(0.0)
    )

    grp_rh = df.groupby("region_id")["RH_avg"]

    df["humidity_rolling3d_mean"] = grp_rh.transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    df["humidity_rolling7d_mean"] = grp_rh.transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )

    rain_cols = [col for col in df.columns if "rainfall" in col]

    for col in rain_cols:
        df[col] = np.log1p(df[col])

    return df


# =========================
# INFERENCE
# =========================
def predict_flood(df_3h: pd.DataFrame, model_data: dict, flood_threshold: float):
    le_region = model_data["le_region"]
    feature_cols = model_data["feature_cols"]
    scaler = model_data["scaler"]
    model = model_data["flood_model"]

    df_daily = preprocess_inference(df_3h, le_region)
    df_feat = feature_engineer_inference(df_daily)

    missing_features = [col for col in feature_cols if col not in df_feat.columns]
    if missing_features:
        raise KeyError(f"Missing feature(s) after preprocessing: {missing_features}")

    X_infer = df_feat[feature_cols].values
    X_infer_scaled = scaler.transform(X_infer)

    flood_proba = model.predict_proba(X_infer_scaled)[:, 1]
    flood_pred = (flood_proba >= flood_threshold).astype(int)

    df_result = df_feat[
        ["region_name", "date", "Tn", "Tx", "Tavg", "RH_avg", "RR"]
    ].copy()

    df_result["flood_probability"] = flood_proba.round(4)
    df_result["flood_alert"] = flood_pred
    df_result["alert_label"] = df_result["flood_alert"].map(
        {0: "No Flood", 1: "⚠ FLOOD"}
    )

    return df_result, df_daily, df_feat


def build_summary(df_result: pd.DataFrame):
    summary = df_result[
        [
            "region_name",
            "date",
            "RR",
            "RH_avg",
            "Tavg",
            "flood_probability",
            "alert_label",
        ]
    ].copy()

    summary.columns = [
        "Region",
        "Date",
        "Rain (mm)",
        "Humidity (%)",
        "Temp (°C)",
        "Flood Prob.",
        "Alert",
    ]

    summary["Date"] = summary["Date"].dt.strftime("%Y-%m-%d")
    summary["Rain (mm)"] = summary["Rain (mm)"].round(2)
    summary["Humidity (%)"] = summary["Humidity (%)"].round(2)
    summary["Temp (°C)"] = summary["Temp (°C)"].round(2)

    return summary


def build_region_coordinates(regions):
    return pd.DataFrame(
        [
            {"Region": name, "Latitude": lat, "Longitude": lon}
            for name, (lat, lon) in REGIONS.items()
            if name in regions
        ]
    )


def build_prediction_map_data(summary: pd.DataFrame):
    region_status = (
        summary.groupby("Region", as_index=False)
        .agg(
            **{
                "Max Flood Prob.": ("Flood Prob.", "max"),
                "Flood Alert Days": (
                    "Alert",
                    lambda alerts: int((alerts == "⚠ FLOOD").sum()),
                ),
                "Total Rain (mm)": ("Rain (mm)", "sum"),
                "Avg Humidity (%)": ("Humidity (%)", "mean"),
            }
        )
    )

    map_df = build_region_coordinates(region_status["Region"].tolist())
    map_df = map_df.merge(region_status, on="Region", how="left")
    map_df["Status"] = np.where(
        map_df["Flood Alert Days"] > 0,
        "Flood Alert",
        "No Flood",
    )
    map_df["Max Flood Prob."] = map_df["Max Flood Prob."].round(4)
    map_df["Total Rain (mm)"] = map_df["Total Rain (mm)"].round(2)
    map_df["Avg Humidity (%)"] = map_df["Avg Humidity (%)"].round(2)

    return map_df


def render_map_view(map_df: pd.DataFrame):
    if map_df.empty:
        st.info("Tidak ada region untuk ditampilkan di peta.")
        return

    st.map(
        map_df.rename(columns={"Latitude": "lat", "Longitude": "lon"})[
            ["lat", "lon"]
        ],
        use_container_width=True,
    )


def render_prediction_map_view(map_df: pd.DataFrame, summary: pd.DataFrame):
    if map_df.empty:
        st.info("Tidak ada region untuk ditampilkan di peta.")
        return

    color_map = {
        "Flood Alert": "#d62828",
        "No Flood": "#2a9d8f",
    }

    marker_colors = [
        color_map.get(status, "#457b9d") for status in map_df["Status"]
    ]

    hover_text = [
        (
            f"{row['Region']}<br>"
            f"Status: {row['Status']}<br>"
            f"Max Flood Prob.: {row['Max Flood Prob.']:.2%}<br>"
            f"Flood Alert Days: {int(row['Flood Alert Days'])}<br>"
            f"Total Rain: {row['Total Rain (mm)']:.2f} mm"
        )
        for _, row in map_df.iterrows()
    ]

    fig = go.Figure(
        go.Scattermapbox(
            lat=map_df["Latitude"],
            lon=map_df["Longitude"],
            mode="markers",
            marker={
                "size": 18,
                "color": marker_colors,
            },
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            customdata=map_df[["Region"]].to_numpy(),
        )
    )

    fig.update_layout(
        mapbox={
            "style": "open-street-map",
            "center": {
                "lat": float(map_df["Latitude"].mean()),
                "lon": float(map_df["Longitude"].mean()),
            },
            "zoom": 9,
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        clickmode="event+select",
        height=420,
    )

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        key="prediction_map_chart",
        on_select="rerun",
    )

    st.caption("Klik titik pada peta untuk melihat ringkasan region.")

    selection = event.selection if event else None
    if isinstance(selection, dict):
        points = selection.get("points", [])
    else:
        points = getattr(selection, "points", []) if selection else []

    selected_region = None
    if points:
        selected_region = points[0]["customdata"][0]

    if selected_region:
        selected_summary = summary[summary["Region"] == selected_region].copy()
        st.markdown(f"**Summary for {selected_region}**")
        st.dataframe(selected_summary, use_container_width=True, hide_index=True)
    else:
        st.info("Belum ada region yang dipilih dari peta.")


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(
    page_title="Welcome to Floodium",
    layout="wide",
)

st.title("Welcome to Floodium")
st.caption("Peramal banjir berbasis ML model yang cepat dan akurat")

with st.sidebar:
    st.header("Configuration")

    api_key = "97bc38df5fcf84748795471418d012c6"

    flood_threshold = st.slider(
        "Flood Threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_FLOOD_THRESHOLD,
        step=0.01,
    )

    model_path = DEFAULT_MODEL_PATH

    selected_regions = st.multiselect(
        "Region",
        options=list(REGIONS.keys()),
        default=list(REGIONS.keys()),
    )

    run_button = st.button("Run Prediction", type="primary", use_container_width=True)

st.subheader("Region View")
region_df = build_region_coordinates(selected_regions)
map_tab, coordinate_tab = st.tabs(["Map View", "Coordinates"])

with map_tab:
    render_map_view(region_df)

with coordinate_tab:
    st.dataframe(region_df, use_container_width=True, hide_index=True)

if run_button:
    if not selected_regions:
        st.error("Pilih minimal 1 region.")
        st.stop()

    if not api_key:
        st.error("OpenWeatherMap API Key wajib diisi.")
        st.stop()

    model_data = load_model_from_path(model_path)

    raw_responses = {}
    fetch_errors = []

    progress = st.progress(0)
    status = st.empty()

    for i, region in enumerate(selected_regions, start=1):
        lat, lon = REGIONS[region]

        try:
            status.info(f"Fetching forecast: {region}")
            raw_responses[region] = fetch_owm_forecast(region, lat, lon, api_key)
        except Exception as err:
            fetch_errors.append(f"{region}: {err}")

        progress.progress(i / len(selected_regions))

    status.empty()

    if fetch_errors:
        st.error("Sebagian data forecast gagal diambil.")
        for err in fetch_errors:
            st.write(err)

    if not raw_responses:
        st.stop()

    try:
        frames_3h = [parse_owm_response(data) for data in raw_responses.values()]
        df_3h = pd.concat(frames_3h, ignore_index=True)

        df_result, df_daily, df_feat = predict_flood(
            df_3h=df_3h,
            model_data=model_data,
            flood_threshold=flood_threshold,
        )

        summary = build_summary(df_result)

    except Exception as err:
        st.error(f"Gagal melakukan inference: {err}")
        st.stop()

    total_rows = len(summary)
    flood_rows = int((summary["Alert"] == "⚠ FLOOD").sum())
    max_prob = float(summary["Flood Prob."].max())

    col1, col2, col3 = st.columns(3)
    col1.metric("Forecast Rows", total_rows)
    col2.metric("Flood Alerts", flood_rows)
    col3.metric("Max Probability", f"{max_prob:.2%}")

    st.subheader("Prediction View")
    prediction_map_df = build_prediction_map_data(summary)
    prediction_map_tab, summary_tab, chart_tab = st.tabs(
        ["Map View", "Summary", "Trend"]
    )

    with prediction_map_tab:
        render_prediction_map_view(prediction_map_df, summary)
        st.dataframe(
            prediction_map_df,
            use_container_width=True,
            hide_index=True,
        )

    with summary_tab:
        st.dataframe(summary, use_container_width=True, hide_index=True)

    with chart_tab:
        chart_df = summary.copy()
        chart_df["Date"] = pd.to_datetime(chart_df["Date"])
        chart_df = chart_df.pivot_table(
            index="Date",
            columns="Region",
            values="Flood Prob.",
            aggfunc="mean",
        )
        st.line_chart(chart_df)

  
    csv = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Prediction CSV",
        data=csv,
        file_name=f"flood_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

else:
    st.info("Isi API key dan model, lalu klik Run Prediction.")
