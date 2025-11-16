# AA_forecast_app.py
# AAvenue — Premium Forecast Studio (C1 theme)
# Supports: Prophet, ARIMA, LSTM. Large-file friendly (Dask).
# Run with Python 3.10 for best compatibility.

import io
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np

# plotting
import plotly.express as px
import plotly.graph_objects as go

# optional heavy libs (import gracefully)
_missing = []
try:
    import dask.dataframe as dd
except Exception:
    dd = None
    _missing.append("dask")

try:
    from prophet import Prophet
except Exception:
    Prophet = None
    _missing.append("prophet")

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None
    _missing.append("statsmodels")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
except Exception:
    tf = None
    Sequential = None
    LSTM = None
    Dense = None
    _missing.append("tensorflow")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------
# THEME COLORS for C1 (Pure White Corporate)
# -----------------------
C1 = {
    "bg": "#FFFFFF",           # page background
    "card": "#F7F9FC",         # card bg
    "text": "#0F1B2D",         # main text navy
    "muted": "#5C6B7A",        # muted text
    "accent": "#1B71D1",       # primary accent (chart blue)
    "accent2": "#00A8A8",      # secondary accent (teal)
    "highlight": "#F9A03F",    # highlight (orange)
}

# -----------------------
# PAGE CONFIG + CSS (C1)
# -----------------------
st.set_page_config(page_title="AAvenue — Premium Forecast Studio (C1)",
                   page_icon="📈",
                   layout="wide")

_STYLES = f"""
<style>
/* page */
body {{ background: {C1['bg']}; color: {C1['text']}; }}
.stApp > header {{ background: transparent; }}
.big-title {{ font-size: 36px; font-weight:800; color: {C1['text']}; margin-bottom: 4px; }}
.subtitle {{ font-size:14px; color: {C1['muted']}; margin-bottom: 18px; }}
.card {{
  background: {C1['card']};
  border-left: 4px solid {C1['accent']};
  padding: 18px;
  border-radius: 10px;
  box-shadow: 0 6px 20px rgba(13,27,45,0.06);
  margin-bottom: 14px;
}}
.kpi {{ font-size:22px; font-weight:800; color: {C1['text']}; }}
.kpi-sub {{ font-size:12px; color: {C1['muted']}; }}
.small-muted {{font-size:12px; color: {C1['muted']};}}
.btn-primary {{
  background: linear-gradient(90deg, {C1['accent']}, {C1['accent2']});
  color: white; padding: 8px 12px; border-radius:8px; font-weight:700; border:none;
}}
.model-pill {{ display:inline-block; padding:6px 10px; margin-right:6px; border-radius:999px; background: {C1['card']}; border:1px solid #e6eef8; font-weight:700; color:{C1['text']}; }}
</style>
"""
st.markdown(_STYLES, unsafe_allow_html=True)

# Header
st.markdown(f"<div class='big-title'>📈 AAvenue — Premium Forecast Studio</div>", unsafe_allow_html=True)
st.markdown(f"<div class='subtitle'>White Corporate theme · Prophet | ARIMA | LSTM · Large-file support · Interactive charts</div>", unsafe_allow_html=True)
st.write("---")

# Missing packages alert
if _missing:
    st.warning(
        "Some optional packages are not installed: " + ", ".join(_missing)
        + ".\nFor full functionality, install them with your Python 3.10 executable; e.g.:\n\n"
        "'C:\\\\Users\\\\mahendar.v\\\\AppData\\\\Local\\\\Programs\\\\Python\\\\Python310\\\\python.exe -m pip install {}'".format(
            " ".join(_missing)
        )
    )

# -----------------------
# Sidebar: Data + Options
# -----------------------
st.sidebar.header("📥 Data & Settings")

upload_mode = st.sidebar.radio("File mode", ["Normal upload (pandas)", "Large file (Dask)"])
uploaded = st.sidebar.file_uploader("Upload CSV file (or generate sample)", type=["csv"])

if st.sidebar.button("Generate realistic sample (2 years daily)"):
    # generate seasonal daily sample (2023-2024)
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")
    rng = np.random.default_rng(42)
    seasonal = 50 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
    trend = np.linspace(0, 300, len(dates))
    noise = rng.normal(0, 20, len(dates))
    sales = (200 + seasonal + trend + noise).round().astype(int)
    sample_df = pd.DataFrame({
        "date": dates,
        "sales": sales,
        # extra columns to simulate Walmart-like retail dataset
        "store_id": np.random.choice(range(1, 11), size=len(dates)),
        "item_id": np.random.choice(range(1000, 1010), size=len(dates)),
        "promo": np.random.choice([0, 1], size=len(dates), p=[0.85, 0.15]),
        "on_hand": np.random.randint(20, 500, size=len(dates))
    })
    buf = io.StringIO()
    sample_df.to_csv(buf, index=False)
    buf.seek(0)
    uploaded = buf
    st.sidebar.success("Sample generated — go to main area to load and map columns.")

# file encoding option
encoding = st.sidebar.selectbox("File encoding (if upload fails)", ["utf-8", "latin1", "cp1252"])

# parsing options
st.sidebar.subheader("Parsing / preprocessing")
date_guess = st.sidebar.checkbox("Auto-detect date column", value=True)
drop_na_dates = st.sidebar.checkbox("Drop rows with invalid dates", value=True)
resample_freq = st.sidebar.selectbox("If resample: Frequency", ["None", "D (daily)", "W (weekly)", "M (monthly)"])
resample_freq_map = {"None": None, "D (daily)": "D", "W (weekly)": "W", "M (monthly)": "M"}

# model hyperparams
st.sidebar.subheader("Model controls")
default_periods = st.sidebar.slider("Forecast horizon (periods)", 3, 90, 12)
train_test_split = st.sidebar.slider("Train/Test split (%)", 50, 95, 80)

# LSTM controls
st.sidebar.markdown("**LSTM settings (in-browser quick mode)**")
lstm_window = st.sidebar.number_input("LSTM window size", min_value=3, max_value=120, value=20)
lstm_epochs = st.sidebar.number_input("LSTM epochs (keep low in-browser)", min_value=1, max_value=200, value=20)
lstm_batch = st.sidebar.number_input("LSTM batch size", min_value=1, max_value=256, value=16)

# Prophet seasonalities toggle
prophet_add_seasonality = st.sidebar.checkbox("Prophet add monthly seasonality", value=True)

# -----------------------
# Load Data (pandas or dask)
# -----------------------
@st.cache_data
def load_with_pandas(file_obj, encoding):
    try:
        return pd.read_csv(file_obj, encoding=encoding)
    except Exception:
        return pd.read_csv(file_obj)

@st.cache_data
def load_with_dask(file_obj, encoding):
    if dd is None:
        raise RuntimeError("Dask not installed.")
    return dd.read_csv(file_obj).compute()

df = None
if uploaded:
    try:
        if upload_mode == "Large file (Dask)":
            df = load_with_dask(uploaded, encoding)
        else:
            df = load_with_pandas(uploaded, encoding)
        st.success(f"Loaded dataset — shape {df.shape}")
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.stop()
else:
    st.info("Upload a CSV or generate a sample to begin (sidebar).")

# -----------------------
# If df loaded: mapping, cleaning, explore preview
# -----------------------
if df is not None:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔎 Data Preview & Column Mapping")
    st.write("First 5 rows (map date/target columns on the right):")
    st.dataframe(df.head())
    st.markdown("</div>", unsafe_allow_html=True)

    cols = list(df.columns)

    # auto detect date candidates
    candidate_dates = []
    if date_guess:
        for name in ["date", "ds", "timestamp", "time", "day"]:
            if name in df.columns:
                candidate_dates.append(name)
        for c in df.columns:
            if np.issubdtype(df[c].dtype, np.datetime64):
                candidate_dates.append(c)
        if not candidate_dates:
            for c in df.columns[:6]:
                try:
                    pd.to_datetime(df[c].dropna().iloc[:50], errors="raise")
                    candidate_dates.append(c)
                except Exception:
                    pass

    default_date = candidate_dates[0] if candidate_dates else cols[0]

    st.sidebar.subheader("Column mapping")
    date_col = st.sidebar.selectbox("Date column", cols, index=cols.index(default_date) if default_date in cols else 0)

    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        st.error("No numeric column found to forecast. Please upload a dataset containing a numeric target column (sales, quantity, etc.).")
        st.stop()
    target_col = st.sidebar.selectbox("Target (y) column", numeric_cols, index=0)

    # optional grouping column
    extra_cols = [c for c in cols if c not in (date_col, target_col)]
    group_col = st.sidebar.selectbox("Optional: group/sku column (for multi series)", ["(none)"] + extra_cols)

    # convert and clean date
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if drop_na_dates:
        before = len(df)
        df = df.dropna(subset=[date_col])
        after = len(df)
        if before != after:
            st.sidebar.info(f"Dropped {before-after} rows with invalid dates.")

    # reduce to necessary columns
    if group_col != "(none)":
        df = df[[group_col, date_col, target_col]].copy()
    else:
        df = df[[date_col, target_col]].copy()

    # rename to ds,y
    df = df.rename(columns={date_col: "ds", target_col: "y"})
    df = df.sort_values("ds").reset_index(drop=True)

    # optionally resample
    freq_map = resample_freq_map[resample_freq]
    if freq_map is not None:
        try:
            df = df.set_index("ds").resample(freq_map).sum().reset_index()
            st.sidebar.success(f"Resampled to {freq_map}")
        except Exception as e:
            st.sidebar.warning(f"Resample failed: {e}")

    # KPI numbers
    total_sales = float(df["y"].sum())
    mean_sales = float(df["y"].mean())
    last_value = float(df["y"].iloc[-1])
    prev_period_val = float(df["y"].iloc[-2]) if len(df) > 1 else np.nan
    pct_change = (last_value - prev_period_val) / prev_period_val * 100 if prev_period_val and prev_period_val != 0 else np.nan

    # KPI cards
    col1, col2, col3, col4 = st.columns([1.4, 1.4, 1.4, 2.0])
    with col1:
        st.markdown(f"<div class='card'><div class='kpi'>{total_sales:,.0f}</div><div class='kpi-sub'>Total (sum) of target</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='card'><div class='kpi'>{mean_sales:,.2f}</div><div class='kpi-sub'>Mean (avg)</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='card'><div class='kpi'>{last_value:,.0f}</div><div class='kpi-sub'>Last value</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='card'><div class='kpi'>{(f'{pct_change:.1f}%') if not np.isnan(pct_change) else 'N/A'}</div><div class='kpi-sub'>Change vs prev</div></div>", unsafe_allow_html=True)

    st.write("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Explore", "Insights", "Forecast", "Download"])

    # -----------------------
    # Explore tab
    # -----------------------
    with tab1:
        st.header("Interactive Time Series Explorer")
        fig = px.line(df, x="ds", y="y", title="Target over time", template="plotly_white")
        fig.update_traces(line=dict(color=C1["accent"], width=2))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color=C1["text"]))
        st.plotly_chart(fig, use_container_width=True)

        # date-range selector
        min_date = pd.to_datetime(df["ds"].min()).to_pydatetime()
        max_date = pd.to_datetime(df["ds"].max()).to_pydatetime()
        dr = st.slider("Select date range for inspection", min_value=min_date, max_value=max_date, value=(min_date, max_date), format="YYYY-MM-DD")
        df_range = df[(df["ds"] >= dr[0]) & (df["ds"] <= dr[1])]

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Distribution")
            fig2 = px.histogram(df_range, x="y", nbins=50, marginal="box", title="Value distribution", template="plotly_white")
            fig2.update_traces(marker_line_width=0)
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            st.subheader("Lag plot (lag=1)")
            df_range_local = df_range.copy()
            df_range_local["lag_1"] = df_range_local["y"].shift(1)
            fig3 = px.scatter(df_range_local, x="lag_1", y="y", trendline="ols", title="Lag 1 scatter", template="plotly_white")
            st.plotly_chart(fig3, use_container_width=True)

    # -----------------------
    # Insights tab: rolling stats, autocorrelation heatmap
    # -----------------------
    with tab2:
        st.header("📘 Data Insights & Seasonality Analysis")

        st.markdown("""
        ### 📌 What this section shows
        This page helps you understand **patterns and behavior** in your time series data.

        **1️⃣ Rolling Mean & Rolling Standard Deviation**  
        - Rolling Mean smooths the data and shows long-term trends.  
        - Rolling Standard Deviation shows how much the data fluctuates.  
        If the standard deviation increases over time, it indicates volatility.

        **2️⃣ Autocorrelation (Lag Analysis)**  
        - Lag-1 correlation shows if today’s value depends on yesterday’s value.  
        - Lag-7 correlation helps detect **weekly seasonality**.  
        Higher values mean stronger repeating patterns.

        Understanding these patterns helps the forecasting models (Prophet, ARIMA, LSTM)
        make better predictions.
        """)

        # Rolling stats
        st.subheader("📈 Rolling Statistics (7-day window)")
        window = 7
        # safety: ensure y exists
        if "y" not in df.columns:
            st.error("Target column 'y' not found after mapping. Re-check mapping.")
        else:
            df["rolling_mean"] = df["y"].rolling(window).mean()
            df["rolling_std"] = df["y"].rolling(window).std()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["ds"], y=df["y"],
                name="Original Series",
                line=dict(color=C1["accent"])
            ))
            fig.add_trace(go.Scatter(
                x=df["ds"], y=df["rolling_mean"],
                name=f"{window}-day Rolling Mean",
                line=dict(color=C1["accent2"], width=3)
            ))
            fig.add_trace(go.Scatter(
                x=df["ds"], y=df["rolling_std"],
                name=f"{window}-day Rolling Std",
                line=dict(color=C1["highlight"], width=1, dash="dash")
            ))

            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=C1["text"])
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **Interpretation:**  
            - If Rolling Mean is trending upward → your business is growing.  
            - If Rolling Standard Deviation is rising → demand is becoming more unpredictable.  
            """)

            # Autocorrelation
            st.subheader("🔄 Autocorrelation Heatmap (Lag 1 & Lag 7)")
            df["lag1"] = df["y"].shift(1)
            df["lag7"] = df["y"].shift(7)
            corr_mat = df[["y", "lag1", "lag7"]].corr()

            fig = px.imshow(
                corr_mat,
                text_auto=True,
                color_continuous_scale="Blues",
                title="Correlation Strength Between Lags"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **How to read this:**  
            - **High Lag-1 correlation** → strong short-term dependence  
              (today's value depends on yesterday).  
            - **High Lag-7 correlation** → weekly seasonality (common in retail).  
            """)

    # -----------------------
    # Forecast tab: models and options
    # -----------------------
    with tab3:
        st.header("🔮 Forecast Center")
        st.subheader("🤖 Choose Forecast Model (C1)")
        model_choice_text = st.radio(
            "Select a forecasting model",
            [
                "Prophet (business patterns)",
                "ARIMA (stable trends)",
                "LSTM Neural Network (deep learning)"
            ],
            horizontal=True
        )
        if "Prophet" in model_choice_text:
            model_choice = "Prophet"
        elif "ARIMA" in model_choice_text:
            model_choice = "ARIMA"
        else:
            model_choice = "LSTM"

        st.success(f"Model Selected: **{model_choice}**")

        periods = st.number_input("Forecast horizon (periods)", min_value=1, max_value=365, value=default_periods)
        use_test_split = st.checkbox("Show train/test backtest (split and score)", value=True)

        # split
        split_pct = train_test_split / 100.0
        split_idx = int(len(df) * split_pct)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy() if use_test_split and split_idx < len(df) else None

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write(f"Training rows: {len(train_df)}" + (f" — Test rows: {len(test_df)}" if test_df is not None else ""))
        st.markdown("</div>", unsafe_allow_html=True)

        # helper for freq detection for Prophet
        def infer_prophet_freq(series):
            try:
                freq = pd.infer_freq(series)
                return freq if freq is not None else "D"
            except Exception:
                return "D"

        # Run model
        forecast_out = None
        metrics = {}

        if model_choice == "Prophet":
            if Prophet is None:
                st.error("Prophet is not installed. Install via your Python 3.10: python -m pip install prophet")
            else:
                with st.spinner("Training Prophet model..."):
                    try:
                        m = Prophet(daily_seasonality=True)
                        if prophet_add_seasonality:
                            try:
                                m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                            except Exception:
                                pass
                        m.fit(train_df[["ds", "y"]])
                        freq = infer_prophet_freq(train_df["ds"])
                        future = m.make_future_dataframe(periods=periods, freq=freq)
                        forecast_df = m.predict(future)
                        forecast_out = forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].set_index("ds")
                        if test_df is not None:
                            try:
                                joined = forecast_out.join(test_df.set_index("ds")["y"], how="inner")
                                if not joined.empty:
                                    y_true = joined["y"].values
                                    y_pred = joined["yhat"].values[:len(y_true)]
                                    metrics["MAE"] = mean_absolute_error(y_true, y_pred)
                                    metrics["RMSE"] = mean_squared_error(y_true, y_pred, squared=False)
                            except Exception:
                                pass
                    except Exception as e:
                        st.error(f"Prophet training failed: {e}")

        elif model_choice == "ARIMA":
            if ARIMA is None:
                st.error("statsmodels (ARIMA) not installed. Install via: python -m pip install statsmodels")
            else:
                with st.spinner("Fitting ARIMA (1,1,1)..."):
                    try:
                        model = ARIMA(train_df["y"], order=(1, 1, 1))
                        model_fit = model.fit()
                        preds = model_fit.forecast(steps=periods)
                        last_date = pd.to_datetime(train_df["ds"].iloc[-1])
                        future_idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq="D")
                        forecast_out = pd.DataFrame({"yhat": preds}, index=future_idx)
                        if test_df is not None:
                            try:
                                test_preds = model_fit.forecast(steps=len(test_df))
                                metrics["MAE"] = mean_absolute_error(test_df["y"].values, test_preds)
                                metrics["RMSE"] = mean_squared_error(test_df["y"].values, test_preds, squared=False)
                            except Exception:
                                pass
                    except Exception as e:
                        st.error(f"ARIMA failed: {e}")

        elif model_choice == "LSTM":
            if tf is None:
                st.error("TensorFlow not installed. Install via: python -m pip install tensorflow")
            else:
                if len(train_df) < max(50, lstm_window + 10):
                    st.error(f"Not enough data for LSTM. Need at least {max(50, lstm_window + 10)} rows; you have {len(train_df)}.")
                else:
                    with st.spinner("Training LSTM (this may take time)..."):
                        try:
                            scaler = MinMaxScaler()
                            series = train_df["y"].values.reshape(-1, 1)
                            series_scaled = scaler.fit_transform(series)
                            X, y_seq = [], []
                            w = lstm_window
                            for i in range(w, len(series_scaled)):
                                X.append(series_scaled[i-w:i])
                                y_seq.append(series_scaled[i, 0])
                            X = np.array(X)
                            y_seq = np.array(y_seq)
                            model = Sequential()
                            model.add(LSTM(64, activation="tanh", input_shape=(w, 1)))
                            model.add(Dense(1))
                            model.compile(optimizer="adam", loss="mse")
                            model.fit(X, y_seq, epochs=lstm_epochs, batch_size=lstm_batch, verbose=0)
                            last_seq = series_scaled[-w:].copy()
                            preds = []
                            for _ in range(periods):
                                p = model.predict(last_seq.reshape(1, w, 1), verbose=0)[0][0]
                                preds.append(p)
                                last_seq = np.concatenate([last_seq[1:], [[p]]], axis=0)
                            preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
                            last_date = pd.to_datetime(train_df["ds"].iloc[-1])
                            future_idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq="D")
                            forecast_out = pd.DataFrame({"yhat": preds_inv}, index=future_idx)
                            metrics["MAE"] = None
                            metrics["RMSE"] = None
                        except Exception as e:
                            st.error(f"LSTM training failed: {e}")

        # Display forecast plot
        if forecast_out is not None:
            st.subheader("Forecast Chart")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], name="Historical", mode="lines", line=dict(color=C1["accent"], width=3)))

            # forecast may be DataFrame with index as dates
            if "yhat" in forecast_out.columns:
                fig.add_trace(go.Scatter(x=forecast_out.index, y=forecast_out["yhat"], name="Forecast", mode="lines", line=dict(color=C1["highlight"], width=3)))
            else:
                # if Prophet style with yhat column but different structure
                try:
                    fig.add_trace(go.Scatter(x=forecast_out.index, y=forecast_out.iloc[:, 0], name="Forecast", mode="lines", line=dict(color=C1["highlight"], width=3)))
                except Exception:
                    pass

            if "yhat_lower" in forecast_out.columns:
                fig.add_trace(go.Scatter(x=forecast_out.index, y=forecast_out["yhat_lower"], name="Lower", mode="lines", line=dict(color="#e8e8e8"), opacity=0.5, showlegend=False))
            if "yhat_upper" in forecast_out.columns:
                fig.add_trace(go.Scatter(x=forecast_out.index, y=forecast_out["yhat_upper"], name="Upper", mode="lines", line=dict(color="#e8e8e8"), opacity=0.5, showlegend=False))

            fig.update_layout(template="plotly_white", height=520, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=C1["text"]))
            st.plotly_chart(fig, use_container_width=True)

            # Show forecast table & metrics
            st.subheader("Forecast Output")
            out_df = forecast_out.reset_index().rename(columns={"index": "date"})
            st.dataframe(out_df.head(max(10, len(out_df))))

            if metrics:
                st.subheader("Backtest Metrics (if available)")
                for k, v in metrics.items():
                    if v is None:
                        st.write(f"**{k}**: N/A")
                    else:
                        st.write(f"**{k}**: {v:.3f}")

            csv = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(label="⬇️ Download Forecast CSV",data=csv,file_name=f"forecast_{model_choice}.csv",mime="text/csv")

        else:
            st.info("Model did not produce a forecast. Check messages above.")

    # -----------------------
    # Download / Export
    # -----------------------
    with tab4:
        st.header("Export & Save")
        st.write("Download the processed dataset (ds,y).")
        csv_orig = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download processed dataset (csv)", csv_orig, "processed_data.csv", mime="text/csv")

        st.write("You can also copy the sample code snippet to reproduce the forecast offline.")
        st.code("""# sample snippet to load processed CSV and run Prophet
import pandas as pd
from prophet import Prophet
df = pd.read_csv("processed_data.csv")
df['ds'] = pd.to_datetime(df['ds'])
m = Prophet()
m.fit(df[['ds','y']])
future = m.make_future_dataframe(periods=12)
fc = m.predict(future)
""", language="python")
else:
    # no df loaded
    st.info("When you upload a CSV the app will process and show model options. Try the 'Generate sample' button in the sidebar to test the app.")
