import io
import hashlib

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

import matplotlib.pyplot as plt


# ---------------- Page config ---------------- #
st.set_page_config(page_title="Energy Dashboard", layout="wide")
st.title("⚡ Energy Dashboard")
st.write("This will become a multi-client energy dashboard.")
st.sidebar.header("Controls")
st.sidebar.write("More filters coming soon.")
st.caption("Prototype – client reference data to be added.")
st.write("Use the sidebar to navigate future views.")


# ---------------- Helper functions ---------------- #
def infer_interval_hours(df, time_col):
    dt = df[time_col].diff().dt.total_seconds().dropna()
    if dt.empty:
        return 0.25  # fallback 15 min
    return dt.mode()[0] / 3600.0


def compute_load_duration(df, power_col, dt_hours):
    ldc = df[power_col].sort_values(ascending=False).reset_index(drop=True)
    hours = (ldc.index + 1) * dt_hours
    return pd.DataFrame({"hours": hours, "power_kw": ldc.values})


def load_raw_sheet_with_auto_header(xls, sheet_name):
    """
    Reads the sheet twice max:
    1) header=None (ONLY first 200 rows) to find the header row (looking for 'kanaal')
    2) read again using that header row
    """
    # Big speedup: don't scan entire sheet to find header row
    temp = pd.read_excel(xls, sheet_name=sheet_name, header=None, nrows=200)

    mask = temp.apply(
        lambda row: row.astype(str).str.lower().str.contains("kanaal").any(),
        axis=1,
    )
    header_candidates = temp.index[mask]
    header_row = int(header_candidates[0]) if len(header_candidates) else 0

    df = pd.read_excel(xls, sheet_name=sheet_name, header=header_row)

    df.columns = df.columns.map(lambda c: str(c).strip().lower())

    ts_candidates = [
        c for c in df.columns if any(k in c for k in ["tijd", "time", "datum", "date"])
    ]
    time_col = ts_candidates[0] if ts_candidates else df.columns[0]

    pw_candidates = [
        c for c in df.columns if any(k in c for k in ["waarde", "power", "vermogen", "kw"])
    ]
    power_col = pw_candidates[0] if pw_candidates else df.columns[1]

    df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
    df["power_kw"] = pd.to_numeric(df[power_col], errors="coerce")

    df = df[["timestamp", "power_kw"]].dropna()
    return df


def simulate_battery_peak_shaving(
    df,
    power_col,
    dt_hours,
    batt_capacity_kwh,
    batt_power_kw,
    grid_limit_kw,
    rte=0.85,
):
    """
    Simple time-domain battery peak shaving simulation.
    Positive batt_kw = charging, negative batt_kw = discharging.
    """
    p_arr = df[power_col].to_numpy(dtype=float)
    n = len(p_arr)

    soc = 0.0
    soc_arr = np.empty(n, dtype=float)
    grid_arr = np.empty(n, dtype=float)
    batt_arr = np.empty(n, dtype=float)

    for i in range(n):
        p = p_arr[i]
        p_grid = p
        p_batt = 0.0

        if p > grid_limit_kw and soc > 0:
            max_discharge_kw_energy_limited = soc / dt_hours
            max_discharge_kw = min(batt_power_kw, max_discharge_kw_energy_limited)

            discharge_kw = min(p - grid_limit_kw, max_discharge_kw)
            if discharge_kw < 0:
                discharge_kw = 0.0

            energy_out = discharge_kw * dt_hours
            soc -= energy_out / rte

            p_grid = p - discharge_kw
            p_batt = -discharge_kw

        elif p < grid_limit_kw and soc < batt_capacity_kwh:
            headroom_kw = grid_limit_kw - p
            charge_kw = min(batt_power_kw, headroom_kw)

            energy_in_from_grid = charge_kw * dt_hours
            energy_stored = energy_in_from_grid * rte

            if soc + energy_stored > batt_capacity_kwh:
                energy_stored = batt_capacity_kwh - soc
                energy_in_from_grid = energy_stored / rte
                charge_kw = energy_in_from_grid / dt_hours

            soc += energy_stored
            p_grid = p + charge_kw
            p_batt = charge_kw

        soc = max(0.0, min(batt_capacity_kwh, soc))

        soc_arr[i] = soc
        grid_arr[i] = p_grid
        batt_arr[i] = p_batt

    out = df.copy()
    out["grid_kw"] = grid_arr
    out["batt_kw"] = batt_arr
    out["soc_kwh"] = soc_arr
    return out


def build_batt_sizing_table(
    df,
    power_col,
    dt_hours,
    batt_power_kw,
    grid_limit_kw,
    rte,
    capacities_kwh,
):
    original_peak = df[power_col].max()
    rows = []

    for cap in capacities_kwh:
        sim = simulate_battery_peak_shaving(
            df,
            power_col=power_col,
            dt_hours=dt_hours,
            batt_capacity_kwh=cap,
            batt_power_kw=batt_power_kw,
            grid_limit_kw=grid_limit_kw,
            rte=rte,
        )

        new_peak = sim["grid_kw"].max()
        peak_reduction = original_peak - new_peak

        discharge_kwh = sim.loc[sim["batt_kw"] < 0, "batt_kw"].abs().sum() * dt_hours
        cycles = discharge_kwh / cap if cap > 0 else 0.0

        rows.append(
            {
                "battery capacity (kWh)": cap,
                "peak shaving (new peak kW)": round(new_peak, 0),
                "peak reduction (kW)": round(peak_reduction, 0),
                "cycles/year": round(cycles, 2),
            }
        )

    return pd.DataFrame(rows)


# ---------------- Caching ---------------- #
def _hash_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


@st.cache_data(show_spinner=False)
def cached_sheet_names(xlsx_bytes: bytes):
    xls = pd.ExcelFile(io.BytesIO(xlsx_bytes))
    return xls.sheet_names


@st.cache_data(show_spinner=False)
def cached_read_raw_sheet(xlsx_bytes: bytes, sheet_name: str):
    xls = pd.ExcelFile(io.BytesIO(xlsx_bytes))
    return pd.read_excel(xls, sheet_name=sheet_name)


@st.cache_data(show_spinner=False)
def cached_load_clean_timeseries(xlsx_bytes: bytes, sheet_name: str):
    xls = pd.ExcelFile(io.BytesIO(xlsx_bytes))
    return load_raw_sheet_with_auto_header(xls, sheet_name)


@st.cache_data(show_spinner=False)
def cached_simulation(
    xlsx_bytes: bytes,
    sheet_name: str,
    batt_capacity_kwh: float,
    batt_power_kw: float,
    grid_limit_kw: float,
    rte: float,
):
    df = cached_load_clean_timeseries(xlsx_bytes, sheet_name)
    dt_hours = infer_interval_hours(df, "timestamp")
    sim_df = simulate_battery_peak_shaving(
        df,
        power_col="power_kw",
        dt_hours=dt_hours,
        batt_capacity_kwh=batt_capacity_kwh,
        batt_power_kw=batt_power_kw,
        grid_limit_kw=grid_limit_kw,
        rte=rte,
    )
    return sim_df, dt_hours


@st.cache_data(show_spinner=False)
def cached_batt_sizing_table(
    xlsx_bytes: bytes,
    sheet_name: str,
    batt_power_kw: float,
    grid_limit_kw: float,
    rte: float,
    capacities_kwh: tuple,
):
    df = cached_load_clean_timeseries(xlsx_bytes, sheet_name)
    dt_hours = infer_interval_hours(df, "timestamp")
    return build_batt_sizing_table(
        df=df,
        power_col="power_kw",
        dt_hours=dt_hours,
        batt_power_kw=batt_power_kw,
        grid_limit_kw=grid_limit_kw,
        rte=rte,
        capacities_kwh=list(capacities_kwh),
    )


# ---------------- Sidebar: view selector ---------------- #
view = st.sidebar.selectbox(
    "Choose analysis view",
    [
        "Overview",
        "Load Duration Curve",
        "Peak Load Duration (1h & 4h)",
        "Peak table (Excel reproduction)",
        "Battery simulation",
        "Battery sizing table",
        "All",
    ],
    index=0,
)

uploaded_file = st.sidebar.file_uploader(
    "Upload client Excel file",
    type=["xlsx"],
    help="Upload the raw client export (e.g. Sample file)",
)

if uploaded_file is None:
    st.info("Upload an Excel file from the sidebar to begin.")
    st.stop()

xlsx_bytes = uploaded_file.getvalue()
_ = _hash_bytes(xlsx_bytes)

sheet_names = cached_sheet_names(xlsx_bytes)

raw_candidates = [
    s for s in sheet_names if not any(k in s.lower() for k in ["load", "peak", "batt"])
]
default_sheet = raw_candidates[0] if raw_candidates else sheet_names[0]

sheet_name = st.sidebar.selectbox(
    "Select sheet (raw power data)",
    sheet_names,
    index=sheet_names.index(default_sheet),
)

raw_df = cached_read_raw_sheet(xlsx_bytes, sheet_name)
with st.expander("🧾 Raw sheet preview (as-is)", expanded=False):
    st.dataframe(raw_df.head(30), use_container_width=True)

df = cached_load_clean_timeseries(xlsx_bytes, sheet_name)
if df.empty:
    st.error("Could not detect timestamp/power columns in this sheet.")
    st.stop()

time_col = "timestamp"
power_col = "power_kw"


# ---------------- Sections ---------------- #
if view in ("Overview", "All"):
    st.subheader("✅ Cleaned time series for calculations")
    st.dataframe(df.head(), use_container_width=True)


if view in ("Load Duration Curve", "All"):
    st.subheader("📉 Load Duration Curve")

    df_ldc = df.copy()
    df_ldc["date"] = df_ldc["timestamp"].dt.date
    df_ldc["month"] = df_ldc["timestamp"].dt.to_period("M").astype(str)

    view_mode = st.radio(
        "View load duration for:",
        ["Full period", "Specific day", "Specific month"],
        horizontal=True,
        key="ldc_view_mode",
    )

    if view_mode == "Full period":
        df_slice = df_ldc
        slice_label = "Full period"
    elif view_mode == "Specific day":
        unique_dates = sorted(df_ldc["date"].unique())
        selected_date = st.selectbox("Select a day", unique_dates, key="ldc_day")
        df_slice = df_ldc[df_ldc["date"] == selected_date]
        slice_label = f"Day: {selected_date}"
    else:
        unique_months = sorted(df_ldc["month"].unique())
        selected_month = st.selectbox("Select a month (YYYY-MM)", unique_months, key="ldc_month")
        df_slice = df_ldc[df_ldc["month"] == selected_month]
        slice_label = f"Month: {selected_month}"

    if df_slice.empty:
        st.warning("No data available for this selection.")
    else:
        dt_hours = infer_interval_hours(df_slice, time_col)
        ldc_df = compute_load_duration(df_slice, power_col, dt_hours)

        total_hours = ldc_df["hours"].iloc[-1]
        ldc_df["time_pct"] = ldc_df["hours"] / total_hours * 100

        x_mode = st.radio(
            "X-axis view",
            ["Hours per period", "Percentage of time"],
            horizontal=True,
            key="ldc_x_mode",
        )

        if x_mode == "Hours per period":
            x_col = "hours"
            x_label = f"Cumulative hours ({slice_label})"
        else:
            x_col = "time_pct"
            x_label = f"Cumulative time (%) – {slice_label}"

        max_x = float(ldc_df[x_col].max())
        zoom_limit = st.slider(
            f"Show up to {x_label}",
            min_value=0.0,
            max_value=max_x,
            value=max_x,
            key="ldc_zoom",
        )

        ldc_zoom = ldc_df[ldc_df[x_col] <= zoom_limit]

        fig_ldc = px.line(
            ldc_zoom,
            x=x_col,
            y="power_kw",
            labels={x_col: x_label, "power_kw": "Power (kW)"},
            title=f"Load Duration Curve – {slice_label}",
        )
        fig_ldc.update_traces(mode="lines")
        fig_ldc.update_layout(
            hovermode="x unified",
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
        )
        st.plotly_chart(fig_ldc, use_container_width=True)


if view in ("Peak Load Duration (1h & 4h)", "All"):
    st.subheader("🔥 Peak Load Duration (1h & 4h)")

    dt_hours_full = infer_interval_hours(df, time_col)
    power = df[power_col]

    window_1h = max(int(round(1 / dt_hours_full)), 1)
    window_4h = max(int(round(4 / dt_hours_full)), 1)

    roll_1h = power.rolling(window_1h).mean()
    roll_4h = power.rolling(window_4h).mean()

    pld_1h = roll_1h.sort_values(ascending=False).reset_index(drop=True)
    pld_4h = roll_4h.sort_values(ascending=False).reset_index(drop=True)

    min_len = min(len(pld_1h), len(pld_4h))
    pld_1h = pld_1h.iloc[:min_len]
    pld_4h = pld_4h.iloc[:min_len]

    hours = np.arange(min_len) * dt_hours_full

    pld_df = pd.DataFrame(
        {
            "hours": hours,
            "power_1h_kw": pld_1h.to_numpy(),
            "power_4h_kw": pld_4h.to_numpy(),
        }
    )

    fig_pld = px.line(
        pld_df,
        x="hours",
        y=["power_1h_kw", "power_4h_kw"],
        labels={"hours": "Cumulative hours (full period)", "value": "Power (kW)"},
        title="Peak Load Duration Curves – 1h vs 4h rolling average",
    )
    fig_pld.update_layout(
        hovermode="x unified",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        legend_title_text="Window",
    )
    st.plotly_chart(fig_pld, use_container_width=True)


if view in ("Peak table (Excel reproduction)", "All"):
    st.subheader("📋 Peak load duration table (Excel reproduction)")
    dt_hours_excel = 0.25  # kept EXACTLY as requested

    peak_table = pd.DataFrame()
    peak_table["timestamp"] = df["timestamp"]
    peak_table["kW"] = df["power_kw"]
    peak_table["kWh/15min"] = df["power_kw"] * dt_hours_excel
    peak_table["1631"] = df["power_kw"].rolling(4).mean()
    peak_table["1616"] = df["power_kw"].rolling(8).mean()
    peak_table["1597"] = df["power_kw"].rolling(12).mean()

    st.dataframe(peak_table.head(200), use_container_width=True)


# ---------------- Battery controls (FORM for responsiveness) ---------------- #
needs_battery_controls = view in ("Battery simulation", "Battery sizing table", "All")
if needs_battery_controls:
    st.sidebar.subheader("🔋 Battery parameters")

    default_grid_limit = float(df[power_col].max())

    # Keep last applied params in session state
    if "batt_params" not in st.session_state:
        st.session_state.batt_params = dict(
            batt_capacity_kwh=4000.0,
            batt_power_kw=400.0,
            grid_limit_kw=min(default_grid_limit, 476.0),
            rte=0.85,
        )

    with st.sidebar.form("battery_form", clear_on_submit=False):
        batt_capacity_kwh = st.number_input(
            "Battery capacity (kWh)",
            min_value=500.0,
            max_value=8000.0,
            value=float(st.session_state.batt_params["batt_capacity_kwh"]),
            step=250.0,
        )
        batt_power_kw = st.number_input(
            "Battery power (kW)",
            min_value=50.0,
            max_value=2000.0,
            value=float(st.session_state.batt_params["batt_power_kw"]),
            step=25.0,
        )
        grid_limit_kw = st.number_input(
            "Target grid peak (kW)",
            min_value=0.0,
            max_value=float(df[power_col].max()),
            value=float(st.session_state.batt_params["grid_limit_kw"]),
            step=10.0,
        )
        rte = st.slider(
            "Round-trip efficiency",
            min_value=0.70,
            max_value=0.95,
            value=float(st.session_state.batt_params["rte"]),
            step=0.01,
        )

        apply_btn = st.form_submit_button("Apply parameters")

    if apply_btn:
        st.session_state.batt_params = dict(
            batt_capacity_kwh=float(batt_capacity_kwh),
            batt_power_kw=float(batt_power_kw),
            grid_limit_kw=float(grid_limit_kw),
            rte=float(rte),
        )

    # Always use applied params (prevents rerunning sim while dragging sliders)
    params = st.session_state.batt_params


if view in ("Battery simulation", "All"):
    st.subheader("🔋 Battery sizing – peak shaving simulation")

    with st.spinner("Running simulation..."):
        sim_df, dt_hours_sim = cached_simulation(
            xlsx_bytes=xlsx_bytes,
            sheet_name=sheet_name,
            batt_capacity_kwh=float(params["batt_capacity_kwh"]),
            batt_power_kw=float(params["batt_power_kw"]),
            grid_limit_kw=float(params["grid_limit_kw"]),
            rte=float(params["rte"]),
        )

    original_peak = df[power_col].max()
    new_peak = sim_df["grid_kw"].max()
    total_energy_throughput = sim_df["batt_kw"].abs().sum() * dt_hours_sim
    full_cycles = total_energy_throughput / params["batt_capacity_kwh"] if params["batt_capacity_kwh"] > 0 else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Original peak (kW)", f"{original_peak:.0f}")
    m2.metric("New peak (kW)", f"{new_peak:.0f}")
    m3.metric("Peak reduction (kW)", f"{(original_peak - new_peak):.0f}")
    m4.metric("Battery cycles/year (approx.)", f"{full_cycles:.1f}")

    # ----- Grid chart (Matplotlib: full resolution, fast) ----- #
    st.subheader("📉 Grid power before vs after battery (full resolution)")
    fig1 = plt.figure()
    plt.plot(sim_df[time_col].to_numpy(), sim_df[power_col].to_numpy(), label="Original (kW)")
    plt.plot(sim_df[time_col].to_numpy(), sim_df["grid_kw"].to_numpy(), label="Grid (kW)")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("kW")
    st.pyplot(fig1, clear_figure=True, use_container_width=True)

    # ----- SOC chart (Matplotlib: full resolution, fast) ----- #
    st.subheader("🔋 Battery state-of-charge over time (full resolution)")
    fig2 = plt.figure()
    plt.plot(sim_df[time_col].to_numpy(), sim_df["soc_kwh"].to_numpy())
    plt.xlabel("Time")
    plt.ylabel("SOC (kWh)")
    st.pyplot(fig2, clear_figure=True, use_container_width=True)


if view in ("Battery sizing table", "All"):
    st.subheader("📋 Battery sizing table")

    candidate_capacities = [1000, 2000, 3000, 4000, 5000, 6000]

    with st.spinner("Building sizing table..."):
        batt_table = cached_batt_sizing_table(
            xlsx_bytes=xlsx_bytes,
            sheet_name=sheet_name,
            batt_power_kw=float(params["batt_power_kw"]),
            grid_limit_kw=float(params["grid_limit_kw"]),
            rte=float(params["rte"]),
            capacities_kwh=tuple(candidate_capacities),
        )

    st.dataframe(batt_table, use_container_width=True)
