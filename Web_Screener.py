import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import warnings
import io
import numpy as np
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# --- KONFIGURASI ---
st.set_page_config(layout="wide", page_title="Screener Saham Pro Edition")

st.title("🏆 Dashboard Sniper Saham (Pro Edition)")
st.markdown("""
Mendeteksi fase akumulasi dengan **Format Data Profesional**. 
Kolom Keterangan terpisah dan format harga dalam Rupiah yang rapi.
""")

if 'hasil_scan' not in st.session_state:
    st.session_state['hasil_scan'] = None
if 'status_scan' not in st.session_state:
    st.session_state['status_scan'] = False

# --- SIDEBAR ---
st.sidebar.header("⚙️ Konfigurasi")
default_tickers = "BBCA.JK, BBRI.JK, BMRI.JK, BBNI.JK, TLKM.JK, ASII.JK, GOTO.JK, UNVR.JK, ANTM.JK, ADRO.JK, PTBA.JK, PGAS.JK, KLBF.JK, BRIS.JK, MDKA.JK, INCO.JK, ICBP.JK, INDF.JK, AMRT.JK, JPFA.JK, MEDC.JK, HRUM.JK, TINS.JK, ESSA.JK, AKRA.JK, EXCL.JK, ISAT.JK"
ticker_input = st.sidebar.text_area("Daftar Saham", default_tickers, height=150)

st.sidebar.subheader("Parameter")
period_days = st.sidebar.slider("Periode Data", 30, 90, 60)
max_range_pct = st.sidebar.slider("Max Range (%)", 5, 30, 25) / 100

tombol_scan = st.sidebar.button("🚀 Cari Golden Setup", type="primary")

# --- FUNGSI LOGIKA ---

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_dynamic_snr(df):
    data_points = np.concatenate([df['Low'].values, df['High'].values, df['Close'].values])
    data_points = data_points.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    kmeans.fit(data_points)
    centers = kmeans.cluster_centers_.flatten()
    centers.sort()
    return centers[0], centers[1]

def detect_golden_setup(row, ai_support):
    is_spring = (row['Low'] < ai_support) and (row['Close'] > ai_support * 0.995)
    is_rsi_good = (row['RSI'] > 30) and (row['RSI'] < 60)
    return is_spring and is_rsi_good

def get_ai_status(ticker):
    try:
        df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if len(df) < period_days: return None

        df['RSI'] = calculate_rsi(df['Close'])
        recent = df.tail(period_days)
        
        ai_support, ai_resistance = calculate_dynamic_snr(recent)
        current_candle = recent.iloc[-1]
        
        ai_range = (ai_resistance - ai_support) / ai_support
        
        if ai_range <= max_range_pct:
            pos = (current_candle['Close'] - ai_support) / (ai_resistance - ai_support)
            is_golden = detect_golden_setup(current_candle, ai_support)
            
            status = ""
            signal_label = "NETRAL"
            
            if is_golden:
                status = "✨ GOLDEN SETUP"
                signal_label = "🏆 GOLDEN"
            elif -0.05 <= pos <= 0.25:
                status = "BUY ZONE (Support)"
                signal_label = "✅ BUY"
            elif 0.85 <= pos <= 1.05:
                status = "POTENSI BREAKOUT"
                signal_label = "⚠️ BREAKOUT"
            else:
                status = "SIDEWAYS"
                signal_label = "💤 WAIT"
            
            # --- LOGIKA KETERANGAN DETAIL ---
            if ai_range <= 0.15: ket_range = "😴 Tidur / Kalem"
            elif ai_range <= 0.25: ket_range = "🙂 Normal"
            else: ket_range = "⚡ Liar / Volatil"

            rsi_val = current_candle['RSI']
            if rsi_val < 30: ket_rsi = "🟢 Oversold (Jenuh Jual)"
            elif rsi_val < 45: ket_rsi = "📈 Mulai Bangkit"
            elif rsi_val > 70: ket_rsi = "🔴 Overbought (Jenuh Beli)"
            else: ket_rsi = "⚪ Netral"

            return {
                "Ticker": ticker.replace(".JK", ""),
                "Signal": signal_label,
                "Status": status,
                "Harga": current_candle['Close'],
                "Range %": round(ai_range * 100, 2),
                "Ket. Range": ket_range,
                "RSI": round(rsi_val, 0),
                "Ket. RSI": ket_rsi,
                "Support": round(ai_support, 0),      # Dibulatkan di data
                "Resistance": round(ai_resistance, 0),# Dibulatkan di data
                "Data": recent
            }
    except:
        return None
    return None

def plot_chart(data_dict):
    df = data_dict['Data']
    ticker = data_dict['Ticker']
    signal = data_dict['Signal']
    ket_range = data_dict['Ket. Range']
    ket_rsi = data_dict['Ket. RSI']
    
    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
    
    rsi_lines = [
        mpf.make_addplot(df['RSI'], panel=2, color='purple', ylabel='RSI', width=1.5),
        mpf.make_addplot([70]*len(df), panel=2, color='gray', linestyle='--', width=0.8),
        mpf.make_addplot([30]*len(df), panel=2, color='gray', linestyle='--', width=0.8)
    ]
    
    buf = io.BytesIO()
    title_text = f"{ticker} [{signal}] | {ket_range} | {ket_rsi}"
    
    fig, ax = mpf.plot(
        df, type='candle', style=s, title=title_text, volume=True,
        addplot=rsi_lines, mav=(20),
        hlines=dict(hlines=[data_dict['Support'], data_dict['Resistance']], colors=['b','b'], linestyle='-.', linewidths=(1.5,1.5)),
        panel_ratios=(6,2,2),
        savefig=dict(fname=buf, dpi=100, bbox_inches='tight'),
        returnfig=True
    )
    st.pyplot(fig)

    with st.container():
        st.info(f"💡 **ARTIKEL GRAFIK:** Saham ini tergolong **{ket_range}** dengan kondisi indikator **{ket_rsi}**.")
    st.divider()

# --- FRONTEND ---
if tombol_scan:
    tickers = [t.strip() for t in ticker_input.split(",")]
    results = []
    
    progress_bar = st.progress(0)
    st_text = st.empty()
    
    for i, t in enumerate(tickers):
        st_text.text(f"Analisis Detail: {t}...")
        res = get_ai_status(t)
        if res:
            results.append(res)
        progress_bar.progress((i + 1) / len(tickers))
        
    st_text.text("Selesai!")
    progress_bar.empty()
    st.session_state['hasil_scan'] = results
    st.session_state['status_scan'] = True

if st.session_state['status_scan'] and st.session_state['hasil_scan']:
    results = st.session_state['hasil_scan']
    
    df_res = pd.DataFrame(results)
    
    # Sorting Prioritas
    df_res['Priority'] = df_res['Signal'].apply(lambda x: 0 if '🏆' in x else (1 if '✅' in x else 2))
    df_res = df_res.sort_values(by='Priority')
    
    golden_count = len(df_res[df_res['Signal'].str.contains('🏆')])
    if golden_count > 0:
        st.balloons()
        st.success(f"DITEMUKAN {golden_count} SAHAM GOLDEN SETUP!")

    def color_signal(val):
        color = 'white'
        if '🏆' in val: color = '#ffd700'
        elif '✅' in val: color = '#90ee90'
        elif '⚠️' in val: color = '#ffcccb'
        return f'background-color: {color}; color: black; font-weight: bold'

    cols_order = [
        'Ticker', 'Signal', 'Status', 'Harga', 
        'Range %', 'Ket. Range',
        'RSI', 'Ket. RSI',
        'Support', 'Resistance'
    ]
    
    df_display = df_res[cols_order]
    
    st.dataframe(
        df_display.style.map(color_signal, subset=['Signal']), 
        use_container_width=True,
        column_config={
            "Harga": st.column_config.NumberColumn(format="Rp %d"),
            "Range %": st.column_config.NumberColumn(format="%.2f %%"),
            # FORMAT BARU: Support & Resistance pakai Rupiah
            "Support": st.column_config.NumberColumn(format="Rp %d"),
            "Resistance": st.column_config.NumberColumn(format="Rp %d")
        }
    )
    st.divider()
    
    mode = st.radio("Mode Lihat Chart:", ["Manual Pilih", "Semua Golden Setup", "Semua Hasil"], horizontal=True)
    
    if mode == "Manual Pilih":
        pilihan = st.selectbox("Pilih Saham:", [r['Ticker'] for r in results])
        if pilihan:
            data = next(x for x in results if x['Ticker'] == pilihan)
            plot_chart(data)
            
    elif mode == "Semua Golden Setup":
        golden_results = [r for r in results if "🏆" in r['Signal']]
        if golden_results:
            for r in golden_results:
                plot_chart(r)
        else:
            st.warning("Tidak ada Golden Setup.")
            
    else: 
        for r in results:
            plot_chart(r)

elif st.session_state['status_scan']:
    st.warning("Tidak ditemukan saham yang sesuai parameter.")
