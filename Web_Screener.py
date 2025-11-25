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
st.set_page_config(layout="wide", page_title="Screener Saham Smart Context")

st.title("💎 Dashboard Sniper Saham (Context Aware)")
st.markdown("""
Mendeteksi fase akumulasi dengan **Kecerdasan Kontekstual**. 
Aplikasi membedakan kriteria risiko antara saham Stabil (Blue Chip) dan Saham Agresif (Volatil).
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
# Kita hapus slider Max Range karena sekarang Range dipakai untuk klasifikasi otomatis
st.sidebar.info("ℹ️ Max Range sekarang otomatis menyesuaikan jenis saham.")

tombol_scan = st.sidebar.button("🚀 Analisis Cerdas", type="primary")

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

def detect_spring_pattern(row, ai_support):
    return (row['Low'] < ai_support) and (row['Close'] > ai_support * 0.995)

def get_ai_status(ticker):
    try:
        df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if len(df) < period_days: return None

        df['RSI'] = calculate_rsi(df['Close'])
        recent = df.tail(period_days)
        
        ai_support, ai_resistance = calculate_dynamic_snr(recent)
        classic_low = recent['Low'].min()
        gap_pct = (ai_support - classic_low) / ai_support
        
        current_candle = recent.iloc[-1]
        ai_range = (ai_resistance - ai_support) / ai_support
        
        # --- 1. DETEKSI TIPE SAHAM (KARAKTER) ---
        stock_type = "Normal"
        max_gap_allowed = 0.04  # Default 4%
        
        if ai_range <= 0.12: # Range < 12% (Sangat Stabil / Blue Chip Like)
            stock_type = "🛡️ Stabil (Blue Chip Like)"
            max_gap_allowed = 0.025 # Aturan Ketat: Gap max 2.5%
        elif ai_range <= 0.22: # Range 12-22% (Normal / Second Liner)
            stock_type = "⚖️ Moderat (Second Liner)"
            max_gap_allowed = 0.045 # Aturan Standar: Gap max 4.5%
        else: # Range > 22% (Agresif / Gorengan Like)
            stock_type = "🔥 Agresif (Volatil)"
            max_gap_allowed = 0.07 # Aturan Longgar: Gap max 7%

        # --- 2. ANALISIS KEPUTUSAN DENGAN KONTEKS ---
        pos = (current_candle['Close'] - ai_support) / (ai_resistance - ai_support)
        is_spring = detect_spring_pattern(current_candle, ai_support)
        rsi_good = (current_candle['RSI'] > 30) and (current_candle['RSI'] < 65)
        
        signal_label = "NETRAL"
        status_desc = "Wait and See"
        
        # Logika Keputusan (Adaptif terhadap Tipe Saham)
        if gap_pct <= max_gap_allowed: # Lolos Filter Jurang (Sesuai Tipe)
            
            if is_spring and rsi_good:
                signal_label = "💎 DIAMOND"
                status_desc = f"Perfect Reversal untuk tipe {stock_type}"
            
            elif pos <= 0.15: # Dekat Support AI
                # Bedakan Diamond dan Safe Buy berdasarkan kedekatan dengan Support Klasik
                if gap_pct <= (max_gap_allowed * 0.6): # Gap sangat tipis
                    signal_label = "🥇 GOLDEN" 
                    status_desc = "Sangat Aman (Best Price)"
                else:
                    signal_label = "✅ SAFE BUY"
                    status_desc = "Akumulasi Wajar"
            
            elif 0.85 <= pos <= 1.05:
                signal_label = "⚠️ BREAKOUT"
                status_desc = "Dekat Resistance"
            else:
                signal_label = "💤 WAIT"
                status_desc = "Sideways di Tengah"

        else: # Gagal Filter Jurang (Gap terlalu lebar untuk tipenya)
            signal_label = "❌ TRAP / WAIT"
            status_desc = f"Jurang {gap_pct*100:.1f}% terlalu lebar untuk saham {stock_type}"

        # Keterangan Tambahan
        rsi_val = current_candle['RSI']
        if rsi_val < 30: ket_rsi = "🟢 Oversold"
        elif rsi_val < 45: ket_rsi = "📈 Bangkit"
        elif rsi_val > 70: ket_rsi = "🔴 Overbought"
        else: ket_rsi = "⚪ Netral"

        return {
            "Ticker": ticker.replace(".JK", ""),
            "Keputusan": signal_label,
            "Tipe Saham": stock_type, # Info Baru
            "Harga": current_candle['Close'],
            "Support AI": round(ai_support, 0),
            "Low Klasik": round(classic_low, 0),
            "Gap %": gap_pct,
            "Range %": ai_range,
            "RSI": round(rsi_val, 0),
            "Ket. RSI": ket_rsi,
            "Data": recent,
            "Resistance": ai_resistance
        }

    except:
        return None
    return None

def plot_chart(data_dict):
    df = data_dict['Data']
    ticker = data_dict['Ticker']
    signal = data_dict['Keputusan']
    tipe = data_dict['Tipe Saham']
    
    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
    
    rsi_lines = [
        mpf.make_addplot(df['RSI'], panel=2, color='purple', ylabel='RSI', width=1.5),
        mpf.make_addplot([70]*len(df), panel=2, color='gray', linestyle='--', width=0.8),
        mpf.make_addplot([30]*len(df), panel=2, color='gray', linestyle='--', width=0.8)
    ]
    
    buf = io.BytesIO()
    title_text = f"{ticker} [{signal}] - {tipe}"
    
    fig, ax = mpf.plot(
        df, type='candle', style=s, title=title_text, volume=True,
        addplot=rsi_lines, mav=(20),
        hlines=dict(hlines=[data_dict['Support AI'], data_dict['Resistance'], data_dict['Low Klasik']], 
                    colors=['b','b', 'r'], linestyle=['-.', '-.', ':'], linewidths=(1.5, 1.5, 1.0)), 
        panel_ratios=(6,2,2),
        savefig=dict(fname=buf, dpi=100, bbox_inches='tight'),
        returnfig=True
    )
    st.pyplot(fig)

    with st.container():
        st.info(f"💡 **KONTEKS SAHAM:** Ini adalah saham tipe **{tipe}**. Volatilitasnya {data_dict['Range %']*100:.1f}%.")
        if "TRAP" in signal:
            st.error(f"⚠️ **PERINGATAN:** Jarak ke lantai bawah ({data_dict['Gap %']*100:.1f}%) dianggap terlalu berbahaya untuk karakteristik saham {tipe} ini.")
    st.divider()

# --- FRONTEND ---
if tombol_scan:
    tickers = [t.strip() for t in ticker_input.split(",")]
    results = []
    progress_bar = st.progress(0)
    st_text = st.empty()
    
    for i, t in enumerate(tickers):
        st_text.text(f"Analisis Kontekstual: {t}...")
        res = get_ai_status(t)
        if res: results.append(res)
        progress_bar.progress((i + 1) / len(tickers))
        
    st_text.text("Selesai!")
    progress_bar.empty()
    st.session_state['hasil_scan'] = results
    st.session_state['status_scan'] = True

if st.session_state['status_scan'] and st.session_state['hasil_scan']:
    results = st.session_state['hasil_scan']
    df_res = pd.DataFrame(results)
    
    def assign_priority(sig):
        if '💎' in sig: return 0
        if '🥇' in sig: return 1
        if '✅' in sig: return 2
        if '⚠️' in sig: return 3
        if '❌' in sig: return 5
        return 4

    df_res['Priority'] = df_res['Keputusan'].apply(assign_priority)
    df_res = df_res.sort_values(by=['Priority'])
    
    diamond_count = len(df_res[df_res['Keputusan'].str.contains('💎')])
    if diamond_count > 0:
        st.balloons()
        st.success(f"DITEMUKAN {diamond_count} DIAMOND SETUP (CONTEXT MATCHED)!")

    def color_signal(val):
        if '💎' in val: return 'background-color: #00ced1; color: white; font-weight: bold'
        if '🥇' in val: return 'background-color: #ffd700; color: black; font-weight: bold'
        if '✅' in val: return 'background-color: #90ee90; color: black; font-weight: bold'
        if '⚠️' in val: return 'background-color: #ffcccb; color: black; font-weight: bold'
        if '❌' in val: return 'background-color: #808080; color: white; font-weight: bold'
        return ''

    cols_order = ['Ticker', 'Keputusan', 'Tipe Saham', 'Harga', 'Support AI', 'Low Klasik', 'RSI']
    
    st.dataframe(
        df_res[cols_order].style.map(color_signal, subset=['Keputusan']), 
        use_container_width=True,
        column_config={
            "Harga": st.column_config.NumberColumn(format="Rp %d"),
            "Support AI": st.column_config.NumberColumn(format="Rp %d"),
            "Low Klasik": st.column_config.NumberColumn(format="Rp %d"),
            "RSI": st.column_config.NumberColumn(format="%.0f"),
        }
    )
    st.divider()
    
    mode = st.radio("Filter Chart:", ["Tampilkan Diamond & Golden", "Tampilkan Semua Buy", "Lihat Semua"], horizontal=True)
    
    if mode == "Tampilkan Diamond & Golden":
        top = [r for r in results if '💎' in r['Keputusan'] or '🥇' in r['Keputusan']]
        if top: 
            for r in top: plot_chart(r)
        else: st.warning("Tidak ada setup premium hari ini.")
    elif mode == "Tampilkan Semua Buy":
        buys = [r for r in results if '💎' in r['Keputusan'] or '🥇' in r['Keputusan'] or '✅' in r['Keputusan']]
        for r in buys: plot_chart(r)
    else:
        for r in results: plot_chart(r)

elif st.session_state['status_scan']:
    st.warning("Tidak ditemukan saham yang sesuai.")
