import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import warnings
import io
import numpy as np
from sklearn.cluster import KMeans # <--- OTAK BARU KITA

warnings.filterwarnings("ignore")

# --- KONFIGURASI HALAMAN WEB ---
st.set_page_config(layout="wide", page_title="Screener Saham AI-Powered")

st.title("🧠 Dashboard Akumulasi Saham (AI-Powered)")
st.markdown("""
Aplikasi ini menggunakan **Machine Learning (K-Means)** untuk menentukan area **Support & Resistance Dinamis**, 
bukan sekadar harga tertinggi/terendah.
""")

# --- INISIALISASI SESSION STATE ---
if 'hasil_scan' not in st.session_state:
    st.session_state['hasil_scan'] = None
if 'status_scan' not in st.session_state:
    st.session_state['status_scan'] = False

# --- SIDEBAR ---
st.sidebar.header("⚙️ Konfigurasi AI Screener")
default_tickers = "BBCA.JK, BBRI.JK, BMRI.JK, BBNI.JK, TLKM.JK, ASII.JK, GOTO.JK, UNVR.JK, ANTM.JK, ADRO.JK, PTBA.JK, PGAS.JK, KLBF.JK, BRIS.JK, MDKA.JK, INCO.JK, ICBP.JK, INDF.JK, AMRT.JK, JPFA.JK, MEDC.JK, HRUM.JK, TINS.JK, ESSA.JK"
ticker_input = st.sidebar.text_area("Daftar Saham", default_tickers, height=150)

st.sidebar.subheader("Parameter")
period_days = st.sidebar.slider("Periode Data (Hari)", 30, 90, 60)
max_range_pct = st.sidebar.slider("Max Lebar Kanal (%)", 5, 30, 25) / 100 # Dilonggarkan dikit karena AI lebih ketat

tombol_scan = st.sidebar.button("🚀 Scan dengan AI", type="primary")

# --- FUNGSI AI & MATEMATIKA ---

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_dynamic_snr(df):
    """
    Menggunakan K-Means Clustering untuk mencari 2 pusat gravitasi harga:
    1. Pusat Bawah = Support Zone
    2. Pusat Atas = Resistance Zone
    """
    # Kita ambil data High dan Low, lalu gabungkan jadi satu array panjang
    # Tujuannya agar AI melihat semua jejak harga ekstrem
    data_points = np.concatenate([df['Low'].values, df['High'].values, df['Close'].values])
    data_points = data_points.reshape(-1, 1)
    
    # Minta AI mencari 2 cluster utama (Atap dan Lantai)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    kmeans.fit(data_points)
    
    # Ambil pusat clusternya
    centers = kmeans.cluster_centers_.flatten()
    centers.sort()
    
    support_ai = centers[0]   # Pusat cluster bawah
    resistance_ai = centers[1] # Pusat cluster atas
    
    return support_ai, resistance_ai

def get_ai_status(ticker):
    try:
        df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        
        if len(df) < period_days: return None

        df['RSI'] = calculate_rsi(df['Close'])
        recent = df.tail(period_days)
        
        # --- LOGIKA AI DI SINI ---
        ai_support, ai_resistance = calculate_dynamic_snr(recent)
        
        current_price = recent['Close'].iloc[-1]
        
        # Hitung lebar kanal berdasarkan AI lines, bukan high/low absolut
        # Ini lebih mencerminkan "Effective Trading Range"
        ai_range = (ai_resistance - ai_support) / ai_support
        
        if ai_range <= max_range_pct:
            
            # Posisi harga relatif terhadap garis AI
            # Kita pakai buffer toleransi 5% karena garis AI ada di tengah kerumunan
            pos = (current_price - ai_support) / (ai_resistance - ai_support)
            
            # Logika Status
            if -0.1 <= pos <= 0.25: # Bisa sedikit di bawah garis support AI (False Break)
                status = "✨ AI BUY ZONE"
            elif 0.85 <= pos <= 1.1: # Bisa sedikit di atas resistance
                status = "⚠️ POTENSI BREAKOUT"
            else:
                status = "⏳ WAIT / SIDEWAYS"
            
            return {
                "Ticker": ticker.replace(".JK", ""),
                "Status": status,
                "Harga": current_price,
                "Range %": round(ai_range * 100, 2),
                "RSI": round(recent['RSI'].iloc[-1], 2),
                "Support": ai_support,      # Support AI
                "Resistance": ai_resistance,# Resistance AI
                "Data": recent
            }
    except Exception as e:
        return None
    return None

def plot_chart(data_dict):
    df = data_dict['Data']
    ticker = data_dict['Ticker']
    status = data_dict['Status']
    sup = data_dict['Support']
    res = data_dict['Resistance']
    
    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
    
    # Indikator RSI
    rsi_lines = [
        mpf.make_addplot(df['RSI'], panel=2, color='purple', ylabel='RSI', width=1.5),
        mpf.make_addplot([70]*len(df), panel=2, color='gray', linestyle='--', width=0.8),
        mpf.make_addplot([30]*len(df), panel=2, color='gray', linestyle='--', width=0.8)
    ]
    
    # Render Chart
    buf = io.BytesIO()
    
    # Trik Visualisasi: Garis AI dibuat putus-putus biru agar beda dengan support klasik
    fig, ax = mpf.plot(
        df, type='candle', style=s,
        title=f"{ticker} ({status}) - AI Range: {data_dict['Range %']}%",
        volume=True,
        addplot=rsi_lines,
        mav=(20),
        # Garis Horizontal AI (Blue Dashed)
        hlines=dict(hlines=[sup, res], colors=['b', 'b'], linestyle='-.', linewidths=(1.5, 1.5), alpha=0.8),
        panel_ratios=(6,2,2),
        savefig=dict(fname=buf, dpi=100, bbox_inches='tight'),
        returnfig=True
    )
    st.pyplot(fig)

# --- LOGIKA FRONTEND ---

if tombol_scan:
    tickers = [t.strip() for t in ticker_input.split(",")]
    results = []
    
    progress_bar = st.progress(0)
    st_text = st.empty()
    
    for i, t in enumerate(tickers):
        st_text.text(f"AI sedang berpikir untuk saham: {t}...")
        res = get_ai_status(t)
        if res:
            results.append(res)
        progress_bar.progress((i + 1) / len(tickers))
        
    st_text.text("Analisis AI Selesai!")
    progress_bar.empty()
    
    st.session_state['hasil_scan'] = results
    st.session_state['status_scan'] = True

if st.session_state['status_scan'] and st.session_state['hasil_scan']:
    results = st.session_state['hasil_scan']
    st.success(f"🤖 AI menemukan {len(results)} pola akumulasi yang valid!")
    
    # Tabel
    df_show = pd.DataFrame(results).drop(columns=['Data'])
    st.dataframe(df_show.style.highlight_max(axis=0), use_container_width=True)
    
    st.divider()
    
    # Opsi Tampilan
    mode = st.radio("Mode Tampilan:", ["Manual", "Galeri Penuh"], horizontal=True)
    
    if mode == "Manual":
        pilihan = st.selectbox("Pilih Saham:", [r['Ticker'] for r in results])
        if pilihan:
            data = next(x for x in results if x['Ticker'] == pilihan)
            plot_chart(data)
            st.info(f"ℹ️ **Catatan AI:** Garis putus-putus biru adalah 'Zona Kepadatan Harga'. Perhatikan jika candle menusuk ke bawah garis biru bawah lalu naik lagi (False Break) = Sinyal Akumulasi Kuat.")
    else:
        cols = st.columns(2)
        for i, r in enumerate(results):
            with cols[i%2]:
                plot_chart(r)

elif st.session_state['status_scan']:
    st.warning("AI tidak menemukan pola yang pas. Coba ubah parameter.")
else:
    st.info("👈 Siap untuk analisis? Klik tombol di samping.")
