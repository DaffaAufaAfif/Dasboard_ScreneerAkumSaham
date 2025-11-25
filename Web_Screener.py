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
st.set_page_config(layout="wide", page_title="Screener Saham The Golden Setup")

st.title("🏆 Dashboard Sniper Saham (Golden Setup Edition)")
st.markdown("""
Mendeteksi fase akumulasi dengan fitur spesial **'The Golden Setup'**: 
Perpaduan antara **Spring (False Break)** di Support AI + Momentum RSI.
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
    """
    Logika The Golden Setup:
    1. Spring: Low tembus support, tapi Close balik ke atas support.
    2. RSI: Harus ada tenaga ( > 30) tapi belum jenuh beli (< 60).
    """
    # Toleransi tembus 1-2%
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
        
        # Hitung Range AI
        ai_range = (ai_resistance - ai_support) / ai_support
        
        if ai_range <= max_range_pct:
            pos = (current_candle['Close'] - ai_support) / (ai_resistance - ai_support)
            
            # --- DETEKSI GOLDEN SETUP ---
            is_golden = detect_golden_setup(current_candle, ai_support)
            
            status = ""
            signal_label = "NETRAL" # Label Default
            
            if is_golden:
                status = "✨ GOLDEN SETUP"
                signal_label = "🏆 GOLDEN SETUP" # Label Spesial
            elif -0.05 <= pos <= 0.25:
                status = "BUY ZONE (Support)"
                signal_label = "✅ BUY"
            elif 0.85 <= pos <= 1.05:
                status = "POTENSI BREAKOUT"
                signal_label = "⚠️ BREAKOUT"
            else:
                status = "SIDEWAYS"
                signal_label = "💤 WAIT"
            
            return {
                "Ticker": ticker.replace(".JK", ""),
                "Signal": signal_label, # Kolom Baru
                "Status": status,
                "Harga": current_candle['Close'],
                "Range %": round(ai_range * 100, 2),
                "RSI": round(current_candle['RSI'], 2),
                "Support": ai_support,
                "Resistance": ai_resistance,
                "Data": recent
            }
    except:
        return None
    return None

def plot_chart(data_dict):
    df = data_dict['Data']
    ticker = data_dict['Ticker']
    signal = data_dict['Signal']
    
    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
    
    rsi_lines = [
        mpf.make_addplot(df['RSI'], panel=2, color='purple', ylabel='RSI', width=1.5),
        mpf.make_addplot([70]*len(df), panel=2, color='gray', linestyle='--', width=0.8),
        mpf.make_addplot([30]*len(df), panel=2, color='gray', linestyle='--', width=0.8)
    ]
    
    buf = io.BytesIO()
    
    # Judul Chart akan ada Trophy-nya jika Golden Setup
    title_text = f"{ticker} [{signal}]"
    
    fig, ax = mpf.plot(
        df, type='candle', style=s,
        title=title_text,
        volume=True,
        addplot=rsi_lines,
        mav=(20),
        hlines=dict(hlines=[data_dict['Support'], data_dict['Resistance']], colors=['b','b'], linestyle='-.', linewidths=(1.5,1.5)),
        panel_ratios=(6,2,2),
        savefig=dict(fname=buf, dpi=100, bbox_inches='tight'),
        returnfig=True
    )
    st.pyplot(fig)

# --- FRONTEND ---
if tombol_scan:
    tickers = [t.strip() for t in ticker_input.split(",")]
    results = []
    
    progress_bar = st.progress(0)
    st_text = st.empty()
    
    for i, t in enumerate(tickers):
        st_text.text(f"Mencari Golden Setup di: {t}...")
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
    
    # Pisahkan yang GOLDEN SETUP taruh paling atas
    df_res = pd.DataFrame(results)
    
    # Sorting: Prioritaskan yang punya sinyal '🏆'
    df_res['Priority'] = df_res['Signal'].apply(lambda x: 0 if '🏆' in x else (1 if '✅' in x else 2))
    df_res = df_res.sort_values(by='Priority')
    df_display = df_res.drop(columns=['Data', 'Priority']) # Hapus kolom bantuan
    
    # Hitung jumlah Golden Setup
    golden_count = len(df_res[df_res['Signal'].str.contains('🏆')])
    
    if golden_count > 0:
        st.balloons() # Efek Balon jika ketemu Golden Setup!
        st.success(f"DITEMUKAN {golden_count} SAHAM GOLDEN SETUP! SIAP PANTAU!")
    else:
        st.info("Belum ada 'Golden Setup' sempurna hari ini. Cek saham 'BUY ZONE' di bawah.")

    # Tampilkan Tabel
    def color_signal(val):
        color = 'white'
        if '🏆' in val: color = '#ffd700' # Emas
        elif '✅' in val: color = '#90ee90' # Hijau Muda
        elif '⚠️' in val: color = '#ffcccb' # Merah Muda
        return f'background-color: {color}; color: black; font-weight: bold'

    st.dataframe(df_display.style.map(color_signal, subset=['Signal']), use_container_width=True)
    
    st.divider()
    
    mode = st.radio("Mode Lihat Chart:", ["Manual Pilih", "Semua Golden Setup", "Semua Hasil"], horizontal=True)
    
    if mode == "Manual Pilih":
        pilihan = st.selectbox("Pilih Saham:", [r['Ticker'] for r in results])
        if pilihan:
            data = next(x for x in results if x['Ticker'] == pilihan)
            plot_chart(data)
            if "🏆" in data['Signal']:
                st.warning("🔥 **ANALISIS:** Saham ini membentuk pola **SPRING**. Harga sempat turun menjebol Support AI (Garis Biru Bawah), tapi berhasil ditutup naik kembali. Potensi Reversal Kuat!")
            elif "✅" in data['Signal']:
                st.info("✅ **ANALISIS:** Harga berada di zona aman (Support). Cicil beli dengan risiko rendah.")
                
    elif mode == "Semua Golden Setup":
        golden_results = [r for r in results if "🏆" in r['Signal']]
        if golden_results:
            cols = st.columns(2)
            for i, r in enumerate(golden_results):
                with cols[i%2]:
                    plot_chart(r)
        else:
            st.warning("Tidak ada chart Golden Setup untuk ditampilkan.")
            
    else: # Semua Hasil
        cols = st.columns(2)
        for i, r in enumerate(results):
            with cols[i%2]:
                plot_chart(r)

elif st.session_state['status_scan']:
    st.warning("Tidak ditemukan saham yang sesuai parameter.")
