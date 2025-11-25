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
st.set_page_config(layout="wide", page_title="Screener Saham Decision Maker")

st.title("💎 Dashboard Sniper Saham (Decision Maker)")
st.markdown("""
Mendeteksi fase akumulasi dengan **Satu Keputusan Mutlak**. 
Risiko 'Jurang Lebar' sudah dihitung otomatis ke dalam sinyal. Tidak perlu analisis manual lagi.
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

tombol_scan = st.sidebar.button("🚀 Analisis Keputusan Final", type="primary")

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
    # Pola Spring: Low tembus support, Close balik ke atas support
    return (row['Low'] < ai_support) and (row['Close'] > ai_support * 0.995)

def get_ai_status(ticker):
    try:
        df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if len(df) < period_days: return None

        df['RSI'] = calculate_rsi(df['Close'])
        recent = df.tail(period_days)
        
        # 1. Hitung Support AI & Support Klasik
        ai_support, ai_resistance = calculate_dynamic_snr(recent)
        classic_low = recent['Low'].min()
        
        # 2. Hitung GAP (Risiko Jurang)
        gap_pct = (ai_support - classic_low) / ai_support
        
        # 3. Analisis Posisi
        current_candle = recent.iloc[-1]
        ai_range = (ai_resistance - ai_support) / ai_support
        pos = (current_candle['Close'] - ai_support) / (ai_resistance - ai_support)
        
        # 4. Deteksi Pola
        is_spring = detect_spring_pattern(current_candle, ai_support)
        rsi_good = (current_candle['RSI'] > 30) and (current_candle['RSI'] < 65)
        
        # --- LOGIKA KEPUTUSAN FINAL (THE BRAIN) ---
        signal_label = "NETRAL"
        status_desc = "Wait and See"
        
        if ai_range <= max_range_pct:
            
            # A. KELOMPOK GOLDEN (Spring Pattern)
            if is_spring and rsi_good:
                if gap_pct <= 0.025: # Gap < 2.5% (Sangat Aman)
                    signal_label = "💎 DIAMOND"
                    status_desc = "Perfect Setup: Reversal + Low Risk"
                elif gap_pct <= 0.04: # Gap < 4% (Standar)
                    signal_label = "🥇 GOLDEN"
                    status_desc = "Strong Reversal Pattern"
                else: # Gap Lebar (Bahaya)
                    signal_label = "⚠️ SPECULATIVE"
                    status_desc = "Reversal tapi Jurang Lebar (High Risk)"
            
            # B. KELOMPOK BUY ON SUPPORT (Tanpa Spring)
            elif pos <= 0.15: # Dekat Support AI
                if gap_pct <= 0.03: # Gap Kecil
                    signal_label = "✅ SAFE BUY"
                    status_desc = "Aman di Support (Lantai Kuat)"
                else: # Gap Lebar
                    signal_label = "❌ TRAP / WAIT"
                    status_desc = "Jebakan: Support AI Rapuh (Jurang Dalam)"
            
            # C. LAINNYA
            elif 0.85 <= pos <= 1.05:
                signal_label = "⚠️ WATCH BREAKOUT"
                status_desc = "Dekat Resistance"
            else:
                signal_label = "💤 WAIT"
                status_desc = "No Clear Signal"

            # Keterangan Tambahan
            if ai_range <= 0.15: ket_range = "😴 Tidur"
            else: ket_range = "🙂 Normal"

            return {
                "Ticker": ticker.replace(".JK", ""),
                "Keputusan": signal_label, # Kolom Utama
                "Alasan": status_desc,
                "Gap Risiko": f"{round(gap_pct*100, 1)}%", # Info saja
                "Harga": current_candle['Close'],
                "Support AI": round(ai_support, 0),
                "Low Klasik": round(classic_low, 0),
                "RSI": round(current_candle['RSI'], 0),
                "Data": recent,
                "Resistance": ai_resistance, # Untuk chart
                "Raw Gap": gap_pct # Untuk sorting
            }
    except:
        return None
    return None

def plot_chart(data_dict):
    df = data_dict['Data']
    ticker = data_dict['Ticker']
    signal = data_dict['Keputusan']
    alasan = data_dict['Alasan']
    
    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
    
    rsi_lines = [
        mpf.make_addplot(df['RSI'], panel=2, color='purple', ylabel='RSI', width=1.5),
        mpf.make_addplot([70]*len(df), panel=2, color='gray', linestyle='--', width=0.8),
        mpf.make_addplot([30]*len(df), panel=2, color='gray', linestyle='--', width=0.8)
    ]
    
    buf = io.BytesIO()
    title_text = f"{ticker} [{signal}]"
    
    # Chart 3 Garis (AI + Klasik)
    fig, ax = mpf.plot(
        df, type='candle', style=s, title=title_text, volume=True,
        addplot=rsi_lines, mav=(20),
        hlines=dict(hlines=[data_dict['Support AI'], data_dict['Resistance'], data_dict['Low Klasik']], 
                    colors=['b','b', 'r'], 
                    linestyle=['-.', '-.', ':'],
                    linewidths=(1.5, 1.5, 1.0)), 
        panel_ratios=(6,2,2),
        savefig=dict(fname=buf, dpi=100, bbox_inches='tight'),
        returnfig=True
    )
    st.pyplot(fig)

    # Penjelasan Tegas
    with st.container():
        if "DIAMOND" in signal:
            st.success(f"**KEPUTUSAN: BELI AGRESIF (AMAN).** {alasan}. Support AI dan Klasik berimpit, risiko minim.")
        elif "GOLDEN" in signal:
            st.success(f"**KEPUTUSAN: BELI STANDARD.** {alasan}. Masuk bertahap.")
        elif "SAFE BUY" in signal:
            st.info(f"**KEPUTUSAN: CICIL BELI (AMAN).** {alasan}. Harga di dasar yang solid.")
        elif "SPECULATIVE" in signal:
            st.warning(f"**KEPUTUSAN: SPEKULASI / HATI-HATI.** {alasan}. Ada jurang di bawah support. Siapkan Cut Loss ketat.")
        elif "TRAP" in signal:
            st.error(f"**KEPUTUSAN: JANGAN BELI! (SKIP).** {alasan}. Risiko jatuh ke Low Klasik ({data_dict['Low Klasik']:,.0f}) terlalu besar.")
        else:
            st.write(f"**KEPUTUSAN: WAIT.** {alasan}.")
    st.divider()

# --- FRONTEND ---
if tombol_scan:
    tickers = [t.strip() for t in ticker_input.split(",")]
    results = []
    progress_bar = st.progress(0)
    st_text = st.empty()
    
    for i, t in enumerate(tickers):
        st_text.text(f"Menghitung Risiko & Peluang: {t}...")
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
    
    # Sorting Prioritas Mutlak (Diamond > Golden > Safe > Lainnya)
    def assign_priority(sig):
        if '💎' in sig: return 0
        if '🥇' in sig: return 1
        if '✅' in sig: return 2
        if '⚠️' in sig: return 3 # Speculative & Breakout
        if '❌' in sig: return 5 # Trap (Paling bawah atau warning)
        return 4

    df_res['Priority'] = df_res['Keputusan'].apply(assign_priority)
    df_res = df_res.sort_values(by=['Priority', 'Raw Gap']) # Sort by Priority, then by Gap Terkecil
    
    diamond_count = len(df_res[df_res['Keputusan'].str.contains('💎')])
    if diamond_count > 0:
        st.balloons()
        st.success(f"DITEMUKAN {diamond_count} SAHAM DIAMOND SETUP (PERFECT)!")

    def color_signal(val):
        if '💎' in val: return 'background-color: #00ced1; color: white; font-weight: bold' # Turquoise
        if '🥇' in val: return 'background-color: #ffd700; color: black; font-weight: bold' # Gold
        if '✅' in val: return 'background-color: #90ee90; color: black; font-weight: bold' # Green
        if '⚠️' in val: return 'background-color: #ffcccb; color: black; font-weight: bold' # Reddish
        if '❌' in val: return 'background-color: #808080; color: white; font-weight: bold' # Grey
        return ''

    cols_order = ['Ticker', 'Keputusan', 'Gap Risiko', 'Harga', 'Support AI', 'Low Klasik', 'RSI']
    
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
