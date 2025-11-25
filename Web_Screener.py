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

st.title("🏆 Dashboard Sniper Saham (Smart Analysis)")
st.markdown("""
Mendeteksi fase akumulasi dengan fitur **AI-Explanation**. 
Aplikasi akan menjelaskan arti grafik dan memberikan kesimpulan strategi secara otomatis.
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
    # Logika The Golden Setup: Low tembus support, Close balik ke atas support + RSI Kondusif
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
                signal_label = "🏆 GOLDEN SETUP"
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
                "Signal": signal_label,
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
    sup = data_dict['Support']
    res = data_dict['Resistance']
    rsi_val = data_dict['RSI']
    price = data_dict['Harga']
    
    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
    
    rsi_lines = [
        mpf.make_addplot(df['RSI'], panel=2, color='purple', ylabel='RSI', width=1.5),
        mpf.make_addplot([70]*len(df), panel=2, color='gray', linestyle='--', width=0.8),
        mpf.make_addplot([30]*len(df), panel=2, color='gray', linestyle='--', width=0.8)
    ]
    
    buf = io.BytesIO()
    title_text = f"{ticker} [{signal}]"
    
    fig, ax = mpf.plot(
        df, type='candle', style=s, title=title_text, volume=True,
        addplot=rsi_lines, mav=(20),
        hlines=dict(hlines=[sup, res], colors=['b','b'], linestyle='-.', linewidths=(1.5,1.5)),
        panel_ratios=(6,2,2),
        savefig=dict(fname=buf, dpi=100, bbox_inches='tight'),
        returnfig=True
    )
    st.pyplot(fig)

    # --- BAGIAN PENJELASAN (INTERPRETASI AI) ---
    with st.container():
        st.markdown(f"#### 📝 Analisis AI untuk {ticker}")
        
        # 1. Penjelasan Sinyal (Kesimpulan)
        if "🏆" in signal:
            st.success(f"""
            **KESIMPULAN: STRONG BUY (REVERSAL)**
            Saham ini membentuk pola **SPRING**. Harga sempat turun menjebol Support AI ({sup:,.0f}) untuk memancing panic selling, tapi berhasil ditutup naik kembali. 
            Ini adalah jejak **Smart Money** yang melakukan akumulasi di harga bawah.
            """)
        elif "✅" in signal:
            st.info(f"""
            **KESIMPULAN: BUY ON WEAKNESS (AKUMULASI)**
            Harga saat ini ({price:,.0f}) berada di **Zona Support AI** ({sup:,.0f}). 
            Ini adalah area beli yang aman dengan risiko rendah. Strategi yang disarankan: **Cicil Beli Bertahap**.
            """)
        elif "⚠️" in signal:
            st.warning(f"""
            **KESIMPULAN: WATCH FOR BREAKOUT**
            Harga mendekati **Resistance AI** ({res:,.0f}). Jangan buru-buru beli. 
            Tunggu sampai harga berhasil menembus resistance dengan volume besar (Breakout) baru ikutan beli.
            """)
        else:
            st.write(f"""
            **KESIMPULAN: WAIT AND SEE**
            Harga berada di tengah-tengah rentang konsolidasi ("No Man's Land"). Rasio Risk/Reward kurang menarik.
            Tunggu harga turun ke {sup:,.0f} atau naik menembus {res:,.0f}.
            """)

        # 2. Legenda Singkat (Expander agar rapi)
        with st.expander("📖 Cara Membaca Grafik Ini"):
            st.markdown(f"""
            * **Candlestick:** Menunjukkan pergerakan harga harian.
            * **Garis Putus-putus Biru (Bawah):** Support Kuat AI. Area di mana pembeli biasanya masuk.
            * **Garis Putus-putus Biru (Atas):** Resistance AI. Area di mana penjual biasanya menekan harga.
            * **Grafik Ungu (Bawah):** Indikator RSI ({rsi_val}).
                * Jika RSI < 30: Jenuh Jual (Murah).
                * Jika RSI > 70: Jenuh Beli (Mahal).
                * Jika RSI naik saat harga turun: Sinyal Divergence (Bagus).
            """)
    st.divider()

# --- FRONTEND ---
if tombol_scan:
    tickers = [t.strip() for t in ticker_input.split(",")]
    results = []
    
    progress_bar = st.progress(0)
    st_text = st.empty()
    
    for i, t in enumerate(tickers):
        st_text.text(f"Menganalisis: {t}...")
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
    df_res['Priority'] = df_res['Signal'].apply(lambda x: 0 if '🏆' in x else (1 if '✅' in x else 2))
    df_res = df_res.sort_values(by='Priority')
    
    golden_count = len(df_res[df_res['Signal'].str.contains('🏆')])
    if golden_count > 0:
        st.balloons()
        st.success(f"DITEMUKAN {golden_count} SAHAM GOLDEN SETUP!")

    # Tabel
    def color_signal(val):
        color = 'white'
        if '🏆' in val: color = '#ffd700'
        elif '✅' in val: color = '#90ee90'
        elif '⚠️' in val: color = '#ffcccb'
        return f'background-color: {color}; color: black; font-weight: bold'

    df_display = df_res.drop(columns=['Data', 'Priority'])
    st.dataframe(df_display.style.map(color_signal, subset=['Signal']), use_container_width=True)
    st.divider()
    
    mode = st.radio("Mode Lihat Chart:", ["Manual Pilih", "Semua Golden Setup", "Semua Hasil"], horizontal=True)
    
    if mode == "Manual Pilih":
        pilihan = st.selectbox("Pilih Saham:", [r['Ticker'] for r in results])
        if pilihan:
            data = next(x for x in results if x['Ticker'] == pilihan)
            plot_chart(data) # Chart + Penjelasan akan muncul
            
    elif mode == "Semua Golden Setup":
        golden_results = [r for r in results if "🏆" in r['Signal']]
        if golden_results:
            for r in golden_results: # Tidak pakai kolom (1 per baris) agar penjelasan terbaca jelas
                plot_chart(r)
        else:
            st.warning("Tidak ada Golden Setup.")
            
    else: # Semua Hasil
        for r in results: # Tampilkan satu per satu ke bawah agar tidak sempit
            plot_chart(r)

elif st.session_state['status_scan']:
    st.warning("Tidak ditemukan saham yang sesuai parameter.")
