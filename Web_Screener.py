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
st.set_page_config(layout="wide", page_title="Screener Saham Ranked Edition")

st.title("🏆 Dashboard Sniper Saham (Ranked Priority)")
st.markdown("""
Mendeteksi fase akumulasi dengan **Ranking Prioritas Beli**. 
Sinyal dibedakan menjadi **Golden**, **Strong Buy**, dan **Buy** berdasarkan jarak ke support.
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

tombol_scan = st.sidebar.button("🚀 Cari Sinyal Prioritas", type="primary")

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
            # Posisi Harga (0 = Support, 1 = Resistance)
            pos = (current_candle['Close'] - ai_support) / (ai_resistance - ai_support)
            is_golden = detect_golden_setup(current_candle, ai_support)
            
            status = ""
            signal_label = "NETRAL"
            
            # --- LOGIKA PRIORITAS BARU ---
            if is_golden:
                status = "✨ GOLDEN SETUP"
                signal_label = "🥇 GOLDEN"
            elif pos <= 0.10: # Sangat dekat dengan support (0% - 10% dari lantai)
                status = "STRONG BUY (Best Price)"
                signal_label = "⭐ STRONG BUY"
            elif pos <= 0.25: # Masih di area support (10% - 25% dari lantai)
                status = "BUY ZONE (Accumulation)"
                signal_label = "✅ BUY"
            elif 0.85 <= pos <= 1.05:
                status = "POTENSI BREAKOUT"
                signal_label = "⚠️ BREAKOUT"
            else:
                status = "SIDEWAYS"
                signal_label = "💤 WAIT"
            
            if ai_range <= 0.15: ket_range = "😴 Tidur / Kalem"
            elif ai_range <= 0.25: ket_range = "🙂 Normal"
            else: ket_range = "⚡ Liar / Volatil"

            rsi_val = current_candle['RSI']
            if rsi_val < 30: ket_rsi = "🟢 Oversold"
            elif rsi_val < 45: ket_rsi = "📈 Mulai Bangkit"
            elif rsi_val > 70: ket_rsi = "🔴 Overbought"
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
                "Support": round(ai_support, 0),
                "Resistance": round(ai_resistance, 0),
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
        st_text.text(f"Mengukur Kualitas Sinyal: {t}...")
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
    
    # --- LOGIKA SORTING PRIORITAS YANG BARU ---
    # 0 = Golden, 1 = Strong Buy, 2 = Buy, 3 = Breakout, 4 = Wait
    def assign_priority(sig):
        if '🥇' in sig: return 0
        if '⭐' in sig: return 1
        if '✅' in sig: return 2
        if '⚠️' in sig: return 3
        return 4

    df_res['Priority'] = df_res['Signal'].apply(assign_priority)
    df_res = df_res.sort_values(by='Priority')
    
    golden_count = len(df_res[df_res['Signal'].str.contains('🥇')])
    strong_buy_count = len(df_res[df_res['Signal'].str.contains('⭐')])

    if golden_count > 0:
        st.balloons()
        st.success(f"DITEMUKAN {golden_count} SAHAM GOLDEN SETUP!")
    elif strong_buy_count > 0:
        st.success(f"Ditemukan {strong_buy_count} saham STRONG BUY (Sangat Dekat Support).")

    # --- KONFIGURASI WARNA BARU ---
    def color_signal(val):
        color = 'white'
        if '🥇' in val: color = '#ffd700' # Emas
        elif '⭐' in val: color = '#7CFC00' # Lawn Green (Hijau Terang) - Best Price
        elif '✅' in val: color = '#98FB98' # Pale Green - Standard Buy
        elif '⚠️' in val: color = '#FFB6C1' # Light Pink - Hati-hati
        return f'background-color: {color}; color: black; font-weight: bold'

    cols_order = [
        'Ticker', 'Signal', 'Status', 
        'Harga', 'Support', 'Resistance',
        'Range %', 'Ket. Range',
        'RSI', 'Ket. RSI'
    ]
    
    df_display = df_res[cols_order]
    
    st.dataframe(
        df_display.style.map(color_signal, subset=['Signal']), 
        use_container_width=True,
        column_config={
            "Harga": st.column_config.NumberColumn(format="Rp %d"),
            "Support": st.column_config.NumberColumn(format="Rp %d"),
            "Resistance": st.column_config.NumberColumn(format="Rp %d"),
            "Range %": st.column_config.NumberColumn(format="%.2f %%"),
            "RSI": st.column_config.NumberColumn(format="%.0f"),
        }
    )
    st.divider()
    
    mode = st.radio("Mode Lihat Chart:", ["Manual Pilih", "Top Priority Only", "Semua Hasil"], horizontal=True)
    
    if mode == "Manual Pilih":
        pilihan = st.selectbox("Pilih Saham:", [r['Ticker'] for r in results])
        if pilihan:
            data = next(x for x in results if x['Ticker'] == pilihan)
            plot_chart(data)
            
    elif mode == "Top Priority Only":
        # Tampilkan Golden dan Strong Buy saja
        top_results = [r for r in results if ('🥇' in r['Signal'] or '⭐' in r['Signal'])]
        if top_results:
            for r in top_results:
                plot_chart(r)
        else:
            st.warning("Tidak ada saham Golden atau Strong Buy saat ini.")
            
    else: 
        for r in results:
            plot_chart(r)

elif st.session_state['status_scan']:
    st.warning("Tidak ditemukan saham yang sesuai parameter.")
