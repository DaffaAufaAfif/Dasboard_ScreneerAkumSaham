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
st.set_page_config(layout="wide", page_title="Screener Saham Risk Aware")

st.title("🏆 Dashboard Sniper Saham (Risk Aware Edition)")
st.markdown("""
Mendeteksi fase akumulasi dengan fitur **Risk Gap Detection**. 
Waspada jika jarak antara Support AI dan Support Klasik (Low) terlalu lebar.
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

tombol_scan = st.sidebar.button("🚀 Cari Sinyal Aman", type="primary")

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
        
        # 1. Hitung Support AI
        ai_support, ai_resistance = calculate_dynamic_snr(recent)
        
        # 2. Hitung Support Klasik (Absolute Low)
        classic_low = recent['Low'].min()
        
        # 3. Hitung RISK GAP (Jarak antara AI dan Klasik)
        risk_gap_pct = (ai_support - classic_low) / ai_support
        
        current_candle = recent.iloc[-1]
        ai_range = (ai_resistance - ai_support) / ai_support
        
        if ai_range <= max_range_pct:
            pos = (current_candle['Close'] - ai_support) / (ai_resistance - ai_support)
            is_golden = detect_golden_setup(current_candle, ai_support)
            
            status = ""
            signal_label = "NETRAL"
            
            # Labeling Signal
            if is_golden:
                signal_label = "🥇 GOLDEN"
                status = "Reversal Pattern Detected"
            elif pos <= 0.10: 
                signal_label = "⭐ STRONG BUY"
                status = "Near AI Support"
            elif pos <= 0.25: 
                signal_label = "✅ BUY"
                status = "Accumulation Zone"
            elif 0.85 <= pos <= 1.05:
                signal_label = "⚠️ BREAKOUT"
                status = "Near Resistance"
            else:
                signal_label = "💤 WAIT"
                status = "Sideways"

            # --- FILTER RISK GAP (OBATNYA) ---
            # Jika Gap terlalu lebar (> 3%), turunkan rating sinyal atau beri warning
            risk_note = "Aman"
            if risk_gap_pct > 0.03: # Jika jarak lebih dari 3%
                risk_note = "⚠️ JURANG LEBAR"
                # Jika sinyalnya Buy, kita tambah warning
                if "BUY" in signal_label or "GOLDEN" in signal_label:
                    status += " (High Risk Gap)"
            
            # Keterangan Range & RSI
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
                "Risk Gap": risk_note,      # Kolom Baru
                "Gap %": round(risk_gap_pct*100, 1), # Angka Gap
                "Harga": current_candle['Close'],
                "Range %": round(ai_range * 100, 2),
                "Ket. Range": ket_range,
                "RSI": round(rsi_val, 0),
                "Ket. RSI": ket_rsi,
                "Support AI": round(ai_support, 0),
                "Low Klasik": round(classic_low, 0), # Data Low Klasik
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
    risk_note = data_dict['Risk Gap']
    gap_val = data_dict['Gap %']
    low_classic = data_dict['Low Klasik']
    
    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
    
    rsi_lines = [
        mpf.make_addplot(df['RSI'], panel=2, color='purple', ylabel='RSI', width=1.5),
        mpf.make_addplot([70]*len(df), panel=2, color='gray', linestyle='--', width=0.8),
        mpf.make_addplot([30]*len(df), panel=2, color='gray', linestyle='--', width=0.8)
    ]
    
    buf = io.BytesIO()
    title_text = f"{ticker} [{signal}] - Risk Gap: {gap_val}%"
    
    fig, ax = mpf.plot(
        df, type='candle', style=s, title=title_text, volume=True,
        addplot=rsi_lines, mav=(20),
        # Tampilkan Garis AI (Biru) DAN Garis Klasik (Merah Putus)
        hlines=dict(hlines=[data_dict['Support AI'], data_dict['Resistance'], low_classic], 
                    colors=['b','b', 'r'], 
                    linestyle=['-.', '-.', ':'], # Klasik pakai titik-titik merah
                    linewidths=(1.5, 1.5, 1.0)), 
        panel_ratios=(6,2,2),
        savefig=dict(fname=buf, dpi=100, bbox_inches='tight'),
        returnfig=True
    )
    st.pyplot(fig)

    with st.container():
        # Logika Penjelasan Risk Gap
        if gap_val > 3.0:
            st.warning(f"""
            **⚠️ PERINGATAN RISIKO (GAP DETECTED)**
            Terdapat jarak sebesar **{gap_val}%** antara Support AI ({data_dict['Support AI']:,.0f}) dan Support Klasik Terbawah ({low_classic:,.0f}).
            * **Analisis:** Jika harga jebol garis biru, ada risiko harga ditarik jatuh ke garis merah putus-putus.
            * **Saran:** Jangan *All-in*. Tunggu pantulan (rebound) yang valid di garis biru, atau pasang bid antrian di garis merah.
            """)
        else:
            st.success(f"""
            **✅ STRUKTUR SUPPORT KUAT**
            Jarak antara Support AI dan Low Klasik sangat tipis (**{gap_val}%**). 
            Ini menandakan area demand yang padat dan solid. Risiko jebakan "False Break" yang dalam relatif kecil.
            """)
    st.divider()

# --- FRONTEND ---
if tombol_scan:
    tickers = [t.strip() for t in ticker_input.split(",")]
    results = []
    
    progress_bar = st.progress(0)
    st_text = st.empty()
    
    for i, t in enumerate(tickers):
        st_text.text(f"Analisis Risiko Gap: {t}...")
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
    
    # Priority Sorting
    def assign_priority(sig):
        if '🥇' in sig: return 0
        if '⭐' in sig: return 1
        if '✅' in sig: return 2
        if '⚠️' in sig: return 3
        return 4

    df_res['Priority'] = df_res['Signal'].apply(assign_priority)
    df_res = df_res.sort_values(by='Priority')
    
    golden_count = len(df_res[df_res['Signal'].str.contains('🥇')])
    if golden_count > 0:
        st.balloons()
        st.success(f"DITEMUKAN {golden_count} SAHAM GOLDEN SETUP!")

    def color_signal(val):
        color = 'white'
        if '🥇' in val: color = '#ffd700'
        elif '⭐' in val: color = '#7CFC00'
        elif '✅' in val: color = '#98FB98'
        elif '⚠️' in val: color = '#FFB6C1'
        return f'background-color: {color}; color: black; font-weight: bold'

    # Warna Warning untuk Risk Gap
    def color_risk(val):
        if '⚠️' in val: return 'color: red; font-weight: bold;'
        return 'color: green;'

    cols_order = [
        'Ticker', 'Signal', 'Risk Gap', 'Gap %', # Kolom Baru
        'Harga', 'Support AI', 'Low Klasik',     # Bandingkan AI vs Klasik
        'Range %', 'RSI'
    ]
    
    df_display = df_res[cols_order]
    
    st.dataframe(
        df_display.style.map(color_signal, subset=['Signal']).map(color_risk, subset=['Risk Gap']), 
        use_container_width=True,
        column_config={
            "Harga": st.column_config.NumberColumn(format="Rp %d"),
            "Support AI": st.column_config.NumberColumn(format="Rp %d"),
            "Low Klasik": st.column_config.NumberColumn(format="Rp %d"),
            "Gap %": st.column_config.NumberColumn(format="%.1f %%"),
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
        top_results = [r for r in results if ('🥇' in r['Signal'] or '⭐' in r['Signal'])]
        if top_results:
            for r in top_results: plot_chart(r)
        else: st.warning("Tidak ada saham prioritas.")
    else: 
        for r in results: plot_chart(r)

elif st.session_state['status_scan']:
    st.warning("Tidak ditemukan saham yang sesuai parameter.")
