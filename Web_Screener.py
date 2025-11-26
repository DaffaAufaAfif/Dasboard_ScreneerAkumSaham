import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import warnings
import io
import numpy as np
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# --- KONFIGURASI HALAMAN ---
st.set_page_config(layout="wide", page_title="Screener Saham Click-to-View")

st.title("💎 Dashboard Sniper Saham (Interactive)")
st.markdown("""
**Cara Pakai:** Lakukan Scan, lalu **KLIK Baris/Centang** pada tabel di bawah untuk memunculkan Grafik & Analisis Detail saham tersebut.
""")

if 'hasil_scan' not in st.session_state:
    st.session_state['hasil_scan'] = None
if 'status_scan' not in st.session_state:
    st.session_state['status_scan'] = False

# --- DATABASE PRESETS ---
PRESETS = {
    "Manual (Ketik Sendiri)": "",
    "💎 LQ45 (Big Cap)": "ACES.JK, ADRO.JK, AKRA.JK, AMRT.JK, ANTM.JK, ARTO.JK, ASII.JK, BBCA.JK, BBNI.JK, BBRI.JK, BBTN.JK, BMRI.JK, BRIS.JK, BRPT.JK, BUKA.JK, CPIN.JK, EMTK.JK, ESSA.JK, EXCL.JK, GOTO.JK, HRUM.JK, ICBP.JK, INCO.JK, INDF.JK, INKP.JK, INTP.JK, ISAT.JK, ITMG.JK, JPFA.JK, KLBF.JK, MAPI.JK, MDKA.JK, MEDC.JK, MBMA.JK, MIKA.JK, MTEL.JK, PGAS.JK, PGEO.JK, PTBA.JK, SIDO.JK, SMGR.JK, SRTG.JK, TBIG.JK, TINS.JK, TLKM.JK, TOWR.JK, UNTR.JK, UNVR.JK",
    "🔥 Kompas100 (Likuid & Aktif)": "ACES.JK, ADRO.JK, AKRA.JK, AMRT.JK, ANTM.JK, ARTO.JK, ASII.JK, BBCA.JK, BBNI.JK, BBRI.JK, BBTN.JK, BMRI.JK, BRIS.JK, BRPT.JK, BUKA.JK, CPIN.JK, EMTK.JK, ESSA.JK, EXCL.JK, GOTO.JK, HRUM.JK, ICBP.JK, INCO.JK, INDF.JK, INKP.JK, INTP.JK, ISAT.JK, ITMG.JK, JPFA.JK, KLBF.JK, MAPI.JK, MDKA.JK, MEDC.JK, MBMA.JK, MIKA.JK, MTEL.JK, PGAS.JK, PGEO.JK, PTBA.JK, SIDO.JK, SMGR.JK, SRTG.JK, TBIG.JK, TINS.JK, TLKM.JK, TOWR.JK, UNTR.JK, UNVR.JK, ABMM.JK, ADMR.JK, AGRO.JK, APIC.JK, ASSA.JK, AUTO.JK, AVIA.JK, BBHI.JK, BDMN.JK, BFIN.JK, BJBR.JK, BJTM.JK, BIRD.JK, BUMI.JK, CTRA.JK, DEWA.JK, DOID.JK, DSNG.JK, ELSA.JK, ENRG.JK, ERAA.JK, FREN.JK, GGRM.JK, GJTL.JK, HEAL.JK, HMSP.JK, HOKI.JK, INDY.JK, INKP.JK, JSMR.JK, KAEF.JK, KPIG.JK, LPPF.JK, LSIP.JK, MDKA.JK, MNCN.JK, MPMX.JK, MYOR.JK, PALS.JK, PANI.JK, PNLF.JK, PNBN.JK, PTBA.JK, PWON.JK, RAJA.JK, RALS.JK, SCMA.JK, SIDO.JK, SIMP.JK, SMDR.JK, SMRA.JK, TAPG.JK, TPIA.JK, WIKA.JK, WOOD.JK",
    "🕌 Syariah Populer (JII)": "ADRO.JK, AKRA.JK, ANTM.JK, ASII.JK, BRIS.JK, BRPT.JK, CPIN.JK, ESSA.JK, EXCL.JK, HRUM.JK, ICBP.JK, INCO.JK, INDF.JK, INKP.JK, INTP.JK, ISAT.JK, ITMG.JK, JPFA.JK, KLBF.JK, MAPI.JK, MDKA.JK, MIKA.JK, PGAS.JK, PTBA.JK, SIDO.JK, SMGR.JK, TINS.JK, TLKM.JK, UNTR.JK, UNVR.JK"
}

# --- SIDEBAR ---
st.sidebar.header("⚙️ Konfigurasi Scan")
selected_preset = st.sidebar.selectbox("Pilih Grup Saham:", list(PRESETS.keys()), index=2)

if selected_preset == "Manual (Ketik Sendiri)":
    default_text = ""
else:
    default_text = PRESETS[selected_preset]

ticker_input = st.sidebar.text_area("Daftar Ticker", default_text, height=150)
st.sidebar.caption(f"Jumlah Saham: {len(ticker_input.split(',')) if ticker_input else 0}")

st.sidebar.subheader("Parameter")
period_days = st.sidebar.slider("Periode Data", 30, 90, 60)
tombol_scan = st.sidebar.button("🚀 Mulai Scan Masal", type="primary")

# --- FUNGSI LOGIKA (Engine) ---

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
        
        stock_type = "Normal"
        max_gap_allowed = 0.045
        if ai_range <= 0.12:
            stock_type = "🛡️ Stabil"
            max_gap_allowed = 0.025
        elif ai_range <= 0.22:
            stock_type = "⚖️ Moderat"
            max_gap_allowed = 0.045
        else:
            stock_type = "🔥 Agresif"
            max_gap_allowed = 0.07

        pos = (current_candle['Close'] - ai_support) / (ai_resistance - ai_support)
        is_spring = detect_spring_pattern(current_candle, ai_support)
        rsi_good = (current_candle['RSI'] > 30) and (current_candle['RSI'] < 65)
        
        signal_label = "NETRAL"
        
        if gap_pct <= max_gap_allowed:
            if is_spring and rsi_good: signal_label = "💎 DIAMOND"
            elif pos <= 0.15:
                if gap_pct <= (max_gap_allowed * 0.6): signal_label = "🥇 GOLDEN"
                else: signal_label = "✅ SAFE BUY"
            elif 0.85 <= pos <= 1.05: signal_label = "⚠️ BREAKOUT"
            else: signal_label = "💤 WAIT"
        else:
            signal_label = "❌ TRAP"

        rsi_val = current_candle['RSI']
        if rsi_val < 30: ket_rsi = "🟢 Oversold"
        elif rsi_val < 45: ket_rsi = "📈 Bangkit"
        elif rsi_val > 70: ket_rsi = "🔴 Overbought"
        else: ket_rsi = "⚪ Netral"

        return {
            "Ticker": ticker.replace(".JK", ""),
            "Keputusan": signal_label,
            "Tipe": stock_type,
            "Range %": ai_range,
            "Harga": current_candle['Close'],
            "Support AI": round(ai_support, 0),
            "Gap %": gap_pct,
            "RSI": round(rsi_val, 0),
            "Ket. RSI": ket_rsi,
            "Data": recent,
            "Resistance": ai_resistance,
            "Low Klasik": round(classic_low, 0)
        }
    except:
        return None
    return None

def plot_chart(data_dict):
    df = data_dict['Data']
    ticker = data_dict['Ticker']
    signal = data_dict['Keputusan']
    
    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
    
    rsi_lines = [
        mpf.make_addplot(df['RSI'], panel=2, color='purple', ylabel='RSI', width=1.5),
        mpf.make_addplot([70]*len(df), panel=2, color='gray', linestyle='--', width=0.8),
        mpf.make_addplot([30]*len(df), panel=2, color='gray', linestyle='--', width=0.8)
    ]
    
    buf = io.BytesIO()
    title_text = f"{ticker} [{signal}] - {data_dict['Tipe']}"
    
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
    
    # --- PENJELASAN (SHOW & HIDE) ---
    
    # 1. Ringkasan (Selalu Muncul)
    if "DIAMOND" in signal:
        st.success(f"**💎 REKOMENDASI: DIAMOND SETUP** | Reversal Sempurna + Risiko Minim.")
    elif "GOLDEN" in signal:
        st.success(f"**🥇 REKOMENDASI: BEST PRICE** | Harga Murah di Support Kuat.")
    elif "SAFE BUY" in signal:
        st.info(f"**✅ REKOMENDASI: ACCUMULATE** | Area beli wajar.")
    elif "TRAP" in signal:
        st.error(f"**❌ PERINGATAN: JEBAKAN (TRAP)** | Gap Jurang Terlalu Lebar.")
    elif "BREAKOUT" in signal:
        st.warning(f"**⚠️ PERINGATAN: RESISTANCE** | Dekat Atap.")
    else:
        st.write(f"**💤 STATUS: WAIT AND SEE** | Belum Ada Sinyal Kuat.")
    
    # 2. Detail (Expandable)
    with st.expander(f"🕵️‍♂️ Klik untuk Analisis Detail & Strategi {ticker}"):
        gap_status = "Aman" if data_dict['Gap %'] <= 0.04 else "Berisiko Tinggi"
        
        saran = "Pantau saja."
        if "TRAP" in signal: saran = "JANGAN MASUK. Tunggu harga turun lagi."
        elif "DIAMOND" in signal: saran = "ENTRY SEKARANG. Potensi Reversal tinggi."
        elif "SAFE" in signal or "GOLDEN" in signal: saran = "CICIL BELI di area Support AI."
        
        st.markdown(f"""
        **Bedah Data {ticker}:**
        * **Posisi:** Rp {data_dict['Harga']:,} (Support AI: Rp {data_dict['Support AI']:,})
        * **Jurang Bawah:** Rp {data_dict['Low Klasik']:,} (Gap: {data_dict['Gap %']*100:.1f}% - {gap_status})
        * **Karakter:** {data_dict['Tipe']} (Volatilitas {data_dict['Range %']*100:.1f}%)
        * **RSI:** {data_dict['RSI']} ({data_dict['Ket. RSI']})
        
        **🎯 Action Plan:**
        {saran}
        """)

# --- FRONTEND EXECUTION ---
if tombol_scan:
    raw_tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]
    
    if len(raw_tickers) > 200:
        st.warning(f"⚠️ Scan {len(raw_tickers)} saham sedang berjalan...")
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, t in enumerate(raw_tickers):
        status_text.text(f"Scanning ({i+1}/{len(raw_tickers)}): {t}...")
        res = get_ai_status(t)
        if res: results.append(res)
        progress_bar.progress((i + 1) / len(raw_tickers))
        
    status_text.success("Selesai! Silakan pilih baris di tabel untuk melihat grafik.")
    progress_bar.empty()
    st.session_state['hasil_scan'] = results
    st.session_state['status_scan'] = True

if st.session_state['status_scan'] and st.session_state['hasil_scan']:
    results = st.session_state['hasil_scan']
    df_res = pd.DataFrame(results)
    
    # Sorting Priority
    def assign_priority(sig):
        if '💎' in sig: return 0
        if '🥇' in sig: return 1
        if '✅' in sig: return 2
        if '⚠️' in sig: return 3
        if '💤' in sig: return 4
        if '❌' in sig: return 5
        return 6
    
    df_res['Priority'] = df_res['Keputusan'].apply(assign_priority)
    df_res = df_res.sort_values(by=['Priority', 'Gap %'])
    
    # Counts
    diamond = len(df_res[df_res['Keputusan'].str.contains('💎')])
    golden = len(df_res[df_res['Keputusan'].str.contains('🥇')])
    
    if diamond > 0: st.balloons()
    
    st.info(f"📊 Hasil: **{diamond} Diamond**, **{golden} Golden** dari **{len(results)}** saham.")

    def color_signal(val):
        if '💎' in val: return 'background-color: #00ced1; color: white; font-weight: bold'
        if '🥇' in val: return 'background-color: #ffd700; color: black; font-weight: bold'
        if '✅' in val: return 'background-color: #90ee90; color: black; font-weight: bold'
        if '❌' in val: return 'background-color: #808080; color: white; font-weight: bold'
        return ''

    cols_order = ['Ticker', 'Keputusan', 'Tipe', 'Range %', 'Harga', 'Support AI', 'Gap %', 'RSI', 'Ket. RSI']
    
    # --- TABEL INTERAKTIF (KLIK UNTUK LIHAT GRAFIK) ---
    # selection_mode='single-row' membuat tabel punya checkbox di kiri
    event = st.dataframe(
        df_res[cols_order].style.map(color_signal, subset=['Keputusan']), 
        use_container_width=True,
        on_select="rerun",  # PENTING: Rerun script saat diklik
        selection_mode="single-row",
        column_config={
            "Harga": st.column_config.NumberColumn(format="Rp %d"),
            "Support AI": st.column_config.NumberColumn(format="Rp %d"),
            "Range %": st.column_config.NumberColumn(format="%.1f %%"),
            "Gap %": st.column_config.NumberColumn(format="%.1f %%"),
            "RSI": st.column_config.NumberColumn(format="%.0f"),
        }
    )
    
    # --- LOGIKA TAMPILKAN GRAFIK BERDASARKAN PILIHAN ---
    if len(event.selection.rows) > 0:
        # Ambil index baris yang dipilih
        selected_index = event.selection.rows[0]
        # Ambil data Ticker dari dataframe yang sudah di-sort
        selected_ticker = df_res.iloc[selected_index]['Ticker']
        
        # Cari data lengkapnya di hasil scan asli
        selected_data = next(r for r in results if r['Ticker'] == selected_ticker)
        
        st.divider()
        st.subheader(f"📈 Analisis Saham Terpilih: {selected_ticker}")
        plot_chart(selected_data)
    else:
        st.write("👈 **Pilih salah satu baris di tabel (klik kotak di kiri)** untuk melihat Grafik & Analisisnya.")

elif st.session_state['status_scan']:
    st.warning("Tidak ditemukan hasil.")
