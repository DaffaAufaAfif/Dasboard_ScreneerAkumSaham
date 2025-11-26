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
st.set_page_config(layout="wide", page_title="Screener Saham Full Inspector")

st.title("💎 Dashboard Sniper Saham (Full Inspector)")
st.markdown("""
Mendeteksi fase akumulasi dengan transparansi penuh.
**Tabel** menampilkan seluruh hasil scan. **Grafik** bisa dipilih manual untuk analisis mendalam (termasuk saham Wait/Trap).
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
        
        # Klasifikasi Tipe Saham
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
        
        # Logika Keputusan
        if gap_pct <= max_gap_allowed:
            if is_spring and rsi_good:
                signal_label = "💎 DIAMOND"
            elif pos <= 0.15:
                if gap_pct <= (max_gap_allowed * 0.6):
                    signal_label = "🥇 GOLDEN"
                else:
                    signal_label = "✅ SAFE BUY"
            elif 0.85 <= pos <= 1.05:
                signal_label = "⚠️ BREAKOUT"
            else:
                signal_label = "💤 WAIT"
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
    
    # --- BAGIAN PENJELASAN LENGKAP (SHOW/HIDE) ---
    
    # 1. Ringkasan (Selalu Muncul)
    if "DIAMOND" in signal:
        st.success(f"**💎 REKOMENDASI: DIAMOND SETUP** | Reversal Sempurna + Risiko Minim. Sangat Layak Beli.")
    elif "GOLDEN" in signal:
        st.success(f"**🥇 REKOMENDASI: BEST PRICE** | Harga Murah di Support Kuat. Akumulasi Bertahap.")
    elif "SAFE BUY" in signal:
        st.info(f"**✅ REKOMENDASI: ACCUMULATE** | Aman di Support. Potensi Upside Wajar.")
    elif "TRAP" in signal:
        st.error(f"**❌ PERINGATAN: JEBAKAN (TRAP)** | Gap Jurang Terlalu Lebar. Jangan Tergiur Harga Murah.")
    elif "BREAKOUT" in signal:
        st.warning(f"**⚠️ PERINGATAN: RESISTANCE** | Harga Dekat Atap. Rawan Pantulan ke Bawah.")
    else: # WAIT
        st.write(f"**💤 STATUS: WAIT AND SEE** | Belum Ada Sinyal Kuat. Pantau Terus.")
    
    # 2. Detail Analisis (Expandable untuk SEMUA signal)
    with st.expander(f"🕵️‍♂️ Lihat Analisis Detail {ticker} (Risiko & Strategi)"):
        
        # Logic Narasi Berdasarkan Sinyal
        saran_strategi = ""
        analisis_gap = ""
        
        if data_dict['Gap %'] > 0.04:
            analisis_gap = f"⚠️ **HATI-HATI!** Jarak ke lantai dasar (Support Klasik) cukup jauh ({data_dict['Gap %']*100:.1f}%). Jika garis biru jebol, harga bisa jatuh dalam."
        else:
            analisis_gap = f"✅ **AMAN.** Jarak ke lantai dasar sangat tipis ({data_dict['Gap %']*100:.1f}%). Risiko 'False Break' yang dalam relatif kecil."

        if "TRAP" in signal:
            saran_strategi = "**JANGAN BELI SEKARANG.** Risiko jatuh ke Support Klasik lebih besar daripada potensi naik. Tunggu harga turun lagi atau Gap mengecil."
        elif "WAIT" in signal:
            saran_strategi = "**HARGA TANGGUNG.** Posisi harga berada di tengah-tengah (No Man's Land). Risk/Reward ratio tidak menarik."
        elif "BREAKOUT" in signal:
            saran_strategi = "**JANGAN FOMO.** Tunggu konfirmasi: Apakah harga berhasil tembus Resistance dengan volume besar? Atau malah memantul turun? Beli jika Breakout valid atau tunggu di Support."
        else: # Buy Signals
            saran_strategi = f"**BELI BERTAHAP.** Masuk di area Rp {data_dict['Support AI']:,.0f}. Pasang Stop Loss ketat di bawah Rp {data_dict['Low Klasik']:,.0f}."

        st.markdown(f"""
        **1. Profil Risiko:**
        * Tipe Saham: **{data_dict['Tipe']}** (Range: {data_dict['Range %']*100:.1f}%)
        * Analisis Gap: {analisis_gap}
        
        **2. Peta Harga:**
        * Harga Sekarang: **Rp {data_dict['Harga']:,.0f}**
        * Support AI (Area Beli): **Rp {data_dict['Support AI']:,.0f}**
        * Support Klasik (Dasar Jurang): **Rp {data_dict['Low Klasik']:,.0f}**
        * Resistance (Target Jual): **Rp {data_dict['Resistance']:,.0f}**
        
        **3. Indikator Tenaga (RSI):**
        * Posisi RSI: **{data_dict['RSI']} ({data_dict['Ket. RSI']})**
        
        **🧠 KESIMPULAN & SARAN:**
        {saran_strategi}
        """)
        
    st.divider()

# --- FRONTEND EXECUTION ---
if tombol_scan:
    raw_tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]
    
    if len(raw_tickers) > 200:
        st.warning(f"⚠️ Scan {len(raw_tickers)} saham sedang berjalan. Mohon tunggu...")
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, t in enumerate(raw_tickers):
        status_text.text(f"Scanning ({i+1}/{len(raw_tickers)}): {t}...")
        res = get_ai_status(t)
        if res: results.append(res)
        progress_bar.progress((i + 1) / len(raw_tickers))
        
    status_text.success("Selesai!")
    progress_bar.empty()
    st.session_state['hasil_scan'] = results
    st.session_state['status_scan'] = True

if st.session_state['status_scan'] and st.session_state['hasil_scan']:
    results = st.session_state['hasil_scan']
    df_res = pd.DataFrame(results)
    
    # Priority Sorting (Hanya untuk urutan tabel, tidak membuang data)
    def assign_priority(sig):
        if '💎' in sig: return 0
        if '🥇' in sig: return 1
        if '✅' in sig: return 2
        if '⚠️' in sig: return 3 # Breakout
        if '💤' in sig: return 4 # Wait
        if '❌' in sig: return 5 # Trap
        return 6
    
    df_res['Priority'] = df_res['Keputusan'].apply(assign_priority)
    df_res = df_res.sort_values(by=['Priority', 'Gap %']) # Sort terbaik di atas, tapi SEMUA ada
    
    diamond_count = len(df_res[df_res['Keputusan'].str.contains('💎')])
    if diamond_count > 0:
        st.balloons()
        st.success(f"🔥 DITEMUKAN {diamond_count} DIAMOND SETUP!")
    
    def color_signal(val):
        if '💎' in val: return 'background-color: #00ced1; color: white; font-weight: bold'
        if '🥇' in val: return 'background-color: #ffd700; color: black; font-weight: bold'
        if '✅' in val: return 'background-color: #90ee90; color: black; font-weight: bold'
        if '❌' in val: return 'background-color: #808080; color: white; font-weight: bold'
        if '⚠️' in val: return 'background-color: #ffcccb; color: black; font-weight: bold'
        return ''

    # TABEL LENGKAP (SEMUA SIGNAL)
    st.subheader("📋 Tabel Hasil Scan (Semua Status)")
    cols_order = ['Ticker', 'Keputusan', 'Tipe', 'Range %', 'Harga', 'Support AI', 'Gap %', 'RSI', 'Ket. RSI']
    
    st.dataframe(
        df_res[cols_order].style.map(color_signal, subset=['Keputusan']), 
        use_container_width=True,
        column_config={
            "Harga": st.column_config.NumberColumn(format="Rp %d"),
            "Support AI": st.column_config.NumberColumn(format="Rp %d"),
            "Range %": st.column_config.NumberColumn(format="%.1f %%"),
            "Gap %": st.column_config.NumberColumn(format="%.1f %%"),
            "RSI": st.column_config.NumberColumn(format="%.0f"),
        }
    )
    st.divider()
    
    # --- INSPEKTOR GRAFIK MANUAL (UNTUK SEMUA SAHAM) ---
    st.subheader("🔍 Inspektur Grafik & Analisis")
    st.caption("Pilih saham apa saja dari hasil scan di atas (termasuk Wait/Trap) untuk melihat detail risikonya.")
    
    # List saham untuk dropdown (Diurutkan berdasarkan prioritas tabel biar enak carinya)
    list_saham_sorted = df_res['Ticker'].tolist()
    
    selected_ticker = st.selectbox("Pilih Saham:", list_saham_sorted)
    
    if selected_ticker:
        # Ambil data saham terpilih
        selected_data = next(r for r in results if r['Ticker'] == selected_ticker)
        plot_chart(selected_data)

elif st.session_state['status_scan']:
    st.warning("Tidak ditemukan hasil.")
