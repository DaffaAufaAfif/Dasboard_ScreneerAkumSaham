import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import warnings
import io
import numpy as np
import datetime as dt
from datetime import timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

warnings.filterwarnings("ignore")

# --- KONFIGURASI HALAMAN ---
st.set_page_config(layout="wide", page_title="Maverick AI Dashboard")

st.title("💎 MAVERICK AI: Integrated Trading System")
st.markdown("""
**Sistem Hybrid:**
1. **Radar (K-Means):** Screening fase akumulasi & support/resistance.
2. **Oracle (LSTM):** Prediksi harga masa depan (Deep Learning).
""")

if 'hasil_scan' not in st.session_state:
    st.session_state['hasil_scan'] = None
if 'status_scan' not in st.session_state:
    st.session_state['status_scan'] = False

# --- PRESETS ---
PRESETS = {
    "Manual": "",
    "💎 LQ45": "ACES.JK, ADRO.JK, AKRA.JK, AMRT.JK, ANTM.JK, ARTO.JK, ASII.JK, BBCA.JK, BBNI.JK, BBRI.JK, BBTN.JK, BMRI.JK, BRIS.JK, BRPT.JK, BUKA.JK, CPIN.JK, EMTK.JK, ESSA.JK, EXCL.JK, GOTO.JK, HRUM.JK, ICBP.JK, INCO.JK, INDF.JK, INKP.JK, INTP.JK, ISAT.JK, ITMG.JK, JPFA.JK, KLBF.JK, MAPI.JK, MDKA.JK, MEDC.JK, MBMA.JK, MIKA.JK, MTEL.JK, PGAS.JK, PGEO.JK, PTBA.JK, SIDO.JK, SMGR.JK, SRTG.JK, TBIG.JK, TINS.JK, TLKM.JK, TOWR.JK, UNTR.JK, UNVR.JK",
    "🔥 Kompas100": "ACES.JK, ADRO.JK, AKRA.JK, AMRT.JK, ANTM.JK, ARTO.JK, ASII.JK, BBCA.JK, BBNI.JK, BBRI.JK, BBTN.JK, BMRI.JK, BRIS.JK, BRPT.JK, BUKA.JK, CPIN.JK, EMTK.JK, ESSA.JK, EXCL.JK, GOTO.JK, HRUM.JK, ICBP.JK, INCO.JK, INDF.JK, INKP.JK, INTP.JK, ISAT.JK, ITMG.JK, JPFA.JK, KLBF.JK, MAPI.JK, MDKA.JK, MEDC.JK, MBMA.JK, MIKA.JK, MTEL.JK, PGAS.JK, PGEO.JK, PTBA.JK, SIDO.JK, SMGR.JK, SRTG.JK, TBIG.JK, TINS.JK, TLKM.JK, TOWR.JK, UNTR.JK, UNVR.JK, ABMM.JK, ADMR.JK, AGRO.JK, APIC.JK, ASSA.JK, AUTO.JK, AVIA.JK, BBHI.JK, BDMN.JK, BFIN.JK, BJBR.JK, BJTM.JK, BIRD.JK, BUMI.JK, CTRA.JK, DEWA.JK, DOID.JK, DSNG.JK, ELSA.JK, ENRG.JK, ERAA.JK, FREN.JK, GGRM.JK, GJTL.JK, HEAL.JK, HMSP.JK, HOKI.JK, INDY.JK, INKP.JK, JSMR.JK, KAEF.JK, KPIG.JK, LPPF.JK, LSIP.JK, MDKA.JK, MNCN.JK, MPMX.JK, MYOR.JK, PALS.JK, PANI.JK, PNLF.JK, PNBN.JK, PTBA.JK, PWON.JK, RAJA.JK, RALS.JK, SCMA.JK, SIDO.JK, SIMP.JK, SMDR.JK, SMRA.JK, TAPG.JK, TPIA.JK, WIKA.JK, WOOD.JK",
    "🕌 JII Syariah": "ADRO.JK, AKRA.JK, ANTM.JK, ASII.JK, BRIS.JK, BRPT.JK, CPIN.JK, ESSA.JK, EXCL.JK, HRUM.JK, ICBP.JK, INCO.JK, INDF.JK, INKP.JK, INTP.JK, ISAT.JK, ITMG.JK, JPFA.JK, KLBF.JK, MAPI.JK, MDKA.JK, MIKA.JK, PGAS.JK, PTBA.JK, SIDO.JK, SMGR.JK, TINS.JK, TLKM.JK, UNTR.JK, UNVR.JK"
}

# --- SIDEBAR ---
st.sidebar.header("⚙️ Konfigurasi")
selected_preset = st.sidebar.selectbox("Grup Saham:", list(PRESETS.keys()), index=2)
default_text = PRESETS[selected_preset] if selected_preset != "Manual" else ""
ticker_input = st.sidebar.text_area("Ticker", default_text, height=100)
period_days = st.sidebar.slider("Periode Radar (Hari)", 30, 90, 60)
tombol_scan = st.sidebar.button("🚀 SCAN RADAR (K-MEANS)", type="primary")

# ==========================================
# 🧠 FUNGSI 1: NEURAL NETWORK (LSTM MARK-2)
# ==========================================
def run_maverick_prediction(ticker):
    # Konfigurasi LSTM
    PREDICTION_DAYS = 60
    EPOCHS = 30 # Kita set 30 biar tidak terlalu lama nunggu di web
    
    # Download Data Panjang (4 Tahun) untuk Belajar
    today = dt.datetime.now()
    tomorrow = today + timedelta(days=1)
    start_learn = "2020-01-01"
    end_learn = tomorrow.strftime("%Y-%m-%d")
    
    try:
        data = yf.download(ticker, start=start_learn, end=end_learn, progress=False)
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(1)
        
        # Hitung Indikator (RSI)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        data = data.dropna()
        
        # Siapkan Dataset (Close, Volume, RSI)
        dataset = data[['Close', 'Volume', 'RSI']].values
        
        # Scaling
        scaler_features = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler_features.fit_transform(dataset)
        
        scaler_target = MinMaxScaler(feature_range=(0, 1))
        scaler_target.fit_transform(dataset[:, 0].reshape(-1, 1)) # Target cuma Close
        
        # Training Data
        x_train, y_train = [], []
        for i in range(PREDICTION_DAYS, len(scaled_data)):
            x_train.append(scaled_data[i-PREDICTION_DAYS:i, :])
            y_train.append(scaled_data[i, 0])
            
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 3))
        
        # Build Model (Mark-2)
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 3)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Training (Silent Mode)
        model.fit(x_train, y_train, epochs=EPOCHS, batch_size=32, verbose=0)
        
        # Prediksi Masa Depan
        last_60 = data[['Close', 'Volume', 'RSI']].tail(PREDICTION_DAYS).values
        last_60_scaled = scaler_features.transform(last_60)
        
        X_test = []
        X_test.append(last_60_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 3))
        
        pred_scaled = model.predict(X_test)
        pred_price = scaler_target.inverse_transform(pred_scaled)
        
        final_price = float(pred_price[0][0])
        last_real = data['Close'].iloc[-1]
        
        return final_price, last_real
        
    except Exception as e:
        st.error(f"Error AI: {e}")
        return None, None

# ==========================================
# 📊 FUNGSI 2: SCREENER (K-MEANS)
# ==========================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    rs = gain.rolling(window=period).mean() / loss.rolling(window=period).mean()
    return 100 - (100 / (1 + rs))

def calculate_dynamic_snr(df):
    data_points = np.concatenate([df['Low'].values, df['High'].values, df['Close'].values]).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(data_points)
    centers = sorted(kmeans.cluster_centers_.flatten())
    return centers[0], centers[1]

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
        
        # Tipe Saham
        stock_type = "Normal"
        max_gap = 0.045
        if ai_range <= 0.12: stock_type, max_gap = "🛡️ Stabil", 0.025
        elif ai_range > 0.22: stock_type, max_gap = "🔥 Agresif", 0.07

        pos = (current_candle['Close'] - ai_support) / (ai_resistance - ai_support)
        is_spring = (current_candle['Low'] < ai_support) and (current_candle['Close'] > ai_support * 0.995)
        rsi_good = (30 < current_candle['RSI'] < 65)
        
        signal = "NETRAL"
        if gap_pct <= max_gap:
            if is_spring and rsi_good: signal = "💎 DIAMOND"
            elif pos <= 0.15:
                signal = "🥇 GOLDEN" if gap_pct <= (max_gap*0.6) else "✅ SAFE BUY"
            elif 0.85 <= pos <= 1.05: signal = "⚠️ BREAKOUT"
            else: signal = "💤 WAIT"
        else: signal = "❌ TRAP"

        return {
            "Ticker": ticker.replace(".JK", ""),
            "Keputusan": signal,
            "Tipe": stock_type,
            "Range %": ai_range,
            "Harga": current_candle['Close'],
            "Support AI": ai_support,
            "Gap %": gap_pct,
            "RSI": current_candle['RSI'],
            "Data": recent,
            "Resistance": ai_resistance,
            "Low Klasik": classic_low
        }
    except: return None

def plot_chart(data_dict):
    df = data_dict['Data']
    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
    rsi_p = [mpf.make_addplot(df['RSI'], panel=2, color='purple', width=1.5)]
    
    buf = io.BytesIO()
    fig, ax = mpf.plot(
        df, type='candle', style=s, title=f"{data_dict['Ticker']} [{data_dict['Keputusan']}]",
        volume=True, addplot=rsi_p, mav=(20),
        hlines=dict(hlines=[data_dict['Support AI'], data_dict['Resistance'], data_dict['Low Klasik']], 
                    colors=['b','b','r'], linestyle=['-.','-.',':']),
        panel_ratios=(6,2,2), savefig=dict(fname=buf, dpi=100, bbox_inches='tight'), returnfig=True
    )
    st.pyplot(fig)

# --- HALAMAN UTAMA ---

# 1. SCANNING SECTION
if tombol_scan:
    raw_tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]
    if len(raw_tickers) > 200: st.warning("Sedang menscan banyak saham, mohon bersabar...")
    
    results = []
    prog = st.progress(0)
    for i, t in enumerate(raw_tickers):
        res = get_ai_status(t)
        if res: results.append(res)
        prog.progress((i+1)/len(raw_tickers))
    
    prog.empty()
    st.session_state['hasil_scan'] = results
    st.session_state['status_scan'] = True

# 2. RESULT SECTION
if st.session_state['status_scan'] and st.session_state['hasil_scan']:
    results = st.session_state['hasil_scan']
    df_res = pd.DataFrame(results)
    
    # Priority Sorting
    def prio(x): return {'💎':0,'🥇':1,'✅':2,'⚠️':3,'💤':4,'❌':5}.get(x.split(' ')[0], 6)
    df_res['Prio'] = df_res['Keputusan'].apply(prio)
    df_res = df_res.sort_values(by=['Prio', 'Gap %'])
    
    cnt_diamond = len(df_res[df_res['Keputusan'].str.contains('💎')])
    if cnt_diamond > 0: st.success(f"🔥 Ditemukan {cnt_diamond} DIAMOND Setup!")
    
    # Tabel Selection
    def color_row(val):
        c = {'💎':'#00ced1','🥇':'#ffd700','✅':'#90ee90','❌':'#808080'}.get(val.split(' ')[0], '')
        return f'background-color: {c}; color: {"white" if c in ["#00ced1","#808080"] else "black"}; font-weight: bold'

    cols = ['Ticker','Keputusan','Tipe','Harga','Support AI','Gap %','RSI']
    event = st.dataframe(
        df_res[cols].style.map(color_row, subset=['Keputusan']),
        use_container_width=True, on_select="rerun", selection_mode="single-row",
        column_config={"Harga": st.column_config.NumberColumn(format="Rp %d"), "Support AI": st.column_config.NumberColumn(format="Rp %d"), "Gap %": st.column_config.NumberColumn(format="%.1f %%"), "RSI": st.column_config.NumberColumn(format="%.0f")}
    )

    # 3. DETAIL & PREDICTION SECTION
    if len(event.selection.rows) > 0:
        idx = event.selection.rows[0]
        sel_ticker = df_res.iloc[idx]['Ticker']
        sel_data = next(r for r in results if r['Ticker'] == sel_ticker)
        
        st.divider()
        st.subheader(f"🔍 Analisis Deep Dive: {sel_ticker}.JK")
        
        col_grafik, col_ai = st.columns([2, 1])
        
        with col_grafik:
            plot_chart(sel_data)
            with st.expander("📖 Penjelasan Teknis (K-Means)"):
                st.write(f"Saham ini bertipe **{sel_data['Tipe']}**. Support AI terdeteksi di **Rp {sel_data['Support AI']:,.0f}**. Gap risiko ke lantai dasar adalah **{sel_data['Gap %']*100:.1f}%**.")

        with col_ai:
            st.info("🤖 **MAVERICK AI ENGINE**")
            st.write("Ingin tahu prediksi harga besok menggunakan Neural Network?")
            
            # TOMBOL SAKTI
            if st.button(f"🧠 Jalankan Prediksi {sel_ticker}", type="primary"):
                with st.spinner(f"Maverick sedang melatih otak untuk {sel_ticker}... (Mungkin butuh 30-60 detik)"):
                    pred_price, real_price = run_maverick_prediction(sel_ticker + ".JK")
                
                if pred_price:
                    diff = pred_price - real_price
                    arah = "NAIK 📈" if diff > 0 else "TURUN 📉"
                    pct = (diff / real_price) * 100
                    
                    st.success("✅ PREDIKSI SELESAI")
                    st.metric(label="Target Harga Besok", value=f"Rp {pred_price:,.0f}", delta=f"{pct:.2f}%")
                    st.write(f"**Arah:** {arah}")
                    st.caption("*Disclaimer: Prediksi AI berbasis probabilitas data historis.*")
                else:
                    st.error("Gagal menjalankan prediksi.")

elif st.session_state['status_scan']:
    st.warning("Tidak ditemukan hasil.")
