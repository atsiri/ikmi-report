import streamlit as st
import pandas as pd
import geopandas as gpd
import json
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- 1. PASSWORD PROTECTION ---
def check_password():
    """Returns True if the user had the correct password."""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    placeholder = st.empty()
    with placeholder.container():
        st.write("## 🔒 Dashboard Login")
        if "APP_PASSWORD" not in st.secrets:
            st.warning("⚠️ 'APP_PASSWORD' not found in secrets.toml. allowing access for demo.")
            return True
            
        password = st.text_input("Password", type="password")
        if password:
            if password == st.secrets["APP_PASSWORD"]:
                st.session_state["password_correct"] = True
                placeholder.empty()
                st.rerun()
            else:
                st.error("😕 Password incorrect")
    return False

if check_password():  
    st.set_page_config(page_title="Indeks Keamanan Manusia Indonesia", layout="wide")

    # --- 1. Data Loading & Preprocessing ---
    @st.cache_data
    def load_data():
        with open('data.json', 'r') as f:
            data = json.load(f)
            
        rows = []
        indeks_rows = []
        
        for prov, content in data.items():
            prov_name = str(prov).upper().strip()
            
            # 1a. Extract Index and Dimension Scores
            indeks_record = {'PROVINSI': prov_name}
            if "IKMI_SCORE" in content:
                indeks_record['INDEKS_KEAMANAN_MANUSIA_INDONESIA'] = content["IKMI_SCORE"]
                
            if "DIMENSION_SCORES" in content:
                dim_scores = content["DIMENSION_SCORES"]
                indeks_record['KESEJAHTERAAN_SOSIAL_INDEKS'] = dim_scores.get("KESEJAHTERAAN SOSIAL", 0)
                indeks_record['KEAMANAN_DARI_BENCANA_INDEKS'] = dim_scores.get("KEAMANAN DARI BENCANA", 0)
                indeks_record['KEAMANAN_DARI_KEKERASAN_FISIK_INDEKS'] = dim_scores.get("KEAMANAN DARI KEKERASAN FISIK", 0)
                indeks_record['KEBHINNEKAAN_INDEKS'] = dim_scores.get("PERLINDUNGAN DAN PEMANFAATAN ATAS KEBHINNEKAAN", 0) 
                
            indeks_rows.append(indeks_record)
            
            # 1b. Extract Dimensions and Indicator Variables
            for key, val in content.items():
                if key not in ["IKMI_SCORE", "DIMENSION_SCORES"]:
                    dim_name = key
                    for var_name, records in val.items():
                        for rec in records:
                            rec_copy = rec.copy()
                            rec_copy['Dimensi_Key'] = dim_name
                            rec_copy['Variable_Key'] = var_name
                            rec_copy['PROVINSI'] = prov_name
                            rows.append(rec_copy)
                        
        df = pd.DataFrame(rows)
        indeks_df = pd.DataFrame(indeks_rows)
        
        # Include 'JUMLAH DESA' and 'JUMLAH_DESA' to prevent them from being treated as indicators
        metadata_cols = ['PROVINSI', 'TAHUN', 'KODE_PROVINSI', 'DIMENSI', 'VARIABEL', 'Dimensi_Key', 'Variable_Key', 'Province_Key', 'JUMLAH DESA', 'JUMLAH_DESA']
        
        for col in df.columns:
            if col not in metadata_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        indicator_cols = [c for c in numeric_cols if c not in metadata_cols]
        
        df = df.groupby(['PROVINSI', 'Dimensi_Key', 'Variable_Key'])[indicator_cols].mean().reset_index()
        
        for col in indicator_cols:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isna(c_min) or c_max == c_min:
                df[f"{col}_norm"] = 0.0
            else:
                df[f"{col}_norm"] = (df[col] - c_min) / (c_max - c_min)
                
        # Generate mapping dictionaries (Indikator -> Variabel -> Dimensi)
        ind_to_var = {}
        var_to_dim = {}
        for var in df['Variable_Key'].unique():
            var_df = df[df['Variable_Key'] == var]
            dim = var_df['Dimensi_Key'].iloc[0]
            var_to_dim[var] = dim
            for col in indicator_cols:
                if var_df[col].notna().any():
                    ind_to_var[col] = var
                    
        # Load GeoJSON
        gdf = gpd.read_file('indonesia_provinces.geojson')
        gdf['PROVINSI_GEO'] = gdf['PROVINSI'].astype(str).str.upper().str.strip()
        
        return df, gdf, indicator_cols, indeks_df, ind_to_var, var_to_dim

    df, gdf, all_indicators, indeks_df, ind_to_var, var_to_dim = load_data()

    # Aggregasi Data Untuk Tabel
    norm_cols_all = [c + "_norm" for c in all_indicators]
    prov_agg = df.groupby('PROVINSI')[all_indicators + norm_cols_all].mean().reset_index()
    prov_agg = prov_agg.merge(indeks_df, on='PROVINSI', how='left')

    indeks_data = [
        'INDEKS_KEAMANAN_MANUSIA_INDONESIA',
        'KESEJAHTERAAN_SOSIAL_INDEKS',
        'KEAMANAN_DARI_BENCANA_INDEKS',
        'KEBHINNEKAAN_INDEKS',
        'KEAMANAN_DARI_KEKERASAN_FISIK_INDEKS'
    ]

    for col in indeks_data:
        if col in prov_agg.columns:
            prov_agg[col] = prov_agg[col].fillna(0)

    # --- 2. Sidebar Filters (Hanya 2 Filter Utama) ---
    st.sidebar.header("Filter Data")

    # 1. Filter INDEKS
    selected_indeks = st.sidebar.selectbox("1. Filter by INDEKS", indeks_data)

    # 2. Filter INDIKATOR (Hanya yang diawali RASIO atau RERATA)
    valid_map_indicators = sorted([col for col in all_indicators if str(col).startswith("RASIO") or str(col).startswith("RERATA") or str(col).startswith("INDEKS")])
    selected_indikator = st.sidebar.selectbox("2. Filter by INDIKATOR (Pilih 'None' untuk Peta berbasis Indeks)", ["None"] + valid_map_indicators)

    # --- 3. Logic Setup (Map & Table) ---
    # Pemetaan Indeks Key ke Dimensi Key yang ada di dataset
    dimensi_mapping = {
        'KESEJAHTERAAN_SOSIAL_INDEKS': 'KESEJAHTERAAN SOSIAL',
        'KEAMANAN_DARI_BENCANA_INDEKS': 'KEAMANAN DARI BENCANA',
        'KEBHINNEKAAN_INDEKS': 'PERLINDUNGAN DAN PEMANFAATAN ATAS KEBHINNEKAAN',
        'KEAMANAN_DARI_KEKERASAN_FISIK_INDEKS': 'KEAMANAN DARI KEKERASAN FISIK',
        'INDEKS_KEAMANAN_MANUSIA_INDONESIA': 'ALL'
    }
    
    # Reverse mapping untuk mencari Indeks berdasarkan nama Dimensi
    dimensi_mapping_reverse = {v: k for k, v in dimensi_mapping.items()}
    
    # Baseline index untuk selalu ditampilkan
    active_indeks_cols = ['INDEKS_KEAMANAN_MANUSIA_INDONESIA']

    if selected_indikator != "None":
        # Jika Indikator dipilih: Map menampilkan indikator tsb, Tabel menampilkan 1 VARIABEL penuh
        map_metric = selected_indikator
        map_range = None # Range dinamis karena indikator tidak selalu 0-1
        
        target_var = ind_to_var.get(selected_indikator)
        if target_var:
            table_raw_cols = [col for col, var in ind_to_var.items() if var == target_var]
            
            # Tambahkan Indeks Dimensi yang menaungi Variabel ini agar tabel relevan
            rel_dim = var_to_dim.get(target_var)
            rel_indeks = dimensi_mapping_reverse.get(rel_dim)
            if rel_indeks and rel_indeks not in active_indeks_cols:
                active_indeks_cols.append(rel_indeks)
        else:
            table_raw_cols = [selected_indikator]
            
    else:
        # Jika Indikator None: Map menampilkan Indeks, Tabel menampilkan 1 DIMENSI penuh
        map_metric = selected_indeks
        map_range = (0, 100) if map_metric != 'INDEKS_KEAMANAN_MANUSIA_INDONESIA' else (0, 100) # Updated range for index scaling 0-100
        
        target_dim = dimensi_mapping.get(selected_indeks)
        if target_dim == 'ALL':
            table_raw_cols = []
            active_indeks_cols = indeks_data # Jika ALL, tampilkan seluruh indeks
        else:
            target_vars = [var for var, dim in var_to_dim.items() if dim == target_dim]
            table_raw_cols = [col for col, var in ind_to_var.items() if var in target_vars]
            
            # Tambahkan indeks dimensi yang dipilih agar muncul di tabel
            if selected_indeks not in active_indeks_cols:
                active_indeks_cols.append(selected_indeks)

    # Menggabungkan data provinsi dengan geometri map
    prov_agg = prov_agg.loc[:, ~(prov_agg.columns.str.contains('norm'))]
    merged_gdf = gdf.merge(prov_agg, left_on='PROVINSI_GEO', right_on='PROVINSI', how='left')

    if map_metric in merged_gdf.columns:
        merged_gdf['IKMI_Score'] = merged_gdf[map_metric].fillna(0)
    else:
        merged_gdf['IKMI_Score'] = 0

    # Menyiapkan data hover infobox peta
    hover_data = {ind: ':.2f' for ind in indeks_data}
    if selected_indikator != "None":
        hover_data[selected_indikator] = ':.2f'
        if f"{selected_indikator}_norm" in merged_gdf.columns:
            hover_data[f"{selected_indikator}_norm"] = ':.2f'

    # --- 4. Render Choropleth Map ---
    st.title("Peta Indeks Keamanan Manusia Indonesia")
    geojson_data = json.loads(merged_gdf.to_json())
    # --- NEW AUTOMATIC COLOR SCALE LOGIC ---
    # Daftar indikator di mana nilai tinggi = buruk (Skala dibalik menjadi Hijau -> Merah)
    negative_indicators = [
        'RASIO_PENGANGGURAN_PEKERJA',
        'RASIO_PENCURIAN_DENGAN_KEKERASAN',
        'RASIO_PENGANIAYAAN',
        'RASIO_PERKOSAAN',
        'RASIO_PEMBUNUHAN',
        'RASIO_PERDAGANGAN_ORANG',
        'RASIO_KONFLIK_ANTAR_KELOMPOK_WARGA',
        'RASIO_KONFLIK_WARGA_ANTAR_DESA',
        'RASIO_KONFLIK_ANTAR_SUKU',
        'RASIO_KONFLIK_WARGA_DENGAN_APARAT_KEAMANAN',
        'RASIO_KONFLIK_WARGA_DENGAN_APARAT_PEMERINTAH'
    ]

    # Default Skala: Merah (Buruk/Rendah) ke Hijau (Baik/Tinggi)
    map_color_scale = ["maroon", "red", "orange", "yellow", "green", "darkgreen"]

    # Jika indikator yang dipilih termasuk dalam list negative_indicators, balik warnanya
    if selected_indikator in negative_indicators:
        map_color_scale = map_color_scale[::-1]  # Menjadi Hijau (Rendah) -> Merah (Tinggi)
    # ---------------------------------------

    fig = px.choropleth_mapbox(
        merged_gdf,
        geojson=geojson_data,
        locations='PROVINSI_GEO',
        featureidkey="properties.PROVINSI_GEO",
        color='IKMI_Score', 
        color_continuous_scale=map_color_scale, 
        range_color=map_range,
        mapbox_style="carto-positron",
        zoom=3.5,
        center={"lat": -2.5, "lon": 118.0},
        opacity=0.7,
        hover_name='PROVINSI_GEO',
        hover_data=hover_data,
    )

    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)
    

    # --- 5. Display Data Table ---
    st.subheader("Detail Data per Provinsi")

    # Ambil raw kolom untuk tabel beserta versi normalisasinya
    table_norm_cols = [c + "_norm" for c in table_raw_cols]
    
    # Kolom yang ditampilkan: Provinsi + Indeks Terkait + Raw Kolom Terpilih + Norm Kolom Terpilih
    display_cols = ['PROVINSI'] + active_indeks_cols + sorted(table_raw_cols) + sorted(table_norm_cols)
    display_cols = [c for c in display_cols if c in prov_agg.columns]

    table_data = prov_agg[display_cols].sort_values(by='INDEKS_KEAMANAN_MANUSIA_INDONESIA', ascending=False)
    numeric_display_cols = [col for col in display_cols if col != 'PROVINSI']

    # 1. Define color list
    color_list = ["maroon", "red", "orange", "yellow", "green", "darkgreen"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("my_cmap", color_list)

    st.dataframe(
        table_data.style
        .format("{:.2f}", subset=numeric_display_cols)
        .background_gradient(cmap='RdYlGn', subset=['INDEKS_KEAMANAN_MANUSIA_INDONESIA'], vmin=0, vmax=100),
        #.background_gradient(cmap=custom_cmap, subset=['INDEKS_KEAMANAN_MANUSIA_INDONESIA']), #'RdYlGn'
        use_container_width=True, 
        hide_index=True
    )