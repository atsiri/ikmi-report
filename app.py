import streamlit as st
import pandas as pd
import networkx as nx
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components
from shapely import wkt
import geopandas as gpd
from streamlit_agraph import agraph, Node, Edge, Config

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
    st.set_page_config(page_title="Indeks Keamanan MasyarakatIndonesia Dashboard", layout="wide")

    # --- 1. Data Loading & Preprocessing ---
    @st.cache_data
    def load_data():
        # Load JSON data
        with open('data.json', 'r') as f:
            data = json.load(f)
            
        rows = []
        for prov, dims in data.items():
            for dim, vars_ in dims.items():
                for var, records in vars_.items():
                    for rec in records:
                        rec['Dimensi_Key'] = dim
                        rec['Variable_Key'] = var
                        rows.append(rec)
                        
        df = pd.DataFrame(rows)
        
        # Standardize province names to uppercase
        df['PROVINSI'] = df['PROVINSI'].astype(str).str.upper().str.strip()
        
        # Identify indicator columns (Exclude standard metadata columns)
        metadata_cols = ['PROVINSI', 'TAHUN', 'KODE_PROVINSI', 'DIMENSI', 'VARIABEL', 'Dimensi_Key', 'Variable_Key', 'Province_Key']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        indicator_cols = [c for c in numeric_cols if c not in metadata_cols]
        
        # Group by Province, Dimension, and Variable (take the mean for aggregations over years)
        df = df.groupby(['PROVINSI', 'Dimensi_Key', 'Variable_Key'])[indicator_cols].mean().reset_index()
        
        # Load GeoJSON data
        gdf = gpd.read_file('indonesia_provinces.geojson')
        gdf['PROVINSI_GEO'] = gdf['PROVINSI'].astype(str).str.upper().str.strip()
        
        return df, gdf, indicator_cols

    df, gdf, all_indicators = load_data()

    # --- 2. Sidebar Filters ---
    st.sidebar.header("Filter Data")

    # Filter Dimensi
    available_dims = df['Dimensi_Key'].unique()
    selected_dims = st.sidebar.multiselect("Select Dimensi", available_dims, default=available_dims[0] if len(available_dims) > 0 else None)

    # Filter Variable
    if selected_dims:
        available_vars = df[df['Dimensi_Key'].isin(selected_dims)]['Variable_Key'].unique()
    else:
        available_vars = df['Variable_Key'].unique()
    selected_vars = st.sidebar.multiselect("Select Variable", available_vars, default=available_vars[0] if len(available_vars) > 0 else None)

    # Dynamic Indicator Filtering Logic
    relevant_indicators = []
    all_table_cols = []

    if selected_vars:
        for var in selected_vars:
            # Get data subset for this variable
            var_df = df[df['Variable_Key'] == var]
            
            # Keep columns that actually have values for this variable (not completely NaN)
            valid_cols = [col for col in all_indicators if var_df[col].notna().any()]
            
            # Save all valid columns for the table display
            all_table_cols.extend(valid_cols)
            
            # Apply the specific rule: If KESIAPSIAGAAN BENCANA, only allow RASIO_ in the dropdown
            if var == "KESIAPSIAGAAN BENCANA":
                valid_cols = [col for col in valid_cols if str(col).startswith("RASIO_")]
                
            relevant_indicators.extend(valid_cols)
            
    # Remove duplicates
    relevant_indicators = list(set(relevant_indicators))
    all_table_cols = list(set(all_table_cols))

    # Filter Indicators Dropdown (Map Display)
    selected_inds = st.sidebar.multiselect("Select Indicator (for Map)", relevant_indicators, default=relevant_indicators[0] if len(relevant_indicators) > 0 else None)

    # --- 3. Data Calculation & Normalization ---
    if not selected_dims or not selected_vars or not selected_inds:
        st.warning("Please select at least one Dimensi, Variable, and Indicator from the sidebar.")
    else:
        # Filter the dataframe based on selections
        filtered_df = df[(df['Dimensi_Key'].isin(selected_dims)) & (df['Variable_Key'].isin(selected_vars))]
        
        # Aggregate actual values per province for ALL table columns (this keeps all data intact)
        prov_agg = filtered_df.groupby('PROVINSI')[all_table_cols].mean().reset_index()
        
        # Calculate Normalized Ratios (Min-Max Normalization) ONLY for Map Indicators
        normalized_cols = []
        for ind in selected_inds:
            col_min = prov_agg[ind].min()
            col_max = prov_agg[ind].max()
            norm_col_name = f"{ind}_normalized"
            normalized_cols.append(norm_col_name)
            
            if col_max - col_min == 0:
                prov_agg[norm_col_name] = 0.0 
            else:
                prov_agg[norm_col_name] = (prov_agg[ind] - col_min) / (col_max - col_min)
                
        # Calculate the average of chosen normalized indicators for the Choropleth
        prov_agg['IKMI_Ratio'] = prov_agg[normalized_cols].mean(axis=1)
        
        # Merge with GeoDataFrame
        merged_gdf = gdf.merge(prov_agg, left_on='PROVINSI_GEO', right_on='PROVINSI', how='left')
        merged_gdf['IKMI_Ratio'] = merged_gdf['IKMI_Ratio'].fillna(0)
        
        # --- 4. Render Choropleth Map ---
        st.title("Peta Indeks Keamanan Masyarakat Indonesia")
        
        geojson_data = json.loads(merged_gdf.to_json())
        
        # Prepare hover data (Show actual values for the selected map indicators)
        hover_data = {ind: ':.4f' for ind in selected_inds}
        hover_data['IKMI_Ratio'] = ':.4f'
        
        fig = px.choropleth_mapbox(
            merged_gdf,
            geojson=geojson_data,
            locations='PROVINSI_GEO',
            featureidkey="properties.PROVINSI_GEO",
            color='IKMI_Ratio',
            color_continuous_scale=["red", "yellow", "green"],
            range_color=(0, 1),
            mapbox_style="carto-positron",
            zoom=3.5,
            center={"lat": -2.5, "lon": 118.0},
            opacity=0.7,
            hover_name='PROVINSI_GEO',
            hover_data=hover_data,
            #title="Choropleth Map: Average Normalized Ratio by Province"
        )
        
        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
        
        # --- 5. Display Data Table ---
        st.subheader("Detail Data per Provinsi")
        
        # Prepare columns: Display all underlying metrics + the map score
        display_cols = ['PROVINSI'] + sorted(all_table_cols) + ['IKMI_Ratio']
        table_data = prov_agg[display_cols].sort_values(by='IKMI_Ratio', ascending=False)
        
        st.dataframe(
            table_data.style.background_gradient(cmap='RdYlGn_r', subset=['IKMI_Ratio']),
            use_container_width=True, 
            hide_index=True
        )