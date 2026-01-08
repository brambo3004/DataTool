import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import LineString
import folium
from streamlit_folium import st_folium
import requests

# --- CONFIGURATIE ---
st.set_page_config(layout="wide", page_title="iASSET Tool - Final Complete")

FILE_NIET_RIJSTROOK = "N-allemaal-niet-rijstrook.csv"
FILE_WEL_RIJSTROOK = "N-allemaal-alleen-rijstrook.csv"

# Hiërarchie: Lager = Belangrijker
HIERARCHY_RANK = {
    'rijstrook': 1,
    'parallelweg': 2,
    'fietspad': 3
}

ALL_META_COLS = [
    'subthema', 'Situering', 'verhardingssoort', 'Soort deklaag specifiek', 
    'Jaar aanleg', 'Jaar deklaag', 'Onderhoudsproject', 
    'Advies_Onderhoudsproject', 'validation_error', 'Spoor_ID', 
    'Is_Project_Grens'
]

# --- FUNCTIES ---

@st.cache_data
def load_data():
    df1 = pd.read_csv(FILE_NIET_RIJSTROOK, low_memory=False)
    df2 = pd.read_csv(FILE_WEL_RIJSTROOK, low_memory=False)
    df = pd.concat([df1, df2], ignore_index=True)
    
    def parse_geom(x):
        try:
            return wkt.loads(x)
        except:
            return None
    
    df['geometry'] = df['rds coordinaten'].apply(parse_geom)
    df = df.dropna(subset=['geometry'])
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.set_crs(epsg=28992, inplace=True, allow_override=True)
    
    if 'id' not in gdf.columns:
        gdf['id'] = range(len(gdf))
        
    for col in ALL_META_COLS:
        if col not in gdf.columns:
            gdf[col] = ''
            
    if 'Situering' in gdf.columns:
        gdf['Situering'] = gdf['Situering'].astype(str).str.strip().str.title().replace('Nan', 'Onbekend')
    else:
        gdf['Situering'] = 'Onbekend'
        
    gdf['subthema_clean'] = gdf['subthema'].astype(str).str.lower().str.strip()
    
    def get_rank(thema):
        return HIERARCHY_RANK.get(thema, 4) 
    
    gdf['Rank'] = gdf['subthema_clean'].apply(get_rank)
    
    return gdf

def get_pdok_hectopunten_visual_only(bounds):
    """
    Haalt alle hectometerpaaltjes op in de bounding box.
    Geen strenge filtering op wegnummer meer om te voorkomen dat ze verdwijnen.
    """
    wfs_url = "https://service.pdok.nl/rws/nwbwegen/wfs/v1_0"
    buffer = 200
    minx, miny, maxx, maxy = bounds
    bbox_str = f"{minx-buffer},{miny-buffer},{maxx+buffer},{maxy+buffer}"
    
    params = {
        "service": "WFS", "version": "1.0.0", "request": "GetFeature",
        "typeName": "hectopunten", "outputFormat": "json", "bbox": bbox_str
    }
    
    try:
        r = requests.get(wfs_url, params=params, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if data.get('features'):
                gdf = gpd.GeoDataFrame.from_features(data['features'])
                gdf.set_crs(epsg=28992, inplace=True)
                
                if not gdf.empty:
                    # Probeer hectometrering kolom te vinden en om te zetten naar float
                    # PDOK gebruikt vaak 'hectomtrng' (afgekapt) of 'hectometrering'
                    col_name = 'hectometrering'
                    if 'hectomtrng' in gdf.columns:
                        col_name = 'hectomtrng'
                    
                    if col_name in gdf.columns:
                        gdf['hm_val'] = gdf[col_name].astype(float)
                    else:
                        gdf['hm_val'] = 0.0
                        
                return gdf
    except: pass
    return gpd.GeoDataFrame()

def build_hierarchical_network_lines(gdf_subset):
    """Gevectoriseerde netwerkopbouw."""
    all_lines = []
    
    for situering in gdf_subset['Situering'].unique():
        group = gdf_subset[gdf_subset['Situering'] == situering].copy()
        if len(group) < 2: continue
        
        group['geom_buffer'] = group.geometry.buffer(0.5)
        group_buffered = group.set_geometry('geom_buffer')
        
        joined = gpd.sjoin(
            group_buffered[['geom_buffer', 'Rank', 'subthema_clean']], 
            group[['geometry', 'Rank', 'subthema_clean']], 
            how='inner', predicate='intersects'
        )
        
        joined = joined[joined.index != joined['index_right']]
        if joined.empty: continue

        # Primair: zelfde type
        mask_prim = (joined['Rank_left'] < 4) & (joined['subthema_clean_left'] == joined['subthema_clean_right'])
        valid_prim = joined[mask_prim].copy()
        
        # Secundair: verbind met hoogste rang primair
        mask_sec = (joined['Rank_left'] >= 4) & (joined['Rank_right'] < 4)
        candidates_sec = joined[mask_sec].copy()
        
        valid_sec = pd.DataFrame()
        if not candidates_sec.empty:
            candidates_sec = candidates_sec.sort_values('Rank_right')
            valid_sec = candidates_sec[~candidates_sec.index.duplicated(keep='first')]
            
        final_edges = pd.concat([valid_prim, valid_sec])
        if final_edges.empty: continue
        
        centroids = group.geometry.centroid
        sources = centroids.loc[final_edges.index]
        targets = centroids.loc[final_edges['index_right'].values]
        
        lines = [LineString([s, t]) for s, t in zip(sources, targets)]
        all_lines.extend(lines)

    if all_lines:
        return gpd.GeoDataFrame(geometry=all_lines, crs="EPSG:28992")
    return gpd.GeoDataFrame(crs="EPSG:28992")

# --- UI OPBOUW ---

st.title("🛠️ iASSET Tool - Final Complete")

if 'data_complete' not in st.session_state:
    with st.spinner('Data laden...'):
        st.session_state['data_complete'] = load_data()

raw_gdf = st.session_state['data_complete']

all_roads = sorted([str(x) for x in raw_gdf['Wegnummer'].dropna().unique()])
selected_road = st.sidebar.selectbox("Kies een Wegnummer", all_roads)

road_df = raw_gdf[raw_gdf['Wegnummer'] == selected_road].copy()

col_map, col_data = st.columns([2, 1])

with col_map:
    st.subheader(f"Kaart: {selected_road}")
    
    if not road_df.empty:
        with st.spinner('Kaart opbouwen...'):
            # 1. Netwerk
            network_gdf = build_hierarchical_network_lines(road_df)
            # 2. Paaltjes (Zonder strenge filter)
            pdok_hm = get_pdok_hectopunten_visual_only(road_df.total_bounds)
        
        road_df_web = road_df.to_crs(epsg=4326)
        
        if not network_gdf.empty: network_web = network_gdf.to_crs(epsg=4326)
        else: network_web = gpd.GeoDataFrame()
            
        if not pdok_hm.empty: pdok_web = pdok_hm.to_crs(epsg=4326)
        else: pdok_web = gpd.GeoDataFrame()

        centroid = road_df_web.unary_union.centroid
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=14, tiles="CartoDB positron")

        # A. VLAKKEN
        safe_cols = [c for c in ALL_META_COLS if c in road_df_web.columns]
        folium.GeoJson(
            road_df_web[['geometry'] + safe_cols],
            style_function=lambda x: {
                'fillColor': 'orange' if x['properties'].get('Is_Project_Grens') else 'green',
                'color': 'black', 'weight': 0.5, 'fillOpacity': 0.4
            },
            tooltip=folium.GeoJsonTooltip(fields=safe_cols, style="font-size: 11px;")
        ).add_to(m)

        # B. LIJNEN
        if not network_web.empty:
            folium.GeoJson(
                network_web,
                style_function=lambda x: {'color': 'blue', 'weight': 2, 'opacity': 0.8},
                tooltip="Verbinding"
            ).add_to(m)

        # C. NODEN
        for _, row in road_df_web.iterrows():
            c = row.geometry.centroid
            folium.CircleMarker(
                [c.y, c.x], 
                radius=2, color='black', fill=True, fill_color='black', fill_opacity=1.0
            ).add_to(m)

        # D. PAALTJES
        if not pdok_web.empty:
            for _, row in pdok_web.iterrows():
                if row.geometry is None: continue
                g = row.geometry
                if g.geom_type == 'MultiPoint': g = g.centroid
                
                try: val = float(row.get('hm_val'))/10
                except: val = 0
                lbl = f"{val:.1f} {row.get('hecto_lttr','')}".strip()
                
                folium.CircleMarker(
                    [g.y, g.x], radius=4, color='red', weight=1, fill=True, fill_color='white', fill_opacity=1.0
                ).add_to(m)
                
                folium.Marker(
                    [g.y, g.x], 
                    icon=folium.DivIcon(
                        icon_size=(100,20), icon_anchor=(0,0),
                        html=f'<div style="font-size:9pt; font-weight:bold; color:red; padding-left:8px; text-shadow: -1px -1px 0 #fff, 1px -1px 0 #fff;">{lbl}</div>'
                    )
                ).add_to(m)

        st_folium(m, width=900, height=600)

with col_data:
    st.subheader("Data Mutatie")
    cols = [c for c in ALL_META_COLS if c in road_df.columns]
    st.data_editor(road_df[cols], key="editor", num_rows="dynamic", height=600)
    
    if st.button("✅ Advies Overnemen"):
        st.info("Functie tijdelijk uitgeschakeld.")

st.divider()
csv = st.session_state['data_complete'].drop(columns=['geometry', 'Rank', 'subthema_clean'], errors='ignore').to_csv(index=False).encode('utf-8')
st.download_button("📥 Download iASSET CSV", csv, "iASSET_New.csv", "text/csv")