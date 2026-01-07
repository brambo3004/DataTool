import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import LineString, Point
import folium
from streamlit_folium import st_folium
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import requests

# --- CONFIGURATIE ---
st.set_page_config(layout="wide", page_title="iASSET Tool - Smooth Centerline")

FILE_NIET_RIJSTROOK = "N-allemaal-niet-rijstrook.csv"
FILE_WEL_RIJSTROOK = "N-allemaal-alleen-rijstrook.csv"

REQ_ONDERHOUDSPROJECT = ['afrit en entree', 'fietspad', 'inrit en doorsteek', 'parallelweg', 'rijstrook']
FORBIDDEN_ONDERHOUDSPROJECT = ['fietsstalling', 'parkeerplaats', 'rotonderand', 'verkeerseiland of middengeleider']
BOUNDARY_COLS = ['verhardingssoort', 'Soort deklaag specifiek', 'Jaar aanleg', 'Jaar deklaag']

# Alle kolommen die we ooit in een tooltip of tabel willen zien
ALL_META_COLS = [
    'subthema', 'verhardingssoort', 'Soort deklaag specifiek', 
    'Jaar aanleg', 'Jaar deklaag', 'Onderhoudsproject', 
    'Advies_Onderhoudsproject', 'validation_error', 'Spoor_ID', 
    'offset', 'volgorde_score', 'Is_Project_Grens'
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
    
    # CRASH PREVENTIE: Zorg dat alle kolommen bestaan (ook al zijn ze leeg)
    for col in ALL_META_COLS:
        if col not in gdf.columns:
            gdf[col] = np.nan if col in ['offset', 'volgorde_score'] else ''
            
    # Zorg dat Is_Project_Grens een boolean is (voorkomt kleur-fouten)
    gdf['Is_Project_Grens'] = False
    
    return gdf

def get_pdok_hectopunten_v1(bounds):
    """Haalt hectometerpunten op via WFS 1.0.0."""
    wfs_url = "https://service.pdok.nl/rws/nwbwegen/wfs/v1_0"
    # Ruime buffer van 300m om zeker te zijn
    buffer = 300
    minx, miny, maxx, maxy = bounds
    bbox_str = f"{minx-buffer},{miny-buffer},{maxx+buffer},{maxy+buffer}"
    
    params = {
        "service": "WFS", "version": "1.0.0", "request": "GetFeature",
        "typeName": "hectopunten", "outputFormat": "json", "bbox": bbox_str
    }
    
    try:
        response = requests.get(wfs_url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            features = data.get('features', [])
            if len(features) > 0:
                gdf_hm = gpd.GeoDataFrame.from_features(features)
                gdf_hm.set_crs(epsg=28992, inplace=True)
                return gdf_hm, len(features)
    except Exception:
        pass
    return gpd.GeoDataFrame(), 0

def calculate_smooth_centerline(subset):
    """
    Berekent een VLOEIENDE as door het midden van de weg.
    Lost het zigzag-probleem op door te middelen (rolling mean).
    """
    rijstroken = subset[subset['subthema_clean'] == 'rijstrook'].copy()
    
    if len(rijstroken) < 2:
        return subset, None # Te weinig data

    # 1. Ruwe Sortering met PCA (Lengterichting bepalen)
    coords = np.array([(g.centroid.x, g.centroid.y) for g in rijstroken.geometry])
    pca = PCA(n_components=1)
    scores = pca.fit_transform(coords).flatten()
    rijstroken['pca_temp'] = scores
    
    # Sorteer de rijstrook-tegels op volgorde
    rijstroken = rijstroken.sort_values('pca_temp')
    
    # 2. DE TRUC: Rolling Mean (Gladstrijken)
    # We pakken de coordinaten van de gesorteerde tegels
    xs = rijstroken.geometry.centroid.x
    ys = rijstroken.geometry.centroid.y
    
    # We middelen over 10 tegels. Dit "trekt" de lijn naar het midden van de weg
    # en haalt de zigzag (L/R/L/R) eruit.
    window_size = min(len(rijstroken), 15) # Dynamische window
    xs_smooth = xs.rolling(window=window_size, center=True, min_periods=1).mean()
    ys_smooth = ys.rolling(window=window_size, center=True, min_periods=1).mean()
    
    # Maak de Centerline
    smooth_points = [Point(x, y) for x, y in zip(xs_smooth, ys_smooth)]
    
    # Verwijder dubbele punten die te dicht op elkaar liggen (schoont op)
    clean_points = []
    if smooth_points:
        clean_points.append(smooth_points[0])
        for pt in smooth_points[1:]:
            if pt.distance(clean_points[-1]) > 0.5: # Minimaal halve meter stap
                clean_points.append(pt)
                
    if len(clean_points) < 2:
        return subset, None
        
    centerline = LineString(clean_points)
    
    # 3. Projecteer ALLES op deze gladde lijn
    stationing = []
    offsets = []
    
    for idx, row in subset.iterrows():
        pt = row.geometry.centroid
        dist_along = centerline.project(pt)
        stationing.append(dist_along)
        
        # Offset berekenen (Links/Rechts)
        proj_pt = centerline.interpolate(dist_along)
        # Vector wiskunde voor Links/Rechts bepaling
        vec_x = pt.x - proj_pt.x
        vec_y = pt.y - proj_pt.y
        # Tangent van de lijn
        next_pt = centerline.interpolate(dist_along + 1.0) # 1 meter verder
        tan_x = next_pt.x - proj_pt.x
        tan_y = next_pt.y - proj_pt.y
        # Cross product
        cross = (tan_x * vec_y) - (tan_y * vec_x)
        
        dist_real = pt.distance(centerline)
        offsets.append(dist_real * (1 if cross >= 0 else -1))
        
    subset['stationing'] = stationing
    subset['offset'] = offsets
    
    return subset, centerline

def calculate_logic(gdf):
    gdf['subthema_clean'] = gdf['subthema'].astype(str).str.lower().str.strip()
    gdf['project_clean'] = gdf['Onderhoudsproject'].astype(str).str.strip().replace('nan', '')
    
    # Reset analyse kolommen
    gdf['validation_error'] = ''
    gdf['Advies_Onderhoudsproject'] = ''
    gdf['Is_Project_Grens'] = False
    gdf['volgorde_score'] = np.nan
    gdf['offset'] = np.nan
    gdf['Spoor_ID'] = 'Overig' 
    
    # Validatie regels
    mask_missing = (gdf['subthema_clean'].isin(REQ_ONDERHOUDSPROJECT)) & (gdf['project_clean'] == '')
    gdf.loc[mask_missing, 'validation_error'] += 'Missend Project; '
    mask_excess = (gdf['subthema_clean'].isin(FORBIDDEN_ONDERHOUDSPROJECT)) & (gdf['project_clean'] != '')
    gdf.loc[mask_excess, 'validation_error'] += 'Onterecht Project; '
    
    # Buren Advies
    targets = gdf[mask_missing].copy()
    sources = gdf[gdf['project_clean'] != ''].copy()
    if not targets.empty and not sources.empty:
        targets['geometry_buffer'] = targets.geometry.buffer(0.5)
        targets.set_geometry('geometry_buffer', inplace=True)
        joined = gpd.sjoin(targets[['geometry_buffer']], sources[['geometry', 'Onderhoudsproject']], how='left', predicate='intersects')
        advice = joined.groupby(joined.index)['Onderhoudsproject'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        gdf.loc[advice.index, 'Advies_Onderhoudsproject'] = advice

    debug_storage = {}

    # Per weg verwerken
    for weg in gdf['Wegnummer'].unique():
        if pd.isna(weg): continue
        mask = gdf['Wegnummer'] == weg
        subset = gdf[mask].copy()
        if len(subset) < 2: continue
        
        try:
            # 1. Bereken de Vloeiende As & Projecties
            subset, centerline_geom = calculate_smooth_centerline(subset)
            
            if centerline_geom:
                debug_storage[weg] = centerline_geom
                subset['volgorde_score'] = subset['stationing']
            else:
                continue # Skip als geen as gevonden

            # 2. Spoorvorming (Clusteren op stabiele OFFSET)
            for thema in ['rijstrook', 'fietspad', 'parallelweg']:
                thema_mask = subset['subthema_clean'] == thema
                thema_data = subset[thema_mask]
                
                if len(thema_data) > 1:
                    offsets = thema_data[['offset']]
                    # Als er duidelijke spreiding is (>2.5m) en genoeg data
                    if offsets.std().iloc[0] > 2.5 and len(thema_data) >= 5:
                        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(offsets)
                        means = thema_data.groupby(clusters)['offset'].mean()
                        
                        # Check of clusters echt uit elkaar liggen
                        if abs(means.iloc[0] - means.iloc[1]) > 3.0:
                            # Sorteer: negatief = Rechts, positief = Links (of andersom)
                            if means.iloc[0] < means.iloc[1]:
                                map_dict = {0: f"{thema}_R", 1: f"{thema}_L"}
                            else:
                                map_dict = {0: f"{thema}_L", 1: f"{thema}_R"}
                            subset.loc[thema_mask, 'Spoor_ID'] = [map_dict[c] for c in clusters]
                        else:
                            subset.loc[thema_mask, 'Spoor_ID'] = f"{thema}_C"
                    else:
                        subset.loc[thema_mask, 'Spoor_ID'] = f"{thema}_C"
            
            subset = subset.sort_values(['Spoor_ID', 'volgorde_score'])

            # Grensbepaling
            primair_mask = subset['subthema_clean'].isin(['rijstrook', 'parallelweg', 'fietspad'])
            subset_primair = subset[primair_mask].copy()
            for col in BOUNDARY_COLS:
                if col in subset_primair.columns:
                    prev_values = subset_primair.groupby('Spoor_ID')[col].shift(1)
                    verandering = subset_primair[col].ne(prev_values) & prev_values.notna()
                    grens_indices = subset_primair[verandering].index
                    if not grens_indices.empty:
                        gdf.loc[grens_indices, 'Is_Project_Grens'] = True
                        gdf.loc[grens_indices, 'validation_error'] += f'Mogelijke Projectgrens ({col} wijzigt); '

            cols_to_update = ['volgorde_score', 'offset', 'Spoor_ID', 'Is_Project_Grens', 'validation_error']
            gdf.loc[mask, cols_to_update] = subset[cols_to_update]
            
        except Exception:
            pass

    return gdf, debug_storage

# --- UI OPBOUW ---

st.title("🛠️ iASSET Tool - Anti-Zigzag (Smooth)")

# Gebruik data_v3 om cache te forceren en KeyErrors te voorkomen
if 'data_v3' not in st.session_state:
    with st.spinner('Wegennetwerk analyseren en rechttrekken...'):
        raw_gdf = load_data()
        result_gdf, debug_info = calculate_logic(raw_gdf)
        st.session_state['data_v3'] = result_gdf
        st.session_state['debug_v3'] = debug_info
else:
    result_gdf = st.session_state['data_v3']

all_roads = sorted([str(x) for x in result_gdf['Wegnummer'].dropna().unique()])
selected_road = st.sidebar.selectbox("Kies een Wegnummer", all_roads)
show_debug = st.sidebar.checkbox("☑️ Toon Berekende As", value=False)

road_df = result_gdf[result_gdf['Wegnummer'] == selected_road].copy()
if 'volgorde_score' in road_df.columns:
    road_df = road_df.sort_values(['Spoor_ID', 'volgorde_score'])

col_map, col_data = st.columns([2, 1])

with col_map:
    st.subheader(f"Kaart: {selected_road}")
    
    if not road_df.empty:
        road_df_web = road_df.to_crs(epsg=4326)
        
        with st.spinner('PDOK laden...'):
            bounds = road_df.total_bounds
            pdok_hm, hm_count = get_pdok_hectopunten_v1(bounds)
            
        pdok_hm_web = gpd.GeoDataFrame()
        if not pdok_hm.empty:
            pdok_hm_web = pdok_hm.to_crs(epsg=4326)

        # Centrum
        try:
            if not pdok_hm_web.empty: centroid = pdok_hm_web.union_all().centroid
            else: centroid = road_df_web.union_all().centroid
        except AttributeError:
             if not pdok_hm_web.empty: centroid = pdok_hm_web.unary_union.centroid
             else: centroid = road_df_web.unary_union.centroid
            
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=14, tiles="CartoDB positron")

        # DEBUG: Toon de berekende as
        if show_debug and selected_road in st.session_state['debug_v3']:
            centerline = st.session_state['debug_v3'][selected_road]
            line_gdf = gpd.GeoDataFrame(geometry=[centerline], crs="EPSG:28992").to_crs(epsg=4326)
            folium.GeoJson(
                line_gdf,
                style_function=lambda x: {'color': 'black', 'weight': 4, 'dashArray': '5, 5'},
                tooltip="Berekende Middenas (Smooth)"
            ).add_to(m)

        # 1. VLAKKEN
        # Veilige kolom selectie
        safe_tooltip = [c for c in ALL_META_COLS if c in road_df_web.columns]
        
        folium.GeoJson(
            road_df_web[['geometry'] + safe_tooltip],
            style_function=lambda x: {
                'fillColor': 'orange' if x['properties'].get('Is_Project_Grens') else 
                             ('red' if 'Missend' in x['properties'].get('validation_error','') else 
                             ('purple' if 'Onterecht' in x['properties'].get('validation_error','') else 'green')),
                'color': 'black', 'weight': 0.5, 'fillOpacity': 0.4
            },
            tooltip=folium.GeoJsonTooltip(fields=safe_tooltip, style="font-size: 11px;")
        ).add_to(m)

        # 2. SPOREN (Lijnen)
        color_map = {'rijstrook': 'blue', 'fietspad': 'red', 'parallelweg': 'green'}
        unique_sporen = road_df_web['Spoor_ID'].dropna().unique()
        
        for spoor in unique_sporen:
            if any(k in str(spoor) for k in color_map.keys()):
                spoor_data = road_df_web[road_df_web['Spoor_ID'] == spoor].sort_values('volgorde_score')
                if len(spoor_data) > 1:
                    base_color = 'gray'
                    for key, color in color_map.items():
                        if key in str(spoor): base_color = color
                    
                    points = [(geom.centroid.y, geom.centroid.x) for geom in spoor_data.geometry]
                    folium.PolyLine(points, color=base_color, weight=2, opacity=0.8, tooltip=f"Spoor: {spoor}").add_to(m)

        # 3. PAALTJES
        if not pdok_hm_web.empty:
            for _, row in pdok_hm_web.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty: continue
                if geom.geom_type == 'MultiPoint': geom = geom.centroid
                
                hm_val = row.get('hectomtrng') or row.get('hectometrering', '?')
                try: 
                    label = f"{float(hm_val)/10:.1f}"
                except: 
                    label = str(hm_val)
                
                label += f" {row.get('hecto_lttr','') or ''}".strip()
                
                if 50 < geom.y < 54 and 3 < geom.x < 8:
                    folium.CircleMarker([geom.y, geom.x], radius=2, color='black', fill=True, fill_color='black', weight=1).add_to(m)
                    folium.Marker(
                        [geom.y, geom.x],
                        icon=folium.DivIcon(
                            icon_size=(100,20), icon_anchor=(0,0),
                            html=f'<div style="font-size:9pt; font-weight:bold; padding-left:6px;">{label}</div>'
                        )
                    ).add_to(m)

        st_folium(m, width=900, height=600)

with col_data:
    st.subheader("Data Mutatie")
    cols_to_show = [c for c in ALL_META_COLS if c in road_df.columns]
    edited_df = st.data_editor(road_df[cols_to_show], key="editor", num_rows="dynamic", height=600)
    
    if st.button("✅ Advies Overnemen"):
        mask = (road_df['Advies_Onderhoudsproject'].notna()) & (road_df['Advies_Onderhoudsproject'] != '') & (road_df['Onderhoudsproject'].isna() | (road_df['Onderhoudsproject'] == ''))
        idx = road_df[mask].index
        st.session_state['data_v3'].loc[idx, 'Onderhoudsproject'] = road_df.loc[idx, 'Advies_Onderhoudsproject']
        st.success("Bijgewerkt!")
        st.rerun()

st.divider()
csv = st.session_state['data_v3'].drop(columns=['geometry', 'geometry_buffer', 'subthema_clean', 'project_clean', 'volgorde_score', 'offset', 'Spoor_ID', 'validation_error', 'Advies_Onderhoudsproject', 'Is_Project_Grens'], errors='ignore').to_csv(index=False).encode('utf-8')
st.download_button("📥 Download iASSET CSV", csv, "iASSET_New.csv", "text/csv")