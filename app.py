import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import LineString, Point
import folium
from streamlit_folium import st_folium
import numpy as np
import requests
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# --- CONFIGURATIE ---
st.set_page_config(layout="wide", page_title="iASSET Tool - Dynamic Split")

FILE_NIET_RIJSTROOK = "N-allemaal-niet-rijstrook.csv"
FILE_WEL_RIJSTROOK = "N-allemaal-alleen-rijstrook.csv"

REQ_ONDERHOUDSPROJECT = ['afrit en entree', 'fietspad', 'inrit en doorsteek', 'parallelweg', 'rijstrook']
FORBIDDEN_ONDERHOUDSPROJECT = ['fietsstalling', 'parkeerplaats', 'rotonderand', 'verkeerseiland of middengeleider']
BOUNDARY_COLS = ['verhardingssoort', 'Soort deklaag specifiek', 'Jaar aanleg', 'Jaar deklaag']

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
    
    for col in ALL_META_COLS:
        if col not in gdf.columns:
            gdf[col] = np.nan if col in ['offset', 'volgorde_score'] else ''
    gdf['Is_Project_Grens'] = False
    
    return gdf

def get_pdok_hectopunten_filtered(bounds, wegnummer_str):
    wfs_url = "https://service.pdok.nl/rws/nwbwegen/wfs/v1_0"
    buffer = 500
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
                
                zoek_num = "".join(filter(str.isdigit, str(wegnummer_str)))
                if not zoek_num: return gdf

                mask = pd.Series(False, index=gdf.index)
                if 'stt_naam' in gdf.columns:
                    mask |= gdf['stt_naam'].astype(str).str.contains(zoek_num, na=False)
                if 'wegnummer' in gdf.columns:
                    mask |= gdf['wegnummer'].astype(str).str.contains(zoek_num, na=False)
                
                filtered = gdf[mask].copy()
                if not filtered.empty:
                    filtered['hm_val'] = filtered['hectomtrng'].fillna(filtered.get('hectometrering', 0)).astype(float)
                return filtered
    except: pass
    return gpd.GeoDataFrame()

def create_axis_from_hm_unique(gdf_hm):
    if gdf_hm.empty or len(gdf_hm) < 2: return None
    gdf_unique = gdf_hm.dissolve(by='hm_val', as_index=False).copy()
    gdf_unique['geometry'] = gdf_unique.geometry.centroid
    gdf_sorted = gdf_unique.sort_values('hm_val')
    
    # Outlier filter (Bezempluim)
    points = [gdf_sorted.iloc[0].geometry]
    for i in range(1, len(gdf_sorted)):
        curr = gdf_sorted.iloc[i].geometry
        if curr.distance(points[-1]) < 800:
            points.append(curr)
            
    if len(points) > 1: return LineString(points)
    return None

def sort_points_nearest_neighbor(df_subset):
    if len(df_subset) < 2: return df_subset
    
    # Start bij laagste stationing
    start_idx = df_subset['volgorde_score'].idxmin()
    
    coords = np.array([(g.centroid.x, g.centroid.y) for g in df_subset.geometry])
    ids = df_subset.index.tolist()
    id_to_pos = {pid: i for i, pid in enumerate(ids)}
    
    sorted_ids = [start_idx]
    current_idx = start_idx
    remaining = set(ids)
    remaining.remove(current_idx)
    
    while remaining:
        curr_pos = coords[id_to_pos[current_idx]].reshape(1, -1)
        candidates = list(remaining)
        cand_pos = coords[[id_to_pos[c] for c in candidates]]
        
        dists = cdist(curr_pos, cand_pos, metric='euclidean').flatten()
        nearest_rel_idx = np.argmin(dists)
        next_idx = candidates[nearest_rel_idx]
        
        sorted_ids.append(next_idx)
        remaining.remove(next_idx)
        current_idx = next_idx
        
    return df_subset.loc[sorted_ids]

def calculate_logic(gdf):
    gdf['subthema_clean'] = gdf['subthema'].astype(str).str.lower().str.strip()
    gdf['project_clean'] = gdf['Onderhoudsproject'].astype(str).str.strip().replace('nan', '')
    cols_reset = ['validation_error', 'Advies_Onderhoudsproject', 'Spoor_ID']
    for c in cols_reset: gdf[c] = ''
    gdf['Is_Project_Grens'] = False
    gdf['volgorde_score'] = np.nan
    gdf['offset'] = np.nan
    gdf['Spoor_ID'] = 'Overig'
    gdf['draw_order'] = np.nan
    
    mask_missing = (gdf['subthema_clean'].isin(REQ_ONDERHOUDSPROJECT)) & (gdf['project_clean'] == '')
    gdf.loc[mask_missing, 'validation_error'] += 'Missend Project; '
    mask_excess = (gdf['subthema_clean'].isin(FORBIDDEN_ONDERHOUDSPROJECT)) & (gdf['project_clean'] != '')
    gdf.loc[mask_excess, 'validation_error'] += 'Onterecht Project; '
    
    targets = gdf[mask_missing].copy()
    sources = gdf[gdf['project_clean'] != ''].copy()
    if not targets.empty and not sources.empty:
        targets['geometry_buffer'] = targets.geometry.buffer(0.5)
        targets.set_geometry('geometry_buffer', inplace=True)
        joined = gpd.sjoin(targets[['geometry_buffer']], sources[['geometry', 'Onderhoudsproject']], how='left', predicate='intersects')
        advice = joined.groupby(joined.index)['Onderhoudsproject'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        gdf.loc[advice.index, 'Advies_Onderhoudsproject'] = advice

    debug_storage = {}

    for weg in gdf['Wegnummer'].unique():
        if pd.isna(weg): continue
        mask = gdf['Wegnummer'] == weg
        subset = gdf[mask].copy()
        if len(subset) < 2: continue
        
        try:
            gdf_hm = get_pdok_hectopunten_filtered(subset.total_bounds, weg)
            centerline = create_axis_from_hm_unique(gdf_hm)
            
            if centerline is None:
                rijstr = subset[subset['subthema_clean'] == 'rijstrook']
                if not rijstr.empty:
                    rijstr = rijstr.sort_values(by=['rds coordinaten']) 
                    pts = [g.centroid for g in rijstr.geometry]
                    if len(pts) > 1: centerline = LineString(pts).simplify(5)
            
            if centerline is None: continue
            debug_storage[weg] = centerline
            
            # Projecteren
            stationing = []
            offsets = []
            for idx, row in subset.iterrows():
                pt = row.geometry.centroid
                dist_along = centerline.project(pt)
                stationing.append(dist_along)
                
                proj_pt = centerline.interpolate(dist_along)
                vec_x = pt.x - proj_pt.x
                vec_y = pt.y - proj_pt.y
                next_pt = centerline.interpolate(min(dist_along + 1.0, centerline.length))
                tan_x = next_pt.x - proj_pt.x
                tan_y = next_pt.y - proj_pt.y
                cross = (tan_x * vec_y) - (tan_y * vec_x)
                dist_real = pt.distance(centerline)
                offsets.append(dist_real * (1 if cross >= 0 else -1))
                
            subset['volgorde_score'] = stationing
            subset['offset'] = offsets

            # --- DYNAMISCHE SPOORVORMING (K-MEANS) ---
            # Hier vervangen we de harde 3.5m grens door slimme clustering
            for thema in ['rijstrook', 'fietspad', 'parallelweg']:
                thema_mask = subset['subthema_clean'] == thema
                thema_data = subset[thema_mask]
                
                if len(thema_data) > 1:
                    off_vals = thema_data[['offset']]
                    # Check spreiding: als de standaardafwijking klein is (<1m), is het waarschijnlijk 1 baan
                    # Als de spreiding groter is, zijn het er 2 (L/R)
                    
                    std_dev = off_vals['offset'].std()
                    
                    if std_dev > 1.5 and len(thema_data) >= 2:
                        # Duidelijke spreiding -> Splitsen in L en R
                        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(off_vals)
                        means = thema_data.groupby(clusters)['offset'].mean()
                        
                        # Welke cluster is Links (hoogste waarde) en welke Rechts (laagste)?
                        if means.iloc[0] < means.iloc[1]:
                            map_dict = {0: f"{thema}_R", 1: f"{thema}_L"}
                        else:
                            map_dict = {0: f"{thema}_L", 1: f"{thema}_R"}
                            
                        subset.loc[thema_mask, 'Spoor_ID'] = [map_dict[c] for c in clusters]
                    else:
                        # Weinig spreiding -> Centrum
                        subset.loc[thema_mask, 'Spoor_ID'] = f"{thema}_C"
            
            # Sorteren per Spoor
            unique_tracks = subset['Spoor_ID'].unique()
            for track in unique_tracks:
                track_mask = subset['Spoor_ID'] == track
                track_data = subset[track_mask]
                
                if len(track_data) > 1:
                    sorted_track = sort_points_nearest_neighbor(track_data)
                    order_mapping = pd.Series(range(len(sorted_track)), index=sorted_track.index)
                    subset.loc[track_mask, 'draw_order'] = subset.index.map(order_mapping)
                else:
                    subset.loc[track_mask, 'draw_order'] = 0

            # Grensbepaling
            subset = subset.sort_values(['Spoor_ID', 'draw_order'])
            primair_mask = subset['subthema_clean'].isin(['rijstrook', 'parallelweg', 'fietspad'])
            subset_primair = subset[primair_mask].copy()
            for col in BOUNDARY_COLS:
                if col in subset_primair.columns:
                    prev = subset_primair.groupby('Spoor_ID')[col].shift(1)
                    chg = subset_primair[col].ne(prev) & prev.notna()
                    idx_chg = subset_primair[chg].index
                    if not idx_chg.empty:
                        gdf.loc[idx_chg, 'Is_Project_Grens'] = True
                        gdf.loc[idx_chg, 'validation_error'] += f'Mogelijke Projectgrens ({col} wijzigt); '

            cols_upd = ['volgorde_score', 'offset', 'Spoor_ID', 'Is_Project_Grens', 'validation_error', 'draw_order']
            if 'draw_order' not in gdf.columns: gdf['draw_order'] = np.nan
            gdf.loc[mask, cols_upd] = subset[cols_upd]
            
        except Exception as e:
            pass

    return gdf, debug_storage

# --- UI OPBOUW ---

st.title("🛠️ iASSET Tool - Dynamic Clustering (No Zigzag)")

if 'data_v12' not in st.session_state:
    with st.spinner('Slimme spoorvorming toepassen...'):
        raw_gdf = load_data()
        res_gdf, dbg = calculate_logic(raw_gdf)
        st.session_state['data_v12'] = res_gdf
        st.session_state['debug_v12'] = dbg
else:
    res_gdf = st.session_state['data_v12']

all_roads = sorted([str(x) for x in res_gdf['Wegnummer'].dropna().unique()])
selected_road = st.sidebar.selectbox("Kies een Wegnummer", all_roads)
show_axis = st.sidebar.checkbox("☑️ Toon Berekende As", value=True)

road_df = res_gdf[res_gdf['Wegnummer'] == selected_road].copy()
if 'draw_order' in road_df.columns:
    road_df = road_df.sort_values(['Spoor_ID', 'draw_order'])

col_map, col_data = st.columns([2, 1])

with col_map:
    st.subheader(f"Kaart: {selected_road}")
    
    if not road_df.empty:
        road_df_web = road_df.to_crs(epsg=4326)
        
        with st.spinner('Map laden...'):
            pdok_hm = get_pdok_hectopunten_filtered(road_df.total_bounds, selected_road)
        if not pdok_hm.empty: pdok_hm = pdok_hm.to_crs(epsg=4326)

        try:
            if not pdok_hm.empty: c = pdok_hm.union_all().centroid
            else: c = road_df_web.union_all().centroid
        except: c = road_df_web.unary_union.centroid
            
        m = folium.Map(location=[c.y, c.x], zoom_start=14, tiles="CartoDB positron")

        # 1. VLAKKEN
        safe_cols = [c for c in ALL_META_COLS if c in road_df_web.columns]
        folium.GeoJson(
            road_df_web[['geometry'] + safe_cols],
            style_function=lambda x: {
                'fillColor': 'orange' if x['properties'].get('Is_Project_Grens') else 
                             ('red' if 'Missend' in x['properties'].get('validation_error','') else 
                             ('purple' if 'Onterecht' in x['properties'].get('validation_error','') else 'green')),
                'color': 'black', 'weight': 0.5, 'fillOpacity': 0.4
            },
            tooltip=folium.GeoJsonTooltip(fields=safe_cols, style="font-size: 11px;")
        ).add_to(m)

        # 2. SPOREN
        cmap = {'rijstrook': 'blue', 'fietspad': 'red', 'parallelweg': 'green'}
        for spoor in road_df_web['Spoor_ID'].dropna().unique():
            if any(k in str(spoor) for k in cmap.keys()):
                dat = road_df_web[road_df_web['Spoor_ID'] == spoor]
                
                curr_seg = []
                last_pt = None
                
                for _, r in dat.iterrows():
                    g = r.geometry.centroid
                    if last_pt is not None:
                        if last_pt.distance(r.geometry.centroid) > 300: 
                            if len(curr_seg)>1:
                                bc='gray'
                                for k,v in cmap.items(): 
                                    if k in str(spoor): bc=v
                                folium.PolyLine([(y,x) for x,y in curr_seg], color=bc, weight=2, opacity=0.8, tooltip=spoor).add_to(m)
                            curr_seg = []
                    curr_seg.append((g.x, g.y))
                    last_pt = r.geometry.centroid
                
                if len(curr_seg)>1:
                    bc='gray'
                    for k,v in cmap.items(): 
                        if k in str(spoor): bc=v
                    folium.PolyLine([(y,x) for x,y in curr_seg], color=bc, weight=2, opacity=0.8, tooltip=spoor).add_to(m)

        # 3. AS
        if show_axis and selected_road in st.session_state['debug_v12']:
            cline = st.session_state['debug_v12'][selected_road]
            line_gdf = gpd.GeoDataFrame(geometry=[cline], crs="EPSG:28992").to_crs(epsg=4326)
            folium.GeoJson(line_gdf, style_function=lambda x: {'color': 'black', 'weight': 3, 'dashArray': '5,5'}).add_to(m)

        # 4. PAALTJES
        if not pdok_hm.empty:
            for _, row in pdok_hm.iterrows():
                if row.geometry is None: continue
                g = row.geometry.centroid if row.geometry.geom_type=='MultiPoint' else row.geometry
                try: val = float(row.get('hm_val'))/10
                except: val = 0
                lbl = f"{val:.1f} {row.get('hecto_lttr','')}".strip()
                if 50<g.y<54 and 3<g.x<8:
                    folium.CircleMarker([g.y, g.x], radius=2, color='black', fill=True, fill_color='black').add_to(m)
                    folium.Marker([g.y, g.x], icon=folium.DivIcon(icon_size=(100,20), html=f'<div style="font-size:9pt; font-weight:bold; padding-left:6px;">{lbl}</div>')).add_to(m)

        st_folium(m, width=900, height=600)

with col_data:
    st.subheader("Data Mutatie")
    cols = [c for c in ALL_META_COLS if c in road_df.columns]
    st.data_editor(road_df[cols], key="editor", num_rows="dynamic", height=600)
    
    if st.button("✅ Advies Overnemen"):
        mask = (road_df['Advies_Onderhoudsproject'].notna()) & (road_df['Advies_Onderhoudsproject'] != '') & (road_df['Onderhoudsproject'].isna() | (road_df['Onderhoudsproject'] == ''))
        idx = road_df[mask].index
        st.session_state['data_v12'].loc[idx, 'Onderhoudsproject'] = road_df.loc[idx, 'Advies_Onderhoudsproject']
        st.success("Bijgewerkt!")
        st.rerun()

st.divider()
csv = st.session_state['data_v12'].drop(columns=['geometry', 'geometry_buffer', 'subthema_clean', 'project_clean', 'volgorde_score', 'offset', 'Spoor_ID', 'validation_error', 'Advies_Onderhoudsproject', 'Is_Project_Grens', 'draw_order'], errors='ignore').to_csv(index=False).encode('utf-8')
st.download_button("📥 Download iASSET CSV", csv, "iASSET_New.csv", "text/csv")