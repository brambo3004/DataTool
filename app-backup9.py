import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely import wkt
import folium
from streamlit_folium import st_folium
import requests
import networkx as nx
from datetime import datetime
import numpy as np
import io

# --- CONFIGURATIE ---
st.set_page_config(layout="wide", page_title="iASSET Tool - Smart Advisor")

FILE_NIET_RIJSTROOK = "N-allemaal-niet-rijstrook.csv"
FILE_WEL_RIJSTROOK = "N-allemaal-alleen-rijstrook.csv"

HIERARCHY_RANK = {'rijstrook': 1, 'parallelweg': 2, 'fietspad': 3}

SUBTHEMA_MUST_HAVE_PROJECT = ['afrit en entree', 'fietspad', 'inrit en doorsteek', 'parallelweg', 'rijstrook']
SUBTHEMA_MUST_NOT_HAVE_PROJECT = ['fietsstalling', 'parkeerplaats', 'rotonderand', 'verkeerseiland of middengeleider']
MUTATION_REQUIRED_COLS = ['subthema', 'naam', 'Gebruikersfunctie', 'Type onderdeel', 'verhardingssoort', 'Onderhoudsproject']
SEGMENTATION_ATTRIBUTES = ['verhardingssoort', 'Soort deklaag specifiek', 'Jaar aanleg', 'Jaar deklaag', 'Besteknummer']

# Labels voor leesbare weergave in de adviseur
FRIENDLY_LABELS = {
    'verhardingssoort': 'Verharding',
    'Soort deklaag specifiek': 'Deklaag',
    'Jaar aanleg': 'Aanleg',
    'Jaar deklaag': 'Deklaagjaar',
    'Besteknummer': 'Bestek',
    'Onderhoudsproject': 'Huidig Project'
}

ALL_META_COLS = [
    'subthema', 'Situering', 'verhardingssoort', 'Soort deklaag specifiek', 
    'Jaar aanleg', 'Jaar deklaag', 'Onderhoudsproject', 
    'Advies_Onderhoudsproject', 'validation_error', 'Spoor_ID', 
    'Is_Project_Grens', 'Advies_Bron', 'Wegnummer', 'Besteknummer',
    'tijdstipRegistratie', 'nummer', 'gps coordinaten', 'rds coordinaten'
]

# --- HULPFUNCTIES ---

def clean_display_value(val):
    """Verwijdert .0 van getallen en maakt strings netjes."""
    if pd.isna(val) or val == '' or str(val).lower() == 'nan':
        return ""
    s = str(val).strip()
    if s.endswith(".0"):
        return s[:-2]
    return s

# --- FUNCTIES: DATA & NETWERK ---

@st.cache_data
def load_data():
    df1 = pd.read_csv(FILE_NIET_RIJSTROOK, low_memory=False)
    df2 = pd.read_csv(FILE_WEL_RIJSTROOK, low_memory=False)
    df = pd.concat([df1, df2], ignore_index=True)
    
    if 'id' in df.columns: 
        df.rename(columns={'id': 'bron_id'}, inplace=True)
    
    df['sys_id'] = range(len(df))
    
    def parse_geom(x):
        try: return wkt.loads(x)
        except: return None
    
    df['geometry'] = df['rds coordinaten'].apply(parse_geom)
    df = df.dropna(subset=['geometry'])
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.set_crs(epsg=28992, inplace=True, allow_override=True)
    
    gdf.set_index('sys_id', drop=False, inplace=True)
    gdf.index.name = None
            
    for col in ALL_META_COLS:
        if col not in gdf.columns: gdf[col] = ''
    
    if 'Situering' in gdf.columns:
        gdf['Situering'] = gdf['Situering'].astype(str).str.strip().str.title().replace('Nan', 'Onbekend')
    else:
        gdf['Situering'] = 'Onbekend'
        
    gdf['subthema_clean'] = gdf['subthema'].astype(str).str.lower().str.strip()
    gdf['Rank'] = gdf['subthema_clean'].apply(lambda x: HIERARCHY_RANK.get(x, 4))
    
    def parse_date_info(x):
        s = str(x).strip()
        if s.endswith('.0'):
            s = s[:-2]
        if not s or s.lower() == 'nan':
            return 0, 0
        if len(s) == 4 and s.isdigit():
            return int(s), 0
        try:
            dt = pd.to_datetime(s, errors='coerce')
            if pd.notna(dt):
                return dt.year, dt.month
        except: 
            pass
        if len(s) >= 4 and s[:4].isdigit():
             return int(s[:4]), 0
        return 0, 0
    
    if 'tijdstipRegistratie' in gdf.columns:
        parsed = gdf['tijdstipRegistratie'].apply(parse_date_info)
        gdf['reg_jaar'] = [p[0] for p in parsed]
        gdf['reg_maand'] = [p[1] for p in parsed]
    else:
        gdf['reg_jaar'] = 0
        gdf['reg_maand'] = 0

    for col in ['Jaar aanleg', 'Jaar deklaag']:
        if col in gdf.columns:
            gdf[col] = gdf[col].apply(clean_display_value)

    return gdf

def build_graph_from_geometry(gdf):
    gdf_buffer = gdf.copy()
    gdf_buffer['geometry'] = gdf_buffer.geometry.buffer(1.5)
    
    cols = ['geometry', 'subthema_clean', 'Rank', 'sys_id']
    
    left_df = gdf_buffer[cols].copy()
    left_df.index.name = None
    right_df = gdf[cols].copy()
    right_df.index.name = None
    
    joined = gpd.sjoin(
        left_df, right_df,
        how='inner', predicate='intersects', lsuffix='left', rsuffix='right'
    )
    
    joined = joined[joined['sys_id_left'] != joined['sys_id_right']]
    
    G = nx.Graph()
    G.add_nodes_from(gdf.index)
    
    edges = []
    for _, row in joined.iterrows():
        idx_left = row['sys_id_left']
        idx_right = row['sys_id_right']
        sub_left = row['subthema_clean_left']
        sub_right = row['subthema_clean_right']
        
        rel_type = 'lateral'
        if sub_left == sub_right: rel_type = 'longitudinal'
        edges.append((idx_left, idx_right, {'type': rel_type}))
        
    G.add_edges_from(edges)
    return G

# --- FUNCTIES: ANALYSE ---

def check_rules(gdf):
    violations = []
    
    mask_missing = (
        gdf['subthema_clean'].isin([x.lower() for x in SUBTHEMA_MUST_HAVE_PROJECT]) &
        (gdf['Onderhoudsproject'].isna() | (gdf['Onderhoudsproject'] == ''))
    )
    for idx, row in gdf[mask_missing].iterrows():
        violations.append({
            'type': 'error', 'id': idx, 'subthema': row['subthema'],
            'msg': 'Mist verplicht onderhoudsproject',
            'missing_cols': ['Onderhoudsproject']
        })

    mask_unexpected = (
        gdf['subthema_clean'].isin([x.lower() for x in SUBTHEMA_MUST_NOT_HAVE_PROJECT]) &
        (gdf['Onderhoudsproject'].notna()) & (gdf['Onderhoudsproject'] != '')
    )
    for idx, row in gdf[mask_unexpected].iterrows():
        violations.append({
            'type': 'warning', 'id': idx, 'subthema': row['subthema'],
            'msg': 'Heeft onverwacht onderhoudsproject',
            'missing_cols': ['Onderhoudsproject'] 
        })
        
    mask_recent = gdf['reg_jaar'].isin([2025, 2026])
    if mask_recent.any():
        for idx, row in gdf[mask_recent].iterrows():
            missing_cols = [c for c in MUTATION_REQUIRED_COLS if pd.isna(row.get(c)) or str(row.get(c)).strip() == '']
            if missing_cols:
                maand_str = f"-{int(row['reg_maand']):02d}" if row['reg_maand'] > 0 else ""
                datum_label = f"{int(row['reg_jaar'])}{maand_str}"
                
                violations.append({
                    'type': 'mutation', 'id': idx, 'subthema': row['subthema'],
                    'msg': f"Mutatie {datum_label}: Incompleet ({', '.join(missing_cols)})",
                    'missing_cols': missing_cols
                })
    return violations

def generate_grouped_proposals(gdf, G):
    groups = {}
    node_to_group = {}
    
    BACKBONES = {
        'rijstrook':   {'prefix': 'GRP_RIJBAAN',   'rank': 1},
        'parallelweg': {'prefix': 'GRP_PARALLEL',  'rank': 2},
        'fietspad':    {'prefix': 'GRP_FIETSPAD',  'rank': 3}
    }
    
    processed_ids = set()
    
    # FASE 1: Ruggengraat
    for subthema_key, config in BACKBONES.items():
        nodes_of_type = [n for n in G.nodes if gdf.loc[n, 'subthema_clean'] == subthema_key]
        if not nodes_of_type: continue
            
        G_sub = G.subgraph(nodes_of_type).copy()
        
        edges_to_remove = []
        for u, v in G_sub.edges():
            row_u = gdf.loc[u]
            row_v = gdf.loc[v]
            match = True
            for col in SEGMENTATION_ATTRIBUTES:
                val_u = clean_display_value(row_u.get(col, ''))
                val_v = clean_display_value(row_v.get(col, ''))
                if val_u != val_v:
                    match = False
                    break
            if not match: edges_to_remove.append((u, v))
                
        G_sub.remove_edges_from(edges_to_remove)
        components = list(nx.connected_components(G_sub))
        
        for i, comp in enumerate(components):
            group_id = f"{config['prefix']}_{i+1}"
            node_list = list(comp)
            first_node = gdf.loc[node_list[0]]
            
            # --- VERBETERDE LABELS ---
            specs = []
            
            for attr in SEGMENTATION_ATTRIBUTES:
                val = clean_display_value(first_node.get(attr, ''))
                if val:
                    label = FRIENDLY_LABELS.get(attr, attr)
                    specs.append(f"{label}: {val}")
            
            curr_proj = clean_display_value(first_node.get('Onderhoudsproject', ''))
            if curr_proj:
                specs.append(f"Huidig: {curr_proj}")
            
            reason_txt = ", ".join(specs) if specs else "Geen specifieke kenmerken"

            groups[group_id] = {
                'ids': node_list,
                'subthema': subthema_key,
                'category': 'Ruggengraat',
                'reason': reason_txt,
                'rank': config['rank']
            }
            for n in node_list:
                node_to_group[n] = group_id
                processed_ids.add(n)

    # FASE 2: Pac-Man Absorptie
    remaining = set([n for n in G.nodes if n not in processed_ids])
    
    changes = True
    while changes:
        changes = False
        to_remove_from_remaining = set()
        
        for node_id in remaining:
            neighbors = list(G.neighbors(node_id))
            found_groups = set()
            for n in neighbors:
                if n in node_to_group:
                    found_groups.add(node_to_group[n])
            
            if found_groups:
                candidates = []
                for gid in found_groups:
                    rank = groups[gid]['rank']
                    candidates.append((rank, gid))
                
                candidates.sort(key=lambda x: x[0])
                best_group_id = candidates[0][1]
                
                groups[best_group_id]['ids'].append(node_id)
                node_to_group[node_id] = best_group_id
                
                to_remove_from_remaining.add(node_id)
                processed_ids.add(node_id)
                changes = True
        
        remaining -= to_remove_from_remaining

    return {k: v for k, v in groups.items() if v['ids']}

def get_pdok_hectopunten_visual_only(road_gdf):
    """
    Haalt hectometerpaaltjes op via PDOK WFS.
    """
    wfs_url = "https://service.pdok.nl/rws/nwbwegen/wfs/v1_0"
    buffer_meters = 200 
    chunk_size = 50     
    
    if road_gdf.empty:
        return gpd.GeoDataFrame()

    road_sorted = road_gdf.copy()
    road_sorted['sort_x'] = road_sorted.geometry.centroid.x
    road_sorted = road_sorted.sort_values('sort_x')

    all_features = []
    
    num_segments = len(road_sorted)
    
    for i in range(0, num_segments, chunk_size):
        chunk = road_sorted.iloc[i : i+chunk_size]
        minx, miny, maxx, maxy = chunk.total_bounds
        bbox_str = f"{minx-buffer_meters},{miny-buffer_meters},{maxx+buffer_meters},{maxy+buffer_meters}"
        
        params = {
            "service": "WFS", "version": "1.0.0", "request": "GetFeature", 
            "typeName": "hectopunten", "outputFormat": "json", 
            "bbox": bbox_str, "maxFeatures": 5000 
        }
        
        try:
            r = requests.get(wfs_url, params=params, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if data.get('features'):
                    all_features.extend(data['features'])
        except:
            continue

    if not all_features:
        return gpd.GeoDataFrame()

    gdf_result = gpd.GeoDataFrame.from_features(all_features)
    gdf_result.set_crs(epsg=28992, inplace=True)
    
    hm_col = None
    if 'hectometrering' in gdf_result.columns: hm_col = 'hectometrering'
    elif 'hectomtrng' in gdf_result.columns: hm_col = 'hectomtrng'

    if 'id' in gdf_result.columns:
        gdf_result = gdf_result.drop_duplicates(subset=['id'])
    elif hm_col:
        gdf_result = gdf_result.drop_duplicates(subset=[hm_col])
    
    if hm_col:
        gdf_result['hm_val'] = pd.to_numeric(gdf_result[hm_col], errors='coerce').fillna(0)
    else:
        gdf_result['hm_val'] = 0.0

    return gdf_result

def log_change(oid, field, old_val, new_val, status="Succes"):
    if 'change_log' not in st.session_state:
        st.session_state['change_log'] = []
    
    st.session_state['change_log'].append({
        'Tijd': datetime.now().strftime("%H:%M:%S"),
        'ID': oid,
        'Veld': field,
        'Oud': str(old_val),
        'Nieuw': str(new_val),
        'Status': status
    })

# --- START APPLICATIE ---

if 'data_complete' not in st.session_state:
    with st.spinner('Data laden...'):
        st.session_state['data_complete'] = load_data()
else:
    if 'sys_id' not in st.session_state['data_complete'].columns:
        st.cache_data.clear()
        with st.spinner('Reloading data...'):
            st.session_state['data_complete'] = load_data()
            st.rerun()

raw_gdf = st.session_state['data_complete']

# --- STATE ---
if 'processed_groups' not in st.session_state: st.session_state['processed_groups'] = set()
if 'ignored_groups' not in st.session_state: st.session_state['ignored_groups'] = set()
if 'change_log' not in st.session_state: st.session_state['change_log'] = []

if 'selected_group_id' not in st.session_state: st.session_state['selected_group_id'] = None
if 'selected_error_id' not in st.session_state: st.session_state['selected_error_id'] = None
if 'zoom_bounds' not in st.session_state: st.session_state['zoom_bounds'] = None

# --- SIDEBAR ---
st.sidebar.title("iASSET Advisor")
all_roads = sorted([str(x) for x in raw_gdf['Wegnummer'].dropna().unique()])
selected_road = st.sidebar.selectbox("Kies Wegnummer", all_roads)

road_gdf = raw_gdf[raw_gdf['Wegnummer'] == selected_road].copy()

if 'graph_current' not in st.session_state or st.session_state.get('last_road') != selected_road:
    with st.spinner('Netwerk analyseren...'):
        st.session_state['graph_current'] = build_graph_from_geometry(road_gdf)
        st.session_state['last_road'] = selected_road
        st.session_state['computed_groups'] = None
        st.session_state['zoom_bounds'] = None
        st.session_state['selected_error_id'] = None
        st.session_state['selected_group_id'] = None
        st.session_state['change_log'] = [] 

G_road = st.session_state['graph_current']

# --- LAYOUT ---
col_map, col_inspector = st.columns([3, 2])

# --- RECHTS: INSPECTOR ---
with col_inspector:
    st.subheader("Werklijst")
    
    def on_mode_change():
        st.session_state['selected_error_id'] = None
        st.session_state['selected_group_id'] = None
        st.session_state['zoom_bounds'] = None
        
    mode = st.radio("Modus:", ["🔍 Data Kwaliteit", "🏗️ Project Adviseur"], 
                    horizontal=True, on_change=on_mode_change)
    st.divider()

    # --- MODUS 1: KWALITEIT ---
    if mode == "🔍 Data Kwaliteit":
        violations = check_rules(road_gdf)
        if not violations:
            st.success("Schoon! Geen datakwaliteit issues.")
        else:
            st.write(f"**{len(violations)} issues gevonden**")
            
            with st.container(height=400):
                for i, v in enumerate(violations):
                    vid = v['id']
                    unique_key = f"btn_err_{vid}_{i}" 
                    
                    is_selected = (st.session_state['selected_error_id'] == vid)
                    
                    if is_selected:
                        with st.container(border=True):
                            st.markdown("**:blue-background[GESELECTEERD]**")
                            c1, c2 = st.columns([3, 1])
                            with c1:
                                st.markdown(f"**{v['subthema']}**")
                                st.caption(f"{v['msg']}")
                            with c2:
                                if st.button("Toon", key=unique_key):
                                    st.session_state['selected_error_id'] = vid
                                    obj_geom = road_gdf.loc[vid].geometry
                                    st.session_state['zoom_bounds'] = obj_geom.bounds
                                    st.rerun()
                    else:
                        with st.container():
                            c1, c2 = st.columns([3, 1])
                            with c1:
                                st.markdown(f"**{v['subthema']}**")
                                st.caption(f"{v['msg']}")
                            with c2:
                                if st.button("Toon", key=unique_key):
                                    st.session_state['selected_error_id'] = vid
                                    obj_geom = road_gdf.loc[vid].geometry
                                    st.session_state['zoom_bounds'] = obj_geom.bounds
                                    st.rerun()
                            st.divider()

            if st.session_state['selected_error_id']:
                err_id = st.session_state['selected_error_id']
                if err_id in road_gdf.index:
                    st.divider()
                    st.markdown(f"#### Corrigeer ID {err_id}")
                    
                    row = road_gdf.loc[err_id]
                    viol_info = next((v for v in violations if v['id'] == err_id), None)
                    cols_to_fix = viol_info['missing_cols'] if viol_info else ['Onderhoudsproject']
                    
                    inputs = {}
                    for col in cols_to_fix:
                        curr_val = clean_display_value(row.get(col, ''))
                        inputs[col] = st.text_input(f"Vul in: {col}", value=curr_val, key=f"fix_{col}_{err_id}")
                    
                    if st.button("Opslaan Correctie"):
                        for col, new_val in inputs.items():
                            old_val = raw_gdf.at[err_id, col]
                            raw_gdf.at[err_id, col] = new_val
                            log_change(err_id, col, old_val, new_val)
                        
                        st.success("Opgeslagen!")
                        st.session_state['selected_error_id'] = None
                        st.rerun()

    # --- MODUS 2: PROJECT ADVISEUR ---
    elif mode == "🏗️ Project Adviseur":
        if 'computed_groups' not in st.session_state or st.session_state['computed_groups'] is None:
            with st.spinner("AI berekent groepen (incl. absorptie)..."):
                groups = generate_grouped_proposals(road_gdf, G_road)
                st.session_state['computed_groups'] = groups
        
        all_groups = st.session_state['computed_groups']
        
        active_groups = {
            k:v for k,v in all_groups.items() 
            if k not in st.session_state['processed_groups'] 
            and k not in st.session_state['ignored_groups']
        }
        
        if not active_groups:
            st.success("Geen adviezen meer beschikbaar.")
            if st.button("Herberekenen / Reset"):
                st.session_state['computed_groups'] = None
                st.session_state['processed_groups'] = set()
                st.rerun()
        else:
            st.write(f"**{len(active_groups)} suggesties beschikbaar**")
            
            with st.container(height=400):
                for g_id in sorted(active_groups.keys()):
                    g_data = active_groups[g_id]
                    count = len(g_data['ids'])
                    icon = "🛣️" if "RIJBAAN" in g_id else "🚲" if "FIETSPAD" in g_id else "🛤️" if "PARALLEL" in g_id else "🌳"
                    
                    is_sel = (st.session_state['selected_group_id'] == g_id)
                    
                    if is_sel:
                         with st.container(border=True):
                            st.markdown("**:blue-background[GESELECTEERD]**")
                            st.markdown(f"**{icon} {g_data['subthema'].title()}** ({count} obj)")
                            st.caption(f"{g_data['reason']}")
                            b1, b2, b3 = st.columns(3)
                            with b1: # Toon
                                if st.button("👁️ Toon", key=f"vis_{g_id}"):
                                    st.session_state['selected_group_id'] = g_id
                                    grp_geom = road_gdf.loc[g_data['ids']].unary_union
                                    st.session_state['zoom_bounds'] = grp_geom.bounds
                                    st.rerun()
                            with b2: # Naam
                                if st.button("✏️ Naam", key=f"edit_{g_id}"):
                                    st.session_state['selected_group_id'] = g_id
                                    grp_geom = road_gdf.loc[g_data['ids']].unary_union
                                    st.session_state['zoom_bounds'] = grp_geom.bounds
                                    st.rerun()
                            with b3: # Negeer
                                if st.button("🗑️ Negeer", key=f"ign_{g_id}"):
                                    st.session_state['ignored_groups'].add(g_id)
                                    st.rerun()
                    else:
                        with st.container():
                            st.markdown(f"**{icon} {g_data['subthema'].title()}** ({count} obj)")
                            st.caption(f"{g_data['reason']}")
                            b1, b2, b3 = st.columns(3)
                            with b1:
                                if st.button("👁️ Toon", key=f"vis_{g_id}"):
                                    st.session_state['selected_group_id'] = g_id
                                    grp_geom = road_gdf.loc[g_data['ids']].unary_union
                                    st.session_state['zoom_bounds'] = grp_geom.bounds
                                    st.rerun()
                            with b2:
                                if st.button("✏️ Naam", key=f"edit_{g_id}"):
                                    st.session_state['selected_group_id'] = g_id
                                    grp_geom = road_gdf.loc[g_data['ids']].unary_union
                                    st.session_state['zoom_bounds'] = grp_geom.bounds
                                    st.rerun()
                            with b3:
                                if st.button("🗑️ Negeer", key=f"ign_{g_id}"):
                                    st.session_state['ignored_groups'].add(g_id)
                                    st.rerun()
                            st.divider()

            if st.session_state['selected_group_id'] and st.session_state['selected_group_id'] in active_groups:
                sel_gid = st.session_state['selected_group_id']
                sel_data = active_groups[sel_gid]
                
                st.info(f"📍 Groep: {sel_gid} ({len(sel_data['ids'])} objecten)")
                
                name_input = st.text_input("Projectnaam (bv. N351-HRB-20.1-24.3)", key="proj_name_input")
                
                if st.button("✅ Opslaan & Toepassen"):
                    if name_input.strip():
                        for oid in sel_data['ids']:
                            if oid in raw_gdf.index:
                                old_v = raw_gdf.at[oid, 'Onderhoudsproject']
                                raw_gdf.at[oid, 'Onderhoudsproject'] = name_input
                                raw_gdf.at[oid, 'Advies_Bron'] = sel_data['reason']
                                log_change(oid, 'Onderhoudsproject', old_v, name_input)
                        
                        st.session_state['processed_groups'].add(sel_gid)
                        st.session_state['selected_group_id'] = None
                        st.session_state['zoom_bounds'] = None
                        st.success("Opgeslagen!")
                        st.rerun()

# --- LINKS: KAART ---
with col_map:
    st.subheader(f"Kaart: {selected_road}")
    
    road_web = road_gdf.to_crs(epsg=4326)
    
    if st.session_state['zoom_bounds']:
        minx, miny, maxx, maxy = st.session_state['zoom_bounds']
        b_poly = wkt.loads(f"POLYGON(({minx} {miny}, {minx} {maxy}, {maxx} {maxy}, {maxx} {miny}, {minx} {miny}))")
        b_gseries = gpd.GeoSeries([b_poly], crs="EPSG:28992").to_crs(epsg=4326)
        b = b_gseries.total_bounds
        fit_b = [[b[1], b[0]], [b[3], b[2]]]
        m = folium.Map(location=[(b[1]+b[3])/2, (b[0]+b[2])/2], zoom_start=16, tiles="CartoDB positron")
        m.fit_bounds(fit_b)
    else:
        c = road_web.unary_union.centroid
        m = folium.Map(location=[c.y, c.x], zoom_start=14, tiles="CartoDB positron")

    def style_fn(feature):
        oid = feature['properties']['sys_id']
        
        if oid == st.session_state['selected_error_id']:
             return {'fillColor': 'red', 'color': 'black', 'weight': 3, 'fillOpacity': 0.8}
             
        if st.session_state['selected_group_id']:
            active_grp = st.session_state['computed_groups'][st.session_state['selected_group_id']]
            if oid in active_grp['ids']:
                return {'fillColor': 'cyan', 'color': 'black', 'weight': 2, 'fillOpacity': 0.8}

        proj = feature['properties'].get('Onderhoudsproject')
        if proj:
            return {'fillColor': '#00cc00', 'color': 'gray', 'weight': 0.5, 'fillOpacity': 0.4}
            
        return {'fillColor': 'gray', 'color': 'black', 'weight': 0.5, 'fillOpacity': 0.2}

    # TOOLTIP VELDEN UITBREIDEN
    meta_cols = [c for c in ALL_META_COLS if c in road_web.columns]
    cols_to_select = ['geometry', 'sys_id'] + meta_cols
    
    tooltip_fields = ['subthema', 'Onderhoudsproject'] + [c for c in SEGMENTATION_ATTRIBUTES if c in road_web.columns]
    
    folium.GeoJson(
        road_web[cols_to_select],
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, style="font-size: 11px;")
    ).add_to(m)

    pdok_hm = get_pdok_hectopunten_visual_only(road_gdf)
    
    if not pdok_hm.empty:
        pdok_web = pdok_hm.to_crs(epsg=4326)
        for _, row in pdok_web.iterrows():
            if row.geometry:
                g = row.geometry.centroid
                val = float(row.get('hm_val', 0))/10
                icon_html = f"""<div style="font-size: 10pt; font-weight: bold; color: black; text-shadow: 1px 1px 0 #fff;">{val:.1f}</div>"""
                folium.Marker([g.y, g.x], icon=folium.DivIcon(icon_size=(40,20), icon_anchor=(10,10), html=icon_html)).add_to(m)
                folium.CircleMarker([g.y, g.x], radius=2, color='red', fill=True).add_to(m)

    # CLICK HANDLER
    output = st_folium(m, width=None, height=600, returned_objects=["last_object_clicked"], key="folium_map")

st.divider()
st.subheader("📝 Logboek Wijzigingen & Export")

if st.session_state['change_log']:
    st.dataframe(pd.DataFrame(st.session_state['change_log']), use_container_width=True)
else:
    st.caption("Nog geen wijzigingen aangebracht.")

# --- EXPORT CONFIGURATIE (ALLEEN MUTATIES) ---
EXPORT_COLUMNS = [
    'bron_id',              # De unieke sleutel 
    'nummer',               # <--- NIEUW
    'Wegnummer',
    'subthema', 
    'Onderhoudsproject',    
    'verhardingssoort',     
    'Soort deklaag specifiek',
    'Jaar aanleg',
    'Jaar deklaag',
    'Besteknummer',
    'gps coordinaten',      # <--- NIEUW
    'rds coordinaten'       # <--- NIEUW
]

valid_export_cols = [c for c in EXPORT_COLUMNS if c in st.session_state['data_complete'].columns]

changed_ids = set()
if 'change_log' in st.session_state and st.session_state['change_log']:
    for entry in st.session_state['change_log']:
        changed_ids.add(entry['ID'])

if changed_ids:
    df_export = st.session_state['data_complete'].loc[list(changed_ids)][valid_export_cols].copy()
    
    # 1. Jaartallen opschonen
    for col in ['Jaar aanleg', 'Jaar deklaag']:
        if col in df_export.columns:
            df_export[col] = df_export[col].apply(clean_display_value)
            
    # 2. TERUG HERNOEMEN: 'bron_id' -> 'id' (belangrijk voor iASSET!)
    if 'bron_id' in df_export.columns:
        df_export.rename(columns={'bron_id': 'id'}, inplace=True)
        
    st.success(f"📦 Er staan {len(df_export)} gewijzigde objecten klaar voor export.")
    
    # KOLOMMEN MAKEN VOOR KNOPPEN
    c_dl1, c_dl2 = st.columns(2)
    
    with c_dl1:
        # OPTIE A: CSV (zoals voorheen)
        csv = df_export.to_csv(index=False, sep=';').encode('utf-8-sig')
        st.download_button(
            label="📥 Download CSV", 
            data=csv, 
            file_name="iASSET_Mutaties.csv", 
            mime="text/csv"
        )
        
    with c_dl2:
        # OPTIE B: EXCEL (Nieuw)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # AANGEPAST: Sheetnaam 'Verhardingen'
            df_export.to_excel(writer, index=False, sheet_name='Verhardingen')
            
        st.download_button(
            label="📊 Download Excel (.xlsx)",
            data=buffer,
            file_name="iASSET_Mutaties.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    st.info("Er zijn nog geen wijzigingen aangebracht. Voer eerst wijzigingen door om te kunnen exporteren.")