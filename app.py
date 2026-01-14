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
import os

# --- CONFIGURATIE ---
st.set_page_config(layout="wide", page_title="iASSET Tool - Smart Advisor")

FILE_NIET_RIJSTROOK = "N-allemaal-niet-rijstrook.csv"
FILE_WEL_RIJSTROOK = "N-allemaal-alleen-rijstrook.csv"
AUTOSAVE_FILE = "autosave_log.csv" 

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
    'tijdstipRegistratie', 'nummer', 'gps coordinaten', 'rds coordinaten', 'Metrering'
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

def save_autosave():
    if 'change_log' in st.session_state and st.session_state['change_log']:
        df_log = pd.DataFrame(st.session_state['change_log'])
        df_log.to_csv(AUTOSAVE_FILE, index=False, sep=';')
    else:
        if os.path.exists(AUTOSAVE_FILE):
            os.remove(AUTOSAVE_FILE)

def apply_change_to_data(oid, field, new_val):
    raw_gdf = st.session_state['data_complete']
    if oid in raw_gdf.index:
        raw_gdf.at[oid, field] = new_val

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
    
    # Hectometrering opschonen (fallback)
    if 'Metrering' in gdf.columns:
        gdf['hm_sort'] = pd.to_numeric(gdf['Metrering'].astype(str).str.replace(',', '.'), errors='coerce')
        gdf['hm_sort'] = gdf['hm_sort'].fillna(99999.9)
    else:
        gdf['hm_sort'] = 99999.9

    def parse_date_info(x):
        s = str(x).strip()
        if s.endswith('.0'): s = s[:-2]
        if not s or s.lower() == 'nan': return 0, 0
        if len(s) == 4 and s.isdigit(): return int(s), 0
        try:
            dt = pd.to_datetime(s, errors='coerce')
            if pd.notna(dt): return dt.year, dt.month
        except: pass
        if len(s) >= 4 and s[:4].isdigit(): return int(s[:4]), 0
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

def check_rules(gdf, G=None):
    """
    Checkt datakwaliteit:
    1. Missende attributen (bestaande logica).
    2. Topologische eilandjes (nieuwe logica: objecten die nergens aan vast zitten).
    """
    violations = []
    
    # --- STAP 1: Attribuut checks (bestaande logica) ---
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
    
    # --- STAP 2: Topologische checks (Nieuw: Weeskinderen) ---
    if G is not None:
        # We negeren objecten die sowieso geen project hoeven (zoals parkeerplaatsen)
        # voor de bepaling of iets 'fout' is, maar ze tellen wel mee als verbinding.
        
        # Haal alle backbones op (rijstrook, parallelweg, fietspad)
        backbone_types = ['rijstrook', 'parallelweg', 'fietspad']
        ignore_types = [x.lower() for x in SUBTHEMA_MUST_NOT_HAVE_PROJECT]
        
        # Vind alle losse componenten in het netwerk
        components = list(nx.connected_components(G))
        
        for comp in components:
            # Check of dit eilandje een ruggengraat heeft
            has_backbone = False
            valid_nodes_in_island = []
            
            for node_id in comp:
                sub = gdf.loc[node_id, 'subthema_clean']
                
                if sub in backbone_types:
                    has_backbone = True
                
                # Check of dit een object is dat wél een project zou moeten hebben
                if sub not in ignore_types:
                    valid_nodes_in_island.append(node_id)
            
            # Als er GEEN ruggengraat in dit eiland zit, zijn de valide objecten 'verloren'
            if not has_backbone and valid_nodes_in_island:
                for vid in valid_nodes_in_island:
                    row = gdf.loc[vid]
                    violations.append({
                        'type': 'error', 
                        'id': vid, 
                        'subthema': row['subthema'],
                        'msg': 'Geïsoleerd object: Geen verbinding met hoofdnetwerk (Rijbaan/Parallel/Fiets)',
                        'missing_cols': []
                    })

    return violations

def generate_grouped_proposals(gdf, G):
    groups = {}
    node_to_group = {}
    
    # Configuratie van de hiërarchie
    BACKBONES = {
        'rijstrook':   {'prefix': 'GRP_RIJBAAN',   'rank': 1},
        'parallelweg': {'prefix': 'GRP_PARALLEL',  'rank': 2},
        'fietspad':    {'prefix': 'GRP_FIETSPAD',  'rank': 3}
    }
    
    processed_ids = set()
    backbone_types = set(BACKBONES.keys())
    ignore_types = set([x.lower() for x in SUBTHEMA_MUST_NOT_HAVE_PROJECT])
    
    # Hulpfunctie om eigenschappen te vergelijken
    def get_seg_hash(node_id):
        row = gdf.loc[node_id]
        vals = [clean_display_value(row.get(c, '')) for c in SEGMENTATION_ATTRIBUTES]
        return tuple(vals)

    # --- FASE 1: Ruggengraat Formeren ---
    initial_groups = [] 
    
    for subthema_key, config in BACKBONES.items():
        nodes_of_type = [n for n in G.nodes if gdf.loc[n, 'subthema_clean'] == subthema_key]
        if not nodes_of_type: continue
            
        G_sub = G.subgraph(nodes_of_type).copy()
        
        edges_to_remove = []
        for u, v in G_sub.edges():
            if get_seg_hash(u) != get_seg_hash(v):
                 edges_to_remove.append((u, v))
                
        G_sub.remove_edges_from(edges_to_remove)
        components = list(nx.connected_components(G_sub))
        
        for i, comp in enumerate(components):
            temp_id = f"{config['prefix']}_TEMP_{i}"
            node_list = list(comp)
            first_node = gdf.loc[node_list[0]]
            
            # Metadata voor labels en vergelijking
            seg_props = get_seg_hash(node_list[0])
            
            specs = []
            for idx, attr in enumerate(SEGMENTATION_ATTRIBUTES):
                val = seg_props[idx]
                if val:
                    specs.append(f"{FRIENDLY_LABELS.get(attr, attr)}: {val}")
            
            curr_proj = clean_display_value(first_node.get('Onderhoudsproject', ''))
            if curr_proj: specs.append(f"Huidig: {curr_proj}")
            reason_txt = ", ".join(specs) if specs else "Geen specifieke kenmerken"

            groups[temp_id] = {
                'ids': node_list,
                'subthema': subthema_key,
                'category': 'Ruggengraat',
                'reason': reason_txt,
                'rank': config['rank'],
                'prefix': config['prefix'],
                'seg_props': seg_props # Opslaan voor merge check
            }
            
            initial_groups.append((config['rank'], temp_id))
            for n in node_list:
                node_to_group[n] = temp_id
                processed_ids.add(n)

    # --- FASE 2: Gelaagde Expansie (Olievlek) ---
    initial_groups.sort(key=lambda x: x[0])
    
    for rank, group_id in initial_groups:
        queue = list(groups[group_id]['ids'])
        idx = 0
        while idx < len(queue):
            current_node = queue[idx]
            idx += 1
            
            neighbors = G.neighbors(current_node)
            for buur in neighbors:
                if buur in processed_ids: continue
                buur_sub = gdf.loc[buur, 'subthema_clean']
                
                if buur_sub in backbone_types: continue
                if buur_sub in ignore_types: continue
                
                groups[group_id]['ids'].append(buur)
                node_to_group[buur] = group_id
                processed_ids.add(buur)
                queue.append(buur)

    # --- FASE 3: Ritsen (Merging) ---
    # We proberen groepen samen te voegen die via secundaire objecten (nu toegevoegd) aan elkaar raken
    # en dezelfde eigenschappen hebben.
    
    merged_map = {} # map oude_group_id -> nieuwe_group_id
    active_group_ids = list(groups.keys())
    
    changed = True
    while changed:
        changed = False
        # Bouw graaf van groepen
        G_groups = nx.Graph()
        G_groups.add_nodes_from(active_group_ids)
        
        # Check connecties tussen groepen
        # Dit is zwaar, dus we doen het slim: check buren van nodes in groep
        for gid in active_group_ids:
            g_data = groups[gid]
            my_props = g_data['seg_props']
            my_sub = g_data['subthema']
            
            # Verzamel alle buren van deze hele groep
            boundary_nodes = set()
            for node in g_data['ids']:
                for buur in G.neighbors(node):
                    if buur in node_to_group and node_to_group[buur] != gid:
                        other_gid = node_to_group[buur]
                        # Check criteria voor merge:
                        # 1. Zelfde subthema (rijbaan <-> rijbaan)
                        # 2. Zelfde eigenschappen
                        if other_gid in groups:
                            other_data = groups[other_gid]
                            if (other_data['subthema'] == my_sub and 
                                other_data['seg_props'] == my_props):
                                G_groups.add_edge(gid, other_gid)

        # Zoek componenten (groepen van groepen die samen horen)
        meta_components = list(nx.connected_components(G_groups))
        
        # Als er componenten zijn met >1 groep, moeten we mergen
        new_active_ids = []
        
        for comp in meta_components:
            comp_list = list(comp)
            if len(comp_list) > 1:
                # Merge deze groepen
                primary_id = comp_list[0]
                changed = True
                
                for other_id in comp_list[1:]:
                    # Voeg nodes toe aan primary
                    groups[primary_id]['ids'].extend(groups[other_id]['ids'])
                    # Update pointers
                    for n in groups[other_id]['ids']:
                        node_to_group[n] = primary_id
                    # Verwijder oude
                    del groups[other_id]
                
                new_active_ids.append(primary_id)
            else:
                new_active_ids.append(comp_list[0])
        
        active_group_ids = new_active_ids

    # --- FASE 4: Sorteren en Hernoemen ---
    if not groups: return {}

    minx, miny, maxx, maxy = gdf.total_bounds
    width = maxx - minx
    height = maxy - miny
    use_x_axis = width > height
    
    sortable_groups = []
    for g_id, g_data in groups.items():
        if not g_data['ids']: continue
        nodes_geom = gdf.loc[g_data['ids'], 'geometry']
        avg_x = nodes_geom.centroid.x.mean()
        avg_y = nodes_geom.centroid.y.mean()
        score = avg_x if use_x_axis else avg_y
        sortable_groups.append({'score': score, 'data': g_data, 'rank': g_data['rank']})

    # Richting detectie
    reverse_order = False
    hm_col = next((c for c in gdf.columns if 'hect' in c.lower() or 'hm' == c.lower()), None)
    if hm_col:
        try:
            valid_hm = gdf[pd.to_numeric(gdf[hm_col], errors='coerce').notna()].copy()
            if not valid_hm.empty:
                valid_hm['hm_num'] = pd.to_numeric(valid_hm[hm_col])
                node_start = valid_hm.loc[valid_hm['hm_num'].idxmin()]
                node_end = valid_hm.loc[valid_hm['hm_num'].idxmax()]
                p_s = node_start.geometry.centroid.x if use_x_axis else node_start.geometry.centroid.y
                p_e = node_end.geometry.centroid.x if use_x_axis else node_end.geometry.centroid.y
                if p_s > p_e: reverse_order = True
        except: pass

    key_func = (lambda x: (x['rank'], -x['score'])) if reverse_order else (lambda x: (x['rank'], x['score']))
    sortable_groups.sort(key=key_func)
    
    final_groups = {}
    counters = {} 
    for item in sortable_groups:
        data = item['data']
        prefix = data['prefix']
        if prefix not in counters: counters[prefix] = 1
        new_id = f"{prefix}_{counters[prefix]}"
        counters[prefix] += 1
        final_groups[new_id] = data
        
    return final_groups

def get_pdok_hectopunten_visual_only(road_gdf):
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
    save_autosave()

# --- START APPLICATIE & AUTOLOAD ---

if 'data_complete' not in st.session_state:
    with st.spinner('Data laden...'):
        st.session_state['data_complete'] = load_data()
        
        if os.path.exists(AUTOSAVE_FILE):
            try:
                df_auto = pd.read_csv(AUTOSAVE_FILE, sep=';')
                st.session_state['change_log'] = df_auto.to_dict('records')
                count_restored = 0
                for row in st.session_state['change_log']:
                    apply_change_to_data(row['ID'], row['Veld'], row['Nieuw'])
                    count_restored += 1
                if count_restored > 0:
                    st.toast(f"🔄 {count_restored} wijzigingen hersteld uit autosave!", icon="💾")
            except Exception as e:
                st.error(f"Kon autosave niet laden: {e}")
else:
    if 'sys_id' not in st.session_state['data_complete'].columns:
        st.cache_data.clear()
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
        violations = check_rules(road_gdf, G_road) # <--- NIEUWE REGEL MET GRAAF
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
                            # --- AANGEPAST: Alleen opslaan als waarde verschilt ---
                            old_val = raw_gdf.at[err_id, col]
                            val_old_clean = clean_display_value(old_val)
                            val_new_clean = clean_display_value(new_val)
                            
                            if val_old_clean != val_new_clean:
                                raw_gdf.at[err_id, col] = new_val
                                log_change(err_id, col, val_old_clean, new_val)
                        
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
            
            # Sorteer met negatieve spatial score (draait richting om)
            def sort_key_advisor(item):
                gid, data = item
                r = data.get('rank', 99)
                spatial = data.get('spatial_sort_val', 0)
                return (r, -spatial) 
            
            sorted_items = sorted(active_groups.items(), key=sort_key_advisor)
            
            with st.container(height=400):
                for g_id, g_data in sorted_items:
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
                        val_new_clean = clean_display_value(name_input)
                        
                        count_updates = 0
                        for oid in sel_data['ids']:
                            if oid in raw_gdf.index:
                                # --- AANGEPAST: Alleen opslaan als waarde verschilt ---
                                old_v = raw_gdf.at[oid, 'Onderhoudsproject']
                                val_old_clean = clean_display_value(old_v)
                                
                                if val_old_clean != val_new_clean:
                                    raw_gdf.at[oid, 'Onderhoudsproject'] = name_input
                                    raw_gdf.at[oid, 'Advies_Bron'] = sel_data['reason']
                                    log_change(oid, 'Onderhoudsproject', val_old_clean, name_input)
                                    count_updates += 1
                        
                        st.session_state['processed_groups'].add(sel_gid)
                        st.session_state['selected_group_id'] = None
                        st.session_state['zoom_bounds'] = None
                        
                        if count_updates > 0:
                            st.success(f"Opgeslagen! {count_updates} objecten bijgewerkt.")
                        else:
                            st.info("Geen wijzigingen nodig, alle objecten hadden deze naam al.")
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
        # Fix voor DeprecationWarning van unary_union
        try:
            # Nieuwe methode in recente GeoPandas versies
            geom_union = road_web.geometry.union_all()
        except AttributeError:
            # Fallback voor oudere versies
            geom_union = road_web.unary_union
            
        c = geom_union.centroid
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

    # --- DEBUG NETWERK LAAG ---
    st.write("### 🛠️ Debug Tools")
    show_network = st.toggle("🕸️ Toon Netwerk & Verbindingen", value=False)
    
    if show_network and 'graph_current' in st.session_state:
        G_debug = st.session_state['graph_current']
        
        # Mapping maken van node -> groep
        node_group_map = {}
        if 'computed_groups' in st.session_state and st.session_state['computed_groups']:
            for grp_id, grp_data in st.session_state['computed_groups'].items():
                for node_id in grp_data['ids']:
                    node_group_map[node_id] = grp_id
        
        lines_internal = []
        lines_external = []
        
        # BELANGRIJK: We gebruiken road_web (EPSG:4326) voor de coordinaten, 
        # niet road_gdf (RD), anders tekenen we buiten beeld!
        for u, v in G_debug.edges():
            if u in road_web.index and v in road_web.index:
                p1 = road_web.loc[u].geometry.centroid
                p2 = road_web.loc[v].geometry.centroid
                
                # Checken of groep gelijk is
                grp_u = node_group_map.get(u)
                grp_v = node_group_map.get(v)
                
                # Folium verwacht [Lat, Lon] -> oftewel [y, x]
                coords = [[p1.y, p1.x], [p2.y, p2.x]]
                
                if grp_u and grp_v and grp_u == grp_v:
                    lines_internal.append(coords)
                else:
                    lines_external.append(coords)

        #if lines_external:
        #    folium.PolyLine(
        #        lines_external, color="red", weight=1.5, opacity=0.6, 
        #        tooltip="Fysieke verbinding (Geen groep)"
        #    ).add_to(m)
            
        if lines_internal:
            folium.PolyLine(
                lines_internal, color="#00FF00", weight=3, opacity=0.8, 
                tooltip="Gegroepeerde verbinding"
            ).add_to(m)
            
        # Puntjes tekenen (optioneel, kan zwaar zijn bij veel data)
        # for node_id in G_debug.nodes():
        #      if node_id in road_web.index:
        #          pt = road_web.loc[node_id].geometry.centroid
        #          folium.CircleMarker([pt.y, pt.x], radius=2, color="blue").add_to(m)
            
        # Voeg ook de centroïden toe als stippen
        # Dit helpt om te zien waar de lijnen vandaan komen
        for node_id in G_debug.nodes():
             if node_id in road_gdf.index:
                 pt = road_gdf.loc[node_id].geometry.centroid
                 folium.CircleMarker(
                     location=[pt.y, pt.x],
                     radius=2,
                     color="blue",
                     fill=True,
                     fillOpacity=1
                 ).add_to(m)

    # CLICK HANDLER
    output = st_folium(m, width=None, height=600, returned_objects=["last_object_clicked"], key="folium_map")

st.divider()
st.subheader("📝 Logboek Wijzigingen & Export")

if st.session_state['change_log']:
    # We tonen de lijst omgekeerd (nieuwste bovenaan)
    reversed_log = list(reversed(list(enumerate(st.session_state['change_log']))))
    
    with st.container(height=300):
        for idx, entry in reversed_log:
            c1, c2, c3, c4 = st.columns([1, 2, 4, 1])
            c1.text(entry['Tijd'])
            c2.text(f"ID: {entry['ID']}")
            c3.text(f"{entry['Veld']}: {entry['Oud']} ➡ {entry['Nieuw']}")
            
            if c4.button("↩️ Herstel", key=f"undo_{idx}"):
                apply_change_to_data(entry['ID'], entry['Veld'], entry['Oud'])
                del st.session_state['change_log'][idx]
                save_autosave()
                st.success("Wijziging ongedaan gemaakt!")
                st.rerun()
else:
    st.caption("Nog geen wijzigingen aangebracht.")

# --- EXPORT CONFIGURATIE (ALLEEN MUTATIES) ---
EXPORT_COLUMNS = [
    'bron_id',              # De unieke sleutel 
    'nummer',               
    'Wegnummer',
    'subthema', 
    'Onderhoudsproject',    
    'verhardingssoort',     
    'Soort deklaag specifiek',
    'Jaar aanleg',
    'Jaar deklaag',
    'Besteknummer',
    'gps coordinaten',      
    'rds coordinaten'       
]

valid_export_cols = [c for c in EXPORT_COLUMNS if c in st.session_state['data_complete'].columns]

changed_ids = set()
if 'change_log' in st.session_state and st.session_state['change_log']:
    for entry in st.session_state['change_log']:
        changed_ids.add(entry['ID'])

if changed_ids:
    df_export = st.session_state['data_complete'].loc[list(changed_ids)][valid_export_cols].copy()
    
    for col in ['Jaar aanleg', 'Jaar deklaag']:
        if col in df_export.columns:
            df_export[col] = df_export[col].apply(clean_display_value)
            
    if 'bron_id' in df_export.columns:
        df_export.rename(columns={'bron_id': 'id'}, inplace=True)
        
    st.success(f"📦 Er staan {len(df_export)} gewijzigde objecten klaar voor export.")
    
    c_dl1, c_dl2 = st.columns(2)
    
    with c_dl1:
        csv = df_export.to_csv(index=False, sep=';').encode('utf-8-sig')
        st.download_button(
            label="📥 Download CSV", 
            data=csv, 
            file_name="iASSET_Mutaties.csv", 
            mime="text/csv"
        )
        
    with c_dl2:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_export.to_excel(writer, index=False, sheet_name='Verhardingen')
            
        st.download_button(
            label="📊 Download Excel (.xlsx)",
            data=buffer,
            file_name="iASSET_Mutaties.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    st.info("Er zijn nog geen wijzigingen aangebracht. Voer eerst wijzigingen door om te kunnen exporteren.")