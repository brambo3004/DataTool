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

# AANGEPAST: Landbouwpad en Busbaan toegevoegd (Rank 2)
HIERARCHY_RANK = {
    'rijstrook': 1, 
    'parallelweg': 2, 
    'landbouwpad': 2, 
    'busbaan': 2, 
    'fietspad': 3
}

# AANGEPAST: Lijst met objecten die een project MOETEN hebben
SUBTHEMA_MUST_HAVE_PROJECT = [
    'afrit en entree', 
    'fietspad', 
    'inrit en doorsteek', 
    'parallelweg', 
    'rijstrook', 
    'landbouwpad', 
    'busbaan'
]

# --- CONFIGURATIE AANPASSING ---

# Dit blijven de 'ruggengraat' types voor de netwerk-analyse (voor de eilandjes-check)
BACKBONE_TYPES = ['rijstrook', 'parallelweg', 'landbouwpad', 'busbaan', 'fietspad']

# DIT IS DE NIEUWE LOGICA:
# Alles moet een project hebben, BEHALVE deze lijst.
# Hier kun jij later handmatig nieuwe types aan toevoegen.
SUBTHEMA_EXCEPTIONS = [
    'fietsstalling', 
    'parkeerplaats', 
    'rotonderand', 
    'verkeerseiland of middengeleider'
]

# (De oude lijst SUBTHEMA_MUST_HAVE_PROJECT hebben we voor de validatie niet meer nodig, 
#  want we draaien de logica om: alles is verplicht, tenzij...)

MUTATION_REQUIRED_COLS = ['subthema', 'naam', 'Gebruikersfunctie', 'Type onderdeel', 'verhardingssoort', 'Onderhoudsproject']

# DEZE REGEL ONTBREK BIJ JOU EN VEROORZAAKTE DE ERROR:
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
    """
    Verwijdert .0 van getallen (zoals jaartallen 2005.0 -> 2005), 
    maar LAAT HET STAAN bij tekst/projectnamen (N351...35.0 -> N351...35.0).
    """
    if pd.isna(val) or val == '' or str(val).lower() == 'nan':
        return ""
    
    s = str(val).strip()
    
    if s.endswith(".0"):
        # Check: Zitten er letters (a-z) in de string?
        # JA -> Waarschijnlijk een projectnaam of code. Laat de .0 staan.
        if any(c.isalpha() for c in s):
            return s
        
        # NEE -> Waarschijnlijk een puur getal (jaartal/float). Haal .0 weg.
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
    
    # --- FIX START: Gebruik GPS en transformeer naar Meters ---
    # We lezen de GPS coordinaten (WGS84 / Graden)
    df['geometry'] = df['gps coordinaten'].apply(parse_geom)
    
    # Gooi rijen weg zonder geldige geometrie
    df = df.dropna(subset=['geometry'])
    
    # Maak GeoDataFrame en vertel: Dit is WGS84 (Graden)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
    
    # Transformeer NU naar RD New (Meters)
    # Hierdoor kloppen je buffers van 1.5 straks precies (1.5 meter)
    gdf.to_crs(epsg=28992, inplace=True)
    # --- FIX END ---
    
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
    violations = []
    
    exceptions_clean = [x.lower() for x in SUBTHEMA_EXCEPTIONS]
    # Alle ruggengraat types verzamelen
    all_backbones = ['rijstrook', 'parallelweg', 'landbouwpad', 'busbaan', 'fietspad']
    
    # --- STAP 1: Basis Attribuut Checks ---
    # Alles wat GEEN uitzondering is, moet een project hebben
    mask_missing = (
        ~gdf['subthema_clean'].isin(exceptions_clean) & 
        (gdf['Onderhoudsproject'].isna() | (gdf['Onderhoudsproject'] == ''))
    )
    for idx, row in gdf[mask_missing].iterrows():
        violations.append({
            'type': 'error', 'id': idx, 'subthema': row['subthema'],
            'msg': 'Mist verplicht onderhoudsproject',
            'missing_cols': ['Onderhoudsproject']
        })
    
    # --- STAP 2: Topologische Checks (Weeskinderen & Integriteit) ---
    if G is not None:
        
        # A. Wees-Secundaire Objecten Check
        # Een secundair object (geen backbone, geen uitzondering) MOET vastzitten aan een backbone
        for node_id in G.nodes:
            sub = gdf.loc[node_id, 'subthema_clean']
            
            # Als het een backbone of exception is, slaan we over
            if sub in all_backbones or sub in exceptions_clean:
                continue
                
            # Dit is een secundair object (bv inrit). Check buren.
            neighbors = G.neighbors(node_id)
            connected_to_backbone = False
            for buur in neighbors:
                buur_sub = gdf.loc[buur, 'subthema_clean']
                if buur_sub in all_backbones:
                    connected_to_backbone = True
                    break
            
            if not connected_to_backbone:
                violations.append({
                    'type': 'warning', 'id': node_id, 'subthema': gdf.loc[node_id, 'subthema'],
                    'msg': 'Zwevend secundair object: Grenst nergens aan een hoofdroute (Rijbaan/Fiets/etc).',
                    'missing_cols': []
                })

        # B. "Onverwacht Gedrag" / Project Integriteit Check
        # Als een object Project X heeft, is het dan verbonden met een Ruggengraat die OOK Project X heeft?
        # Zo niet, dan is het "verdwaald" of "ver weg".
        
        # We checken alleen objecten die al een projectnaam hebben
        mask_has_project = (gdf['Onderhoudsproject'].notna()) & (gdf['Onderhoudsproject'] != '')
        for idx, row in gdf[mask_has_project].iterrows():
            my_proj = str(row['Onderhoudsproject']).strip()
            my_sub = row['subthema_clean']
            
            # Als ik zelf een backbone ben, check ik of ik aan mijn eigen soort vastzit (optioneel),
            # maar deze check is vooral bedoeld voor secundaire objecten of 'verloren' stukjes.
            
            # We zoeken via de graaf: Kan ik vanaf hier een Backbone bereiken met HETZELFDE project?
            # We doen een kleine 'Breadth First Search' of checken directe buren.
            # Voor performance checken we eerst directe buren.
            
            neighbors = list(G.neighbors(idx))
            match_found = False
            
            # Check 1: Directe buur met zelfde project?
            for buur in neighbors:
                buur_proj = str(gdf.loc[buur, 'Onderhoudsproject']).strip()
                if buur_proj == my_proj:
                    match_found = True
                    break
            
            if not match_found:
                 # Check 2: Als ik zelf backbone ben, en ik heb geen buren met zelfde project, 
                 # dan ben ik een eilandje van dat project.
                 violations.append({
                    'type': 'warning', 'id': idx, 'subthema': row['subthema'],
                    'msg': f"Geïsoleerd t.o.v. project '{my_proj}'. Geen directe buren met dit project.",
                    'missing_cols': ['Onderhoudsproject']
                })
            else:
                # Check 3 (Geavanceerd): Zit ik vast aan een Backbone met dit project?
                # Als ik een secundair object ben, en mijn buren hebben wel dit project,
                # maar GEEN van die buren is een backbone... dan zweven we samen.
                if my_sub not in all_backbones:
                    connected_to_project_backbone = False
                    for buur in neighbors:
                        buur_sub = gdf.loc[buur, 'subthema_clean']
                        buur_proj = str(gdf.loc[buur, 'Onderhoudsproject']).strip()
                        if buur_proj == my_proj and buur_sub in all_backbones:
                            connected_to_project_backbone = True
                            break
                    
                    if not connected_to_project_backbone:
                        # Nog één stap dieper kijken? (voor inrit -> berm -> rijbaan)
                        # Voor nu houden we het simpel:
                         violations.append({
                            'type': 'info', 'id': idx, 'subthema': row['subthema'],
                            'msg': f"Verbonden met '{my_proj}', maar raakt niet direct de hoofdrijbaan/fietspad van dit project.",
                            'missing_cols': []
                        })

    return violations

def generate_grouped_proposals(gdf, G):
    groups = {}
    node_to_group = {}
    
    # --- CONFIGURATIE VAN DE HIËRARCHIE (De Waterval) ---
    HIERARCHY_CONFIG = [
        {'rank': 1, 'types': ['rijstrook'], 'prefix': 'GRP_RIJBAAN'},
        {'rank': 2, 'types': ['parallelweg', 'landbouwpad', 'busbaan'], 'prefix': 'GRP_PARALLEL'},
        {'rank': 3, 'types': ['fietspad'], 'prefix': 'GRP_FIETSPAD'}
    ]
    
    processed_ids = set()
    exceptions_clean = [x.lower() for x in SUBTHEMA_EXCEPTIONS]
    
    def get_seg_hash(node_id):
        row = gdf.loc[node_id]
        vals = [clean_display_value(row.get(c, '')) for c in SEGMENTATION_ATTRIBUTES]
        return tuple(vals)

    for layer in HIERARCHY_CONFIG:
        rank = layer['rank']
        target_types = layer['types']
        prefix = layer['prefix']
        
        # --- STAP A: VORM DE RUGGENGRAAT ---
        candidates = [
            n for n in G.nodes 
            if gdf.loc[n, 'subthema_clean'] in target_types 
            and n not in processed_ids
        ]
        
        if not candidates: continue
            
        G_sub = G.subgraph(candidates).copy()
        edges_to_remove = []
        for u, v in G_sub.edges():
            if get_seg_hash(u) != get_seg_hash(v):
                 edges_to_remove.append((u, v))
        G_sub.remove_edges_from(edges_to_remove)
        
        components = list(nx.connected_components(G_sub))
        current_layer_groups = []
        
        for i, comp in enumerate(components):
            temp_id = f"{prefix}_{rank}_{i}"
            node_list = list(comp)
            first_node = gdf.loc[node_list[0]]
            seg_props = get_seg_hash(node_list[0])
            
            # --- NIEUW: Huidig project ophalen voor weergave ---
            curr_proj = clean_display_value(first_node.get('Onderhoudsproject', ''))
            
            specs = []
            for idx, attr in enumerate(SEGMENTATION_ATTRIBUTES):
                val = seg_props[idx]
                if val: specs.append(f"{FRIENDLY_LABELS.get(attr, attr)}: {val}")
            reason_txt = ", ".join(specs) if specs else "Basis kenmerken"
            
            groups[temp_id] = {
                'ids': node_list,
                'subthema': target_types[0],
                'rank': rank,
                'prefix': prefix,
                'reason': reason_txt,
                'current_project': curr_proj, # <--- OPGESLAGEN
                'seg_props': seg_props,
                'spatial_sort_val': 0 
            }
            
            for n in node_list:
                processed_ids.add(n)
                node_to_group[n] = temp_id
                
            current_layer_groups.append(temp_id)
            
        # --- STAP B: EXPANSIE ---
        for group_id in current_layer_groups:
            queue = list(groups[group_id]['ids'])
            idx = 0
            while idx < len(queue):
                current_node = queue[idx]
                idx += 1
                neighbors = G.neighbors(current_node)
                for buur in neighbors:
                    if buur in processed_ids: continue
                    buur_sub = gdf.loc[buur, 'subthema_clean']
                    if buur_sub in exceptions_clean: continue
                    
                    is_other_backbone = False
                    for check_layer in HIERARCHY_CONFIG:
                        if buur_sub in check_layer['types']:
                            is_other_backbone = True
                            break
                    if is_other_backbone: continue
                    
                    groups[group_id]['ids'].append(buur)
                    node_to_group[buur] = group_id
                    processed_ids.add(buur)
                    queue.append(buur)

    # --- STAP C: SORTEREN ---
    if not groups: return {}

    minx, miny, maxx, maxy = gdf.total_bounds
    use_x_axis = (maxx - minx) > (maxy - miny)
    
    for g_id, g_data in groups.items():
        nodes_geom = gdf.loc[g_data['ids'], 'geometry']
        avg_x = nodes_geom.centroid.x.mean()
        avg_y = nodes_geom.centroid.y.mean()
        g_data['spatial_sort_val'] = avg_x if use_x_axis else avg_y

    sorted_groups = sorted(groups.items(), key=lambda x: (x[1]['rank'], x[1]['spatial_sort_val']))
    
    final_groups = {}
    counters = {} 
    for _, data in sorted_groups:
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
        # Reset ook de klik-selectie als je van modus wisselt
        if 'folium_map' in st.session_state:
            st.session_state['folium_map'] = None 
        
    # AANGEPAST: 3e optie toegevoegd
    mode = st.radio("Modus:", 
                    ["🔍 Data Kwaliteit", "🏗️ Project Adviseur", "✏️ Individueel Bewerken"], 
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
                    
                    if "RIJBAAN" in g_id: icon = "🛣️"
                    elif "FIETSPAD" in g_id: icon = "🚲" 
                    elif "PARALLEL" in g_id: icon = "🛤️"
                    else: icon = "🌳"
                    
                    is_sel = (st.session_state['selected_group_id'] == g_id)
                    container_args = {"border": True} if is_sel else {}
                    
                    with st.container(**container_args):
                        if is_sel:
                            st.markdown("**:blue-background[GESELECTEERD]**")
                        
                        # Titel
                        st.markdown(f"**{icon} {g_data['subthema'].title()}** ({count} obj)")
                        
                        # De technische specificaties
                        st.caption(f"{g_data['reason']}")
                        
                        # OUDE PROJECT TONEN (STAP 11)
                        old_p = g_data.get('current_project', '')
                        old_p_display = old_p if old_p else "Geen"
                        st.markdown(f"<small>Huidig: *{old_p_display}*</small>", unsafe_allow_html=True)

                        # KNOPPEN (STAP 9)
                        b1, b2 = st.columns([1, 1])
                        
                        with b1: # Selecteer / Toon
                            btn_label = "📍 Geselecteerd" if is_sel else "👁️ Selecteer"
                            if st.button(btn_label, key=f"vis_{g_id}", disabled=is_sel):
                                st.session_state['selected_group_id'] = g_id
                                grp_geom = road_gdf.loc[g_data['ids']].unary_union
                                st.session_state['zoom_bounds'] = grp_geom.bounds
                                st.rerun()
                                
                        with b2: # Negeer
                            if st.button("🗑️ Negeer", key=f"ign_{g_id}"):
                                st.session_state['ignored_groups'].add(g_id)
                                if is_sel:
                                    st.session_state['selected_group_id'] = None
                                    st.session_state['zoom_bounds'] = None
                                st.rerun()
                                
                        if not is_sel:
                            st.divider()

            # --- INVULVELD VERSCHIJNT HIERONDER ALS ER IETS GESELECTEERD IS ---
            # DIT STOND VERKEERD INGESPRONGEN BIJ JOU
            if st.session_state['selected_group_id'] and st.session_state['selected_group_id'] in active_groups:
                sel_gid = st.session_state['selected_group_id']
                sel_data = active_groups[sel_gid]
                
                st.divider()
                st.markdown(f"#### 🏷️ Naamgeven: {sel_gid}")
                st.info(f"Bevat {len(sel_data['ids'])} objecten. ({sel_data['reason']})")
                
                # Als er een oud project is, kunnen we dat als suggestie in de placeholder zetten
                old_p_hint = sel_data.get('current_project', '')
                placeholder_txt = old_p_hint if old_p_hint else "bv. N351-HRB-20.1-24.3"
                
                name_input = st.text_input("Projectnaam", value="", placeholder=placeholder_txt, key="proj_name_input")
                
                if st.button("✅ Opslaan & Toepassen", type="primary"):
                    if name_input.strip():
                        val_new_clean = clean_display_value(name_input)
                        
                        count_updates = 0
                        for oid in sel_data['ids']:
                            if oid in raw_gdf.index:
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
                            st.info("Geen wijzigingen nodig, naam bestond al.")
                        st.rerun()
    
    # --- MODUS 3: INDIVIDUEEL BEWERKEN ---
    elif mode == "✏️ Individueel Bewerken":
        
        # We kijken of er iets aangeklikt is in de sessie-state van de kaart
        clicked_id = None
        map_state = st.session_state.get('folium_map')
        
        if map_state and map_state.get('last_object_clicked'):
            # Haal properties op uit de klik
            props = map_state['last_object_clicked'].get('properties')
            if props:
                clicked_id = props.get('sys_id')
        
        if clicked_id is not None and clicked_id in road_gdf.index:
            row = road_gdf.loc[clicked_id]
            
            st.markdown(f"#### ✏️ Object Bewerken (ID: {clicked_id})")
            st.info(f"**{row['subthema'].title()}**")
            
            # We maken een formulier zodat de pagina niet herlaadt bij elke toetsaanslag
            with st.form(key=f"edit_form_{clicked_id}"):
                c1, c2 = st.columns(2)
                
                # Welke velden wil je kunnen aanpassen?
                # Hier pakken we de belangrijkste mutatie-velden + project
                editable_fields = [
                    'Onderhoudsproject',
                    'verhardingssoort',
                    'Soort deklaag specifiek',
                    'Jaar aanleg',
                    'Jaar deklaag',
                    'Besteknummer'
                ]
                
                inputs = {}
                for i, field in enumerate(editable_fields):
                    # Wissel tussen linker en rechter kolom
                    col_obj = c1 if i % 2 == 0 else c2
                    
                    current_val = clean_display_value(row.get(field, ''))
                    
                    with col_obj:
                        inputs[field] = st.text_input(FRIENDLY_LABELS.get(field, field), value=current_val)
                
                st.divider()
                submitted = st.form_submit_button("💾 Wijzigingen Opslaan", use_container_width=True)
                
                if submitted:
                    changes_made = 0
                    for field, new_val in inputs.items():
                        old_val = raw_gdf.at[clicked_id, field]
                        val_old_clean = clean_display_value(old_val)
                        val_new_clean = clean_display_value(new_val)
                        
                        if val_old_clean != val_new_clean:
                            # Update de ruwe data
                            raw_gdf.at[clicked_id, field] = new_val
                            # Log voor undo/export
                            log_change(clicked_id, field, val_old_clean, new_val)
                            changes_made += 1
                    
                    if changes_made > 0:
                        st.success(f"✅ {changes_made} veld(en) bijgewerkt!")
                        st.cache_data.clear() # Cache clearen als data verandert
                        
                        # Kleine hack om direct te verversen zonder de selectie kwijt te raken
                        st.rerun()
                    else:
                        st.info("Geen wijzigingen gedetecteerd.")
                        
        else:
            st.info("👆 **Selecteer een object op de kaart** om de eigenschappen te wijzigen.")
            st.caption("Klik op een lijn of vlak in het kaartvenster hiernaast.")

# --- LINKS: KAART ---
with col_map:
    st.subheader(f"Kaart: {selected_road}")
    
    road_web = road_gdf.to_crs(epsg=4326)
    
    # 1. Bepaal de zoom en center
    if st.session_state['zoom_bounds']:
        minx, miny, maxx, maxy = st.session_state['zoom_bounds']
        b_poly = wkt.loads(f"POLYGON(({minx} {miny}, {minx} {maxy}, {maxx} {maxy}, {maxx} {miny}, {minx} {miny}))")
        b_gseries = gpd.GeoSeries([b_poly], crs="EPSG:28992").to_crs(epsg=4326)
        b = b_gseries.total_bounds
        fit_b = [[b[1], b[0]], [b[3], b[2]]]
        m = folium.Map(location=[(b[1]+b[3])/2, (b[0]+b[2])/2], zoom_start=16, tiles="CartoDB positron")
        m.fit_bounds(fit_b)
    else:
        try:
            geom_union = road_web.geometry.union_all()
        except AttributeError:
            geom_union = road_web.unary_union
            
        c = geom_union.centroid
        m = folium.Map(location=[c.y, c.x], zoom_start=14, tiles="CartoDB positron")

    # --- VOORBEREIDING KLEURING ---
    # We maken een set van alle ID's die momenteel 'geadviseerd' worden (Geel)
    suggested_ids = set()
    if 'computed_groups' in st.session_state and st.session_state['computed_groups']:
        for g_id, g_data in st.session_state['computed_groups'].items():
            # We tonen alleen suggesties die nog niet verwerkt of genegeerd zijn
            if g_id not in st.session_state['processed_groups'] and g_id not in st.session_state['ignored_groups']:
                suggested_ids.update(g_data['ids'])

    # 2. De Style Functie (AANGEPAST OP KLEURBEPALING)
    def style_fn(feature):
        oid = feature['properties']['sys_id']
        props = feature['properties']
        
        # A. SELECTIE (Lichtblauw / Cyan) - Hoogste prioriteit
        is_selected = False
        
        # - Check: Edit Modus Klik
        if mode == "✏️ Individueel Bewerken":
            map_state = st.session_state.get('folium_map')
            if map_state and map_state.get('last_object_clicked'):
                clicked_props = map_state['last_object_clicked'].get('properties')
                if clicked_props and clicked_props.get('sys_id') == oid:
                    is_selected = True
        
        # - Check: Error Selectie
        if oid == st.session_state.get('selected_error_id'):
            is_selected = True
            
        # - Check: Groep Selectie (Adviseur)
        if st.session_state.get('selected_group_id'):
            active_grp = st.session_state['computed_groups'].get(st.session_state['selected_group_id'])
            if active_grp and oid in active_grp['ids']:
                is_selected = True
        
        if is_selected:
            return {'fillColor': '#00FFFF', 'color': 'black', 'weight': 3, 'fillOpacity': 0.8}

        # B. SUGGESTIE (Geel) - Prioriteit boven bestaand project
        # "Onderhoudsprojecten waarvoor een andere naam wordt gesuggereerd"
        if oid in suggested_ids:
             return {'fillColor': '#FFFF00', 'color': 'black', 'weight': 1, 'fillOpacity': 0.6}

        # C. BESTAAND PROJECT (Groen)
        # "Onderhoudsprojecten die al kloppen"
        proj = props.get('Onderhoudsproject')
        if proj:
            return {'fillColor': '#00CC00', 'color': 'gray', 'weight': 0.5, 'fillOpacity': 0.5}

        # D. OVERIG / GEEN NODIG (Grijs)
        # "Objecten die geen onderhoudsproject nodig hebben" (en rest)
        return {'fillColor': '#808080', 'color': 'gray', 'weight': 0.5, 'fillOpacity': 0.3}

    # 3. Tooltip velden en GeoJson Laag
    meta_cols = [c for c in ALL_META_COLS if c in road_web.columns]
    cols_to_select = ['geometry', 'sys_id'] + meta_cols
    
    tooltip_fields = ['subthema', 'Onderhoudsproject'] + [c for c in SEGMENTATION_ATTRIBUTES if c in road_web.columns]
    
    folium.GeoJson(
        road_web[cols_to_select],
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, style="font-size: 11px;")
    ).add_to(m)

    # 4. Hectometrering Laag
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

    # 5. Debug Netwerk Laag
    st.write("### 🛠️ Debug Tools")
    show_network = st.toggle("🕸️ Toon Netwerk & Verbindingen", value=False)
    
    if show_network and 'graph_current' in st.session_state:
        G_debug = st.session_state['graph_current']
        
        node_group_map = {}
        if st.session_state.get('computed_groups'):
            for grp_id, grp_data in st.session_state['computed_groups'].items():
                for node_id in grp_data['ids']:
                    node_group_map[node_id] = grp_id
        
        lines_internal = []
        
        for u, v in G_debug.edges():
            if u in road_web.index and v in road_web.index:
                grp_u = node_group_map.get(u)
                grp_v = node_group_map.get(v)
                
                if grp_u and grp_v and grp_u == grp_v:
                    p1 = road_web.loc[u].geometry.centroid
                    p2 = road_web.loc[v].geometry.centroid
                    lines_internal.append([[p1.y, p1.x], [p2.y, p2.x]])

        if lines_internal:
            folium.PolyLine(
                lines_internal, color="#00FF00", weight=3, opacity=0.8, 
                tooltip="Gegroepeerde verbinding"
            ).add_to(m)
            
        for node_id in G_debug.nodes():
             if node_id in road_gdf.index:
                 pt = road_gdf.loc[node_id].geometry.centroid
                 folium.CircleMarker(
                     location=[pt.y, pt.x], radius=2, color="blue", fill=True, fillOpacity=1
                 ).add_to(m)

    # 6. Render de kaart
    st_folium(m, width=None, height=600, returned_objects=["last_object_clicked"], key="folium_map")

st.divider()
st.subheader("📝 Logboek Wijzigingen & Export")

if st.session_state['change_log']:
    # --- KNOP: ALLES TERUGDRAAIEN ---
    c_all_1, c_all_2 = st.columns([1, 5])
    with c_all_1:
        if st.button("⚠️ Alles Herstellen", type="primary", help="Draai alle wijzigingen in één keer terug"):
            # We werken van nieuw naar oud terug
            for entry in reversed(st.session_state['change_log']):
                # 1. Data Herstellen
                apply_change_to_data(entry['ID'], entry['Veld'], entry['Oud'])
                
                # 2. Adviesgroep Status Herstellen (Logic van stap 8)
                if entry['Veld'] == 'Onderhoudsproject':
                    if 'computed_groups' in st.session_state and st.session_state['computed_groups']:
                        groups = st.session_state['computed_groups']
                        target_id = entry['ID']
                        
                        group_id_to_restore = None
                        for gid, gdata in groups.items():
                            if target_id in gdata['ids']:
                                group_id_to_restore = gid
                                break
                        
                        if group_id_to_restore and group_id_to_restore in st.session_state['processed_groups']:
                            st.session_state['processed_groups'].discard(group_id_to_restore)

            # 3. Log leegmaken en opslaan
            st.session_state['change_log'] = []
            save_autosave() # Dit maakt het CSV bestand leeg
            st.success("Alle wijzigingen zijn ongedaan gemaakt!")
            st.rerun()

    with c_all_2:
        st.caption(f"Er staan {len(st.session_state['change_log'])} wijzigingen in de wachtrij.")

    st.divider()

    # --- INDIVIDUELE LIJST ---
    # We tonen de lijst omgekeerd (nieuwste bovenaan)
    reversed_log = list(reversed(list(enumerate(st.session_state['change_log']))))
    
    with st.container(height=300):
        for idx, entry in reversed_log:
            c1, c2, c3, c4 = st.columns([1, 2, 4, 1])
            c1.text(entry['Tijd'])
            c2.text(f"ID: {entry['ID']}")
            c3.text(f"{entry['Veld']}: {entry['Oud']} ➡ {entry['Nieuw']}")
            
            if c4.button("↩️ Herstel", key=f"undo_{idx}"):
                # 1. Data Waarde Herstellen
                apply_change_to_data(entry['ID'], entry['Veld'], entry['Oud'])
                
                # 2. Adviesgroep Status Herstellen
                if entry['Veld'] == 'Onderhoudsproject':
                    if 'computed_groups' in st.session_state and st.session_state['computed_groups']:
                        groups = st.session_state['computed_groups']
                        target_id = entry['ID']
                        
                        group_id_to_restore = None
                        for gid, gdata in groups.items():
                            if target_id in gdata['ids']:
                                group_id_to_restore = gid
                                break
                        
                        if group_id_to_restore and group_id_to_restore in st.session_state['processed_groups']:
                            st.session_state['processed_groups'].discard(group_id_to_restore)
                            st.toast(f"Adviesgroep {group_id_to_restore} is teruggezet.", icon="back")

                # 3. Log item verwijderen en opslaan
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