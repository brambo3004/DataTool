import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely import wkt
import folium
from streamlit_folium import st_folium
import requests
import networkx as nx
from datetime import datetime

# --- CONFIGURATIE ---
st.set_page_config(layout="wide", page_title="iASSET Tool - Smart Advisor")

FILE_NIET_RIJSTROOK = "N-allemaal-niet-rijstrook.csv"
FILE_WEL_RIJSTROOK = "N-allemaal-alleen-rijstrook.csv"

HIERARCHY_RANK = {'rijstrook': 1, 'parallelweg': 2, 'fietspad': 3}

SUBTHEMA_MUST_HAVE_PROJECT = ['afrit en entree', 'fietspad', 'inrit en doorsteek', 'parallelweg', 'rijstrook']
SUBTHEMA_MUST_NOT_HAVE_PROJECT = ['fietsstalling', 'parkeerplaats', 'rotonderand', 'verkeerseiland of middengeleider']
MUTATION_REQUIRED_COLS = ['subthema', 'naam', 'Gebruikersfunctie', 'Type onderdeel', 'verhardingssoort', 'Onderhoudsproject']
SEGMENTATION_ATTRIBUTES = ['verhardingssoort', 'Soort deklaag specifiek', 'Jaar aanleg', 'Jaar deklaag', 'Jaar conservering', 'Jaar herstrating', 'Besteknummer']

ALL_META_COLS = [
    'subthema', 'Situering', 'verhardingssoort', 'Soort deklaag specifiek', 
    'Jaar aanleg', 'Jaar deklaag', 'Onderhoudsproject', 
    'Advies_Onderhoudsproject', 'validation_error', 'Spoor_ID', 
    'Is_Project_Grens', 'Advies_Bron', 'Wegnummer'
]

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
            
    for col in ALL_META_COLS:
        if col not in gdf.columns: gdf[col] = ''
    
    if 'Situering' in gdf.columns:
        gdf['Situering'] = gdf['Situering'].astype(str).str.strip().str.title().replace('Nan', 'Onbekend')
    else:
        gdf['Situering'] = 'Onbekend'
        
    gdf['subthema_clean'] = gdf['subthema'].astype(str).str.lower().str.strip()
    gdf['Rank'] = gdf['subthema_clean'].apply(lambda x: HIERARCHY_RANK.get(x, 4))
    
    def parse_year(x):
        try:
            s = str(x).strip()
            if len(s) >= 4 and s[:4].isdigit(): return int(s[:4])
        except: pass
        return 0
    
    if 'tijdstipRegistratie' in gdf.columns:
        gdf['reg_jaar'] = gdf['tijdstipRegistratie'].apply(parse_year)
    else:
        gdf['reg_jaar'] = 0

    return gdf

def build_graph_from_geometry(gdf):
    gdf_buffer = gdf.copy()
    gdf_buffer['geometry'] = gdf_buffer.geometry.buffer(0.5)
    
    joined = gpd.sjoin(
        gdf_buffer[['geometry', 'subthema_clean', 'Rank']],
        gdf[['geometry', 'subthema_clean', 'Rank']],
        how='inner', predicate='intersects', lsuffix='left', rsuffix='right'
    )
    joined = joined[joined.index != joined.index_right]
    
    G = nx.Graph()
    G.add_nodes_from(gdf.index)
    
    edges = []
    for idx_left, row in joined.iterrows():
        idx_right = row['index_right']
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
    
    # Regel 1: Mist Project
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

    # Regel 2: Onverwacht Project
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
        
    # Regel 3: Mutaties
    mask_recent = gdf['reg_jaar'].isin([2025, 2026])
    if mask_recent.any():
        for idx, row in gdf[mask_recent].iterrows():
            missing_cols = [c for c in MUTATION_REQUIRED_COLS if pd.isna(row.get(c)) or str(row.get(c)).strip() == '']
            if missing_cols:
                violations.append({
                    'type': 'mutation', 'id': idx, 'subthema': row['subthema'],
                    'msg': f"Incompleet: {', '.join(missing_cols)}",
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
                val_u = str(row_u.get(col, '')).strip().replace('nan', '')
                val_v = str(row_v.get(col, '')).strip().replace('nan', '')
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
            
            specs = []
            if str(first_node.get('Jaar deklaag', '')).strip() not in ['nan', '']:
                specs.append(f"Bj:{int(float(first_node['Jaar deklaag']))}")
            if str(first_node.get('verhardingssoort', '')).strip() not in ['nan', '']:
                specs.append(str(first_node['verhardingssoort'])[:15])
            
            reason_txt = ", ".join(specs) if specs else "Geen details"

            groups[group_id] = {
                'ids': node_list,
                'subthema': subthema_key,
                'category': 'Ruggengraat',
                'reason': f"{subthema_key.title()} ({reason_txt})",
                'rank': config['rank']
            }
            for n in node_list:
                node_to_group[n] = group_id
                processed_ids.add(n)

    # FASE 2: Iteratieve Absorptie
    remaining = set([n for n in G.nodes if n not in processed_ids])
    
    changes = True
    while changes:
        changes = False
        to_remove_from_remaining = set()
        
        for node_id in remaining:
            neighbors = list(G.neighbors(node_id))
            candidates = []
            
            for n in neighbors:
                if n in node_to_group:
                    p_group = node_to_group[n]
                    p_rank = groups[p_group]['rank']
                    candidates.append((p_rank, p_group))
            
            if candidates:
                candidates.sort(key=lambda x: x[0])
                best_parent_group_id = candidates[0][1]
                
                node_to_group[node_id] = best_parent_group_id 
                
                sec_group_id = f"{best_parent_group_id}_SEC"
                
                if sec_group_id not in groups:
                    parent_info = groups[best_parent_group_id]
                    groups[sec_group_id] = {
                        'ids': [],
                        'subthema': 'secundair',
                        'category': 'Secundair',
                        'reason': f"Ligt aan {parent_info['subthema']}",
                        'rank': 99 
                    }
                    groups[sec_group_id]['rank'] = 99 
                    
                groups[sec_group_id]['ids'].append(node_id)
                to_remove_from_remaining.add(node_id)
                processed_ids.add(node_id)
                changes = True
        
        remaining -= to_remove_from_remaining

    return {k: v for k, v in groups.items() if v['ids']}

def get_pdok_hectopunten_visual_only(bounds):
    wfs_url = "https://service.pdok.nl/rws/nwbwegen/wfs/v1_0"
    buffer = 500
    minx, miny, maxx, maxy = bounds
    bbox_str = f"{minx-buffer},{miny-buffer},{maxx+buffer},{maxy+buffer}"
    params = {
        "service": "WFS", "version": "1.0.0", "request": "GetFeature", 
        "typeName": "hectopunten", "outputFormat": "json", 
        "bbox": bbox_str, "maxFeatures": 5000
    }
    try:
        r = requests.get(wfs_url, params=params, timeout=4)
        if r.status_code == 200:
            data = r.json()
            if data.get('features'):
                gdf = gpd.GeoDataFrame.from_features(data['features'])
                gdf.set_crs(epsg=28992, inplace=True)
                if not gdf.empty:
                    col = 'hectometrering' if 'hectometrering' in gdf.columns else 'hectomtrng'
                    if col in gdf.columns: gdf['hm_val'] = gdf[col].astype(float)
                    else: gdf['hm_val'] = 0.0
                return gdf
    except: pass
    return gpd.GeoDataFrame()

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
if 'manual_selection' not in st.session_state: st.session_state['manual_selection'] = set()
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
        st.session_state['manual_selection'] = set()
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
        st.session_state['manual_selection'] = set()
        st.session_state['zoom_bounds'] = None
        
    mode = st.radio("Modus:", ["🔍 Data Kwaliteit", "🏗️ Project Adviseur", "👆 Handmatige Selectie"], 
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
                for v in violations:
                    vid = v['id']
                    is_selected = (st.session_state['selected_error_id'] == vid)
                    
                    # GEBRUIK HIER border=True IPV ST.SUCCESS
                    if is_selected:
                        with st.container(border=True):
                            st.markdown("**:blue-background[GESELECTEERD]**")
                            c1, c2 = st.columns([3, 1])
                            with c1:
                                st.markdown(f"**{v['subthema']}**")
                                st.caption(f"{v['msg']}")
                            with c2:
                                if st.button("Toon", key=f"btn_err_{vid}"):
                                    st.session_state['selected_error_id'] = vid
                                    obj_geom = road_gdf.loc[vid].geometry
                                    st.session_state['zoom_bounds'] = obj_geom.bounds
                                    st.rerun()
                    else:
                        # Normale weergave
                        with st.container():
                            c1, c2 = st.columns([3, 1])
                            with c1:
                                st.markdown(f"**{v['subthema']}**")
                                st.caption(f"{v['msg']}")
                            with c2:
                                if st.button("Toon", key=f"btn_err_{vid}"):
                                    st.session_state['selected_error_id'] = vid
                                    obj_geom = road_gdf.loc[vid].geometry
                                    st.session_state['zoom_bounds'] = obj_geom.bounds
                                    st.rerun()
                            st.divider()

            # BEWERK PANEEL
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
                        curr_val = row.get(col, '')
                        if pd.isna(curr_val): curr_val = ""
                        inputs[col] = st.text_input(f"Vul in: {col}", value=str(curr_val), key=f"fix_{col}_{err_id}")
                    
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

            # BEWERK PANEEL
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

    # --- MODUS 3: HANDMATIG ---
    elif mode == "👆 Handmatige Selectie":
        st.info("Klik op objecten in de kaart om ze aan/uit te vinken.")
        
        selection = list(st.session_state['manual_selection'])
        
        if selection:
            st.write(f"**{len(selection)} objecten geselecteerd**")
            
            sel_df = road_gdf.loc[selection][['subthema', 'Onderhoudsproject']]
            st.dataframe(sel_df, height=150)
            
            if st.button("🧹 Wis Selectie"):
                st.session_state['manual_selection'] = set()
                st.rerun()
            
            st.divider()
            manual_name = st.text_input("Projectnaam voor selectie", key="man_proj_input")
            
            if st.button("💾 Opslaan op Selectie"):
                if manual_name.strip():
                    for oid in selection:
                        if oid in raw_gdf.index:
                            old_v = raw_gdf.at[oid, 'Onderhoudsproject']
                            raw_gdf.at[oid, 'Onderhoudsproject'] = manual_name
                            raw_gdf.at[oid, 'Advies_Bron'] = "Handmatige Selectie"
                            log_change(oid, 'Onderhoudsproject', old_v, manual_name)
                    
                    st.success(f"{len(selection)} objecten bijgewerkt!")
                    st.session_state['manual_selection'] = set()
                    st.rerun()
        else:
            st.warning("Selecteer objecten op de kaart.")

# --- LINKS: KAART ---
with col_map:
    st.subheader(f"Kaart: {selected_road}")
    
    road_web = road_gdf.to_crs(epsg=4326)
    
    # Smart Zoom
    if st.session_state['zoom_bounds']:
        minx, miny, maxx, maxy = st.session_state['zoom_bounds']
        # Converteer bounds naar WGS84 voor folium
        b_poly = wkt.loads(f"POLYGON(({minx} {miny}, {minx} {maxy}, {maxx} {maxy}, {maxx} {miny}, {minx} {miny}))")
        b_gseries = gpd.GeoSeries([b_poly], crs="EPSG:28992").to_crs(epsg=4326)
        b = b_gseries.total_bounds
        # Fit bounds [[lat1, lon1], [lat2, lon2]]
        fit_b = [[b[1], b[0]], [b[3], b[2]]]
        
        m = folium.Map(location=[(b[1]+b[3])/2, (b[0]+b[2])/2], zoom_start=16, tiles="CartoDB positron")
        m.fit_bounds(fit_b)
    else:
        c = road_web.unary_union.centroid
        m = folium.Map(location=[c.y, c.x], zoom_start=14, tiles="CartoDB positron")

    def style_fn(feature):
        oid = feature['properties']['sys_id']
        
        # 1. Handmatige selectie
        if oid in st.session_state['manual_selection']:
            return {'fillColor': 'magenta', 'color': 'black', 'weight': 2, 'fillOpacity': 0.8}
        
        # 2. Error selectie
        if oid == st.session_state['selected_error_id']:
             return {'fillColor': 'red', 'color': 'black', 'weight': 3, 'fillOpacity': 0.8}
             
        # 3. Groep selectie
        if st.session_state['selected_group_id']:
            active_grp = st.session_state['computed_groups'][st.session_state['selected_group_id']]
            if oid in active_grp['ids']:
                return {'fillColor': 'cyan', 'color': 'black', 'weight': 2, 'fillOpacity': 0.8}

        # 4. Heeft al project?
        proj = feature['properties'].get('Onderhoudsproject')
        if proj:
            return {'fillColor': '#00cc00', 'color': 'gray', 'weight': 0.5, 'fillOpacity': 0.4}
            
        return {'fillColor': 'gray', 'color': 'black', 'weight': 0.5, 'fillOpacity': 0.2}

    meta_cols = [c for c in ALL_META_COLS if c in road_web.columns]
    cols_to_select = ['geometry', 'sys_id'] + meta_cols
    
    folium.GeoJson(
        road_web[cols_to_select],
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=['subthema', 'Onderhoudsproject'], style="font-size: 11px;")
    ).add_to(m)

    pdok_hm = get_pdok_hectopunten_visual_only(road_gdf.total_bounds)
    if not pdok_hm.empty:
        pdok_web = pdok_hm.to_crs(epsg=4326)
        for _, row in pdok_web.iterrows():
            if row.geometry:
                g = row.geometry.centroid
                val = float(row.get('hm_val', 0))/10
                icon_html = f"""<div style="font-size: 10pt; font-weight: bold; color: black; text-shadow: 1px 1px 0 #fff;">{val:.1f}</div>"""
                folium.Marker([g.y, g.x], icon=folium.DivIcon(icon_size=(40,20), icon_anchor=(10,10), html=icon_html)).add_to(m)
                folium.CircleMarker([g.y, g.x], radius=2, color='red', fill=True).add_to(m)

    # LET OP: We gebruiken st_folium om clicks terug te krijgen
    output = st_folium(m, width=None, height=600, returned_objects=["last_object_clicked"])
    
    # Click Handler voor Handmatige Selectie
    if mode == "👆 Handmatige Selectie" and output and output.get("last_object_clicked"):
        clicked_props = output["last_object_clicked"].get("properties")
        if clicked_props and 'sys_id' in clicked_props:
            click_id = clicked_props['sys_id']
            
            # Toggle de selectie
            if click_id in st.session_state['manual_selection']:
                st.session_state['manual_selection'].remove(click_id)
            else:
                st.session_state['manual_selection'].add(click_id)
            
            st.rerun()

# --- CHANGE LOG & EXPORT ---
st.divider()
st.subheader("📝 Logboek Wijzigingen & Export")

if st.session_state['change_log']:
    st.dataframe(pd.DataFrame(st.session_state['change_log']), use_container_width=True)
else:
    st.caption("Nog geen wijzigingen aangebracht.")

csv = st.session_state['data_complete'].drop(columns=['geometry', 'Rank', 'subthema_clean', 'reg_jaar', 'sys_id'], errors='ignore').to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Definitieve CSV", csv, "iASSET_Smart_Export.csv", "text/csv")