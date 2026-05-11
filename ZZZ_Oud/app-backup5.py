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
SEGMENTATION_ATTRIBUTES = ['verhardingssoort', 'Soort deklaag specifiek', 'Jaar aanleg', 'Jaar deklaag', 'Besteknummer']

ALL_META_COLS = [
    'subthema', 'Situering', 'verhardingssoort', 'Soort deklaag specifiek', 
    'Jaar aanleg', 'Jaar deklaag', 'Onderhoudsproject', 
    'Advies_Onderhoudsproject', 'validation_error', 'Spoor_ID', 
    'Is_Project_Grens', 'Advies_Bron', 'Wegnummer', 'Besteknummer'
]

# --- FUNCTIES: DATA & NETWERK ---

@st.cache_data
def load_data():
    df1 = pd.read_csv(FILE_NIET_RIJSTROOK, low_memory=False)
    df2 = pd.read_csv(FILE_WEL_RIJSTROOK, low_memory=False)
    df = pd.concat([df1, df2], ignore_index=True)
    
    # Behoud origineel ID als bron_id indien aanwezig
    if 'id' in df.columns: 
        df.rename(columns={'id': 'bron_id'}, inplace=True)
    
    # We maken sys_id aan als string om type-verwarring te voorkomen
    # Dit fungeert als de unieke sleutel voor de applicatie logica
    df['sys_id'] = range(len(df))
    df['sys_id'] = df['sys_id'].astype(str)
    
    def parse_geom(x):
        try: return wkt.loads(x)
        except: return None
    
    df['geometry'] = df['rds coordinaten'].apply(parse_geom)
    df = df.dropna(subset=['geometry'])
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    
    # --- INTELLIGENTE CRS DETECTIE ---
    if not gdf.empty:
        first_geom = gdf['geometry'].iloc[0]
        # Als x < 180 is het waarschijnlijk GPS (WGS84)
        if first_geom and first_geom.centroid.x < 180:
            gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
            # Converteer naar RD (Meters) voor berekeningen
            gdf = gdf.to_crs(epsg=28992)
        else:
            # Anders nemen we aan dat het al RD is
            gdf.set_crs(epsg=28992, inplace=True, allow_override=True)
    else:
        gdf.set_crs(epsg=28992, inplace=True, allow_override=True)
    
    # Index instellen op sys_id, maar behoud kolom ook voor tooltip/properties
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
            for attr in SEGMENTATION_ATTRIBUTES:
                val = str(first_node.get(attr, '')).strip()
                if val and val.lower() != 'nan':
                    specs.append(f"{val}")
            
            reason_txt = ", ".join(specs) if specs else "Geen specifieke kenmerken"

            groups[group_id] = {
                'ids': node_list,
                'subthema': subthema_key,
                'category': 'Ruggengraat',
                'reason': f"Gegroepeerd op: {reason_txt}",
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

def get_pdok_hectopunten_visual_only(bounds):
    wfs_url = "https://service.pdok.nl/rws/nwbwegen/wfs/v1_0"
    buffer = 500
    try:
        minx, miny, maxx, maxy = bounds
        bbox_str = f"{minx-buffer},{miny-buffer},{maxx+buffer},{maxy+buffer}"
        params = {
            "service": "WFS", "version": "1.0.0", "request": "GetFeature", 
            "typeName": "hectopunten", "outputFormat": "json", 
            "bbox": bbox_str, "maxFeatures": 5000
        }
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
    # Verzeker dat sys_id bestaat
    if 'sys_id' not in st.session_state['data_complete'].columns:
        st.cache_data.clear()
        st.session_state['data_complete'] = load_data()

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

if st.sidebar.button("⚠️ Cache Legen / Reset"):
    st.cache_data.clear()
    st.session_state.clear()
    st.rerun()

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
        # Logboek niet wissen bij wissel weg
        st.session_state['last_click_ts'] = 0

G_road = st.session_state['graph_current']

# --- LAYOUT ---
col_map, col_inspector = st.columns([3, 2])

# --- RECHTS: INSPECTOR ---
with col_inspector:
    
    def on_mode_change():
        st.session_state['selected_error_id'] = None
        st.session_state['selected_group_id'] = None
        st.session_state['zoom_bounds'] = None
        
    mode = st.radio("Modus:", ["🔍 Data Kwaliteit", "🏗️ Project Adviseur", "👆 Handmatige Selectie"], 
                    horizontal=True, on_change=on_mode_change)
    st.divider()

    # --- MODUS 1: KWALITEIT ---
    if mode == "🔍 Data Kwaliteit":
        st.subheader("Data Validatie")
        violations = check_rules(road_gdf)
        if not violations:
            st.success("Schoon! Geen datakwaliteit issues.")
        else:
            st.write(f"**{len(violations)} issues gevonden**")
            with st.container(height=500):
                for v in violations:
                    vid = v['id']
                    if st.button(f"{v['subthema']} - {v['msg']}", key=f"err_{vid}", use_container_width=True):
                        st.session_state['selected_error_id'] = vid
                        obj_geom = road_gdf.loc[vid].geometry
                        st.session_state['zoom_bounds'] = obj_geom.bounds
                        st.rerun()

            if st.session_state['selected_error_id']:
                err_id = st.session_state['selected_error_id']
                if err_id in road_gdf.index:
                    st.divider()
                    st.markdown(f"#### Corrigeer ID: {err_id}")
                    row = road_gdf.loc[err_id]
                    viol_info = next((v for v in violations if v['id'] == err_id), None)
                    cols_to_fix = viol_info['missing_cols'] if viol_info else ['Onderhoudsproject']
                    inputs = {}
                    for col in cols_to_fix:
                        curr_val = row.get(col, '')
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
        st.subheader("AI Adviezen")
        if 'computed_groups' not in st.session_state or st.session_state['computed_groups'] is None:
            with st.spinner("AI berekent groepen..."):
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
            if st.button("Reset Adviezen"):
                st.session_state['computed_groups'] = None
                st.session_state['processed_groups'] = set()
                st.rerun()
        else:
            st.write(f"**{len(active_groups)} suggesties**")
            with st.container(height=500):
                for g_id in sorted(active_groups.keys()):
                    g_data = active_groups[g_id]
                    count = len(g_data['ids'])
                    icon = "🛣️" if "RIJBAAN" in g_id else "🚲" if "FIETSPAD" in g_id else "🛤️" if "PARALLEL" in g_id else "🌳"
                    
                    with st.expander(f"{icon} {g_data['subthema'].title()} ({count} obj)"):
                        st.caption(g_data['reason'])
                        c1, c2, c3 = st.columns(3)
                        if c1.button("Toon", key=f"v_{g_id}"):
                            st.session_state['selected_group_id'] = g_id
                            grp_geom = road_gdf.loc[g_data['ids']].unary_union
                            st.session_state['zoom_bounds'] = grp_geom.bounds
                            st.rerun()
                        if c2.button("Bewerk", key=f"e_{g_id}"):
                            st.session_state['selected_group_id'] = g_id
                            st.rerun()
                        if c3.button("Negeer", key=f"i_{g_id}"):
                            st.session_state['ignored_groups'].add(g_id)
                            st.rerun()

            if st.session_state['selected_group_id'] and st.session_state['selected_group_id'] in active_groups:
                sel_gid = st.session_state['selected_group_id']
                st.divider()
                st.info(f"Geselecteerd: {sel_gid}")
                name_input = st.text_input("Projectnaam:", key="proj_name_input")
                if st.button("Toepassen"):
                    for oid in active_groups[sel_gid]['ids']:
                        if oid in raw_gdf.index:
                            old_v = raw_gdf.at[oid, 'Onderhoudsproject']
                            raw_gdf.at[oid, 'Onderhoudsproject'] = name_input
                            log_change(oid, 'Onderhoudsproject', old_v, name_input)
                    st.session_state['processed_groups'].add(sel_gid)
                    st.session_state['selected_group_id'] = None
                    st.rerun()

    # --- MODUS 3: HANDMATIG ---
    elif mode == "👆 Handmatige Selectie":
        selection = st.session_state['manual_selection']
        st.subheader(f"Selectie: {len(selection)} objecten")
        
        if selection:
            sel_df = road_gdf[road_gdf['sys_id'].isin(selection)]
            st.dataframe(sel_df[['subthema', 'Onderhoudsproject', 'naam']], height=200, use_container_width=True)
            
            col_act1, col_act2 = st.columns(2)
            if col_act1.button("🗑️ Wis Selectie"):
                st.session_state['manual_selection'] = set()
                st.rerun()
            
            st.markdown("#### Massa Update")
            manual_name = st.text_input("Nieuw Project:", key="man_proj_input")
            if st.button("Update Selectie"):
                for oid in selection:
                    if oid in raw_gdf.index:
                        old_v = raw_gdf.at[oid, 'Onderhoudsproject']
                        raw_gdf.at[oid, 'Onderhoudsproject'] = manual_name
                        log_change(oid, 'Onderhoudsproject', old_v, manual_name)
                st.success("Bijgewerkt!")
                st.rerun()
        else:
            st.info("Klik op de kaart om te selecteren.")

# --- LINKS: KAART (ROBUUSTE VERSIE) ---
with col_map:
    st.subheader(f"Kaart: {selected_road}")
    
    # 1. GeoDataFrames voorbereiden
    road_web = road_gdf.to_crs(epsg=4326)
    
    # Check bounds
    if st.session_state['zoom_bounds']:
        minx, miny, maxx, maxy = st.session_state['zoom_bounds']
        c_x, c_y = (minx+maxx)/2, (miny+maxy)/2
        zoom = 18
    else:
        if not road_web.empty:
            b = road_web.total_bounds
            c_x, c_y = (b[0]+b[2])/2, (b[1]+b[3])/2
            zoom = 14
        else:
            c_x, c_y, zoom = 5.1, 52.1, 8

    m = folium.Map(location=[c_y, c_x], zoom_start=zoom, tiles="CartoDB positron")

    # 2. BASIS LAAG (Alle objecten)
    # Standaard styling (grijs of groen als project al bestaat)
    def base_style(feature):
        proj = feature['properties'].get('Onderhoudsproject', '')
        if proj:
            return {'fillColor': '#00cc00', 'color': 'gray', 'weight': 0.5, 'fillOpacity': 0.4}
        return {'fillColor': 'gray', 'color': 'black', 'weight': 0.5, 'fillOpacity': 0.2}
    
    # Zorg dat sys_id in properties zit
    cols_base = ['geometry', 'sys_id', 'subthema', 'Onderhoudsproject']
    road_web_json = road_web[cols_base].reset_index(drop=True)
    
    folium.GeoJson(
        road_web_json,
        style_function=base_style,
        tooltip=folium.GeoJsonTooltip(fields=['subthema', 'Onderhoudsproject', 'sys_id']),
        name="Alle Wegen"
    ).add_to(m)
    
    # 3. SELECTIE HIGHLIGHT LAAG (Rood)
    # In plaats van style_function dynamisch te maken, voegen we fysiek een laag toe
    # bovenop de andere voor de geselecteerde items.
    selected_ids = st.session_state['manual_selection']
    if selected_ids:
        # Filter alleen de geselecteerde rijen
        selected_gdf = road_web[road_web['sys_id'].isin(selected_ids)]
        if not selected_gdf.empty:
            selected_json = selected_gdf[['geometry', 'sys_id', 'subthema']].reset_index(drop=True)
            folium.GeoJson(
                selected_json,
                style_function=lambda x: {'fillColor': '#ff0000', 'color': 'black', 'weight': 2, 'fillOpacity': 0.7},
                interactive=True, # Moet interactief zijn om ook te kunnen deselecteren
                name="Selectie"
            ).add_to(m)
            
    # 4. ERROR / GROEP HIGHLIGHTS
    if st.session_state['selected_error_id']:
        eid = st.session_state['selected_error_id']
        e_gdf = road_web[road_web['sys_id'] == str(eid)]
        if not e_gdf.empty:
            folium.GeoJson(e_gdf, style_function=lambda x: {'color': 'red', 'weight': 3, 'fillOpacity':0}).add_to(m)

    if st.session_state['selected_group_id']:
        gid = st.session_state['selected_group_id']
        g_ids = st.session_state['computed_groups'][gid]['ids']
        g_gdf = road_web[road_web['sys_id'].isin([str(i) for i in g_ids])]
        if not g_gdf.empty:
             folium.GeoJson(
                 g_gdf, 
                 style_function=lambda x: {'fillColor': 'cyan', 'color': 'blue', 'weight': 2, 'fillOpacity': 0.6}
             ).add_to(m)

    # 5. RENDER EN CLICK HANDLING
    output = st_folium(m, width=None, height=600, returned_objects=["last_object_clicked"])

    # 6. LOGICA
    if output and output.get("last_object_clicked"):
        clicked = output["last_object_clicked"]
        props = clicked.get("properties", {})
        
        # Probeer ID te vinden
        clicked_id = props.get("sys_id")
        
        if clicked_id:
            clicked_id = str(clicked_id)
            
            # Voorkom oneindige loop door te checken of we deze click al verwerkt hebben
            # We genereren een pseudo-hash van de click
            click_hash = f"{clicked_id}_{datetime.now().timestamp()}"
            
            # Omdat streamlit herlaadt, is een eenvoudige state check vaak genoeg
            # Maar we moeten wel weten of de gebruiker NOG EENS op hetzelfde klikte (deselecteren)
            # of dat dit de oude event is. 
            # st_folium geeft last_object_clicked terug, die blijft staan na rerun.
            # We moeten de 'last_object_clicked' alleen verwerken als hij anders is dan de vorige verwerking?
            # Nee, want je kunt toggle doen.
            
            # TRUC: We kijken of de selectie logica moet omdraaien.
            # Echter, omdat st_folium de oude output terugstuurt bij een rerun die NIET door de kaart
            # getriggerd werd (bv. knop in sidebar), moeten we oppassen.
            # Voor nu gaan we ervan uit dat een user interactie de rerun triggert.
            
            # Om "stale clicks" te voorkomen bij knopdrukken elders:
            # Helaas is dat lastig in Streamlit zonder custom JS.
            # We accepteren de klik, en updaten de set.
            
            current_selection = st.session_state['manual_selection']
            
            # We moeten checken of we deze ID net hebben verwerkt.
            # Dit doen we door te kijken naar de interne state. 
            # Omdat we 'st.rerun' doen direct na de logica, is de output bij de volgende run
            # nog steeds DEZELFDE 'last_object_clicked'. Dit zou leiden tot infinite toggle loop.
            # OPLOSSING: We negeren de click als 'last_object_clicked' identiek is aan wat we in session_state hebben opgeslagen.
            
            prev_click = st.session_state.get('prev_click_obj')
            
            # Vergelijk geometry bounds of coordinates om te zien of het ECHT een nieuwe click is
            # Of simpeler: als ID hetzelfde is, negeren we hem tenzij... tja.
            # Workaround: We kijken of de visuele staat overeenkomt met de data state.
            
            # Beter: We gebruiken de ID en togglen hem.
            # Om de infinite loop te breken:
            # 1. User klikt -> output heeft ID -> Wij togglen -> Rerun.
            # 2. Script runt -> output heeft NOG STEEDS ID -> Wij togglen WEER -> Rerun.
            # Dit is het probleem.
            
            # Oplossing: check if 'last_object_clicked' changed.
            if output["last_object_clicked"] != st.session_state.get("last_processed_click"):
                if clicked_id in current_selection:
                    current_selection.remove(clicked_id)
                    st.toast(f"Gedeselecteerd: {clicked_id}")
                else:
                    current_selection.add(clicked_id)
                    st.toast(f"Geselecteerd: {clicked_id}")
                
                st.session_state['manual_selection'] = current_selection
                st.session_state["last_processed_click"] = output["last_object_clicked"]
                st.rerun()

st.divider()
if st.session_state['change_log']:
    st.markdown("### 📝 Logboek")
    st.dataframe(pd.DataFrame(st.session_state['change_log']))