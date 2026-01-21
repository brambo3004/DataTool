import streamlit as st          # De 'bouwdoos' om de website te maken (knoppen, tekst, layout).
import pandas as pd             # 'Excel op steroïden'. Hiermee rekenen we met tabellen.
import geopandas as gpd         # Hetzelfde als pandas, maar dan voor kaarten (weet wat coordinaten zijn).
from shapely import wkt         # Een vertaler die tekst ("POINT(5 5)") omzet naar een echte vorm voor de computer.
import folium                   # De software om de interactieve kaart (zoals Google Maps) te tekenen.
from streamlit_folium import st_folium # Het bruggetje om die Folium-kaart in de Streamlit-app te tonen.
import requests                 # Hiermee kan de app praten met het internet (bijv. PDOK server).
import networkx as nx           # De wiskundige bibliotheek om netwerken (verbindingen tussen wegen) te snappen.
from datetime import datetime   # Om te weten hoe laat het nu is (voor het logboek).
import numpy as np              # Voor zware wiskundige berekeningen (wordt hier minimaal gebruikt).
import io                       # Om bestanden in het geheugen klaar te zetten voor download.
import os                       # Om met het besturingssysteem te praten (bijv. bestand verwijderen).

# --- CONFIGURATIE (DE SPELREGELS) ---
# Hier stellen we de basis in. Als je iets wilt veranderen aan hoe de app werkt, doe je dat vaak hier.

# Vertel de browser dat we de hele breedte van het scherm willen gebruiken.
st.set_page_config(layout="wide", page_title="iASSET Tool - Smart Advisor")

# De bestandsnamen van de data die we gaan inladen.
FILE_NIET_RIJSTROOK = "N-allemaal-niet-rijstrook.csv"
FILE_WEL_RIJSTROOK = "N-allemaal-alleen-rijstrook.csv"
AUTOSAVE_FILE = "autosave_log.csv" # Hierin slaan we tijdelijk wijzigingen op voor als de app crasht.

# AANGEPAST: De rangorde van wegen.
# Dit vertelt de computer wat 'belangrijker' is. Een rijstrook (1) is de baas over een fietspad (3).
# Dit gebruiken we later om te bepalen wie bij wie hoort (clustering).
HIERARCHY_RANK = {
    'rijstrook': 1, 
    'parallelweg': 2, 
    'landbouwpad': 2, 
    'busbaan': 2, 
    'fietspad': 3
}

# --- CONFIGURATIE WEGRICHTINGEN ---
# NTS = North to South (Noord -> Zuid) | Startpunt is hoogste Y (Noord)
# STN = South to North (Zuid -> Noord) | Startpunt is laagste Y (Zuid)
# WTE = West to East   (West -> Oost)  | Startpunt is laagste X (West)
# ETW = East to West   (Oost -> West)  | Startpunt is hoogste X (Oost)

# Alleen gekeken naar het eerste stukje van de weg
ROAD_DIRECTIONS = {
    'N351': 'ETW', # Oost naar West
    'N353': 'STN', # Zuid naar Noord
    'N354': 'ETW', # Oost naar West
    'N355': 'WTE', # West naar Oost
    'N356': 'NTS',
    'N357': 'STN',
    'N358': 'WTE',
    'N359': 'ETW',
    'N361': 'ETW',
    'N369': 'STN',
    'N380': 'WTE',
    'N381': 'NTS',
    'N383': 'STN',
    'N384': 'STN',
    'N390': 'ETW',
    'N392': 'WTE',
    'N393': 'ETW',
    'N398': 'ETW',
    'N910': 'WTE',
    'N913': 'ETW',
    'N917': 'WTE',
    'N918': 'STN',
    'N919': 'ETW',
    'N924': 'ETW',
    'N927': 'ETW',
    'N928': 'NTS'
}

# --- CONFIGURATIE AANPASSING ---

# De "Ruggengraat". Dit zijn de hoofdwegen waaraan we andere dingen (zoals bermen) koppelen.
# Als we eilandjes zoeken, kijken we of dingen hieraan vast zitten.
BACKBONE_TYPES = ['rijstrook', 'parallelweg', 'landbouwpad', 'busbaan', 'fietspad']

# DIT IS DE NIEUWE LOGICA:
# Alles moet verplicht een projectnaam hebben, BEHALVE deze lijst.
# Dit zijn 'losse' dingen die op zichzelf mogen bestaan zonder project.
SUBTHEMA_EXCEPTIONS = [
    'fietsstalling', 
    'geleideconstructie',
    'parkeerplaats', 
    'rotonderand', 
    'verkeerseiland of middengeleider'
]

# Welke kolommen zijn verplicht aanwezig bij een wijziging?
MUTATION_REQUIRED_COLS = ['subthema', 'naam', 'Gebruikersfunctie', 'Type onderdeel', 'verhardingssoort', 'Onderhoudsproject']

# Welke eigenschappen bepalen of we een weg in stukjes hakken?
# Als de 'verhardingssoort' verandert (bijv. van asfalt naar klinkers), begint er een nieuw stukje weg.
SEGMENTATION_ATTRIBUTES = ['verhardingssoort', 'Soort deklaag specifiek', 'Jaar aanleg', 'Jaar deklaag', 'Besteknummer']

# Dit is een woordenboekje om ingewikkelde kolomnamen (links) te vertalen naar leesbare tekst (rechts) voor de gebruiker.
FRIENDLY_LABELS = {
    'verhardingssoort': 'Verhardingssoort',
    'Soort deklaag specifiek': 'Deklaag',
    'Jaar aanleg': 'Aanleg',
    'Jaar deklaag': 'Deklaagjaar',
    'Besteknummer': 'Bestek',
    'Onderhoudsproject': 'Huidig Project'
}

# Een lijst van alle kolommen die we willen bewaren of tonen in de 'pop-up' op de kaart.
ALL_META_COLS = [
    'subthema', 'Situering', 'verhardingssoort', 'Soort deklaag specifiek', 
    'Jaar aanleg', 'Jaar deklaag', 'Onderhoudsproject', 
    'Advies_Onderhoudsproject', 'validation_error', 'Spoor_ID', 
    'Is_Project_Grens', 'Advies_Bron', 'Wegnummer', 'Besteknummer',
    'tijdstipRegistratie', 'nummer', 'gps coordinaten', 'rds coordinaten', 'Metrering'
]

# --- HULPFUNCTIES (DE KLUSJESMANNEN) ---

def clean_display_value(val):
    """
    Deze functie poetst getallen op.
    Probleem: Computers maken van 2005 soms "2005.0". Dat staat lelijk.
    Oplossing: Als het op .0 eindigt en er staan geen letters in, knip de .0 eraf.
    """
    # Als er niets ingevuld is (nan), geef dan een lege tekst terug.
    if pd.isna(val) or val == '' or str(val).lower() == 'nan':
        return ""
    
    s = str(val).strip() # Haal spaties weg
    
    if s.endswith(".0"):
        # Check: Zitten er letters in? (Bijv. "N351...35.0") -> Laat staan.
        if any(c.isalpha() for c in s):
            return s
        
        # Geen letters? Dan is het een jaartal of getal -> Haal .0 weg.
        return s[:-2]
        
    return s

def save_autosave():
    """
    Slaat de lijst met wijzigingen op in een bestandje.
    Zodat als je browser crasht, je werk niet meteen weg is.
    """
    if 'change_log' in st.session_state and st.session_state['change_log']:
        df_log = pd.DataFrame(st.session_state['change_log'])
        df_log.to_csv(AUTOSAVE_FILE, index=False, sep=';')
    else:
        # Als er geen wijzigingen zijn, ruim dan het oude bestand op.
        if os.path.exists(AUTOSAVE_FILE):
            os.remove(AUTOSAVE_FILE)

def apply_change_to_data(oid, field, new_val):
    """
    De functie die daadwerkelijk een waarde aanpast in de grote database in het geheugen.
    oid = Object ID (welke regel?)
    field = Welke kolom?
    new_val = Wat moet er komen te staan?
    """
    raw_gdf = st.session_state['data_complete']
    if oid in raw_gdf.index:
        raw_gdf.at[oid, field] = new_val

# --- FUNCTIES: DATA & NETWERK (DE MOTOR) ---

@st.cache_data # Dit zegt tegen Streamlit: "Onthoud de uitkomst van deze functie, dat scheelt laadtijd."
def load_data():
    """
    Dit is de zwaarste functie. Hij leest de CSV bestanden,
    repareert de coordinaten en maakt er een kaart-bestand (GeoDataFrame) van.
    """
    # Stap 1: Lees de twee CSV bestanden in
    df1 = pd.read_csv(FILE_NIET_RIJSTROOK, low_memory=False)
    df2 = pd.read_csv(FILE_WEL_RIJSTROOK, low_memory=False)
    # Plak ze onder elkaar tot één grote lijst
    df = pd.concat([df1, df2], ignore_index=True)
    
    # Hernoem 'id' naar 'bron_id' om verwarring te voorkomen
    if 'id' in df.columns: 
        df.rename(columns={'id': 'bron_id'}, inplace=True)
    
    # Geef elke regel een eigen uniek systeem-nummer (sys_id)
    df['sys_id'] = range(len(df))
    
    # Hulpfunctie om tekst-coordinaten te lezen
    def parse_geom(x):
        try: return wkt.loads(x)
        except: return None
    
    # --- BELANGRIJK: COORDINATEN FIX ---
    # We lezen de GPS coordinaten (Dat zijn graden, zoals 52.123, 4.567)
    df['geometry'] = df['gps coordinaten'].apply(parse_geom)
    
    # Gooi rijen weg die geen locatie hebben (daar kunnen we niks mee)
    df = df.dropna(subset=['geometry'])
    
    # Maak er een officieel kaart-bestand van
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    
    # Vertel de computer: "Dit zijn GPS coordinaten (EPSG:4326)"
    gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
    
    # Transformeer ze nu naar het Nederlandse RD-stelsel (Meters, EPSG:28992).
    # Waarom? Omdat GPS in graden is, en we straks willen rekenen met meters (bijv. "ligt dit binnen 0.5 meter?").
    gdf.to_crs(epsg=28992, inplace=True)
    
    # Zet de sys_id als de officiële index (zoek-sleutel)
    gdf.set_index('sys_id', drop=False, inplace=True)
    gdf.index.name = None
            
    # Zorg dat alle kolommen die we verwachten ook echt bestaan, anders vullen we ze met leegte
    for col in ALL_META_COLS:
        if col not in gdf.columns: gdf[col] = ''
    
    # Maak de tekst in 'Situering' netjes (eerste letter hoofdletter)
    if 'Situering' in gdf.columns:
        gdf['Situering'] = gdf['Situering'].astype(str).str.strip().str.title().replace('Nan', 'Onbekend')
    else:
        gdf['Situering'] = 'Onbekend'
        
    # Maak een schone versie van het subthema (alles kleine letters, geen spaties) voor computer-logica
    gdf['subthema_clean'] = gdf['subthema'].astype(str).str.lower().str.strip()
    
    # Ken een 'Rang' toe op basis van de lijst die we bovenin maakten (Rijstrook = 1, etc.)
    gdf['Rank'] = gdf['subthema_clean'].apply(lambda x: HIERARCHY_RANK.get(x, 4))
    
    # Probeer de 'hectometrering' (paaltjes langs de weg) om te zetten naar getallen om te sorteren
    if 'Metrering' in gdf.columns:
        # Vervang komma door punt en maak er een getal van
        gdf['hm_sort'] = pd.to_numeric(gdf['Metrering'].astype(str).str.replace(',', '.'), errors='coerce')
        # Als het niet lukt, geef het een heel hoog nummer (achteraan de lijst)
        gdf['hm_sort'] = gdf['hm_sort'].fillna(99999.9)
    else:
        gdf['hm_sort'] = 99999.9

    # Hulpfunctie om jaartallen uit tekst te peuteren
    def parse_date_info(x):
        # ... (technische code om datums te lezen, bijv "01-01-2020" wordt 2020) ...
        # (ik kort dit iets in voor leesbaarheid, hij probeert gewoon een jaartal te vinden)
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
    
    # Pas de datum-functie toe
    if 'tijdstipRegistratie' in gdf.columns:
        parsed = gdf['tijdstipRegistratie'].apply(parse_date_info)
        gdf['reg_jaar'] = [p[0] for p in parsed]
        gdf['reg_maand'] = [p[1] for p in parsed]
    else:
        gdf['reg_jaar'] = 0
        gdf['reg_maand'] = 0

    # Poets de jaartallen van aanleg en deklaag nog even op (geen .0)
    for col in ['Jaar aanleg', 'Jaar deklaag']:
        if col in gdf.columns:
            gdf[col] = gdf[col].apply(clean_display_value)

    return gdf # Geef de kant-en-klare tabel terug aan de app

def build_graph_from_geometry(gdf):
    # Eerst maken we een kopie van de kaart, zodat we de originelen niet per ongeluk aanpassen.
    gdf_buffer = gdf.copy()
    
    # TRUCJE: We maken elk lijntje of vlakje 0.5 meter dikker ('buffer').
    # Waarom? In GIS-data raken lijnen elkaar soms NET niet (bijv. 1 cm ertussen).
    # Door ze "dikker" te maken, overlappen ze zeker weten wél als ze naast elkaar liggen.
    gdf_buffer['geometry'] = gdf_buffer.geometry.buffer(0.5)
    
    # We pakken alleen de kolommen die we nodig hebben voor de berekening.
    cols = ['geometry', 'subthema_clean', 'Rank', 'sys_id']
    
    left_df = gdf_buffer[cols].copy()
    left_df.index.name = None # Index opschonen
    right_df = gdf[cols].copy()
    right_df.index.name = None
    
    # DIT IS DE MAGIE: 'Spatial Join' (sjoin).
    # De computer kijkt: "Welke dikke lijnen (left) overlappen met de originele lijnen (right)?"
    # Het resultaat is een lijst van alle buren: "ID 1 raakt ID 2", "ID 1 raakt ID 5", etc.
    joined = gpd.sjoin(
        left_df, right_df,
        how='inner', predicate='intersects', lsuffix='left', rsuffix='right'
    )
    
    # We verwijderen regels waar ID links gelijk is aan ID rechts.
    # Want een weg raakt natuurlijk zichzelf, maar dat hoeven we niet te weten.
    joined = joined[joined['sys_id_left'] != joined['sys_id_right']]
    
    # Nu starten we de wiskundige "Graaf" (het netwerk).
    G = nx.Graph()
    # Elk stukje weg uit onze data wordt een 'knooppunt' (node) in dit netwerk.
    G.add_nodes_from(gdf.index)
    
    edges = []
    # Nu lopen we door alle gevonden verbindingen heen om de draadjes te spannen.
    for _, row in joined.iterrows():
        idx_left = row['sys_id_left']
        idx_right = row['sys_id_right']
        sub_left = row['subthema_clean_left']
        sub_right = row['subthema_clean_right']
        
        # We proberen te snappen HOE ze vastzitten:
        # 'lateral' (zijwaarts): Bijv. een Berm naast een Rijbaan (verschillende types).
        rel_type = 'lateral'
        
        # 'longitudinal' (in lengterichting): Bijv. Rijbaan A zit vast aan Rijbaan B (dezelfde types).
        # Dit betekent vaak dat de weg hier doorloopt.
        if sub_left == sub_right: rel_type = 'longitudinal'
        
        # Voeg de verbinding toe aan de lijst.
        edges.append((idx_left, idx_right, {'type': rel_type}))
        
    # Voeg alle draadjes (edges) in één keer toe aan het netwerk.
    G.add_edges_from(edges)
    
    # Geef het complete spinnenweb terug.
    return G

# --- FUNCTIES: ANALYSE ---

def check_rules(gdf, G=None):
    violations = [] # Hier gaan we alle gevonden fouten in verzamelen.
    
    # We maken de lijst met uitzonderingen (dingen die geen project hoeven) even klein.
    exceptions_clean = [x.lower() for x in SUBTHEMA_EXCEPTIONS]
    
    # Dit zijn de 'belangrijke' wegen.
    all_backbones = ['rijstrook', 'parallelweg', 'landbouwpad', 'busbaan', 'fietspad']
    
    # --- STAP 1: De Administratieve Controle ---
    # We zoeken rijen die:
    # 1. NIET in de uitzonderingenlijst staan (dus het is geen boom of bord).
    # 2. EN die GEEN 'Onderhoudsproject' ingevuld hebben.
    mask_missing = (
        ~gdf['subthema_clean'].isin(exceptions_clean) & 
        (gdf['Onderhoudsproject'].isna() | (gdf['Onderhoudsproject'] == ''))
    )
    
    # Voor elk gevonden geval schrijven we een boete uit.
    for idx, row in gdf[mask_missing].iterrows():
        violations.append({
            'type': 'error', 'id': idx, 'subthema': row['subthema'],
            'msg': 'Mist verplicht onderhoudsproject',
            'missing_cols': ['Onderhoudsproject']
        })
    
    # --- STAP 2: De Ruimtelijke Controle (Topologie) ---
    # Dit kan alleen als het netwerk (G) succesvol is gebouwd.
    if G is not None:
        
        # A. Weeskinderen Check (Zwevende objecten)
        # We lopen elk punt in het netwerk af.
        for node_id in G.nodes:
            sub = gdf.loc[node_id, 'subthema_clean']
            
            # Als het een hoofdweg is (backbone) of een uitzondering, slaan we hem over.
            # Die mogen namelijk best 'alleen' liggen of vormen zelf de basis.
            if sub in all_backbones or sub in exceptions_clean:
                continue
                
            # Dit is dus een 'secundair object' (bijv. inrit, berm, goot).
            # We vragen aan het netwerk: "Wie zijn de buren?"
            neighbors = G.neighbors(node_id)
            connected_to_backbone = False
            
            for buur in neighbors:
                buur_sub = gdf.loc[buur, 'subthema_clean']
                # Als een van de buren een hoofdweg is, is het veilig.
                if buur_sub in all_backbones:
                    connected_to_backbone = True
                    break
            
            # Geen hoofdweg als buurman? Dan zweeft dit object in het niets. Fout!
            if not connected_to_backbone:
                violations.append({
                    'type': 'warning', 'id': node_id, 'subthema': gdf.loc[node_id, 'subthema'],
                    'msg': 'Zwevend secundair object: Grenst nergens aan een hoofdroute (Rijbaan/Fiets/etc).',
                    'missing_cols': []
                })

        # B. Integriteit Check (Horen buren bij hetzelfde project?)
        # We kijken alleen naar dingen die al wél een projectnaam hebben.
        mask_has_project = (gdf['Onderhoudsproject'].notna()) & (gdf['Onderhoudsproject'] != '')
        for idx, row in gdf[mask_has_project].iterrows():
            my_proj = str(row['Onderhoudsproject']).strip()
            my_sub = row['subthema_clean']
            
            neighbors = list(G.neighbors(idx))
            match_found = False
            
            # Check 1: Heb ik een directe buur met HETZELFDE project?
            for buur in neighbors:
                buur_proj = str(gdf.loc[buur, 'Onderhoudsproject']).strip()
                if buur_proj == my_proj:
                    match_found = True
                    break
            
            if not match_found:
                 # Niemand in de buurt heeft mijn projectnaam. Ik ben een eilandje.
                 violations.append({
                    'type': 'warning', 'id': idx, 'subthema': row['subthema'],
                    'msg': f"Geïsoleerd t.o.v. project '{my_proj}'. Geen directe buren met dit project.",
                    'missing_cols': ['Onderhoudsproject']
                })
            else:
                # Check 2 (Geavanceerd): Zit ik via-via wel vast aan een hoofdweg van dit project?
                # Als ik bijv. een berm ben, en mijn buurman is een inrit (ook geen hoofdweg)...
                # ...dan zweven we misschien samen los van de hoofdweg.
                if my_sub not in all_backbones:
                    connected_to_project_backbone = False
                    for buur in neighbors:
                        buur_sub = gdf.loc[buur, 'subthema_clean']
                        buur_proj = str(gdf.loc[buur, 'Onderhoudsproject']).strip()
                        # Buurman moet EN zelfde project hebben EN een hoofdweg zijn.
                        if buur_proj == my_proj and buur_sub in all_backbones:
                            connected_to_project_backbone = True
                            break
                    
                    if not connected_to_project_backbone:
                        # Wel buren met dit project, maar geen enkele is een hoofdweg.
                         violations.append({
                            'type': 'info', 'id': idx, 'subthema': row['subthema'],
                            'msg': f"Verbonden met '{my_proj}', maar raakt niet direct de hoofdrijbaan/fietspad van dit project.",
                            'missing_cols': []
                        })

    return violations

def generate_grouped_proposals(gdf, G):
    groups = {}
    node_to_group = {}
    
    # --- CONFIGURATIE VAN DE WATERVAL ---
    # We werken in lagen. Eerst de belangrijkste wegen (Rank 1), dan de rest.
    HIERARCHY_CONFIG = [
        {'rank': 1, 'types': ['rijstrook'], 'prefix': 'GRP_RIJBAAN'},
        {'rank': 2, 'types': ['parallelweg', 'landbouwpad', 'busbaan'], 'prefix': 'GRP_PARALLEL'},
        {'rank': 3, 'types': ['fietspad'], 'prefix': 'GRP_FIETSPAD'}
    ]
    
    processed_ids = set() # Hier houden we bij wie we al gehad hebben.
    exceptions_clean = [x.lower() for x in SUBTHEMA_EXCEPTIONS]
    
    # Hulpfunctie: Maakt een 'vingerafdruk' van de eigenschappen van een weg.
    # Als 'asfalt' en '2010' hetzelfde zijn, is de vingerafdruk (hash) hetzelfde.
    def get_seg_hash(node_id):
        row = gdf.loc[node_id]
        vals = [clean_display_value(row.get(c, '')) for c in SEGMENTATION_ATTRIBUTES]
        return tuple(vals)

    # We lopen de lagen af (Rijbaan -> Parallel -> Fietspad)
    for layer in HIERARCHY_CONFIG:
        rank = layer['rank']
        target_types = layer['types']
        prefix = layer['prefix']
        
        # --- STAP A: VORM DE RUGGENGRAAT ---
        # Zoek alle wegen van dit type die nog niet verwerkt zijn.
        candidates = [
            n for n in G.nodes 
            if gdf.loc[n, 'subthema_clean'] in target_types 
            and n not in processed_ids
        ]
        
        if not candidates: continue
            
        # Maak een tijdelijk mini-netwerkje van alleen deze kandidaten.
        G_sub = G.subgraph(candidates).copy()
        
        # NU KOMT HET KNIP-WERK:
        # Loop alle verbindingen in dit mini-netwerk na.
        edges_to_remove = []
        for u, v in G_sub.edges():
            # Als de vingerafdruk (asfalt, jaar) van buurman U anders is dan buurman V...
            if get_seg_hash(u) != get_seg_hash(v):
                 # ...dan knippen we de draad door! Hier begint een nieuw project.
                 edges_to_remove.append((u, v))
        G_sub.remove_edges_from(edges_to_remove)
        
        # Nu kijken we wat er overblijft: losse eilandjes ('connected components').
        # Elk eilandje is de basis (stam) van een nieuw project.
        components = list(nx.connected_components(G_sub))
        current_layer_groups = []
        
        for i, comp in enumerate(components):
            temp_id = f"{prefix}_{rank}_{i}"
            node_list = list(comp)
            
            # We slaan wat info op over deze groep (voor de weergave later).
            first_node = gdf.loc[node_list[0]]
            seg_props = get_seg_hash(node_list[0])
            curr_proj = clean_display_value(first_node.get('Onderhoudsproject', ''))
            
            # Maak een leesbare tekst van de eigenschappen ("Asfalt, 2010")
            specs = []
            for idx, attr in enumerate(SEGMENTATION_ATTRIBUTES):
                val = seg_props[idx]
                if val: specs.append(f"{FRIENDLY_LABELS.get(attr, attr)}: {val}")
            reason_txt = ", ".join(specs) if specs else "Basis kenmerken"
            
            # Sla de groep op.
            groups[temp_id] = {
                'ids': node_list,
                'subthema': target_types[0],
                'rank': rank,
                'prefix': prefix,
                'reason': reason_txt,
                'current_project': curr_proj, 
                'seg_props': seg_props,
                'spatial_sort_val': 0 
            }
            
            # Markeer deze wegen als 'verwerkt'.
            for n in node_list:
                processed_ids.add(n)
                node_to_group[n] = temp_id
                
            current_layer_groups.append(temp_id)
            
        # --- STAP B: EXPANSIE (Pacman-stijl) ---
        # Nu gaan we de groepen laten groeien. Ze eten alles op wat er direct aan vast zit.
        for group_id in current_layer_groups:
            queue = list(groups[group_id]['ids'])
            idx = 0
            while idx < len(queue):
                current_node = queue[idx]
                idx += 1
                
                # Wie zijn mijn buren?
                neighbors = G.neighbors(current_node)
                for buur in neighbors:
                    # Als de buurman al bezet is, slaan we over.
                    if buur in processed_ids: continue
                    
                    buur_sub = gdf.loc[buur, 'subthema_clean']
                    # Als het een uitzondering is (boom), slaan we over.
                    if buur_sub in exceptions_clean: continue
                    
                    # BELANGRIJK: We eten geen ANDERE hoofdwegen op.
                    # Een fietspad mag geen rijbaan 'opeten'.
                    is_other_backbone = False
                    for check_layer in HIERARCHY_CONFIG:
                        if buur_sub in check_layer['types']:
                            is_other_backbone = True
                            break
                    if is_other_backbone: continue
                    
                    # Oké, voeg de buurman toe aan deze groep!
                    groups[group_id]['ids'].append(buur)
                    node_to_group[buur] = group_id
                    processed_ids.add(buur)
                    # Zet hem in de wachtrij, zodat hij OOK zijn buren weer kan toevoegen.
                    queue.append(buur)

    # --- STAP C: SORTEREN (AANGEPAST: AS-PROJECTIE) ---
    if not groups: return {}

    # 1. Bepaal configuratie en richting
    current_road_label = str(gdf['Wegnummer'].iloc[0]) if 'Wegnummer' in gdf.columns else 'Onbekend'
    direction_code = ROAD_DIRECTIONS.get(current_road_label, 'UNKNOWN')

    # 2. Bereken waarden per groep
    for g_id, g_data in groups.items():
        group_nodes = gdf.loc[g_data['ids']]
        
        # We berekenen het midden van de groep
        center = group_nodes.geometry.unary_union.centroid
        
        # --- TIE-BREAKER LOGICA ---
        # In plaats van "afstand tot startpunt", berekenen we een "positie op de as".
        # Dit getal zorgt voor de volgorde als de hectometrering gelijk is.
        
        tie_breaker_val = 0
        
        if direction_code == 'WTE':   # West naar Oost
            # We willen van West (Laag X) naar Oost (Hoog X).
            # Python sorteert standaard Low->High. Dus X werkt direct.
            tie_breaker_val = center.x
            
        elif direction_code == 'ETW': # Oost naar West
            # We willen van Oost (Hoog X) naar West (Laag X).
            # Om Hoog X als EERSTE te krijgen bij een Low->High sort, maken we hem negatief.
            # (-200 komt voor -100).
            tie_breaker_val = -center.x
            
        elif direction_code == 'STN': # Zuid naar Noord
            # Zuid (Laag Y) -> Noord (Hoog Y).
            tie_breaker_val = center.y
            
        elif direction_code == 'NTS': # Noord naar Zuid
            # Noord (Hoog Y) -> Zuid (Laag Y).
            tie_breaker_val = -center.y
            
        else:
            # Fallback (UNKNOWN):
            # Doe maar gewoon West naar Oost (X-as)
            tie_breaker_val = center.x

        g_data['tie_breaker_dist'] = tie_breaker_val # Voor debug en sortering

        # --- PRIMAIR: HECTOMETRERING ---
        min_hm = group_nodes['hm_sort'].min()
        
        if min_hm < 90000.0:
            g_data['sort_value'] = min_hm
            g_data['sort_mode'] = 'hm'
        else:
            # Als er helemaal geen hectometer is, is onze tie-breaker de hoofdwaarde
            g_data['sort_value'] = tie_breaker_val
            g_data['sort_mode'] = 'axis'

    # 3. Sorteer de lijst
    sorted_groups = sorted(groups.items(), key=lambda x: (
        x[1]['rank'],            # 1. Belangrijkste eerst (Rijbaan)
        x[1]['sort_value'],      # 2. Hectometer (Primair)
        x[1]['tie_breaker_dist'] # 3. De positie op de as (Secundair/Tie-breaker)
    ))
    
    # Hernoemen en returnen...
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
    """
    Haalt hectometerpaaltjes op bij PDOK (Publieke Dienstverlening Op de Kaart).
    """
    wfs_url = "https://service.pdok.nl/rws/nwbwegen/wfs/v1_0"
    buffer_meters = 200  # We kijken 200 meter links en rechts van de weg.
    chunk_size = 50      # We doen het in stapjes van 50 wegdelen tegelijk (anders loopt het vast).
    
    if road_gdf.empty:
        return gpd.GeoDataFrame()

    # Sorteer de wegdelen zodat we ze netjes van begin tot eind aflopen.
    road_sorted = road_gdf.copy()
    road_sorted['sort_x'] = road_sorted.geometry.centroid.x
    road_sorted = road_sorted.sort_values('sort_x')

    all_features = []
    num_segments = len(road_sorted)
    
    # We vragen de data in blokjes op bij de server van PDOK.
    for i in range(0, num_segments, chunk_size):
        chunk = road_sorted.iloc[i : i+chunk_size]
        # Bepaal het vierkantje (Bounding Box) waarbinnen we paaltjes zoeken.
        minx, miny, maxx, maxy = chunk.total_bounds
        bbox_str = f"{minx-buffer_meters},{miny-buffer_meters},{maxx+buffer_meters},{maxy+buffer_meters}"
        
        # De 'bestelling' voor de server.
        params = {
            "service": "WFS", "version": "1.0.0", "request": "GetFeature", 
            "typeName": "hectopunten", "outputFormat": "json", 
            "bbox": bbox_str, "maxFeatures": 5000 
        }
        
        try:
            # Klop aan bij PDOK...
            r = requests.get(wfs_url, params=params, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if data.get('features'):
                    all_features.extend(data['features'])
        except:
            continue # Als het internet hapert, gaan we gewoon door zonder paaltjes.

    if not all_features:
        return gpd.GeoDataFrame()

    # Maak van de antwoorden een kaartlaag.
    gdf_result = gpd.GeoDataFrame.from_features(all_features)
    gdf_result.set_crs(epsg=28992, inplace=True)
    
    # Zoek de kolom waar de tekst op het paaltje staat (soms heet dat anders).
    hm_col = None
    if 'hectometrering' in gdf_result.columns: hm_col = 'hectometrering'
    elif 'hectomtrng' in gdf_result.columns: hm_col = 'hectomtrng'

    # Verwijder dubbele paaltjes.
    if 'id' in gdf_result.columns:
        gdf_result = gdf_result.drop_duplicates(subset=['id'])
    elif hm_col:
        gdf_result = gdf_result.drop_duplicates(subset=[hm_col])
    
    # Maak er een getal van zodat we het later kunnen tonen.
    if hm_col:
        gdf_result['hm_val'] = pd.to_numeric(gdf_result[hm_col], errors='coerce').fillna(0)
    else:
        gdf_result['hm_val'] = 0.0

    return gdf_result

def log_change(oid, field, old_val, new_val, status="Succes"):
    # Als er nog geen logboek is, maak er een.
    if 'change_log' not in st.session_state:
        st.session_state['change_log'] = []
    
    # Schrijf een nieuwe regel in het logboek.
    st.session_state['change_log'].append({
        'Tijd': datetime.now().strftime("%H:%M:%S"),
        'ID': oid,           # Welk object?
        'Veld': field,       # Welke kolom?
        'Oud': str(old_val), # Wat stond er eerst?
        'Nieuw': str(new_val), # Wat staat er nu?
        'Status': status
    })
    # Sla het direct op op de harde schijf (voor als de browser crasht).
    save_autosave()

# --- START APPLICATIE & AUTOLOAD ---

# Check: Hebben we de data al ingeladen? Zo nee, doe het nu.
if 'data_complete' not in st.session_state:
    with st.spinner('Data laden...'):
        st.session_state['data_complete'] = load_data()
        
        # Check: Is er nog een 'autosave' bestandje van de vorige keer?
        if os.path.exists(AUTOSAVE_FILE):
            try:
                # Zo ja, lees het in en speel alle wijzigingen opnieuw af.
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
    # Veiligheidscheck: als de data corrupt is geraakt, herlaad de pagina.
    if 'sys_id' not in st.session_state['data_complete'].columns:
        st.cache_data.clear()
        st.rerun()

# Maak een korte verwijzing naar de data (scheelt typwerk).
raw_gdf = st.session_state['data_complete']

# --- STATE (Het geheugen van de browser) ---
# Hier onthouden we wat je hebt aangeklikt, genegeerd of ingezoomd.
if 'processed_groups' not in st.session_state: st.session_state['processed_groups'] = set()
if 'ignored_groups' not in st.session_state: st.session_state['ignored_groups'] = set()
if 'ignored_errors' not in st.session_state: st.session_state['ignored_errors'] = set() # <--- NIEUW
if 'change_log' not in st.session_state: st.session_state['change_log'] = []

if 'selected_group_id' not in st.session_state: st.session_state['selected_group_id'] = None
if 'selected_error_id' not in st.session_state: st.session_state['selected_error_id'] = None
if 'zoom_bounds' not in st.session_state: st.session_state['zoom_bounds'] = None

# --- SIDEBAR (Linkerkant scherm) ---
st.sidebar.title("iASSET Advisor")
# Maak een lijstje van alle wegnummers (N351, etc.)
all_roads = sorted([str(x) for x in raw_gdf['Wegnummer'].dropna().unique()])
selected_road = st.sidebar.selectbox("Kies Wegnummer", all_roads)

# Filter de data: we werken alleen met de weg die jij kiest.
road_gdf = raw_gdf[raw_gdf['Wegnummer'] == selected_road].copy()

# Als je van weg wisselt, moeten we het netwerk (spinnenweb) opnieuw bouwen.
if 'graph_current' not in st.session_state or st.session_state.get('last_road') != selected_road:
    with st.spinner('Netwerk analyseren...'):
        st.session_state['graph_current'] = build_graph_from_geometry(road_gdf)
        st.session_state['last_road'] = selected_road
        # Reset alle selecties omdat we naar een nieuwe weg kijken.
        st.session_state['computed_groups'] = None
        st.session_state['zoom_bounds'] = None
        st.session_state['selected_error_id'] = None
        st.session_state['selected_group_id'] = None
        
G_road = st.session_state['graph_current']

# --- LAYOUT ---
# We verdelen het scherm in twee kolommen: Kaart (links, breed) en Inspecteur (rechts, smaller).
col_map, col_inspector = st.columns([3, 2])

# --- RECHTS: INSPECTOR ---
with col_inspector:
    st.subheader("Werklijst")
    
    # Functie om alles te resetten als je van modus wisselt.
    def on_mode_change():
        st.session_state['selected_error_id'] = None
        st.session_state['selected_group_id'] = None
        st.session_state['zoom_bounds'] = None
        if 'folium_map' in st.session_state:
            st.session_state['folium_map'] = None 
        
    # De keuze-knop voor wat je wilt doen.
    mode = st.radio("Modus:", 
                    ["🔍 Data Kwaliteit", "🏗️ Project Adviseur"], 
                    horizontal=True, on_change=on_mode_change)
    st.divider()

# --- MODUS 1: KWALITEIT (De Politieagent) ---
    if mode == "🔍 Data Kwaliteit":
        # Roep de 'Inspecteur' aan (uit Deel 2).
        all_violations = check_rules(road_gdf, G_road)
        
        # FILTER: Haal de meldingen weg die in de 'ignored_errors' lijst staan.
        violations = [v for v in all_violations if v['id'] not in st.session_state['ignored_errors']]
        
        if not violations:
            st.success("Schoon! Geen datakwaliteit issues.")
            
            # Als er wél verborgen items zijn, geef een optie om ze terug te halen.
            if len(all_violations) > 0:
                if st.button("🔄 Reset genegeerde meldingen"):
                    st.session_state['ignored_errors'] = set()
                    st.rerun()
        else:
            st.write(f"**{len(violations)} issues gevonden**")
            
            # Maak een scrollbare lijst van foutmeldingen.
            with st.container(height=400):
                for i, v in enumerate(violations):
                    vid = v['id']
                    
                    # Unieke keys maken voor de knoppen
                    key_show = f"btn_err_show_{vid}_{i}" 
                    key_ign = f"btn_err_ign_{vid}_{i}"
                    
                    # Als je op een fout hebt geklikt, wordt hij blauw gemarkeerd.
                    is_selected = (st.session_state['selected_error_id'] == vid)
                    
                    # Bepaal de stijl van de container
                    container_args = {"border": True} if is_selected else {}
                    
                    with st.container(**container_args):
                        if is_selected:
                            st.markdown("**:blue-background[GESELECTEERD]**")
                        
                        # We maken 3 kolommen: Tekst (breed), Toon (smal), Negeer (smal)
                        c1, c2, c3 = st.columns([2, 1, 1])
                        
                        with c1:
                            st.markdown(f"**{v['subthema']}**")
                            st.caption(f"{v['msg']}")
                        
                        with c2:
                            # Knop om in te zoomen (Oogje)
                            if st.button("👁️", key=key_show, help="Toon op kaart"):
                                st.session_state['selected_error_id'] = vid
                                obj_geom = road_gdf.loc[vid].geometry
                                st.session_state['zoom_bounds'] = obj_geom.bounds
                                st.rerun()
                        
                        with c3:
                            # Knop om te negeren (Prullenbakje)
                            if st.button("🗑️", key=key_ign, help="Negeer deze melding"):
                                st.session_state['ignored_errors'].add(vid)
                                # Als we degene negeren die we net bekeken, resetten we de view
                                if is_selected:
                                    st.session_state['selected_error_id'] = None
                                    st.session_state['zoom_bounds'] = None
                                st.rerun()
                                
                        if not is_selected:
                            st.divider()

            # Als er een fout geselecteerd is, tonen we hieronder een reparatie-formulier.
            if st.session_state['selected_error_id']:
                err_id = st.session_state['selected_error_id']
                # Check of het ID nog wel in de gefilterde lijst zit (anders crasht het misschien)
                if err_id in road_gdf.index:
                    st.divider()
                    st.markdown(f"#### Corrigeer ID {err_id}")
                    
                    row = road_gdf.loc[err_id]
                    # Zoek op welke kolommen ontbreken.
                    viol_info = next((v for v in all_violations if v['id'] == err_id), None)
                    cols_to_fix = viol_info['missing_cols'] if viol_info else ['Onderhoudsproject']
                    
                    inputs = {}
                    for col in cols_to_fix:
                        curr_val = clean_display_value(row.get(col, ''))
                        inputs[col] = st.text_input(f"Vul in: {col}", value=curr_val, key=f"fix_{col}_{err_id}")
                    
                    if st.button("Opslaan Correctie"):
                        for col, new_val in inputs.items():
                            old_val = raw_gdf.at[err_id, col]
                            val_old_clean = clean_display_value(old_val)
                            val_new_clean = clean_display_value(new_val)
                            
                            # Alleen opslaan als je echt iets veranderd hebt.
                            if val_old_clean != val_new_clean:
                                raw_gdf.at[err_id, col] = new_val
                                log_change(err_id, col, val_old_clean, new_val)
                        
                        st.success("Opgeslagen!")
                        st.session_state['selected_error_id'] = None
                        st.rerun()

    # --- MODUS 2: PROJECT ADVISEUR (De Slimme Hulp) ---
    elif mode == "🏗️ Project Adviseur":
        # Als we nog geen advies hebben berekend, doe dat nu (met de functie uit Deel 2).
        if 'computed_groups' not in st.session_state or st.session_state['computed_groups'] is None:
            with st.spinner("AI berekent groepen (incl. absorptie)..."):
                groups = generate_grouped_proposals(road_gdf, G_road)
                st.session_state['computed_groups'] = groups
        
        all_groups = st.session_state['computed_groups']
        
        # Filter groepen die je al behandeld hebt eruit.
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
            
            # Sorteren: Belangrijke wegen eerst.
            def sort_key_advisor(item):
                gid, data = item
                r = data.get('rank', 99)
                pos = data.get('sort_value', 0)
                tie = data.get('tie_breaker_dist', 0) # De nieuwe tie-breaker
                return (r, pos, tie) # Sorteer op 3 niveaus
            
            sorted_items = sorted(active_groups.items(), key=sort_key_advisor)
            
            with st.container(height=400):
                for g_id, g_data in sorted_items:
                    count = len(g_data['ids'])
                    
                    # Kies een leuk icoontje.
                    if "RIJBAAN" in g_id: icon = "🛣️"
                    elif "FIETSPAD" in g_id: icon = "🚲" 
                    elif "PARALLEL" in g_id: icon = "🛤️"
                    else: icon = "🌳"
                    
                    is_sel = (st.session_state['selected_group_id'] == g_id)
                    container_args = {"border": True} if is_sel else {}
                    
                    with st.container(**container_args):
                        if is_sel:
                            st.markdown("**:blue-background[GESELECTEERD]**")
                        
                        st.markdown(f"**{icon} {g_data['subthema'].title()}** ({count} obj)")
                        st.caption(f"{g_data['reason']}") # Waarom is dit een groep? (bv "Asfalt, 2010")
                        
                        old_p = g_data.get('current_project', '')
                        old_p_display = old_p if old_p else "Geen"
                        st.markdown(f"<small>Huidig: *{old_p_display}*</small>", unsafe_allow_html=True)

                        # Knoppenbalk
                        b1, b2 = st.columns([1, 1])
                        
                        with b1: # Selecteer knop
                            btn_label = "📍 Geselecteerd" if is_sel else "👁️ Selecteer"
                            if st.button(btn_label, key=f"vis_{g_id}", disabled=is_sel):
                                st.session_state['selected_group_id'] = g_id
                                grp_geom = road_gdf.loc[g_data['ids']].unary_union
                                st.session_state['zoom_bounds'] = grp_geom.bounds
                                st.rerun()
                                
                        with b2: # Negeer knop
                            if st.button("🗑️ Negeer", key=f"ign_{g_id}"):
                                st.session_state['ignored_groups'].add(g_id)
                                if is_sel:
                                    st.session_state['selected_group_id'] = None
                                    st.session_state['zoom_bounds'] = None
                                st.rerun()
                                
                        if not is_sel:
                            st.divider()

            # Het invulvak om de projectnaam te accepteren.
            if st.session_state['selected_group_id'] and st.session_state['selected_group_id'] in active_groups:
                sel_gid = st.session_state['selected_group_id']
                sel_data = active_groups[sel_gid]
                
                st.divider()
                st.markdown(f"#### 🏷️ Naamgeven: {sel_gid}")
                st.info(f"Bevat {len(sel_data['ids'])} objecten. ({sel_data['reason']})")
                
                old_p_hint = sel_data.get('current_project', '')
                placeholder_txt = old_p_hint if old_p_hint else "bv. N351-HRB-20.1-24.3"
                
                name_input = st.text_input("Projectnaam", value="", placeholder=placeholder_txt, key="proj_name_input")
                
                if st.button("✅ Opslaan & Toepassen", type="primary"):
                    if name_input.strip():
                        val_new_clean = clean_display_value(name_input)
                        
                        count_updates = 0
                        # Loop door alle objecten in de groep en geef ze de nieuwe naam.
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

# --- LINKS: KAART ---
with col_map:
    st.subheader(f"Kaart: {selected_road}")

    # 1. DEBUG: DIT STAAT HELEMAAL BOVENAAN (MOET ZICHTBAAR ZIJN)
    st.info(f"🚀 DEBUG START: We gaan de kaart laden voor {selected_road}...")

    # 2. CHECK: IS ER DATA?
    if road_gdf.empty:
        st.error("❌ STOP: De tabel 'road_gdf' is leeg. Er is geen data voor deze weg.")
        st.stop() # We stoppen hier bewust om een crash te voorkomen.

    # 3. VEILIG CONVERTEREN (in een try-blok om crashes te vangen)
    try:
        road_web = road_gdf.to_crs(epsg=4326)
        st.write(f"- Data geconverteerd. Aantal rijen: {len(road_web)}")
    except Exception as e:
        st.error(f"❌ CRASH bij CRS conversie: {e}")
        st.stop()

    # 4. VEILIG DE KAART MAKEN (Gebruik Bounds i.p.v. Centroid)
    # Centroid crasht vaak op 'vuile' data, bounds bijna nooit.
    try:
        if st.session_state.get('zoom_bounds'):
             minx, miny, maxx, maxy = st.session_state['zoom_bounds']
        else:
             minx, miny, maxx, maxy = road_web.total_bounds
        
        # Bereken midden van de box
        center_lat = (miny + maxy) / 2
        center_lon = (minx + maxx) / 2
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")
        
        if st.session_state.get('zoom_bounds'):
            m.fit_bounds([[miny, minx], [maxy, maxx]])
            
    except Exception as e:
        st.error(f"⚠️ Kon kaart niet centreren (waarschijnlijk foute geometrie): {e}")
        # Fallback: Start gewoon op 'Nederland' zodat we toch iets zien
        m = folium.Map(location=[52.2, 5.5], zoom_start=8)

    # 5. HARDE TEST: 5 ECHTE PINNEN (Direct uit de data)
    st.write("De 5 test-pinnen worden nu geplaatst...")
    try:
        pins_placed = 0
        for idx, row in road_web.head(5).iterrows():
            geom = row.geometry
            if geom:
                pt = geom.centroid
                # We dwingen 'float' af om Numpy-problemen te voorkomen
                folium.Marker(
                    [float(pt.y), float(pt.x)],
                    popup=f"Test {idx}",
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)
                pins_placed += 1
        st.success(f"✅ {pins_placed} test-pinnen toegevoegd aan kaart-object.")
    except Exception as e:
        st.error(f"❌ Fout bij plaatsen test-pinnen: {e}")

    # 6. NETWERK NODES (Uw originele vraag)
    if 'graph_current' in st.session_state:
        G_debug = st.session_state['graph_current']
        
        # Checkbox om het aan/uit te zetten (standaard AAN)
        if st.checkbox("Toon Netwerk Nodes (Blauwe stippen)", value=True):
            count_net = 0
            for node_id in G_debug.nodes():
                if count_net > 500: break # Veiligheidslimiet
                
                if node_id in road_web.index:
                    geom = road_web.loc[node_id].geometry
                    if geom:
                        pt = geom.centroid
                        # Blauwe cirkeltjes voor de nodes
                        folium.CircleMarker(
                            [float(pt.y), float(pt.x)],
                            radius=4,
                            color='blue',
                            fill=True,
                            fill_color='blue',
                            fill_opacity=0.8,
                            tooltip=f"Node {node_id}"
                        ).add_to(m)
                        count_net += 1
            st.caption(f"📍 {count_net} netwerk-nodes getekend.")

    # 7. VOEG DE ORIGINELE GEOJSON LAAG TOE (Zoals het was)
    suggested_ids = set()
    if 'computed_groups' in st.session_state and st.session_state['computed_groups']:
        for g_id, g_data in st.session_state['computed_groups'].items():
            if g_id not in st.session_state['processed_groups'] and g_id not in st.session_state['ignored_groups']:
                suggested_ids.update(g_data['ids'])

    def style_fn(feature):
        oid = feature['properties']['sys_id']
        props = feature['properties']
        
        if oid == st.session_state.get('selected_error_id'):
            return {'fillColor': '#00FFFF', 'color': 'black', 'weight': 3, 'fillOpacity': 0.8}
        if st.session_state.get('selected_group_id'):
            active_grp = st.session_state['computed_groups'].get(st.session_state['selected_group_id'])
            if active_grp and oid in active_grp['ids']:
                return {'fillColor': '#00FFFF', 'color': 'black', 'weight': 3, 'fillOpacity': 0.8}
        if oid in suggested_ids:
             return {'fillColor': '#FFFF00', 'color': 'black', 'weight': 1, 'fillOpacity': 0.6}
        if props.get('Onderhoudsproject'):
            return {'fillColor': '#00CC00', 'color': 'gray', 'weight': 0.5, 'fillOpacity': 0.5}
        return {'fillColor': '#808080', 'color': 'gray', 'weight': 0.5, 'fillOpacity': 0.3}

    meta_cols = [c for c in ALL_META_COLS if c in road_web.columns]
    cols_to_select = ['geometry', 'sys_id'] + meta_cols
    tooltip_fields = ['subthema', 'Onderhoudsproject'] + [c for c in SEGMENTATION_ATTRIBUTES if c in road_web.columns]
    
    # Voeg de GeoJson laag toe
    try:
        folium.GeoJson(
            road_web[cols_to_select],
            style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, style="font-size: 11px;")
        ).add_to(m)
    except Exception as e:
        st.error(f"Fout bij GeoJson laag: {e}")

    # 8. TEKEN DE KAART
    st_folium(m, width=None, height=600, returned_objects=["last_object_clicked"], key="folium_map")

    # --- DEBUG TOOL: SORTERING ---
    st.divider()
    st.markdown("### 🕵️ Debug: Sortering Analyse")
    
    if 'computed_groups' in st.session_state and st.session_state['computed_groups']:
        debug_data = []
        # We maken een lijstje van wat de computer heeft berekend
        for gid, data in st.session_state['computed_groups'].items():
            # Alleen tonen als het nog niet verwerkt is
            if gid not in st.session_state['processed_groups']:
                debug_data.append({
                    "ID": gid,
                    "Methode": data.get('sort_mode', '?'),
                    "HM Waarde": data.get('sort_value', 999),
                    "Afstand (Tie-breaker)": int(data.get('tie_breaker_dist', 0)),
                    "Subthema": data.get('subthema')
                })
        
        if debug_data:
            # Toon als tabel, gesorteerd op HM Waarde
            df_debug = pd.DataFrame(debug_data).sort_values(by=['HM Waarde', 'Afstand (Tie-breaker)'])
            st.dataframe(df_debug, use_container_width=True, hide_index=True)
        else:
            st.info("Geen actieve groepen om te analyseren.")

st.divider()

st.subheader("📝 Logboek Wijzigingen & Export")

if st.session_state['change_log']:
    # --- KNOP: ALLES TERUGDRAAIEN ---
    c_all_1, c_all_2 = st.columns([1, 5])
    with c_all_1:
        if st.button("⚠️ Alles Herstellen", type="primary", help="Draai alle wijzigingen in één keer terug"):
            # Loop achteruit door het logboek en draai alles terug.
            for entry in reversed(st.session_state['change_log']):
                # 1. Zet de oude data terug.
                apply_change_to_data(entry['ID'], entry['Veld'], entry['Oud'])
                
                # 2. Als we een projectnaam terugdraaien, moet de 'Groep' weer beschikbaar worden voor advies.
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

            # 3. Log leegmaken.
            st.session_state['change_log'] = []
            save_autosave()
            st.success("Alle wijzigingen zijn ongedaan gemaakt!")
            st.rerun()

    with c_all_2:
        st.caption(f"Er staan {len(st.session_state['change_log'])} wijzigingen in de wachtrij.")

    st.divider()

    # --- INDIVIDUELE LIJST ---
    # Toon de lijst met wijzigingen (nieuwste bovenaan).
    reversed_log = list(reversed(list(enumerate(st.session_state['change_log']))))
    
    with st.container(height=300):
        for idx, entry in reversed_log:
            c1, c2, c3, c4 = st.columns([1, 2, 4, 1])
            c1.text(entry['Tijd'])
            c2.text(f"ID: {entry['ID']}")
            c3.text(f"{entry['Veld']}: {entry['Oud']} ➡ {entry['Nieuw']}")
            
            if c4.button("↩️ Herstel", key=f"undo_{idx}"):
                # Eén specifieke wijziging ongedaan maken.
                apply_change_to_data(entry['ID'], entry['Veld'], entry['Oud'])
                
                # Groep herstellen indien nodig.
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

                del st.session_state['change_log'][idx]
                save_autosave()
                st.success("Wijziging ongedaan gemaakt!")
                st.rerun()
else:
    st.caption("Nog geen wijzigingen aangebracht.")

# --- EXPORT (Downloaden) ---
# Welke kolommen willen we in het Excel-bestand?
EXPORT_COLUMNS = [
    'bron_id', 'nummer', 'Wegnummer', 'subthema', 'Onderhoudsproject',    
    'verhardingssoort', 'Soort deklaag specifiek', 'Jaar aanleg',
    'Jaar deklaag', 'Besteknummer', 'gps coordinaten', 'rds coordinaten'       
]

# Check of deze kolommen ook echt bestaan in onze data.
valid_export_cols = [c for c in EXPORT_COLUMNS if c in st.session_state['data_complete'].columns]

# We exporteren alleen de rijen die daadwerkelijk gewijzigd zijn.
changed_ids = set()
if 'change_log' in st.session_state and st.session_state['change_log']:
    for entry in st.session_state['change_log']:
        changed_ids.add(entry['ID'])

if changed_ids:
    df_export = st.session_state['data_complete'].loc[list(changed_ids)][valid_export_cols].copy()
    
    # Jaartallen netjes maken voor Excel.
    for col in ['Jaar aanleg', 'Jaar deklaag']:
        if col in df_export.columns:
            df_export[col] = df_export[col].apply(clean_display_value)
            
    if 'bron_id' in df_export.columns:
        df_export.rename(columns={'bron_id': 'id'}, inplace=True)
        
    st.success(f"📦 Er staan {len(df_export)} gewijzigde objecten klaar voor export.")
    
    c_dl1, c_dl2 = st.columns(2)
    
    # Download knop 1: CSV
    with c_dl1:
        csv = df_export.to_csv(index=False, sep=';').encode('utf-8-sig')
        st.download_button(
            label="📥 Download CSV", 
            data=csv, 
            file_name="iASSET_Mutaties.csv", 
            mime="text/csv"
        )
        
    # Download knop 2: Excel
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
