import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely import wkt
import folium
from streamlit_folium import st_folium
from sklearn.decomposition import PCA
import numpy as np

# --- CONFIGURATIE ---
st.set_page_config(layout="wide", page_title="iASSET Tool")

FILE_NIET_RIJSTROOK = "N-allemaal-niet-rijstrook.csv"
FILE_WEL_RIJSTROOK = "N-allemaal-alleen-rijstrook.csv"

# Validatie regels
REQ_ONDERHOUDSPROJECT = ['afrit en entree', 'fietspad', 'inrit en doorsteek', 'parallelweg', 'rijstrook']
FORBIDDEN_ONDERHOUDSPROJECT = ['fietsstalling', 'parkeerplaats', 'rotonderand', 'verkeerseiland of middengeleider']
# Kolommen die een 'knip' in een onderhoudsproject veroorzaken als ze veranderen
BOUNDARY_COLS = ['verhardingssoort', 'Soort deklaag specifiek', 'Jaar aanleg', 'Jaar deklaag']

# --- FUNCTIES ---

@st.cache_data
def load_data():
    """Laadt data, voegt samen en fixt geometrie. Wordt gecached voor snelheid."""
    df1 = pd.read_csv(FILE_NIET_RIJSTROOK)
    df2 = pd.read_csv(FILE_WEL_RIJSTROOK)
    df = pd.concat([df1, df2], ignore_index=True)
    
    # Geometrie parsen (RDS voorkeur)
    def parse_geom(x):
        try:
            return wkt.loads(x)
        except:
            return None
    
    df['geometry'] = df['rds coordinaten'].apply(parse_geom)
    df = df.dropna(subset=['geometry'])
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.set_crs(epsg=28992, inplace=True, allow_override=True)
    
    # Maak unieke index om editen makkelijker te maken
    if 'id' not in gdf.columns:
        gdf['id'] = range(len(gdf))
    
    return gdf

def calculate_logic(gdf):
    """Voert alle berekeningen uit: Validatie, Buren-advies, Volgorde & Grenzen."""
    
    # 1. Schoonmaken
    gdf['subthema_clean'] = gdf['subthema'].astype(str).str.lower().str.strip()
    gdf['project_clean'] = gdf['Onderhoudsproject'].astype(str).str.strip().replace('nan', '')
    gdf['validation_error'] = ''
    gdf['Advies_Onderhoudsproject'] = ''
    gdf['Is_Project_Grens'] = False
    
    # 2. Validatie Basis
    # Regel 1: Moet project hebben
    mask_missing = (gdf['subthema_clean'].isin(REQ_ONDERHOUDSPROJECT)) & (gdf['project_clean'] == '')
    gdf.loc[mask_missing, 'validation_error'] += 'Missend Project; '
    
    # Regel 2: Mag geen project hebben
    mask_excess = (gdf['subthema_clean'].isin(FORBIDDEN_ONDERHOUDSPROJECT)) & (gdf['project_clean'] != '')
    gdf.loc[mask_excess, 'validation_error'] += 'Onterecht Project; '
    
    # 3. Buren Advies (Spatial Join)
    # Alleen voor missende projecten
    targets = gdf[mask_missing].copy()
    sources = gdf[gdf['project_clean'] != ''].copy()
    
    if not targets.empty and not sources.empty:
        targets['geometry_buffer'] = targets.geometry.buffer(0.5) # 50cm tolerantie
        targets.set_geometry('geometry_buffer', inplace=True)
        
        joined = gpd.sjoin(targets[['geometry_buffer']], sources[['geometry', 'Onderhoudsproject']], how='left', predicate='intersects')
        
        # Meest voorkomende buur
        advice = joined.groupby(joined.index)['Onderhoudsproject'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else None
        )
        gdf.loc[advice.index, 'Advies_Onderhoudsproject'] = advice

    # 4. Volgorde & Grensbepaling (Per Weg)
    gdf['volgorde_score'] = np.nan
    
    # We itereren per wegnummer
    for weg in gdf['Wegnummer'].unique():
        if pd.isna(weg): continue
        mask = gdf['Wegnummer'] == weg
        subset = gdf[mask].copy()
        
        if len(subset) < 2: continue
            
        # PCA voor volgorde
        centroids = np.array([(g.centroid.x, g.centroid.y) for g in subset.geometry])
        pca = PCA(n_components=1)
        try:
            transformed = pca.fit_transform(centroids)
            subset['volgorde_score'] = transformed.flatten()
            
            # Nu sorteren we de subset op volgorde
            subset = subset.sort_values('volgorde_score')
            
            # --- GRENSBEPALING (Het nieuwe stuk) ---
            # We kijken alleen naar primaire objecten voor grenzen
            primair_mask = subset['subthema_clean'].isin(['rijstrook', 'parallelweg', 'fietspad'])
            subset_primair = subset[primair_mask].copy()
            
            # We vergelijken de huidige rij met de vorige rij (shift)
            # Als er iets verandert in de belangrijke kolommen, is het een grens
            for col in BOUNDARY_COLS:
                if col in subset_primair.columns:
                    # Vergelijk met vorige waarde. 
                    # shift(1) schuift data 1 plek op naar beneden.
                    # ne() betekent 'not equal'
                    verandering = subset_primair[col].ne(subset_primair[col].shift(1))
                    
                    # De eerste regel is altijd 'anders' omdat vorige NaN is, die negeren we
                    verandering.iloc[0] = False
                    
                    # Markeer in de hoofdtabel
                    grens_indices = subset_primair[verandering].index
                    gdf.loc[grens_indices, 'Is_Project_Grens'] = True
                    gdf.loc[grens_indices, 'validation_error'] += f'Mogelijke Projectgrens ({col} wijzigt); '

            # Zet volgorde terug in hoofdtabel
            gdf.loc[mask, 'volgorde_score'] = subset['volgorde_score']
            
        except Exception as e:
            pass 

    return gdf

# --- UI OPBOUW ---

st.title("🛠️ iASSET Data Tool")
st.markdown("Selecteer een weg, bekijk de kaart en muteer de data.")

# 1. Data Laden
if 'data' not in st.session_state:
    with st.spinner('Data wordt geladen en geanalyseerd...'):
        raw_gdf = load_data()
        processed_gdf = calculate_logic(raw_gdf)
        st.session_state['data'] = processed_gdf
else:
    processed_gdf = st.session_state['data']

# 2. Sidebar Filters
all_roads = sorted([str(x) for x in processed_gdf['Wegnummer'].dropna().unique()])
selected_road = st.sidebar.selectbox("Kies een Wegnummer", all_roads)

# Filter dataset
road_df = processed_gdf[processed_gdf['Wegnummer'] == selected_road].copy()

# Sorteer op berekende volgorde
if 'volgorde_score' in road_df.columns:
    road_df = road_df.sort_values('volgorde_score')

# 3. Hoofdscherm Indeling
col_map, col_data = st.columns([1, 1])

with col_map:
    st.subheader(f"Kaart: {selected_road}")
    
    if not road_df.empty:
        # Bereken centrum voor kaart
        centroid = road_df.unary_union.centroid
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=15, tiles="CartoDB positron")
        
        # Functie voor kleur
        def get_color(row):
            if row['Is_Project_Grens']: return 'orange'
            if 'Missend' in row['validation_error']: return 'red'
            if 'Onterecht' in row['validation_error']: return 'purple'
            return 'green'

        # Voeg objecten toe aan kaart
        # We gebruiken GeoJSON voor performance
        folium.GeoJson(
            road_df[['geometry', 'subthema', 'validation_error', 'Onderhoudsproject', 'Advies_Onderhoudsproject', 'id']],
            style_function=lambda x: {
                'fillColor': get_color(x['properties']),
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.6
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['subthema', 'validation_error', 'Onderhoudsproject', 'Advies_Onderhoudsproject'],
                aliases=['Type', 'Status', 'Huidig Project', 'Advies']
            )
        ).add_to(m)
        
        st_folium(m, width=700, height=500)
        
        st.info("Legenda: 🟢 OK | 🔴 Missende Data | 🟠 Mogelijke Grens | 🟣 Onterechte Data")

with col_data:
    st.subheader("Data Mutatie")
    
    # Selecteer kolommen om te tonen
    cols_to_show = ['subthema', 'Onderhoudsproject', 'Advies_Onderhoudsproject', 'validation_error', 
                    'verhardingssoort', 'Jaar deklaag', 'Is_Project_Grens']
    
    # Maak de tabel editbaar
    st.markdown("Je kunt hieronder direct in de tabel typen of vinkjes zetten.")
    
    # We gebruiken data_editor. 
    # Let op: we halen de geometrie even weg voor weergave, dat edit niet lekker.
    edited_df = st.data_editor(
        road_df[cols_to_show],
        key="editor",
        num_rows="dynamic",
        height=500
    )
    
    # KNOP: Advies Overnemen
    if st.button("✅ Neem Advies Over (voor zichtbare regels)"):
        # Zoek rijen waar advies is en huidig leeg is
        mask_update = (road_df['Advies_Onderhoudsproject'].notna()) & (road_df['Advies_Onderhoudsproject'] != '') & (road_df['Onderhoudsproject'].isna() | (road_df['Onderhoudsproject'] == ''))
        
        # Update in de hoofdataset in sessie state
        indices_to_update = road_df[mask_update].index
        st.session_state['data'].loc[indices_to_update, 'Onderhoudsproject'] = road_df.loc[indices_to_update, 'Advies_Onderhoudsproject']
        st.success(f"{len(indices_to_update)} objecten bijgewerkt! Herlaad de pagina om resultaat te zien.")
        st.rerun()

# 4. Export Sectie (Onderaan)
st.divider()
st.subheader("Exporteer Resultaat")
st.markdown("Klaar met bewerken? Download de nieuwe CSV voor iASSET.")

def convert_df(df):
    # Verwijder onze hulpkolommen voor export
    output_cols = [c for c in df.columns if c not in ['geometry', 'geometry_buffer', 'subthema_clean', 'project_clean', 'volgorde_score', 'Is_Project_Grens', 'validation_error', 'Advies_Onderhoudsproject']]
    return df[output_cols].to_csv(index=False).encode('utf-8')

csv = convert_df(st.session_state['data'])

st.download_button(
    label="📥 Download CSV voor iASSET",
    data=csv,
    file_name='iASSET_Import_New.csv',
    mime='text/csv',
)