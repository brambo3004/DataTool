import pandas as pd
import geopandas as gpd
from shapely import wkt
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --- CONFIGURATIE ---
FILE_NIET_RIJSTROOK = "N-allemaal-niet-rijstrook.csv"
FILE_WEL_RIJSTROOK = "N-allemaal-alleen-rijstrook.csv"
OUTPUT_FILE = "iASSET_Verrijkt_Export.csv"

# Regels voor validatie
REQ_ONDERHOUDSPROJECT = ['afrit en entree', 'fietspad', 'inrit en doorsteek', 'parallelweg', 'rijstrook']
FORBIDDEN_ONDERHOUDSPROJECT = ['fietsstalling', 'parkeerplaats', 'rotonderand', 'verkeerseiland of middengeleider']

def load_and_prep_data():
    print("1. Data laden...")
    df1 = pd.read_csv(FILE_NIET_RIJSTROOK)
    df2 = pd.read_csv(FILE_WEL_RIJSTROOK)
    df = pd.concat([df1, df2], ignore_index=True)
    
    # Geometrie parsen (RDS voorkeur)
    print("2. Geometrie verwerken...")
    def parse_geom(x):
        try:
            return wkt.loads(x)
        except:
            return None
    
    # Probeer RDS, anders GPS
    df['geometry'] = df['rds coordinaten'].apply(parse_geom)
    # Valideer geometry
    df = df.dropna(subset=['geometry'])
    
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    # Zet CRS op RD New (EPSG:28992) voor meters
    gdf.set_crs(epsg=28992, inplace=True, allow_override=True)
    return gdf

def run_validations(gdf):
    print("3. Validatieregels toepassen...")
    gdf['validation_error'] = ''
    
    # Cleaning
    gdf['subthema_clean'] = gdf['subthema'].astype(str).str.lower().str.strip()
    gdf['project_clean'] = gdf['Onderhoudsproject'].astype(str).str.strip().replace('nan', '')
    
    # Regel 1: Moet project hebben, heeft het niet
    mask_missing = (gdf['subthema_clean'].isin(REQ_ONDERHOUDSPROJECT)) & (gdf['project_clean'] == '')
    gdf.loc[mask_missing, 'validation_error'] += 'Missend Onderhoudsproject; '
    
    # Regel 2: Mag geen project hebben, heeft het wel
    mask_excess = (gdf['subthema_clean'].isin(FORBIDDEN_ONDERHOUDSPROJECT)) & (gdf['project_clean'] != '')
    gdf.loc[mask_excess, 'validation_error'] += 'Onterecht Onderhoudsproject; '
    
    # Regel 3: Missende attributen (generic check)
    check_cols = ['verhardingssoort', 'naam', 'Gebruikersfunctie']
    for col in check_cols:
        if col in gdf.columns:
            mask_col = (gdf[col].isna()) | (gdf[col].astype(str).str.strip() == '')
            gdf.loc[mask_col, 'validation_error'] += f'Missend {col}; '
            
    return gdf

def generate_network_advice(gdf):
    print("4. Netwerk analyse & Advies genereren...")
    # Buffer van 20cm voor 'raken'
    gdf['geometry_buffer'] = gdf.geometry.buffer(0.2)
    
    # We zoeken buren voor objecten die een project missen
    target_mask = (gdf['validation_error'].str.contains('Missend Onderhoudsproject'))
    targets = gdf[target_mask].copy()
    sources = gdf[gdf['project_clean'] != ''].copy()
    
    if targets.empty:
        print("   Geen objecten gevonden die advies nodig hebben.")
        return gdf

    # Spatial Join
    targets.set_geometry('geometry_buffer', inplace=True)
    sources.set_geometry('geometry_buffer', inplace=True)
    
    joined = gpd.sjoin(targets[['geometry_buffer']], sources[['geometry_buffer', 'Onderhoudsproject']], how='left', predicate='intersects')
    
    # Meest voorkomende buur-project kiezen (Mode)
    advice = joined.groupby(joined.index)['Onderhoudsproject'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None
    )
    
    gdf['Advies_Onderhoudsproject'] = np.nan
    gdf.loc[advice.index, 'Advies_Onderhoudsproject'] = advice
    
    # Opruimen tijdelijke kolommen
    gdf.drop(columns=['geometry_buffer'], inplace=True, errors='ignore')
    return gdf

def calculate_sequencing(gdf):
    print("5. Volgorde bepalen (Geometric Sequencing)...")
    gdf['volgorde_score'] = np.nan
    
    # Per wegnummer verwerken
    for weg in gdf['Wegnummer'].unique():
        if pd.isna(weg): continue
        
        mask = gdf['Wegnummer'] == weg
        subset = gdf[mask]
        
        if len(subset) < 2:
            continue
            
        # Centroids berekenen
        centroids = np.array([(g.centroid.x, g.centroid.y) for g in subset.geometry])
        
        # PCA om de hoofdas van de weg te vinden
        pca = PCA(n_components=1)
        try:
            transformed = pca.fit_transform(centroids)
            # Normaliseer naar 0-1 of meters relative
            gdf.loc[mask, 'volgorde_score'] = transformed.flatten()
        except:
            pass # Te weinig punten of error
            
    return gdf

def main():
    gdf = load_and_prep_data()
    gdf = run_validations(gdf)
    gdf = generate_network_advice(gdf)
    gdf = calculate_sequencing(gdf)
    
    # Sla resultaat op (zonder de geometrie kolom als je CSV wilt voor iASSET)
    # Of behoud geometrie voor GIS controle. Hieronder CSV export.
    out_cols = [c for c in gdf.columns if c not in ['geometry', 'geometry_buffer', 'subthema_clean', 'project_clean']]
    gdf[out_cols].to_csv(OUTPUT_FILE, index=False)
    print(f"Klaar! Resultaat opgeslagen in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()