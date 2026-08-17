import pandas as pd

from iasset_tool.nwegendocument_export import build_nwegendocument_concept_rows
from iasset_tool.visible_complexes import (
    build_visible_maintenance_complex_table,
    visible_maintenance_complexes_for_nwegendocument,
)


def test_zichtbare_complexlaag_voegt_dubbele_projectnamen_samen() -> None:
    """Dubbele projectnamen worden één zichtbare werkregel met ruwe ids eronder."""
    advisor = pd.DataFrame(
        [
            {
                "voorstel_id": "r1",
                "onderhoudsproject_voorgesteld": "N354-HRB-12.8-12.9",
                "project_type": "HRB",
                "project_family": "HRB",
                "fysiek_begin_km": 12.70,
                "fysiek_eind_km": 12.80,
                "fysiek_begin_m": 100.0,
                "fysiek_eind_m": 200.0,
                "fysiek_lengte_m": 100.0,
                "voorstelcategorie": "regulier projectvoorstel",
                "iassetvergelijking": "aandacht",
                "werkadvies": "Controleer iASSET-verschil",
                "werklijst_reden": "Voorgesteld onderhoudsproject wijkt af van bestaande iASSET-indeling.",
            },
            {
                "voorstel_id": "r2",
                "onderhoudsproject_voorgesteld": "N354-HRB-12.8-12.9",
                "project_type": "HRB",
                "project_family": "HRB",
                "fysiek_begin_km": 12.78,
                "fysiek_eind_km": 12.90,
                "fysiek_begin_m": 180.0,
                "fysiek_eind_m": 300.0,
                "fysiek_lengte_m": 120.0,
                "voorstelcategorie": "regulier projectvoorstel",
                "iassetvergelijking": "aandacht",
            },
        ]
    )

    visible = build_visible_maintenance_complex_table(advisor)

    assert len(visible) == 1
    row = visible.iloc[0]
    assert row["aantal_ruwe_voorstellen"] == 2
    assert row["bron_voorstel_ids"] == "r1; r2"
    assert row["zichtbare_klasse"] == "controlecluster - dubbele projectnaam samengevoegd"
    assert bool(row["in_concept_nwegendocument"]) is True


def test_kort_regulier_segment_blijft_controlepunt_buiten_conceptwerkblad() -> None:
    """Zeer korte reguliere technische segmenten worden niet als normaal complex geëxporteerd."""
    advisor = pd.DataFrame(
        [
            {
                "voorstel_id": "kort",
                "onderhoudsproject_voorgesteld": "N354-HRB-13.6-13.7",
                "project_type": "HRB",
                "project_family": "HRB",
                "fysiek_begin_km": 13.60,
                "fysiek_eind_km": 13.664,
                "fysiek_lengte_m": 67.0,
                "voorstelcategorie": "regulier projectvoorstel",
                "iassetvergelijking": "aandacht",
            }
        ]
    )

    visible = build_visible_maintenance_complex_table(advisor)
    for_export = visible_maintenance_complexes_for_nwegendocument(visible)

    assert visible.iloc[0]["zichtbare_klasse"] == "kort technisch segment"
    assert bool(visible.iloc[0]["in_concept_nwegendocument"]) is False
    assert for_export.empty


def test_nwegendocument_export_gebruikt_bron_voorstel_ids_voor_objectcontext() -> None:
    """Een zichtbare samengevoegde regel kan objectcontext uit meerdere ruwe voorstellen lezen."""
    advisor = pd.DataFrame(
        [
            {
                "voorstel_id": "r1",
                "onderhoudsproject_voorgesteld": "N354-HRB-12.8-12.9",
                "project_type": "HRB",
                "project_family": "HRB",
                "fysiek_begin_km": 12.70,
                "fysiek_eind_km": 12.80,
                "fysiek_begin_m": 100.0,
                "fysiek_eind_m": 200.0,
                "fysiek_lengte_m": 100.0,
                "voorstelcategorie": "regulier projectvoorstel",
            },
            {
                "voorstel_id": "r2",
                "onderhoudsproject_voorgesteld": "N354-HRB-12.8-12.9",
                "project_type": "HRB",
                "project_family": "HRB",
                "fysiek_begin_km": 12.80,
                "fysiek_eind_km": 12.90,
                "fysiek_begin_m": 200.0,
                "fysiek_eind_m": 300.0,
                "fysiek_lengte_m": 100.0,
                "voorstelcategorie": "regulier projectvoorstel",
            },
        ]
    )
    objects = pd.DataFrame(
        [
            {"voorstel_id": "r1", "Besteknummer": "A", "Soort deklaag specifiek": "SMA"},
            {"voorstel_id": "r2", "Besteknummer": "B", "Soort deklaag specifiek": "SMA"},
        ]
    )

    visible = build_visible_maintenance_complex_table(advisor, objects)
    concept_rows = build_nwegendocument_concept_rows(visible, objects)

    assert len(concept_rows) == 1
    assert concept_rows.iloc[0]["besteknummer"] == "A / B"
