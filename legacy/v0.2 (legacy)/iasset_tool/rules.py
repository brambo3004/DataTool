"""
Datakwaliteitsregels.

Deze module bevat alleen de controles. De UI bepaalt daarna hoe meldingen
getoond, genegeerd of gecorrigeerd worden.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import networkx as nx

from .config import BACKBONE_TYPES
from .domain import (
    is_maintenance_project_exempt,
    is_project_value_empty,
    maintenance_project_exemption_reason,
)
from .utils import normalize_text


def _clean_list(values: list[str]) -> set[str]:
    """Maak een set met lowercase domeinwaarden."""
    return {normalize_text(value) for value in values}


def _exemption_mask(gdf: gpd.GeoDataFrame):
    """
    Bepaal per rij of het object is uitgezonderd van onderhoudsprojectplicht.

    We doen dit rijgewijs omdat de uitzondering "oorspronkelijke BGT-data" niet
    altijd in dezelfde kolom staat.
    """
    return gdf.apply(is_maintenance_project_exempt, axis=1)


def check_rules(gdf: gpd.GeoDataFrame, graph: nx.Graph | None = None) -> list[dict[str, Any]]:
    """
    Controleer objecten op datakwaliteitsissues.

    Regels:
    1. onderhoudsprojectplichtige objecten zonder Onderhoudsproject;
    2. uitgezonderde objecten die tóch een Onderhoudsproject hebben;
    3. secundaire objecten die los zweven van hoofdobjecten;
    4. objecten met projectnaam die geïsoleerd liggen t.o.v. dat project.

    De uitzonderingslogica volgt het werkproces Grijs: bepaalde subthema's en
    objecten met oorspronkelijke BGT-data krijgen geen onderhoudsproject.
    """
    violations: list[dict[str, Any]] = []

    if gdf is None or gdf.empty:
        return violations

    if "subthema_clean" not in gdf.columns:
        return violations

    if "Onderhoudsproject" not in gdf.columns:
        gdf = gdf.copy()
        gdf["Onderhoudsproject"] = ""

    backbone_clean = _clean_list(BACKBONE_TYPES)
    project_empty = gdf["Onderhoudsproject"].apply(is_project_value_empty)
    is_exempt = _exemption_mask(gdf)

    # 1. Administratieve controle: object hoort een project te hebben, maar mist dit.
    missing_mask = ~is_exempt & project_empty

    for idx, row in gdf[missing_mask].iterrows():
        violations.append(
            {
                "type": "error",
                "id": idx,
                "subthema": row.get("subthema", ""),
                "msg": "Mist verplicht onderhoudsproject",
                "missing_cols": ["Onderhoudsproject"],
            }
        )

    # 2. Administratieve controle: object is uitgezonderd, maar heeft tóch een project.
    # Dit sluit aan op het werkprocesfilter 'objecten die onterecht een onderhoudsproject hebben'.
    wrong_project_mask = is_exempt & ~project_empty

    for idx, row in gdf[wrong_project_mask].iterrows():
        reason = maintenance_project_exemption_reason(row)
        msg = "Heeft onderhoudsproject terwijl dit object is uitgezonderd"
        if reason:
            msg = f"{msg}: {reason}"

        violations.append(
            {
                "type": "warning",
                "id": idx,
                "subthema": row.get("subthema", ""),
                "msg": msg,
                "missing_cols": ["Onderhoudsproject"],
            }
        )

    # 3. Ruimtelijke controles.
    if graph is None:
        return violations

    for node_id in graph.nodes:
        if node_id not in gdf.index:
            continue

        row = gdf.loc[node_id]
        sub = row.get("subthema_clean", "")

        if sub in backbone_clean or is_maintenance_project_exempt(row):
            continue

        connected_to_backbone = False
        for neighbor in graph.neighbors(node_id):
            if neighbor not in gdf.index:
                continue
            neighbor_sub = gdf.loc[neighbor].get("subthema_clean", "")
            if neighbor_sub in backbone_clean:
                connected_to_backbone = True
                break

        if not connected_to_backbone:
            violations.append(
                {
                    "type": "warning",
                    "id": node_id,
                    "subthema": row.get("subthema", ""),
                    "msg": "Zwevend secundair object: grenst nergens aan een hoofdroute (Rijbaan/Fiets/etc).",
                    "missing_cols": [],
                }
            )

    # Uitgezonderde objecten met projectnaam hebben hierboven al een gerichte
    # melding gekregen. We slaan ze hier over om dubbele, verwarrende meldingen te voorkomen.
    has_project_mask = ~project_empty & ~is_exempt

    for idx, row in gdf[has_project_mask].iterrows():
        if idx not in graph:
            continue

        project_name = row.get("Onderhoudsproject", "")
        project_name = "" if is_project_value_empty(project_name) else str(project_name).strip()
        my_sub = row.get("subthema_clean", "")

        neighbors = list(graph.neighbors(idx))
        match_found = False

        for neighbor in neighbors:
            if neighbor not in gdf.index:
                continue

            neighbor_project = gdf.loc[neighbor].get("Onderhoudsproject", "")
            if not is_project_value_empty(neighbor_project) and str(neighbor_project).strip() == project_name:
                match_found = True
                break

        if not match_found:
            violations.append(
                {
                    "type": "warning",
                    "id": idx,
                    "subthema": row.get("subthema", ""),
                    "msg": f"Geïsoleerd t.o.v. project '{project_name}'. Geen directe buren met dit project.",
                    "missing_cols": ["Onderhoudsproject"],
                }
            )
            continue

        if my_sub not in backbone_clean:
            connected_to_project_backbone = False

            for neighbor in neighbors:
                if neighbor not in gdf.index:
                    continue

                neighbor_sub = gdf.loc[neighbor].get("subthema_clean", "")
                neighbor_project = gdf.loc[neighbor].get("Onderhoudsproject", "")

                if (
                    not is_project_value_empty(neighbor_project)
                    and str(neighbor_project).strip() == project_name
                    and neighbor_sub in backbone_clean
                ):
                    connected_to_project_backbone = True
                    break

            if not connected_to_project_backbone:
                violations.append(
                    {
                        "type": "info",
                        "id": idx,
                        "subthema": row.get("subthema", ""),
                        "msg": (
                            f"Verbonden met '{project_name}', maar raakt niet direct "
                            "de hoofdrijbaan/fietspad van dit project."
                        ),
                        "missing_cols": [],
                    }
                )

    return violations
