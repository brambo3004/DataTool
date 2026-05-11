\
"""
Datakwaliteitsregels.

Deze module bevat alleen de controles. De UI bepaalt daarna hoe meldingen
getoond, genegeerd of gecorrigeerd worden.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import networkx as nx
import pandas as pd

from .config import BACKBONE_TYPES, SUBTHEMA_EXCEPTIONS
from .utils import clean_display_value


def _clean_list(values: list[str]) -> set[str]:
    """Maak een set met lowercase domeinwaarden."""
    return {str(value).strip().lower() for value in values}


def _project_is_empty(value: Any) -> bool:
    """Controleer of Onderhoudsproject leeg is."""
    return clean_display_value(value) == ""


def check_rules(gdf: gpd.GeoDataFrame, graph: nx.Graph | None = None) -> list[dict[str, Any]]:
    """
    Controleer objecten op datakwaliteitsissues.

    De regels zijn bewust gelijk gehouden aan de bestaande app:
    1. verplichte onderhoudsprojecten ontbreken;
    2. secundaire objecten zweven los van hoofdobjecten;
    3. objecten met projectnaam liggen geïsoleerd t.o.v. dat project.
    """
    violations: list[dict[str, Any]] = []

    if gdf is None or gdf.empty:
        return violations

    exceptions_clean = _clean_list(SUBTHEMA_EXCEPTIONS)
    backbone_clean = _clean_list(BACKBONE_TYPES)

    if "subthema_clean" not in gdf.columns:
        return violations

    if "Onderhoudsproject" not in gdf.columns:
        gdf = gdf.copy()
        gdf["Onderhoudsproject"] = ""

    # 1. Administratieve controle.
    missing_mask = (
        ~gdf["subthema_clean"].isin(exceptions_clean)
        & gdf["Onderhoudsproject"].apply(_project_is_empty)
    )

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

    # 2. Ruimtelijke controles.
    if graph is None:
        return violations

    for node_id in graph.nodes:
        if node_id not in gdf.index:
            continue

        sub = gdf.loc[node_id, "subthema_clean"]

        if sub in backbone_clean or sub in exceptions_clean:
            continue

        connected_to_backbone = False
        for neighbor in graph.neighbors(node_id):
            if neighbor not in gdf.index:
                continue
            neighbor_sub = gdf.loc[neighbor, "subthema_clean"]
            if neighbor_sub in backbone_clean:
                connected_to_backbone = True
                break

        if not connected_to_backbone:
            violations.append(
                {
                    "type": "warning",
                    "id": node_id,
                    "subthema": gdf.loc[node_id].get("subthema", ""),
                    "msg": "Zwevend secundair object: grenst nergens aan een hoofdroute (Rijbaan/Fiets/etc).",
                    "missing_cols": [],
                }
            )

    has_project_mask = ~gdf["Onderhoudsproject"].apply(_project_is_empty)

    for idx, row in gdf[has_project_mask].iterrows():
        if idx not in graph:
            continue

        project_name = clean_display_value(row.get("Onderhoudsproject", ""))
        my_sub = row.get("subthema_clean", "")

        neighbors = list(graph.neighbors(idx))
        match_found = False

        for neighbor in neighbors:
            if neighbor not in gdf.index:
                continue

            neighbor_project = clean_display_value(gdf.loc[neighbor].get("Onderhoudsproject", ""))
            if neighbor_project == project_name:
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

                neighbor_sub = gdf.loc[neighbor, "subthema_clean"]
                neighbor_project = clean_display_value(gdf.loc[neighbor].get("Onderhoudsproject", ""))

                if neighbor_project == project_name and neighbor_sub in backbone_clean:
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
