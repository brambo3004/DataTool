"""
Kleine performance-hulpen voor de Streamlit-app.

Deze module is bewust simpel gehouden: we meten alleen hoe lang zware stappen
duuren en geven dat terug als tabeldata. De functies kennen geen Streamlit, zodat
ze ook in tests of scripts bruikbaar blijven.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, MutableMapping, TypeVar

import pandas as pd

T = TypeVar("T")


def measure_step(
    performance_log: MutableMapping[str, float],
    label: str,
    function: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """
    Voer een functie uit en registreer de verstreken tijd in seconden.

    De app gebruikt dit voor zware stappen zoals regelchecks, netwerkopbouw,
    PDOK-opvragingen en kaartopbouw. Door de meetlaag centraal te houden hoeven
    we de domeinfuncties zelf niet met Streamlit-code te vervuilen.
    """
    start = perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        performance_log[label] = perf_counter() - start


def performance_dataframe(performance_log: MutableMapping[str, float] | None) -> pd.DataFrame:
    """
    Maak een compacte tabel van de gemeten stappen.

    Lege of ontbrekende logs leveren een lege DataFrame op, zodat de UI zonder
    extra controles kan bepalen of er iets te tonen is.
    """
    if not performance_log:
        return pd.DataFrame(columns=["Stap", "Seconden"])

    rows = [
        {"Stap": label, "Seconden": round(float(seconds), 3)}
        for label, seconds in performance_log.items()
    ]
    return pd.DataFrame(rows)
