from iasset_tool.data_loader import parse_wkt_geometry


def test_parse_wkt_geometry_empty_value():
    geom, error = parse_wkt_geometry("")
    assert geom is None
    assert error is not None


def test_parse_wkt_geometry_invalid_value():
    geom, error = parse_wkt_geometry("DIT IS GEEN WKT")
    assert geom is None
    assert error is not None


def test_parse_wkt_geometry_valid_point():
    geom, error = parse_wkt_geometry("POINT (5 52)")
    assert geom is not None
    assert error is None


def test_load_iasset_data_from_uploaded_csv_bytes():
    csv_text = (
        "id;Wegnummer;subthema;gps coordinaten;Onderhoudsproject\n"
        "abc;N398;rijstrook;POINT (5 52);N398-HRB-00.0-01.0\n"
    )
    result = __import__("iasset_tool.data_loader", fromlist=["load_iasset_data"]).load_iasset_data(
        input_files=(("export.csv", csv_text.encode("utf-8")),)
    )

    assert len(result.gdf) == 1
    assert result.gdf.iloc[0]["Wegnummer"] == "N398"
    assert "bron_id" in result.gdf.columns


def test_load_iasset_data_from_uploaded_excel_bytes():
    import io
    import pandas as pd

    buffer = io.BytesIO()
    df = pd.DataFrame(
        {
            "id": ["abc"],
            "Wegnummer": ["N398"],
            "subthema": ["rijstrook"],
            "gps coordinaten": ["POINT (5 52)"],
            "Onderhoudsproject": ["N398-HRB-00.0-01.0"],
        }
    )

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"niet_relevant": [1]}).to_excel(writer, index=False, sheet_name="Info")
        df.to_excel(writer, index=False, sheet_name="Paspoort")

    result = __import__("iasset_tool.data_loader", fromlist=["load_iasset_data"]).load_iasset_data(
        input_files=(("export.xlsx", buffer.getvalue()),)
    )

    assert len(result.gdf) == 1
    assert result.gdf.iloc[0]["Wegnummer"] == "N398"
    assert any("tabblad 'Paspoort'" in warning for warning in result.warnings)
