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


def test_load_iasset_data_accepts_column_aliases_and_ewkt():
    # De WKT bevat zelf ook een puntkomma door de SRID-prefix. Daarom quoten we
    # de geometrie zoals Excel/CSV-export meestal ook doet.
    csv_text = (
        "ID;WEG NUMMER;Sub thema;GPS Coördinaten;Onderhoud project\n"
        '001;N398;rijstrook;"SRID=4326;POINT (5 52)";N398-HRB-00.0-01.0\n'
    )

    result = __import__("iasset_tool.data_loader", fromlist=["load_iasset_data"]).load_iasset_data(
        input_files=(("export.csv", csv_text.encode("utf-8")),)
    )

    assert len(result.gdf) == 1
    assert result.gdf.iloc[0]["bron_id"] == "001"
    assert result.gdf.iloc[0]["Wegnummer"] == "N398"
    assert result.gdf.iloc[0]["subthema"] == "rijstrook"


def test_load_iasset_data_falls_back_to_rds_geometry():
    csv_text = (
        "id;Wegnummer;subthema;rds coordinaten;Onderhoudsproject\n"
        "abc;N398;rijstrook;POINT (160000 560000);N398-HRB-00.0-01.0\n"
    )

    result = __import__("iasset_tool.data_loader", fromlist=["load_iasset_data"]).load_iasset_data(
        input_files=(("export.csv", csv_text.encode("utf-8")),)
    )

    assert len(result.gdf) == 1
    assert round(float(result.gdf.iloc[0].geometry.x)) == 160000
    assert any("RD-geometrie" in warning for warning in result.warnings)


def test_load_iasset_data_logs_invalid_geometry_rows_without_crashing():
    csv_text = (
        "id;Wegnummer;subthema;gps coordinaten;Onderhoudsproject\n"
        "goed;N398;rijstrook;POINT (5 52);N398-HRB-00.0-01.0\n"
        "leeg;N398;rijstrook;;N398-HRB-01.0-02.0\n"
        "fout;N398;rijstrook;DIT IS GEEN WKT;N398-HRB-02.0-03.0\n"
    )

    result = __import__("iasset_tool.data_loader", fromlist=["load_iasset_data"]).load_iasset_data(
        input_files=(("export.csv", csv_text.encode("utf-8")),)
    )

    assert len(result.gdf) == 1
    assert len(result.invalid_geometry_rows) == 2
    assert set(result.invalid_geometry_rows["bron_id"]) == {"leeg", "fout"}


def test_load_iasset_data_reads_excel_with_header_not_on_first_row():
    import io
    import pandas as pd

    buffer = io.BytesIO()
    raw = pd.DataFrame(
        [
            ["iASSET export", "", "", "", ""],
            ["gegenereerd voor test", "", "", "", ""],
            ["id", "Wegnummer", "subthema", "gps coordinaten", "Onderhoudsproject"],
            ["abc", "N398", "rijstrook", "POINT (5 52)", "N398-HRB-00.0-01.0"],
        ]
    )

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        raw.to_excel(writer, index=False, header=False, sheet_name="Export")

    result = __import__("iasset_tool.data_loader", fromlist=["load_iasset_data"]).load_iasset_data(
        input_files=(("export.xlsx", buffer.getvalue()),)
    )

    assert len(result.gdf) == 1
    assert result.gdf.iloc[0]["Wegnummer"] == "N398"
    assert any("kopregel gevonden op rij 3" in warning for warning in result.warnings)


def test_load_iasset_data_reads_cp1252_csv():
    csv_text = (
        "id;Wegnummer;subthema;gps coordinaten;Onderhoudsproject;naam\n"
        "abc;N398;rijstrook;POINT (5 52);N398-HRB-00.0-01.0;Café Fryslân\n"
    )

    result = __import__("iasset_tool.data_loader", fromlist=["load_iasset_data"]).load_iasset_data(
        input_files=(("export.csv", csv_text.encode("cp1252")),)
    )

    assert len(result.gdf) == 1
    assert result.gdf.iloc[0]["naam"] == "Café Fryslân"
    assert any("encoding cp1252" in warning for warning in result.warnings)
