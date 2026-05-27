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
