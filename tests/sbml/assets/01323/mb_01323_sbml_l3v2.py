from mxlpy import InitialAssignment, Model


def init_a_sbml() -> float:
    return 1.0


def init_a_truncated() -> float:
    return 1.00000003487131


def init_a_1865() -> float:
    return 11.9558792387716


def init_a_1873() -> float:
    return 1.826592661479


def init_a_1890a() -> float:
    return 1.16237714821391


def init_a_1890b() -> float:
    return 1.00960758016294


def init_a_1901() -> float:
    return 1.02289189042824


def init_a_1903() -> float:
    return 1.54430106834134


def init_a_1904() -> float:
    return 1.44466874135157


def init_a_1908() -> float:
    return 0.996323269897636


def init_a_1909() -> float:
    return 1.02289189042824


def init_a_1914a() -> float:
    return 0.996323269897636


def init_a_1914b() -> float:
    return 0.981378420849171


def init_a_1915() -> float:
    return 1.00628650259661


def init_a_1917() -> float:
    return 1.00695071810988


def init_a_1923() -> float:
    return 0.979717882066008


def init_a_1924() -> float:
    return 0.996987485410901


def init_a_1929() -> float:
    return 1.0070171396612


def init_a_1931() -> float:
    return 0.999478293585645


def init_a_1941() -> float:
    return 1.0001142799396


def init_a_1945() -> float:
    return 1.00020560957267


def init_a_1948() -> float:
    return 1.00017571987457


def init_a_1949() -> float:
    return 1.00031354459358


def init_a_1951() -> float:
    return 1.000547680562


def init_a_1965a() -> float:
    return 0.999991067961885


def init_a_1965b() -> float:
    return 1.00006280323732


def init_a_1973() -> float:
    return 0.999983927645118


def init_a_1974() -> float:
    return 0.999992114101319


def init_a_1987a() -> float:
    return 0.999998706440288


def init_a_1987b() -> float:
    return 0.999999154785759


def init_a_1992() -> float:
    return 0.999999088364208


def init_a_1994() -> float:
    return 0.999999354050413


def init_a_1995() -> float:
    return 0.999999121574984


def init_a_1999() -> float:
    return 1.00000219357173


def init_a_2000() -> float:
    return 1.00000003321078


def init_a_2001() -> float:
    return 0.9999986898349


def init_a_2002() -> float:
    return 0.999999951844375


def init_a_2003() -> float:
    return 0.99999892231033


def init_a_2006() -> float:
    return 1.0


def init_a_2011() -> float:
    return 0.999999916973061


def init_a_2015() -> float:
    return 0.999999845071732


def create_model() -> Model:
    return (
        Model()
        .add_parameter(
            "a_sbml",
            value=InitialAssignment(fn=init_a_sbml, args=[]),
        )
        .add_parameter(
            "a_truncated",
            value=InitialAssignment(fn=init_a_truncated, args=[]),
        )
        .add_parameter(
            "a_1865",
            value=InitialAssignment(fn=init_a_1865, args=[]),
        )
        .add_parameter(
            "a_1873",
            value=InitialAssignment(fn=init_a_1873, args=[]),
        )
        .add_parameter(
            "a_1890a",
            value=InitialAssignment(fn=init_a_1890a, args=[]),
        )
        .add_parameter(
            "a_1890b",
            value=InitialAssignment(fn=init_a_1890b, args=[]),
        )
        .add_parameter(
            "a_1901",
            value=InitialAssignment(fn=init_a_1901, args=[]),
        )
        .add_parameter(
            "a_1903",
            value=InitialAssignment(fn=init_a_1903, args=[]),
        )
        .add_parameter(
            "a_1904",
            value=InitialAssignment(fn=init_a_1904, args=[]),
        )
        .add_parameter(
            "a_1908",
            value=InitialAssignment(fn=init_a_1908, args=[]),
        )
        .add_parameter(
            "a_1909",
            value=InitialAssignment(fn=init_a_1909, args=[]),
        )
        .add_parameter(
            "a_1914a",
            value=InitialAssignment(fn=init_a_1914a, args=[]),
        )
        .add_parameter(
            "a_1914b",
            value=InitialAssignment(fn=init_a_1914b, args=[]),
        )
        .add_parameter(
            "a_1915",
            value=InitialAssignment(fn=init_a_1915, args=[]),
        )
        .add_parameter(
            "a_1917",
            value=InitialAssignment(fn=init_a_1917, args=[]),
        )
        .add_parameter(
            "a_1923",
            value=InitialAssignment(fn=init_a_1923, args=[]),
        )
        .add_parameter(
            "a_1924",
            value=InitialAssignment(fn=init_a_1924, args=[]),
        )
        .add_parameter(
            "a_1929",
            value=InitialAssignment(fn=init_a_1929, args=[]),
        )
        .add_parameter(
            "a_1931",
            value=InitialAssignment(fn=init_a_1931, args=[]),
        )
        .add_parameter(
            "a_1941",
            value=InitialAssignment(fn=init_a_1941, args=[]),
        )
        .add_parameter(
            "a_1945",
            value=InitialAssignment(fn=init_a_1945, args=[]),
        )
        .add_parameter(
            "a_1948",
            value=InitialAssignment(fn=init_a_1948, args=[]),
        )
        .add_parameter(
            "a_1949",
            value=InitialAssignment(fn=init_a_1949, args=[]),
        )
        .add_parameter(
            "a_1951",
            value=InitialAssignment(fn=init_a_1951, args=[]),
        )
        .add_parameter(
            "a_1965a",
            value=InitialAssignment(fn=init_a_1965a, args=[]),
        )
        .add_parameter(
            "a_1965b",
            value=InitialAssignment(fn=init_a_1965b, args=[]),
        )
        .add_parameter(
            "a_1973",
            value=InitialAssignment(fn=init_a_1973, args=[]),
        )
        .add_parameter(
            "a_1974",
            value=InitialAssignment(fn=init_a_1974, args=[]),
        )
        .add_parameter(
            "a_1987a",
            value=InitialAssignment(fn=init_a_1987a, args=[]),
        )
        .add_parameter(
            "a_1987b",
            value=InitialAssignment(fn=init_a_1987b, args=[]),
        )
        .add_parameter(
            "a_1992",
            value=InitialAssignment(fn=init_a_1992, args=[]),
        )
        .add_parameter(
            "a_1994",
            value=InitialAssignment(fn=init_a_1994, args=[]),
        )
        .add_parameter(
            "a_1995",
            value=InitialAssignment(fn=init_a_1995, args=[]),
        )
        .add_parameter(
            "a_1999",
            value=InitialAssignment(fn=init_a_1999, args=[]),
        )
        .add_parameter(
            "a_2000",
            value=InitialAssignment(fn=init_a_2000, args=[]),
        )
        .add_parameter(
            "a_2001",
            value=InitialAssignment(fn=init_a_2001, args=[]),
        )
        .add_parameter(
            "a_2002",
            value=InitialAssignment(fn=init_a_2002, args=[]),
        )
        .add_parameter(
            "a_2003",
            value=InitialAssignment(fn=init_a_2003, args=[]),
        )
        .add_parameter(
            "a_2006",
            value=InitialAssignment(fn=init_a_2006, args=[]),
        )
        .add_parameter(
            "a_2011",
            value=InitialAssignment(fn=init_a_2011, args=[]),
        )
        .add_parameter(
            "a_2015",
            value=InitialAssignment(fn=init_a_2015, args=[]),
        )
    )
