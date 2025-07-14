from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from mxlpy import Simulator
from mxlpy.integrators.int_scipy import Scipy
from mxlpy.sbml import read
from mxlpy.types import unwrap

if TYPE_CHECKING:
    from mxlpy.model import Model

try:
    import assimulo  # type: ignore  # noqa: F401

    ASSIMULO_FLAG = True
except ImportError:
    ASSIMULO_FLAG = False

ASSET_PATH = Path("tests") / "sbml" / "assets"


def get_simulation_settings(path: Path, prefix: str) -> dict:
    sim_settings = {}
    settings_file = path / f"{prefix}-settings.txt"
    with settings_file.open() as f:
        for line in f:
            i = line.strip().split(": ")
            if i[0] == "absolute":
                sim_settings["atol"] = float(i[1])
            elif i[0] == "relative":
                sim_settings["rtol"] = float(i[1])
            elif i[0] == "amount":
                sim_settings["result-ids"] = [j.strip() for j in i[1].split(",")]
    return sim_settings


def add_constant_species_to_results(
    model: Model, expected: pd.DataFrame, result: pd.DataFrame
) -> pd.DataFrame:
    """Adds constant species from the expected DataFrame to the result DataFrame.

    This function iterates over the columns in the expected DataFrame that are not present in the result DataFrame.
    For each missing column, it retrieves the corresponding species value from the model and creates a new Series
    with that constant value, matching the length of the expected DataFrame. The new Series is then concatenated
    to the result DataFrame.

    Args:
        model: The model from which to retrieve the constant species values.
        expected: The DataFrame containing the expected results, including constant species.
        result: The DataFrame to which the constant species will be added.

    Returns:
        pd.DataFrame: The updated result DataFrame with the constant species added.

    """
    args = model.get_args()
    for name in expected.columns.difference(result.columns):
        species = args[name]
        species = pd.Series(
            np.ones(len(expected.index)) * species,
            index=expected.index,
            name=name,
        )
        result = pd.concat([result, species], axis=1)
    return result


def get_files(test: int) -> tuple[Model, dict, pd.DataFrame]:
    prefix = f"{test:05d}"
    path = ASSET_PATH / prefix
    sim_settings = get_simulation_settings(path=path, prefix=prefix)
    expected = pd.read_csv(path / f"{prefix}-results.csv", index_col=0).astype(float)
    expected.columns = [i.strip() for i in expected.columns]
    return read(file=path / f"{prefix}-sbml-l3v2.xml"), sim_settings, expected


def routine(test: int) -> bool:
    m, sim_settings, expected = get_files(test=test)

    # Make them a bit harder, such that we guarantee we are getting the required ones
    result = unwrap(
        Simulator(
            m,
            integrator=partial(
                Scipy,
                atol=sim_settings["atol"] / 100,
                rtol=sim_settings["rtol"] / 100,
            ),  # type: ignore
        )
        .simulate_time_course(expected.index)  # type: ignore
        .get_result()
    ).get_combined()

    if result is None:
        pytest.fail("Simulation failed")

    result = add_constant_species_to_results(m, expected, result)
    common = list(expected.columns.intersection(result.columns))
    return np.testing.assert_allclose(
        result.loc[:, common],
        expected.loc[:, common],
        rtol=max(sim_settings["rtol"], 1e-2),
        atol=max(sim_settings["atol"], 1e-4),
    )  # type: ignore


def test_00001() -> None:
    routine(test=1)


def test_00002() -> None:
    routine(test=2)


def test_00003() -> None:
    routine(test=3)


def test_00004() -> None:
    routine(test=4)


def test_00005() -> None:
    routine(test=5)


def test_00006() -> None:
    routine(test=6)


def test_00007() -> None:
    routine(test=7)


def test_00008() -> None:
    routine(test=8)


def test_00009() -> None:
    routine(test=9)


def test_00010() -> None:
    routine(test=10)


def test_00011() -> None:
    routine(test=11)


def test_00012() -> None:
    routine(test=12)


def test_00013() -> None:
    routine(test=13)


def test_00014() -> None:
    routine(test=14)


def test_00015() -> None:
    routine(test=15)


def test_00016() -> None:
    routine(test=16)


def test_00017() -> None:
    routine(test=17)


def test_00018() -> None:
    routine(test=18)


def test_00019() -> None:
    routine(test=19)


def test_00020() -> None:
    routine(test=20)


def test_00021() -> None:
    routine(test=21)


def test_00022() -> None:
    routine(test=22)


def test_00023() -> None:
    routine(test=23)


def test_00024() -> None:
    routine(test=24)


def test_00025() -> None:
    routine(test=25)


def test_00026() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=26)


def test_00027() -> None:
    routine(test=27)


def test_00028() -> None:
    routine(test=28)


def test_00029() -> None:
    routine(test=29)


def test_00030() -> None:
    routine(test=30)


def test_00031() -> None:
    routine(test=31)


def test_00032() -> None:
    routine(test=32)


def test_00033() -> None:
    routine(test=33)


def test_00034() -> None:
    routine(test=34)


def test_00035() -> None:
    routine(test=35)


def test_00036() -> None:
    routine(test=36)


def test_00037() -> None:
    routine(test=37)


def test_00038() -> None:
    routine(test=38)


def test_00039() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=39)


def test_00040() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=40)


def test_00041() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=41)


def test_00042() -> None:
    routine(test=42)


def test_00043() -> None:
    routine(test=43)


def test_00044() -> None:
    routine(test=44)


def test_00045() -> None:
    routine(test=45)


def test_00046() -> None:
    routine(test=46)


def test_00047() -> None:
    routine(test=47)


def test_00048() -> None:
    routine(test=48)


def test_00049() -> None:
    routine(test=49)


def test_00050() -> None:
    routine(test=50)


def test_00051() -> None:
    routine(test=51)


def test_00052() -> None:
    routine(test=52)


def test_00053() -> None:
    routine(test=53)


def test_00054() -> None:
    routine(test=54)


def test_00055() -> None:
    routine(test=55)


def test_00056() -> None:
    routine(test=56)


def test_00057() -> None:
    routine(test=57)


def test_00058() -> None:
    routine(test=58)


def test_00060() -> None:
    routine(test=60)


def test_00061() -> None:
    routine(test=61)


def test_00062() -> None:
    routine(test=62)


def test_00063() -> None:
    routine(test=63)


def test_00064() -> None:
    routine(test=64)


def test_00065() -> None:
    routine(test=65)


def test_00066() -> None:
    routine(test=66)


def test_00067() -> None:
    routine(test=67)


def test_00071() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=71)


def test_00072() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=72)


def test_00073() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=73)


def test_00074() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=74)


def test_00075() -> None:
    routine(test=75)


def test_00076() -> None:
    routine(test=76)


def test_00077() -> None:
    routine(test=77)


def test_00078() -> None:
    routine(test=78)


def test_00079() -> None:
    routine(test=79)


def test_00080() -> None:
    routine(test=80)


def test_00081() -> None:
    routine(test=81)


def test_00082() -> None:
    routine(test=82)


def test_00083() -> None:
    routine(test=83)


def test_00084() -> None:
    routine(test=84)


def test_00085() -> None:
    routine(test=85)


def test_00086() -> None:
    routine(test=86)


def test_00087() -> None:
    routine(test=87)


def test_00088() -> None:
    routine(test=88)


def test_00089() -> None:
    routine(test=89)


def test_00090() -> None:
    routine(test=90)


def test_00091() -> None:
    routine(test=91)


def test_00092() -> None:
    routine(test=92)


def test_00093() -> None:
    routine(test=93)


def test_00094() -> None:
    routine(test=94)


def test_00095() -> None:
    routine(test=95)


def test_00096() -> None:
    routine(test=96)


def test_00097() -> None:
    routine(test=97)


def test_00098() -> None:
    routine(test=98)


def test_00099() -> None:
    routine(test=99)


def test_00100() -> None:
    routine(test=100)


def test_00101() -> None:
    routine(test=101)


def test_00102() -> None:
    routine(test=102)


def test_00103() -> None:
    routine(test=103)


def test_00104() -> None:
    routine(test=104)


def test_00105() -> None:
    routine(test=105)


def test_00106() -> None:
    routine(test=106)


def test_00107() -> None:
    routine(test=107)


def test_00108() -> None:
    routine(test=108)


def test_00109() -> None:
    routine(test=109)


def test_00110() -> None:
    routine(test=110)


def test_00111() -> None:
    routine(test=111)


def test_00112() -> None:
    routine(test=112)


def test_00113() -> None:
    routine(test=113)


def test_00114() -> None:
    routine(test=114)


def test_00115() -> None:
    routine(test=115)


def test_00116() -> None:
    routine(test=116)


def test_00117() -> None:
    routine(test=117)


def test_00118() -> None:
    routine(test=118)


def test_00119() -> None:
    routine(test=119)


def test_00120() -> None:
    routine(test=120)


def test_00121() -> None:
    routine(test=121)


def test_00122() -> None:
    routine(test=122)


def test_00123() -> None:
    routine(test=123)


def test_00124() -> None:
    routine(test=124)


def test_00125() -> None:
    routine(test=125)


def test_00126() -> None:
    routine(test=126)


def test_00127() -> None:
    routine(test=127)


def test_00128() -> None:
    routine(test=128)


def test_00132() -> None:
    routine(test=132)


def test_00133() -> None:
    routine(test=133)


def test_00135() -> None:
    routine(test=135)


def test_00136() -> None:
    routine(test=136)


def test_00137() -> None:
    routine(test=137)


def test_00138() -> None:
    routine(test=138)


def test_00139() -> None:
    routine(test=139)


def test_00140() -> None:
    routine(test=140)


def test_00141() -> None:
    routine(test=141)


def test_00142() -> None:
    routine(test=142)


def test_00143() -> None:
    routine(test=143)


def test_00144() -> None:
    routine(test=144)


def test_00145() -> None:
    routine(test=145)


def test_00146() -> None:
    routine(test=146)


def test_00147() -> None:
    routine(test=147)


def test_00148() -> None:
    routine(test=148)


def test_00149() -> None:
    routine(test=149)


def test_00150() -> None:
    routine(test=150)


def test_00151() -> None:
    routine(test=151)


def test_00152() -> None:
    routine(test=152)


def test_00153() -> None:
    routine(test=153)


def test_00154() -> None:
    routine(test=154)


def test_00155() -> None:
    routine(test=155)


def test_00156() -> None:
    routine(test=156)


def test_00157() -> None:
    routine(test=157)


def test_00158() -> None:
    routine(test=158)


def test_00159() -> None:
    routine(test=159)


def test_00160() -> None:
    routine(test=160)


def test_00161() -> None:
    routine(test=161)


def test_00162() -> None:
    routine(test=162)


def test_00163() -> None:
    routine(test=163)


def test_00164() -> None:
    routine(test=164)


def test_00165() -> None:
    routine(test=165)


def test_00166() -> None:
    routine(test=166)


def test_00167() -> None:
    routine(test=167)


def test_00168() -> None:
    routine(test=168)


def test_00169() -> None:
    routine(test=169)


def test_00170() -> None:
    routine(test=170)


def test_00171() -> None:
    routine(test=171)


def test_00172() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=172)


def test_00173() -> None:
    routine(test=173)


def test_00174() -> None:
    routine(test=174)


def test_00175() -> None:
    routine(test=175)


def test_00176() -> None:
    routine(test=176)


def test_00177() -> None:
    routine(test=177)


def test_00178() -> None:
    routine(test=178)


def test_00179() -> None:
    routine(test=179)


def test_00180() -> None:
    routine(test=180)


def test_00181() -> None:
    routine(test=181)


def test_00182() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=182)


def test_00183() -> None:
    routine(test=183)


def test_00184() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=184)


def test_00185() -> None:
    routine(test=185)


def test_00186() -> None:
    routine(test=186)


def test_00187() -> None:
    routine(test=187)


def test_00188() -> None:
    routine(test=188)


def test_00189() -> None:
    routine(test=189)


def test_00190() -> None:
    routine(test=190)


def test_00191() -> None:
    routine(test=191)


def test_00192() -> None:
    routine(test=192)


def test_00193() -> None:
    routine(test=193)


def test_00194() -> None:
    routine(test=194)


def test_00195() -> None:
    routine(test=195)


def test_00196() -> None:
    routine(test=196)


def test_00197() -> None:
    routine(test=197)


def test_00198() -> None:
    routine(test=198)


def test_00199() -> None:
    routine(test=199)


def test_00200() -> None:
    routine(test=200)


def test_00201() -> None:
    routine(test=201)


def test_00202() -> None:
    routine(test=202)


def test_00203() -> None:
    routine(test=203)


def test_00204() -> None:
    routine(test=204)


def test_00205() -> None:
    routine(test=205)


def test_00206() -> None:
    routine(test=206)


def test_00207() -> None:
    routine(test=207)


def test_00208() -> None:
    routine(test=208)


def test_00209() -> None:
    routine(test=209)


def test_00210() -> None:
    routine(test=210)


def test_00211() -> None:
    routine(test=211)


def test_00212() -> None:
    routine(test=212)


def test_00213() -> None:
    routine(test=213)


def test_00214() -> None:
    routine(test=214)


def test_00215() -> None:
    routine(test=215)


def test_00216() -> None:
    routine(test=216)


def test_00217() -> None:
    routine(test=217)


def test_00218() -> None:
    routine(test=218)


def test_00219() -> None:
    routine(test=219)


def test_00220() -> None:
    routine(test=220)


def test_00221() -> None:
    routine(test=221)


def test_00222() -> None:
    routine(test=222)


def test_00223() -> None:
    routine(test=223)


def test_00224() -> None:
    routine(test=224)


def test_00225() -> None:
    routine(test=225)


def test_00226() -> None:
    routine(test=226)


def test_00227() -> None:
    routine(test=227)


def test_00228() -> None:
    routine(test=228)


def test_00229() -> None:
    routine(test=229)


def test_00230() -> None:
    routine(test=230)


def test_00231() -> None:
    routine(test=231)


def test_00232() -> None:
    routine(test=232)


def test_00233() -> None:
    routine(test=233)


def test_00234() -> None:
    routine(test=234)


def test_00235() -> None:
    routine(test=235)


def test_00236() -> None:
    routine(test=236)


def test_00237() -> None:
    routine(test=237)


def test_00238() -> None:
    routine(test=238)


def test_00239() -> None:
    routine(test=239)


def test_00240() -> None:
    routine(test=240)


def test_00241() -> None:
    routine(test=241)


def test_00242() -> None:
    routine(test=242)


def test_00243() -> None:
    routine(test=243)


def test_00244() -> None:
    routine(test=244)


def test_00245() -> None:
    routine(test=245)


def test_00246() -> None:
    routine(test=246)


def test_00247() -> None:
    routine(test=247)


def test_00248() -> None:
    routine(test=248)


def test_00249() -> None:
    routine(test=249)


def test_00250() -> None:
    routine(test=250)


def test_00251() -> None:
    routine(test=251)


def test_00252() -> None:
    routine(test=252)


def test_00253() -> None:
    routine(test=253)


def test_00254() -> None:
    routine(test=254)


def test_00255() -> None:
    routine(test=255)


def test_00256() -> None:
    routine(test=256)


def test_00257() -> None:
    routine(test=257)


def test_00258() -> None:
    routine(test=258)


def test_00259() -> None:
    routine(test=259)


def test_00260() -> None:
    routine(test=260)


def test_00261() -> None:
    routine(test=261)


def test_00262() -> None:
    routine(test=262)


def test_00263() -> None:
    routine(test=263)


def test_00264() -> None:
    routine(test=264)


def test_00265() -> None:
    routine(test=265)


def test_00266() -> None:
    routine(test=266)


def test_00267() -> None:
    routine(test=267)


def test_00268() -> None:
    routine(test=268)


def test_00269() -> None:
    routine(test=269)


def test_00270() -> None:
    routine(test=270)


def test_00271() -> None:
    routine(test=271)


def test_00272() -> None:
    routine(test=272)


def test_00273() -> None:
    routine(test=273)


def test_00274() -> None:
    routine(test=274)


def test_00275() -> None:
    routine(test=275)


def test_00276() -> None:
    routine(test=276)


def test_00277() -> None:
    routine(test=277)


def test_00278() -> None:
    routine(test=278)


def test_00279() -> None:
    routine(test=279)


def test_00280() -> None:
    routine(test=280)


def test_00281() -> None:
    routine(test=281)


def test_00282() -> None:
    routine(test=282)


def test_00283() -> None:
    routine(test=283)


def test_00284() -> None:
    routine(test=284)


def test_00285() -> None:
    routine(test=285)


def test_00286() -> None:
    routine(test=286)


def test_00287() -> None:
    routine(test=287)


def test_00288() -> None:
    routine(test=288)


def test_00289() -> None:
    routine(test=289)


def test_00290() -> None:
    routine(test=290)


def test_00291() -> None:
    routine(test=291)


def test_00292() -> None:
    routine(test=292)


def test_00293() -> None:
    routine(test=293)


def test_00294() -> None:
    routine(test=294)


def test_00295() -> None:
    routine(test=295)


def test_00296() -> None:
    routine(test=296)


def test_00297() -> None:
    routine(test=297)


def test_00298() -> None:
    routine(test=298)


def test_00299() -> None:
    routine(test=299)


def test_00300() -> None:
    routine(test=300)


def test_00301() -> None:
    routine(test=301)


def test_00302() -> None:
    routine(test=302)


def test_00303() -> None:
    routine(test=303)


def test_00304() -> None:
    routine(test=304)


def test_00305() -> None:
    routine(test=305)


def test_00306() -> None:
    routine(test=306)


def test_00307() -> None:
    routine(test=307)


def test_00308() -> None:
    routine(test=308)


def test_00309() -> None:
    routine(test=309)


def test_00310() -> None:
    routine(test=310)


def test_00311() -> None:
    routine(test=311)


def test_00312() -> None:
    routine(test=312)


def test_00313() -> None:
    routine(test=313)


def test_00314() -> None:
    routine(test=314)


def test_00315() -> None:
    routine(test=315)


def test_00316() -> None:
    routine(test=316)


def test_00317() -> None:
    routine(test=317)


def test_00318() -> None:
    routine(test=318)


def test_00319() -> None:
    routine(test=319)


def test_00320() -> None:
    routine(test=320)


def test_00321() -> None:
    routine(test=321)


def test_00322() -> None:
    routine(test=322)


def test_00323() -> None:
    routine(test=323)


def test_00324() -> None:
    routine(test=324)


def test_00325() -> None:
    routine(test=325)


def test_00326() -> None:
    routine(test=326)


def test_00327() -> None:
    routine(test=327)


def test_00328() -> None:
    routine(test=328)


def test_00329() -> None:
    routine(test=329)


def test_00330() -> None:
    routine(test=330)


def test_00331() -> None:
    routine(test=331)


def test_00332() -> None:
    routine(test=332)


def test_00333() -> None:
    routine(test=333)


def test_00334() -> None:
    routine(test=334)


def test_00335() -> None:
    routine(test=335)


def test_00336() -> None:
    routine(test=336)


def test_00337() -> None:
    routine(test=337)


def test_00338() -> None:
    routine(test=338)


def test_00339() -> None:
    routine(test=339)


def test_00340() -> None:
    routine(test=340)


def test_00341() -> None:
    routine(test=341)


def test_00342() -> None:
    routine(test=342)


def test_00343() -> None:
    routine(test=343)


def test_00344() -> None:
    routine(test=344)


def test_00345() -> None:
    routine(test=345)


def test_00346() -> None:
    routine(test=346)


def test_00347() -> None:
    routine(test=347)


def test_00348() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=348)


def test_00349() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=349)


def test_00350() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=350)


def test_00351() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=351)


def test_00352() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=352)


def test_00353() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=353)


def test_00354() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=354)


def test_00355() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=355)


def test_00356() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=356)


def test_00357() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=357)


def test_00358() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=358)


def test_00359() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=359)


def test_00360() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=360)


def test_00361() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=361)


def test_00362() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=362)


def test_00363() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=363)


def test_00364() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=364)


def test_00365() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=365)


def test_00366() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=366)


def test_00367() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=367)


def test_00368() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=368)


def test_00369() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=369)


def test_00370() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=370)


def test_00371() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=371)


def test_00372() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=372)


def test_00373() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=373)


def test_00374() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=374)


def test_00375() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=375)


def test_00376() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=376)


def test_00377() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=377)


def test_00378() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=378)


def test_00379() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=379)


def test_00380() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=380)


def test_00381() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=381)


def test_00382() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=382)


def test_00383() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=383)


def test_00384() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=384)


def test_00385() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=385)


def test_00386() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=386)


def test_00387() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=387)


def test_00389() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=389)


def test_00390() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=390)


def test_00392() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=392)


def test_00393() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=393)


def test_00395() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=395)


def test_00396() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=396)


def test_00397() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=397)


def test_00398() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=398)


def test_00399() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=399)


def test_00400() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=400)


def test_00401() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=401)


def test_00402() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=402)


def test_00403() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=403)


def test_00404() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=404)


def test_00405() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=405)


def test_00406() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=406)


def test_00407() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=407)


def test_00408() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=408)


def test_00409() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=409)


def test_00410() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=410)


def test_00411() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=411)


def test_00412() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=412)


def test_00413() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=413)


def test_00414() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=414)


def test_00415() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=415)


def test_00416() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=416)


def test_00417() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=417)


def test_00418() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=418)


def test_00419() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=419)


def test_00420() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=420)


def test_00421() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=421)


def test_00422() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=422)


def test_00423() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=423)


def test_00424() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=424)


def test_00425() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=425)


def test_00426() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=426)


def test_00427() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=427)


def test_00428() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=428)


def test_00429() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=429)


def test_00430() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=430)


def test_00431() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=431)


def test_00432() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=432)


def test_00433() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=433)


def test_00434() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=434)


def test_00435() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=435)


def test_00436() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=436)


def test_00437() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=437)


def test_00438() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=438)


def test_00439() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=439)


def test_00440() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=440)


def test_00441() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=441)


def test_00442() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=442)


def test_00443() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=443)


def test_00444() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=444)


def test_00446() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=446)


def test_00447() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=447)


def test_00449() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=449)


def test_00450() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=450)


def test_00452() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=452)


def test_00453() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=453)


def test_00454() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=454)


def test_00455() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=455)


def test_00456() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=456)


def test_00457() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=457)


def test_00458() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=458)


def test_00459() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=459)


def test_00460() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=460)


def test_00461() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=461)


def test_00462() -> None:
    routine(test=462)


def test_00463() -> None:
    routine(test=463)


def test_00464() -> None:
    routine(test=464)


def test_00465() -> None:
    routine(test=465)


def test_00466() -> None:
    routine(test=466)


def test_00467() -> None:
    routine(test=467)


def test_00468() -> None:
    routine(test=468)


def test_00469() -> None:
    routine(test=469)


def test_00470() -> None:
    routine(test=470)


def test_00471() -> None:
    routine(test=471)


def test_00472() -> None:
    routine(test=472)


def test_00473() -> None:
    routine(test=473)


def test_00474() -> None:
    routine(test=474)


def test_00475() -> None:
    routine(test=475)


def test_00476() -> None:
    routine(test=476)


def test_00477() -> None:
    routine(test=477)


def test_00478() -> None:
    routine(test=478)


def test_00479() -> None:
    routine(test=479)


def test_00480() -> None:
    routine(test=480)


def test_00481() -> None:
    routine(test=481)


def test_00482() -> None:
    routine(test=482)


def test_00483() -> None:
    routine(test=483)


def test_00484() -> None:
    routine(test=484)


def test_00485() -> None:
    routine(test=485)


def test_00486() -> None:
    routine(test=486)


def test_00487() -> None:
    routine(test=487)


def test_00488() -> None:
    routine(test=488)


def test_00489() -> None:
    routine(test=489)


def test_00490() -> None:
    routine(test=490)


def test_00491() -> None:
    routine(test=491)


def test_00492() -> None:
    routine(test=492)


def test_00493() -> None:
    routine(test=493)


def test_00494() -> None:
    routine(test=494)


def test_00495() -> None:
    routine(test=495)


def test_00496() -> None:
    routine(test=496)


def test_00497() -> None:
    routine(test=497)


def test_00498() -> None:
    routine(test=498)


def test_00499() -> None:
    routine(test=499)


def test_00500() -> None:
    routine(test=500)


def test_00501() -> None:
    routine(test=501)


def test_00502() -> None:
    routine(test=502)


def test_00503() -> None:
    routine(test=503)


def test_00504() -> None:
    routine(test=504)


def test_00505() -> None:
    routine(test=505)


def test_00506() -> None:
    routine(test=506)


def test_00507() -> None:
    routine(test=507)


def test_00508() -> None:
    routine(test=508)


def test_00509() -> None:
    routine(test=509)


def test_00510() -> None:
    routine(test=510)


def test_00511() -> None:
    routine(test=511)


def test_00512() -> None:
    routine(test=512)


def test_00513() -> None:
    routine(test=513)


def test_00514() -> None:
    routine(test=514)


def test_00515() -> None:
    routine(test=515)


def test_00522() -> None:
    routine(test=522)


def test_00523() -> None:
    routine(test=523)


def test_00524() -> None:
    routine(test=524)


def test_00525() -> None:
    routine(test=525)


def test_00526() -> None:
    routine(test=526)


def test_00527() -> None:
    routine(test=527)


def test_00528() -> None:
    routine(test=528)


def test_00529() -> None:
    routine(test=529)


def test_00530() -> None:
    routine(test=530)


def test_00531() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=531)


def test_00532() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=532)


def test_00533() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=533)


def test_00534() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=534)


def test_00535() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=535)


def test_00536() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=536)


def test_00537() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=537)


def test_00538() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=538)


def test_00539() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=539)


def test_00540() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=540)


def test_00541() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=541)


def test_00542() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=542)


def test_00543() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=543)


def test_00544() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=544)


def test_00545() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=545)


def test_00546() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=546)


def test_00547() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=547)


def test_00548() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=548)


def test_00549() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=549)


def test_00550() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=550)


def test_00551() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=551)


def test_00552() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=552)


def test_00553() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=553)


def test_00554() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=554)


def test_00555() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=555)


def test_00556() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=556)


def test_00557() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=557)


def test_00558() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=558)


def test_00559() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=559)


def test_00560() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=560)


def test_00565() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=565)


def test_00566() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=566)


def test_00567() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=567)


def test_00568() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=568)


def test_00569() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=569)


def test_00570() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=570)


def test_00571() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=571)


def test_00572() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=572)


def test_00573() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=573)


def test_00574() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=574)


def test_00575() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=575)


def test_00576() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=576)


def test_00577() -> None:
    routine(test=577)


def test_00578() -> None:
    routine(test=578)


def test_00579() -> None:
    routine(test=579)


def test_00580() -> None:
    routine(test=580)


def test_00581() -> None:
    routine(test=581)


def test_00582() -> None:
    routine(test=582)


def test_00583() -> None:
    routine(test=583)


def test_00584() -> None:
    routine(test=584)


def test_00585() -> None:
    routine(test=585)


def test_00586() -> None:
    routine(test=586)


def test_00587() -> None:
    routine(test=587)


def test_00588() -> None:
    routine(test=588)


def test_00589() -> None:
    routine(test=589)


def test_00590() -> None:
    routine(test=590)


def test_00591() -> None:
    routine(test=591)


def test_00592() -> None:
    routine(test=592)


def test_00593() -> None:
    routine(test=593)


def test_00594() -> None:
    routine(test=594)


def test_00595() -> None:
    routine(test=595)


def test_00596() -> None:
    routine(test=596)


def test_00598() -> None:
    routine(test=598)


def test_00599() -> None:
    routine(test=599)


def test_00600() -> None:
    routine(test=600)


def test_00601() -> None:
    routine(test=601)


def test_00602() -> None:
    routine(test=602)


def test_00603() -> None:
    routine(test=603)


def test_00604() -> None:
    routine(test=604)


def test_00605() -> None:
    routine(test=605)


def test_00606() -> None:
    routine(test=606)


def test_00607() -> None:
    routine(test=607)


def test_00608() -> None:
    routine(test=608)


def test_00611() -> None:
    routine(test=611)


def test_00612() -> None:
    routine(test=612)


def test_00613() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=613)


def test_00614() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=614)


def test_00615() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=615)


def test_00616() -> None:
    routine(test=616)


def test_00617() -> None:
    routine(test=617)


def test_00618() -> None:
    routine(test=618)


def test_00619() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=619)


def test_00620() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=620)


def test_00621() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=621)


def test_00622() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=622)


def test_00623() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=623)


def test_00624() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=624)


def test_00625() -> None:
    routine(test=625)


def test_00626() -> None:
    routine(test=626)


def test_00627() -> None:
    routine(test=627)


def test_00628() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=628)


def test_00629() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=629)


def test_00630() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=630)


def test_00631() -> None:
    routine(test=631)


def test_00632() -> None:
    routine(test=632)


def test_00633() -> None:
    routine(test=633)


def test_00634() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=634)


def test_00635() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=635)


def test_00636() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=636)


def test_00637() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=637)


def test_00638() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=638)


def test_00639() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=639)


def test_00640() -> None:
    routine(test=640)


def test_00641() -> None:
    routine(test=641)


def test_00642() -> None:
    routine(test=642)


def test_00643() -> None:
    routine(test=643)


def test_00644() -> None:
    routine(test=644)


def test_00645() -> None:
    routine(test=645)


def test_00646() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=646)


def test_00647() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=647)


def test_00648() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=648)


def test_00649() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=649)


def test_00650() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=650)


def test_00651() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=651)


def test_00652() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=652)


def test_00653() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=653)


def test_00654() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=654)


def test_00655() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=655)


def test_00656() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=656)


def test_00657() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=657)


def test_00658() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=658)


def test_00659() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=659)


def test_00660() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=660)


def test_00661() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=661)


def test_00662() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=662)


def test_00663() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=663)


def test_00664() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=664)


def test_00665() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=665)


def test_00666() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=666)


def test_00667() -> None:
    routine(test=667)


def test_00668() -> None:
    routine(test=668)


def test_00669() -> None:
    routine(test=669)


def test_00670() -> None:
    routine(test=670)


def test_00671() -> None:
    routine(test=671)


def test_00672() -> None:
    routine(test=672)


def test_00673() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=673)


def test_00674() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=674)


def test_00675() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=675)


def test_00676() -> None:
    routine(test=676)


def test_00677() -> None:
    routine(test=677)


def test_00678() -> None:
    routine(test=678)


def test_00679() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=679)


def test_00680() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=680)


def test_00681() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=681)


def test_00682() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=682)


def test_00683() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=683)


def test_00684() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=684)


def test_00685() -> None:
    routine(test=685)


def test_00686() -> None:
    routine(test=686)


def test_00687() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=687)


def test_00688() -> None:
    routine(test=688)


def test_00689() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=689)


def test_00690() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=690)


def test_00691() -> None:
    routine(test=691)


def test_00692() -> None:
    routine(test=692)


def test_00693() -> None:
    routine(test=693)


def test_00694() -> None:
    routine(test=694)


def test_00695() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=695)


def test_00696() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=696)


def test_00697() -> None:
    routine(test=697)


def test_00698() -> None:
    routine(test=698)


def test_00699() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=699)


def test_00700() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=700)


def test_00701() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=701)


def test_00702() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=702)


def test_00703() -> None:
    routine(test=703)


def test_00704() -> None:
    routine(test=704)


def test_00705() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=705)


def test_00706() -> None:
    routine(test=706)


def test_00707() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=707)


def test_00708() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=708)


def test_00709() -> None:
    routine(test=709)


def test_00710() -> None:
    routine(test=710)


def test_00711() -> None:
    routine(test=711)


def test_00712() -> None:
    routine(test=712)


def test_00713() -> None:
    routine(test=713)


def test_00714() -> None:
    routine(test=714)


def test_00715() -> None:
    routine(test=715)


def test_00716() -> None:
    routine(test=716)


def test_00717() -> None:
    routine(test=717)


def test_00718() -> None:
    routine(test=718)


def test_00719() -> None:
    routine(test=719)


def test_00720() -> None:
    routine(test=720)


def test_00721() -> None:
    routine(test=721)


def test_00722() -> None:
    routine(test=722)


def test_00723() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=723)


def test_00724() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=724)


def test_00732() -> None:
    routine(test=732)


def test_00733() -> None:
    routine(test=733)


def test_00734() -> None:
    routine(test=734)


def test_00735() -> None:
    routine(test=735)


def test_00736() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=736)


def test_00737() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=737)


def test_00738() -> None:
    routine(test=738)


def test_00739() -> None:
    routine(test=739)


def test_00740() -> None:
    routine(test=740)


def test_00741() -> None:
    routine(test=741)


def test_00742() -> None:
    routine(test=742)


def test_00743() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=743)


def test_00744() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=744)


def test_00745() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=745)


def test_00746() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=746)


def test_00747() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=747)


def test_00748() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=748)


def test_00749() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=749)


def test_00750() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=750)


def test_00751() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=751)


def test_00752() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=752)


def test_00753() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=753)


def test_00754() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=754)


def test_00755() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=755)


def test_00756() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=756)


def test_00757() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=757)


def test_00758() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=758)


def test_00759() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=759)


def test_00760() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=760)


def test_00761() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=761)


def test_00762() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=762)


def test_00763() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=763)


def test_00764() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=764)


def test_00765() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=765)


def test_00766() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=766)


def test_00767() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=767)


def test_00768() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=768)


def test_00769() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=769)


def test_00770() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=770)


def test_00771() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=771)


def test_00772() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=772)


def test_00773() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=773)


def test_00774() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=774)


def test_00775() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=775)


def test_00776() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=776)


def test_00777() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=777)


def test_00778() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=778)


def test_00779() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=779)


def test_00780() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=780)


def test_00781() -> None:
    routine(test=781)


def test_00782() -> None:
    routine(test=782)


def test_00783() -> None:
    routine(test=783)


def test_00784() -> None:
    routine(test=784)


def test_00785() -> None:
    routine(test=785)


def test_00786() -> None:
    routine(test=786)


def test_00787() -> None:
    routine(test=787)


def test_00788() -> None:
    routine(test=788)


def test_00789() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=789)


def test_00790() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=790)


def test_00791() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=791)


def test_00792() -> None:
    routine(test=792)


def test_00793() -> None:
    routine(test=793)


def test_00794() -> None:
    routine(test=794)


def test_00795() -> None:
    routine(test=795)


def test_00796() -> None:
    routine(test=796)


def test_00797() -> None:
    routine(test=797)


def test_00798() -> None:
    routine(test=798)


def test_00799() -> None:
    routine(test=799)


def test_00800() -> None:
    routine(test=800)


def test_00801() -> None:
    routine(test=801)


def test_00802() -> None:
    routine(test=802)


def test_00803() -> None:
    routine(test=803)


def test_00804() -> None:
    routine(test=804)


def test_00805() -> None:
    routine(test=805)


def test_00806() -> None:
    routine(test=806)


def test_00807() -> None:
    routine(test=807)


def test_00808() -> None:
    routine(test=808)


def test_00809() -> None:
    routine(test=809)


def test_00810() -> None:
    routine(test=810)


def test_00811() -> None:
    routine(test=811)


def test_00812() -> None:
    routine(test=812)


def test_00813() -> None:
    routine(test=813)


def test_00814() -> None:
    routine(test=814)


def test_00815() -> None:
    routine(test=815)


def test_00816() -> None:
    routine(test=816)


def test_00817() -> None:
    routine(test=817)


def test_00818() -> None:
    routine(test=818)


def test_00819() -> None:
    routine(test=819)


def test_00820() -> None:
    routine(test=820)


def test_00821() -> None:
    routine(test=821)


def test_00822() -> None:
    routine(test=822)


def test_00823() -> None:
    routine(test=823)


def test_00824() -> None:
    routine(test=824)


def test_00825() -> None:
    routine(test=825)


def test_00826() -> None:
    routine(test=826)


def test_00830() -> None:
    routine(test=830)


def test_00831() -> None:
    routine(test=831)


def test_00832() -> None:
    routine(test=832)


def test_00833() -> None:
    routine(test=833)


def test_00834() -> None:
    routine(test=834)


def test_00835() -> None:
    routine(test=835)


def test_00836() -> None:
    routine(test=836)


def test_00837() -> None:
    routine(test=837)


def test_00838() -> None:
    routine(test=838)


def test_00839() -> None:
    routine(test=839)


def test_00840() -> None:
    routine(test=840)


def test_00841() -> None:
    routine(test=841)


def test_00842() -> None:
    routine(test=842)


def test_00843() -> None:
    routine(test=843)


def test_00844() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=844)


def test_00845() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=845)


def test_00846() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=846)


def test_00847() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=847)


def test_00848() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=848)


def test_00849() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=849)


def test_00850() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=850)


def test_00851() -> None:
    routine(test=851)


def test_00852() -> None:
    routine(test=852)


def test_00853() -> None:
    routine(test=853)


def test_00854() -> None:
    routine(test=854)


def test_00855() -> None:
    routine(test=855)


def test_00856() -> None:
    routine(test=856)


def test_00857() -> None:
    routine(test=857)


def test_00858() -> None:
    routine(test=858)


def test_00859() -> None:
    routine(test=859)


def test_00860() -> None:
    routine(test=860)


def test_00861() -> None:
    routine(test=861)


def test_00862() -> None:
    routine(test=862)


def test_00863() -> None:
    routine(test=863)


def test_00864() -> None:
    routine(test=864)


def test_00865() -> None:
    routine(test=865)


def test_00866() -> None:
    routine(test=866)


def test_00867() -> None:
    routine(test=867)


def test_00868() -> None:
    routine(test=868)


def test_00869() -> None:
    routine(test=869)


def test_00876() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=876)


def test_00877() -> None:
    routine(test=877)


def test_00878() -> None:
    routine(test=878)


def test_00879() -> None:
    routine(test=879)


def test_00880() -> None:
    routine(test=880)


def test_00881() -> None:
    routine(test=881)


def test_00882() -> None:
    routine(test=882)


def test_00883() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=883)


def test_00884() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=884)


def test_00885() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=885)


def test_00886() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=886)


def test_00887() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=887)


def test_00888() -> None:
    routine(test=888)


def test_00889() -> None:
    routine(test=889)


def test_00890() -> None:
    routine(test=890)


def test_00891() -> None:
    routine(test=891)


def test_00892() -> None:
    routine(test=892)


def test_00893() -> None:
    routine(test=893)


def test_00894() -> None:
    routine(test=894)


def test_00895() -> None:
    routine(test=895)


def test_00896() -> None:
    routine(test=896)


def test_00897() -> None:
    routine(test=897)


def test_00901() -> None:
    routine(test=901)


def test_00902() -> None:
    routine(test=902)


def test_00903() -> None:
    routine(test=903)


def test_00904() -> None:
    routine(test=904)


def test_00905() -> None:
    routine(test=905)


def test_00906() -> None:
    routine(test=906)


def test_00907() -> None:
    routine(test=907)


def test_00908() -> None:
    routine(test=908)


def test_00909() -> None:
    routine(test=909)


def test_00910() -> None:
    routine(test=910)


def test_00911() -> None:
    routine(test=911)


def test_00912() -> None:
    routine(test=912)


def test_00913() -> None:
    routine(test=913)


def test_00914() -> None:
    routine(test=914)


def test_00915() -> None:
    routine(test=915)


def test_00916() -> None:
    routine(test=916)


def test_00917() -> None:
    routine(test=917)


def test_00918() -> None:
    routine(test=918)


def test_00919() -> None:
    routine(test=919)


def test_00920() -> None:
    routine(test=920)


def test_00921() -> None:
    routine(test=921)


def test_00922() -> None:
    routine(test=922)


def test_00923() -> None:
    routine(test=923)


def test_00924() -> None:
    routine(test=924)


def test_00925() -> None:
    routine(test=925)


def test_00926() -> None:
    routine(test=926)


def test_00927() -> None:
    routine(test=927)


def test_00928() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=928)


def test_00929() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=929)


def test_00930() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=930)


def test_00931() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=931)


def test_00932() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=932)


def test_00933() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=933)


def test_00934() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=934)


def test_00935() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=935)


def test_00936() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=936)


def test_00937() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=937)


def test_00938() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=938)


def test_00939() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=939)


def test_00940() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=940)


def test_00941() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=941)


def test_00942() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=942)


def test_00943() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=943)


def test_00944() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=944)


def test_00945() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=945)


def test_00946() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=946)


def test_00947() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=947)


def test_00948() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=948)


def test_00949() -> None:
    routine(test=949)


def test_00950() -> None:
    routine(test=950)


def test_00951() -> None:
    routine(test=951)


def test_00952() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=952)


def test_00953() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=953)


def test_00954() -> None:
    routine(test=954)


def test_00955() -> None:
    routine(test=955)


def test_00956() -> None:
    routine(test=956)


def test_00957() -> None:
    routine(test=957)


def test_00958() -> None:
    routine(test=958)


def test_00959() -> None:
    with pytest.raises(ZeroDivisionError):
        routine(test=959)


def test_00960() -> None:
    routine(test=960)


def test_00961() -> None:
    routine(test=961)


def test_00962() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=962)


def test_00963() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=963)


def test_00964() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=964)


def test_00965() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=965)


def test_00966() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=966)


def test_00967() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=967)


def test_00968() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=968)


def test_00969() -> None:
    routine(test=969)


def test_00970() -> None:
    routine(test=970)


def test_00971() -> None:
    routine(test=971)


def test_00972() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=972)


def test_00973() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=973)


def test_00974() -> None:
    routine(test=974)


def test_00975() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=975)


def test_00976() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=976)


def test_00977() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=977)


def test_00978() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=978)


def test_00979() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=979)


def test_00980() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=980)


def test_00981() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=981)


def test_00982() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=982)


def test_00983() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=983)


def test_00984() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=984)


def test_00985() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=985)


def test_00986() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=986)


def test_00987() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=987)


def test_00988() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=988)


def test_00989() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=989)


def test_00990() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=990)


def test_00991() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=991)


def test_00992() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=992)


def test_00993() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=993)


def test_00994() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=994)


def test_00995() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=995)


def test_00996() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=996)


def test_00997() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=997)


def test_00998() -> None:
    routine(test=998)


@pytest.mark.skip("""Various variants of amount / conc and constant / dynamic compartment. \
                  Test suite asks for concentration even though species is amount. \
                  Marked by empty substance units (not documented in SBML spec). \
                  We don't need to support this

                  Affected tests: 999, 1206, 1207, 1208, 1307, 1308, 1309, 1498, 1513, 1514
                  """)
def test_00999() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=999)


def test_01000() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1000)


def test_01001() -> None:
    routine(test=1001)


def test_01002() -> None:
    routine(test=1002)


def test_01003() -> None:
    routine(test=1003)


def test_01004() -> None:
    routine(test=1004)


def test_01005() -> None:
    routine(test=1005)


def test_01006() -> None:
    routine(test=1006)


def test_01007() -> None:
    routine(test=1007)


def test_01008() -> None:
    routine(test=1008)


def test_01009() -> None:
    routine(test=1009)


def test_01010() -> None:
    routine(test=1010)


def test_01011() -> None:
    routine(test=1011)


def test_01012() -> None:
    routine(test=1012)


def test_01013() -> None:
    routine(test=1013)


def test_01014() -> None:
    routine(test=1014)


def test_01015() -> None:
    routine(test=1015)


def test_01016() -> None:
    routine(test=1016)


def test_01017() -> None:
    routine(test=1017)


def test_01018() -> None:
    routine(test=1018)


def test_01019() -> None:
    routine(test=1019)


def test_01020() -> None:
    routine(test=1020)


def test_01021() -> None:
    routine(test=1021)


def test_01022() -> None:
    routine(test=1022)


def test_01023() -> None:
    routine(test=1023)


def test_01024() -> None:
    routine(test=1024)


def test_01025() -> None:
    routine(test=1025)


def test_01026() -> None:
    routine(test=1026)


def test_01027() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1027)


def test_01028() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1028)


def test_01029() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1029)


def test_01030() -> None:
    routine(test=1030)


def test_01031() -> None:
    routine(test=1031)


def test_01032() -> None:
    routine(test=1032)


def test_01033() -> None:
    routine(test=1033)


def test_01034() -> None:
    routine(test=1034)


def test_01035() -> None:
    routine(test=1035)


def test_01036() -> None:
    routine(test=1036)


def test_01037() -> None:
    routine(test=1037)


def test_01038() -> None:
    routine(test=1038)


def test_01039() -> None:
    routine(test=1039)


def test_01040() -> None:
    routine(test=1040)


def test_01041() -> None:
    routine(test=1041)


def test_01042() -> None:
    routine(test=1042)


def test_01043() -> None:
    routine(test=1043)


def test_01044() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1044)


def test_01045() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1045)


def test_01046() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1046)


def test_01047() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1047)


def test_01048() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1048)


def test_01049() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1049)


def test_01050() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1050)


def test_01051() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1051)


def test_01052() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1052)


def test_01053() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1053)


def test_01054() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1054)


def test_01055() -> None:
    routine(test=1055)


def test_01056() -> None:
    routine(test=1056)


def test_01057() -> None:
    routine(test=1057)


def test_01058() -> None:
    routine(test=1058)


def test_01059() -> None:
    routine(test=1059)


def test_01060() -> None:
    routine(test=1060)


def test_01061() -> None:
    routine(test=1061)


def test_01062() -> None:
    routine(test=1062)


def test_01063() -> None:
    routine(test=1063)


def test_01064() -> None:
    routine(test=1064)


def test_01065() -> None:
    routine(test=1065)


def test_01066() -> None:
    routine(test=1066)


def test_01067() -> None:
    routine(test=1067)


def test_01068() -> None:
    routine(test=1068)


def test_01069() -> None:
    routine(test=1069)


def test_01070() -> None:
    routine(test=1070)


def test_01071() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1071)


def test_01072() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1072)


def test_01073() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1073)


def test_01074() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1074)


def test_01075() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1075)


def test_01076() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1076)


def test_01077() -> None:
    routine(test=1077)


def test_01078() -> None:
    routine(test=1078)


def test_01079() -> None:
    routine(test=1079)


def test_01080() -> None:
    routine(test=1080)


def test_01081() -> None:
    routine(test=1081)


def test_01082() -> None:
    routine(test=1082)


def test_01083() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1083)


def test_01084() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1084)


def test_01085() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1085)


def test_01086() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1086)


def test_01087() -> None:
    routine(test=1087)


def test_01088() -> None:
    routine(test=1088)


def test_01089() -> None:
    routine(test=1089)


def test_01090() -> None:
    routine(test=1090)


def test_01091() -> None:
    routine(test=1091)


def test_01092() -> None:
    routine(test=1092)


def test_01093() -> None:
    routine(test=1093)


def test_01094() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1094)


def test_01095() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1095)


def test_01096() -> None:
    routine(test=1096)


def test_01097() -> None:
    routine(test=1097)


def test_01098() -> None:
    routine(test=1098)


def test_01099() -> None:
    routine(test=1099)


def test_01100() -> None:
    routine(test=1100)


def test_01101() -> None:
    routine(test=1101)


def test_01102() -> None:
    routine(test=1102)


def test_01103() -> None:
    routine(test=1103)


def test_01104() -> None:
    routine(test=1104)


def test_01105() -> None:
    routine(test=1105)


def test_01106() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1106)


def test_01107() -> None:
    routine(test=1107)


def test_01108() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1108)


def test_01109() -> None:
    routine(test=1109)


def test_01110() -> None:
    routine(test=1110)


def test_01111() -> None:
    routine(test=1111)


def test_01112() -> None:
    routine(test=1112)


def test_01113() -> None:
    routine(test=1113)


def test_01114() -> None:
    routine(test=1114)


def test_01115() -> None:
    routine(test=1115)


def test_01116() -> None:
    routine(test=1116)


def test_01117() -> None:
    routine(test=1117)


def test_01118() -> None:
    routine(test=1118)


def test_01119() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1119)


def test_01120() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1120)


def test_01121() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1121)


def test_01122() -> None:
    routine(test=1122)


def test_01123() -> None:
    routine(test=1123)


def test_01124() -> None:
    with pytest.raises(NotImplementedError):  # comp package
        routine(test=1124)


def test_01125() -> None:
    with pytest.raises(NotImplementedError):  # comp package
        routine(test=1125)


def test_01126() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1126)


def test_01127() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1127)


def test_01128() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1128)


def test_01129() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1129)


def test_01130() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1130)


def test_01131() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1131)


def test_01132() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1132)


def test_01133() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1133)


def test_01134() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1134)


def test_01135() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1135)


def test_01136() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1136)


def test_01137() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1137)


def test_01138() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1138)


def test_01139() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1139)


def test_01140() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1140)


def test_01141() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1141)


def test_01142() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1142)


def test_01143() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1143)


def test_01144() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1144)


def test_01145() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1145)


def test_01146() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1146)


def test_01147() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1147)


def test_01148() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1148)


def test_01149() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1149)


def test_01150() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1150)


def test_01151() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1151)


def test_01152() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1152)


def test_01153() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1153)


def test_01154() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1154)


def test_01155() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1155)


def test_01156() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1156)


def test_01157() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1157)


def test_01158() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1158)


def test_01159() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1159)


def test_01160() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1160)


def test_01161() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1161)


def test_01162() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1162)


def test_01163() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1163)


def test_01164() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1164)


def test_01165() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1165)


def test_01166() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1166)


def test_01167() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1167)


def test_01168() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1168)


def test_01169() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1169)


def test_01170() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1170)


def test_01171() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1171)


def test_01172() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1172)


def test_01173() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1173)


def test_01174() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1174)


def test_01175() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1175)


def test_01176() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1176)


def test_01177() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1177)


def test_01178() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1178)


def test_01179() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1179)


def test_01180() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1180)


def test_01181() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1181)


def test_01182() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1182)


def test_01183() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1183)


def test_01184() -> None:
    routine(test=1184)


def test_01185() -> None:
    routine(test=1185)


def test_01186() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1186)


def test_01187() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1187)


def test_01188() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1188)


def test_01189() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1189)


def test_01190() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1190)


def test_01191() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1191)


def test_01192() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1192)


def test_01193() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1193)


def test_01194() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1194)


def test_01195() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1195)


def test_01196() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1196)


def test_01197() -> None:
    routine(test=1197)


def test_01198() -> None:
    routine(test=1198)


def test_01199() -> None:
    routine(test=1199)


def test_01200() -> None:
    routine(test=1200)


def test_01201() -> None:
    routine(test=1201)


def test_01202() -> None:
    routine(test=1202)


def test_01203() -> None:
    routine(test=1203)


def test_01204() -> None:
    routine(test=1204)


def test_01205() -> None:
    routine(test=1205)


@pytest.mark.skip("""Test suite asks for concentration even though species is amount. \
                  Marked by empty substance units (not documented in SBML spec). \
                  We don't need to support this

                  Affected tests: 999, 1206, 1207, 1208, 1307, 1308, 1309, 1498, 1513, 1514
                  """)
def test_01206() -> None:
    routine(test=1206)


@pytest.mark.skip("""Test suite asks for concentration even though species is amount. \
                  Marked by empty substance units (not documented in SBML spec). \
                  We don't need to support this

                  Affected tests: 999, 1206, 1207, 1208, 1307, 1308, 1309, 1498, 1513, 1514
                  """)
def test_01207() -> None:
    routine(test=1207)


@pytest.mark.skip("""Test suite asks for concentration even though species is amount. \
                  Marked by empty substance units (not documented in SBML spec). \
                  We don't need to support this

                  Affected tests: 999, 1206, 1207, 1208, 1307, 1308, 1309, 1498, 1513, 1514
                  """)
def test_01208() -> None:
    routine(test=1208)


def test_01209() -> None:
    routine(test=1209)


def test_01210() -> None:
    routine(test=1210)


def test_01211() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1211)


def test_01212() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1212)


def test_01213() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1213)


def test_01214() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1214)


def test_01215() -> None:
    routine(test=1215)


def test_01216() -> None:
    routine(test=1216)


def test_01217() -> None:
    routine(test=1217)


def test_01218() -> None:
    routine(test=1218)


def test_01219() -> None:
    routine(test=1219)


def test_01220() -> None:
    routine(test=1220)


def test_01221() -> None:
    routine(test=1221)


def test_01222() -> None:
    with pytest.raises(NotImplementedError):  # Event
        routine(test=1222)


def test_01223() -> None:
    routine(test=1223)


def test_01224() -> None:
    routine(test=1224)


def test_01225() -> None:
    routine(test=1225)


def test_01226() -> None:
    routine(test=1226)


def test_01227() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1227)


def test_01228() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1228)


def test_01229() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1229)


def test_01230() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1230)


def test_01231() -> None:
    routine(test=1231)


def test_01232() -> None:
    routine(test=1232)


def test_01233() -> None:
    routine(test=1233)


def test_01234() -> None:
    routine(test=1234)


def test_01235() -> None:
    routine(test=1235)


def test_01236() -> None:
    routine(test=1236)


def test_01237() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1237)


def test_01238() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1238)


def test_01239() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1239)


def test_01240() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1240)


def test_01241() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1241)


def test_01242() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1242)


def test_01243() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1243)


def test_01244() -> None:
    routine(test=1244)


def test_01245() -> None:
    routine(test=1245)


def test_01246() -> None:
    routine(test=1246)


def test_01247() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1247)


def test_01248() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1248)


def test_01249() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1249)


def test_01250() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1250)


def test_01251() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1251)


def test_01252() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1252)


def test_01253() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1253)


def test_01254() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1254)


def test_01255() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1255)


def test_01256() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1256)


def test_01257() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1257)


def test_01258() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1258)


def test_01259() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1259)


def test_01260() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1260)


def test_01261() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1261)


def test_01262() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1262)


def test_01263() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1263)


def test_01264() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1264)


def test_01265() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1265)


def test_01266() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1266)


def test_01267() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1267)


def test_01268() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1268)


def test_01269() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1269)


def test_01270() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1270)


def test_01271() -> None:
    routine(test=1271)


def test_01272() -> None:
    routine(test=1272)


def test_01273() -> None:
    routine(test=1273)


def test_01274() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1274)


def test_01275() -> None:
    routine(test=1275)


def test_01276() -> None:
    routine(test=1276)


def test_01277() -> None:
    routine(test=1277)


def test_01278() -> None:
    routine(test=1278)


def test_01279() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1279)


def test_01280() -> None:
    routine(test=1280)


def test_01281() -> None:
    routine(test=1281)


def test_01282() -> None:
    routine(test=1282)


def test_01283() -> None:
    routine(test=1283)


def test_01284() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1284)


def test_01285() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1285)


def test_01286() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1286)


def test_01287() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1287)


def test_01288() -> None:
    routine(test=1288)


def test_01289() -> None:
    routine(test=1289)


def test_01290() -> None:
    routine(test=1290)


def test_01291() -> None:
    routine(test=1291)


def test_01292() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1292)


def test_01293() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1293)


def test_01294() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1294)


def test_01295() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1295)


def test_01296() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1296)


def test_01297() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1297)


def test_01298() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1298)


def test_01299() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1299)


def test_01300() -> None:
    routine(test=1300)


def test_01301() -> None:
    routine(test=1301)


def test_01302() -> None:
    routine(test=1302)


def test_01303() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1303)


def test_01304() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1304)


def test_01305() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1305)


def test_01306() -> None:
    routine(test=1306)


@pytest.mark.skip("""Test suite asks for concentration even though species is amount. \
                  Marked by empty substance units (not documented in SBML spec). \
                  We don't need to support this

                  Affected tests: 999, 1206, 1207, 1208, 1307, 1308, 1309, 1498, 1513, 1514
                  """)
def test_01307() -> None:
    routine(test=1307)


@pytest.mark.skip("""Test suite asks for concentration even though species is amount. \
                  Marked by empty substance units (not documented in SBML spec). \
                  We don't need to support this

                  Affected tests: 999, 1206, 1207, 1208, 1307, 1308, 1309, 1498, 1513, 1514
                  """)
def test_01308() -> None:
    routine(test=1308)


@pytest.mark.skip("""Various variants of amount / conc and constant / dynamic compartment. \
                  Test suite asks for concentration even though species is amount. \
                  Marked by empty substance units (not documented in SBML spec). \
                  We don't need to support this

                  Affected tests: 999, 1206, 1207, 1208, 1307, 1308, 1309, 1498, 1513, 1514
                  """)
def test_01309() -> None:
    routine(test=1309)


def test_01310() -> None:
    routine(test=1310)


def test_01311() -> None:
    routine(test=1311)


def test_01312() -> None:
    routine(test=1312)


def test_01313() -> None:
    routine(test=1313)


def test_01314() -> None:
    routine(test=1314)


def test_01315() -> None:
    routine(test=1315)


def test_01316() -> None:
    routine(test=1316)


def test_01317() -> None:
    routine(test=1317)


def test_01318() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1318)


def test_01319() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1319)


def test_01320() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1320)


def test_01321() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1321)


def test_01322() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1322)


def test_01323() -> None:
    routine(test=1323)


def test_01324() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1324)


def test_01325() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1325)


def test_01326() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1326)


def test_01327() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1327)


def test_01328() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1328)


def test_01329() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1329)


def test_01330() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1330)


def test_01331() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1331)


def test_01332() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1332)


def test_01333() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1333)


def test_01334() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1334)


def test_01335() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1335)


def test_01336() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1336)


def test_01337() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1337)


def test_01338() -> None:
    routine(test=1338)


def test_01339() -> None:
    routine(test=1339)


def test_01340() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1340)


def test_01341() -> None:
    routine(test=1341)


def test_01342() -> None:
    routine(test=1342)


def test_01343() -> None:
    routine(test=1343)


def test_01344() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1344)


def test_01345() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1345)


def test_01346() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1346)


def test_01347() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1347)


def test_01348() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1348)


def test_01349() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1349)


def test_01350() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1350)


def test_01351() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1351)


def test_01352() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1352)


def test_01353() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1353)


def test_01354() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1354)


def test_01355() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1355)


def test_01356() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1356)


def test_01357() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1357)


def test_01358() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1358)


def test_01359() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1359)


def test_01360() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1360)


def test_01361() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1361)


def test_01362() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1362)


def test_01363() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1363)


def test_01364() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1364)


def test_01365() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1365)


def test_01366() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1366)


def test_01367() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1367)


def test_01368() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1368)


def test_01369() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1369)


def test_01370() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1370)


def test_01371() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1371)


def test_01372() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1372)


def test_01373() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1373)


def test_01374() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1374)


def test_01375() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1375)


def test_01376() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1376)


def test_01377() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1377)


def test_01378() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1378)


def test_01379() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1379)


def test_01380() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1380)


def test_01381() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1381)


def test_01382() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1382)


def test_01383() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1383)


def test_01384() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1384)


def test_01385() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1385)


def test_01386() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1386)


def test_01387() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1387)


def test_01388() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1388)


def test_01389() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1389)


def test_01390() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1390)


def test_01391() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1391)


def test_01392() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1392)


def test_01393() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1393)


def test_01394() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1394)


def test_01395() -> None:
    routine(test=1395)


def test_01396() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1396)


def test_01397() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1397)


def test_01398() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1398)


def test_01399() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1399)


def test_01400() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1400)


def test_01401() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1401)


def test_01402() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1402)


def test_01403() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1403)


def test_01404() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1404)


def test_01405() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1405)


def test_01406() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1406)


def test_01407() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1407)


def test_01408() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1408)


def test_01409() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1409)


def test_01410() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1410)


def test_01411() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1411)


def test_01412() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1412)


def test_01413() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1413)


def test_01414() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1414)


def test_01415() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1415)


def test_01416() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1416)


def test_01417() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1417)


def test_01418() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1418)


def test_01419() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1419)


def test_01420() -> None:
    routine(test=1420)


def test_01421() -> None:
    routine(test=1421)


def test_01422() -> None:
    routine(test=1422)


def test_01423() -> None:
    routine(test=1423)


def test_01424() -> None:
    routine(test=1424)


def test_01425() -> None:
    routine(test=1425)


def test_01426() -> None:
    routine(test=1426)


def test_01427() -> None:
    routine(test=1427)


def test_01428() -> None:
    routine(test=1428)


def test_01429() -> None:
    routine(test=1429)


def test_01430() -> None:
    routine(test=1430)


def test_01431() -> None:
    routine(test=1431)


def test_01432() -> None:
    routine(test=1432)


def test_01433() -> None:
    routine(test=1433)


def test_01434() -> None:
    routine(test=1434)


def test_01435() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1435)


def test_01436() -> None:
    routine(test=1436)


def test_01437() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1437)


def test_01438() -> None:
    routine(test=1438)


def test_01439() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1439)


def test_01440() -> None:
    routine(test=1440)


def test_01441() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1441)


def test_01442() -> None:
    routine(test=1442)


def test_01443() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1443)


def test_01444() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1444)


def test_01445() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1445)


def test_01446() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1446)


def test_01447() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1447)


def test_01448() -> None:
    with pytest.raises(NotImplementedError):  # Event
        routine(test=1448)


def test_01449() -> None:
    routine(test=1449)


def test_01450() -> None:
    routine(test=1450)


def test_01451() -> None:
    routine(test=1451)


def test_01452() -> None:
    routine(test=1452)


def test_01453() -> None:
    routine(test=1453)


def test_01454() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1454)


def test_01455() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1455)


def test_01456() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1456)


def test_01457() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1457)


def test_01458() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1458)


def test_01459() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1459)


def test_01460() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1460)


def test_01461() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1461)


def test_01462() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1462)


def test_01463() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1463)


def test_01464() -> None:
    routine(test=1464)


def test_01465() -> None:
    routine(test=1465)


def test_01466() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1466)


def test_01467() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1467)


def test_01468() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1468)


def test_01469() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1469)


def test_01470() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1470)


def test_01471() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1471)


def test_01472() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1472)


def test_01473() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1473)


def test_01474() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1474)


def test_01475() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1475)


def test_01476() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1476)


def test_01477() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1477)


def test_01478() -> None:
    routine(test=1478)


def test_01479() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1479)


def test_01480() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1480)


def test_01481() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1481)


def test_01482() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1482)


def test_01483() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1483)


def test_01484() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1484)


def test_01485() -> None:
    routine(test=1485)


def test_01486() -> None:
    routine(test=1486)


def test_01487() -> None:
    routine(test=1487)


def test_01488() -> None:
    with pytest.raises(ZeroDivisionError):
        routine(test=1488)


def test_01489() -> None:
    routine(test=1489)


def test_01490() -> None:
    routine(test=1490)


def test_01491() -> None:
    routine(test=1491)


def test_01492() -> None:
    routine(test=1492)


def test_01493() -> None:
    routine(test=1493)


def test_01494() -> None:
    routine(test=1494)


def test_01495() -> None:
    routine(test=1495)


def test_01496() -> None:
    routine(test=1496)


def test_01497() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1497)


@pytest.mark.skip("""Test suite asks for concentration even though species is amount. \
                  Marked by empty substance units (not documented in SBML spec). \
                  We don't need to support this

                  Affected tests: 999, 1206, 1207, 1208, 1307, 1308, 1309, 1498, 1513, 1514
                  """)
def test_01498() -> None:
    routine(test=1498)


def test_01499() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1499)


def test_01500() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1500)


def test_01501() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1501)


def test_01502() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1502)


def test_01503() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1503)


def test_01504() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1504)


def test_01505() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1505)


def test_01506() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1506)


def test_01507() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1507)


def test_01508() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1508)


def test_01509() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1509)


def test_01510() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1510)


def test_01511() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1511)


def test_01512() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1512)


@pytest.mark.skip(""""Test suite asks for concentration even though species is amount. \
                  Marked by empty substance units (not documented in SBML spec). \
                  We don't need to support this

                  Affected tests: 999, 1206, 1207, 1208, 1307, 1308, 1309, 1498, 1513, 1514
                  """)
def test_01513() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1513)


@pytest.mark.skip("""Test suite asks for concentration even though species is amount. \
                  Marked by empty substance units (not documented in SBML spec). \
                  We don't need to support this

                  Affected tests: 999, 1206, 1207, 1208, 1307, 1308, 1309, 1498, 1513, 1514
                  """)
def test_01514() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1514)


def test_01515() -> None:
    routine(test=1515)


def test_01516() -> None:
    routine(test=1516)


def test_01517() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1517)


def test_01518() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1518)


def test_01519() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1519)


def test_01520() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1520)


def test_01521() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1521)


def test_01522() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1522)


def test_01523() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1523)


def test_01524() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1524)


def test_01525() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1525)


def test_01526() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1526)


def test_01527() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1527)


def test_01528() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1528)


def test_01529() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1529)


def test_01530() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1530)


def test_01531() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1531)


def test_01532() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1532)


def test_01533() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1533)


def test_01534() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1534)


def test_01535() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1535)


def test_01536() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1536)


def test_01537() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1537)


def test_01538() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1538)


def test_01539() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1539)


def test_01540() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1540)


def test_01541() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1541)


def test_01542() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1542)


def test_01543() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1543)


def test_01544() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1544)


def test_01545() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1545)


def test_01546() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1546)


def test_01547() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1547)


def test_01548() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1548)


def test_01549() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1549)


def test_01550() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1550)


def test_01551() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1551)


def test_01552() -> None:
    routine(test=1552)


def test_01553() -> None:
    routine(test=1553)


def test_01554() -> None:
    routine(test=1554)


def test_01555() -> None:
    routine(test=1555)


def test_01556() -> None:
    routine(test=1556)


def test_01557() -> None:
    routine(test=1557)


def test_01558() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1558)


def test_01559() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1559)


def test_01560() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1560)


def test_01561() -> None:
    routine(test=1561)


def test_01562() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1562)


def test_01563() -> None:
    routine(test=1563)


def test_01564() -> None:
    routine(test=1564)


def test_01565() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1565)


def test_01566() -> None:
    routine(test=1566)


def test_01567() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1567)


def test_01568() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1568)


def test_01569() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1569)


def test_01570() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1570)


def test_01571() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1571)


def test_01572() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1572)


def test_01573() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1573)


def test_01574() -> None:
    routine(test=1574)


def test_01575() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1575)


def test_01576() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1576)


def test_01577() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1577)


def test_01578() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1578)


def test_01579() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1579)


def test_01580() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1580)


def test_01581() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1581)


def test_01582() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1582)


def test_01583() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1583)


def test_01584() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1584)


def test_01585() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1585)


def test_01586() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1586)


def test_01587() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1587)


def test_01588() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1588)


def test_01589() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1589)


def test_01590() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1590)


def test_01591() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1591)


def test_01592() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1592)


def test_01593() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1593)


def test_01594() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1594)


def test_01595() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1595)


def test_01596() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1596)


def test_01597() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1597)


def test_01598() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1598)


def test_01599() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1599)


def test_01600() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1600)


def test_01601() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1601)


def test_01602() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1602)


def test_01603() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1603)


def test_01604() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1604)


def test_01605() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1605)


def test_01606() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1606)


def test_01607() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1607)


def test_01608() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1608)


def test_01609() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1609)


def test_01610() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1610)


def test_01611() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1611)


def test_01612() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1612)


def test_01613() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1613)


def test_01614() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1614)


def test_01615() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1615)


def test_01616() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1616)


def test_01617() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1617)


def test_01618() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1618)


def test_01619() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1619)


def test_01620() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1620)


def test_01621() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1621)


def test_01622() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1622)


def test_01623() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1623)


def test_01624() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1624)


def test_01625() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1625)


def test_01626() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1626)


def test_01627() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1627)


def test_01628() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1628)


def test_01629() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1629)


def test_01630() -> None:
    with pytest.raises(NotImplementedError):  # FBA model
        routine(test=1630)


def test_01631() -> None:
    routine(test=1631)


def test_01632() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1632)


def test_01633() -> None:
    routine(test=1633)


def test_01634() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1634)


def test_01635() -> None:
    routine(test=1635)


def test_01636() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1636)


def test_01637() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1637)


def test_01638() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1638)


def test_01639() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1639)


def test_01640() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1640)


def test_01641() -> None:
    routine(test=1641)


def test_01642() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1642)


def test_01643() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1643)


def test_01644() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1644)


def test_01645() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1645)


def test_01646() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1646)


def test_01647() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1647)


def test_01648() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1648)


def test_01649() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1649)


def test_01650() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1650)


def test_01651() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1651)


def test_01652() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1652)


def test_01653() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1653)


def test_01654() -> None:
    routine(test=1654)


def test_01655() -> None:
    routine(test=1655)


def test_01656() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1656)


def test_01657() -> None:
    routine(test=1657)


def test_01658() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1658)


def test_01659() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1659)


def test_01660() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1660)


def test_01661() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1661)


def test_01662() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1662)


def test_01663() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1663)


def test_01664() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1664)


def test_01665() -> None:
    routine(test=1665)


def test_01666() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1666)


def test_01667() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1667)


def test_01668() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1668)


def test_01669() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1669)


def test_01670() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1670)


def test_01671() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1671)


def test_01672() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1672)


def test_01673() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1673)


def test_01674() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1674)


def test_01675() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1675)


def test_01676() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1676)


def test_01677() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1677)


def test_01678() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1678)


def test_01679() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1679)


def test_01680() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1680)


def test_01681() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1681)


def test_01682() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1682)


def test_01683() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1683)


def test_01684() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1684)


def test_01685() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1685)


def test_01686() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1686)


def test_01687() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1687)


def test_01688() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1688)


def test_01689() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1689)


def test_01690() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1690)


def test_01691() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1691)


def test_01692() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1692)


def test_01693() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1693)


def test_01694() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1694)


def test_01695() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1695)


def test_01696() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1696)


def test_01697() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1697)


def test_01698() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1698)


def test_01699() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1699)


def test_01700() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1700)


def test_01701() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1701)


def test_01702() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1702)


def test_01703() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1703)


def test_01704() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1704)


def test_01705() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1705)


def test_01706() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1706)


def test_01707() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1707)


def test_01708() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1708)


def test_01709() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1709)


def test_01710() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1710)


def test_01711() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1711)


def test_01712() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1712)


def test_01713() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1713)


def test_01714() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1714)


def test_01715() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1715)


def test_01716() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1716)


def test_01717() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1717)


def test_01718() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1718)


def test_01719() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1719)


def test_01720() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1720)


def test_01721() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1721)


def test_01722() -> None:
    routine(test=1722)


def test_01723() -> None:
    routine(test=1723)


def test_01724() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1724)


def test_01725() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1725)


def test_01726() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1726)


def test_01727() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1727)


def test_01728() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1728)


def test_01729() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1729)


def test_01730() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1730)


def test_01731() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1731)


def test_01732() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1732)


def test_01733() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1733)


def test_01734() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1734)


def test_01735() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1735)


def test_01736() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1736)


def test_01737() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1737)


def test_01738() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1738)


def test_01739() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1739)


def test_01740() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1740)


def test_01741() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1741)


def test_01742() -> None:
    routine(test=1742)


def test_01743() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1743)


def test_01744() -> None:
    routine(test=1744)


def test_01745() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1745)


def test_01746() -> None:
    routine(test=1746)


def test_01747() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1747)


def test_01748() -> None:
    routine(test=1748)


def test_01749() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1749)


def test_01750() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1750)


def test_01751() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1751)


def test_01752() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1752)


def test_01753() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1753)


def test_01754() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1754)


def test_01755() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1755)


def test_01756() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1756)


def test_01757() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1757)


def test_01758() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1758)


def test_01759() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1759)


def test_01760() -> None:
    routine(test=1760)


def test_01761() -> None:
    routine(test=1761)


def test_01762() -> None:
    routine(test=1762)


def test_01763() -> None:
    routine(test=1763)


def test_01764() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1764)


def test_01765() -> None:
    with pytest.raises(FileNotFoundError):
        routine(test=1765)


def test_01766() -> None:
    routine(test=1766)


def test_01767() -> None:
    routine(test=1767)


def test_01768() -> None:
    routine(test=1768)


def test_01769() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1769)


def test_01770() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1770)


def test_01771() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1771)


def test_01772() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1772)


def test_01773() -> None:
    routine(test=1773)


def test_01774() -> None:
    routine(test=1774)


def test_01775() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1775)


def test_01776() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1776)


def test_01777() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1777)


def test_01778() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1778)


def test_01779() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1779)


def test_01780() -> None:
    with pytest.raises(NotImplementedError):
        routine(test=1780)
