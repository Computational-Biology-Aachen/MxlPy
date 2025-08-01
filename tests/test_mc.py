from __future__ import annotations

import numpy as np
import pandas as pd

from mxlpy import Model, fns, make_protocol, mc


def get_simple_model() -> Model:
    model = Model()
    model.add_parameters({"k0": 1.0, "k1": 1.0, "k2": 2.0})
    model.add_variables({"S": 0.0, "P": 0.0})

    model.add_reaction(
        "v0",
        fn=fns.constant,
        args=["k0"],
        stoichiometry={"S": 1.0},
    )

    model.add_reaction(
        "v1",
        fn=fns.mass_action_1s,
        args=["S", "k1"],
        stoichiometry={"S": -1.0, "P": 1.0},
    )

    model.add_reaction(
        "v2",
        fn=fns.mass_action_1s,
        args=["P", "k2"],
        stoichiometry={"P": -1.0},
    )

    return model


def get_to_scan() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "k0": [1.0, 1.1, 1.2],
        }
    )


def get_mc_to_scan() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "k1": [1.0, 1.1, 1.2],
            "k2": [2.0, 2.1, 2.2],
        }
    )


def get_protocol() -> pd.DataFrame:
    return make_protocol(
        [
            (1.0, {"k0": 2.0}),
        ]
    )


def test_steady_state() -> None:
    res = mc.steady_state(get_simple_model(), mc_to_scan=get_mc_to_scan())

    # Args
    args = pd.DataFrame(
        {
            "S": {
                (1.0, 2.0): 0.9999999999992708,
                (1.1, 2.1): 0.9090909090898309,
                (1.2, 2.2): 0.8333333333323488,
            },
            "P": {
                (1.0, 2.0): 0.49999999999927136,
                (1.1, 2.1): 0.4761904761892899,
                (1.2, 2.2): 0.45454545454427325,
            },
            "v0": {(1.0, 2.0): 1.0, (1.1, 2.1): 1.0, (1.2, 2.2): 1.0},
            "v1": {
                (1.0, 2.0): 0.9999999999992708,
                (1.1, 2.1): 0.9999999999988141,
                (1.2, 2.2): 0.9999999999988185,
            },
            "v2": {
                (1.0, 2.0): 0.9999999999985427,
                (1.1, 2.1): 0.9999999999975088,
                (1.2, 2.2): 0.9999999999974012,
            },
        },
        dtype=float,
    )
    args.index.names = ["k1", "k2"]
    pd.testing.assert_frame_equal(
        res.get_args(),
        args,
        atol=1e-6,
        rtol=1e-6,
    )

    # Variables
    variables = pd.DataFrame(
        {
            "S": {
                (1.0, 2.0): 0.9999999999992708,
                (1.1, 2.1): 0.9090909090898309,
                (1.2, 2.2): 0.8333333333323488,
            },
            "P": {
                (1.0, 2.0): 0.49999999999927136,
                (1.1, 2.1): 0.4761904761892899,
                (1.2, 2.2): 0.45454545454427325,
            },
        },
        dtype=float,
    )
    variables.index.names = ["k1", "k2"]
    pd.testing.assert_frame_equal(
        res.variables,
        variables,
        atol=1e-6,
        rtol=1e-6,
    )

    # Fluxes
    fluxes = pd.DataFrame(
        {
            "v0": {(1.0, 2.0): 1.0, (1.1, 2.1): 1.0, (1.2, 2.2): 1.0},
            "v1": {
                (1.0, 2.0): 0.9999999999992708,
                (1.1, 2.1): 0.9999999999988141,
                (1.2, 2.2): 0.9999999999988185,
            },
            "v2": {
                (1.0, 2.0): 0.9999999999985427,
                (1.1, 2.1): 0.9999999999975088,
                (1.2, 2.2): 0.9999999999974012,
            },
        },
        dtype=float,
    )
    fluxes.index.names = ["k1", "k2"]
    pd.testing.assert_frame_equal(
        res.fluxes,
        fluxes,
        atol=1e-6,
        rtol=1e-6,
    )


def test_time_course() -> None:
    res = mc.time_course(
        get_simple_model(),
        mc_to_scan=get_mc_to_scan(),
        time_points=np.array([1.0], dtype=float),
    )
    # args
    args = pd.DataFrame(
        {
            "S": {
                (0, 0.0): 0.0,
                (0, 1.0): 0.632120556479658,
                (1, 0.0): 0.0,
                (1, 1.0): 0.6064808307990998,
                (2, 0.0): 0.0,
                (2, 1.0): 0.582338154772543,
            },
            "P": {
                (0, 0.0): 0.0,
                (0, 1.0): 0.19978819487494498,
                (1, 0.0): 0.0,
                (1, 1.0): 0.20746322963145172,
                (2, 0.0): 0.0,
                (2, 1.0): 0.21378932105196793,
            },
            "v0": {
                (0, 0.0): 1.0,
                (0, 1.0): 1.0,
                (1, 0.0): 1.0,
                (1, 1.0): 1.0,
                (2, 0.0): 1.0,
                (2, 1.0): 1.0,
            },
            "v1": {
                (0, 0.0): 0.0,
                (0, 1.0): 0.632120556479658,
                (1, 0.0): 0.0,
                (1, 1.0): 0.6671289138790097,
                (2, 0.0): 0.0,
                (2, 1.0): 0.6988057857270517,
            },
            "v2": {
                (0, 0.0): 0.0,
                (0, 1.0): 0.39957638974988996,
                (1, 0.0): 0.0,
                (1, 1.0): 0.43567278222604866,
                (2, 0.0): 0.0,
                (2, 1.0): 0.4703365063143295,
            },
        },
        dtype=float,
    )
    args.index.names = ["n", "time"]
    pd.testing.assert_frame_equal(
        res.get_args(),
        args,
        atol=1e-6,
        rtol=1e-6,
    )


def test_time_course_over_protocol() -> None:
    res = mc.protocol(
        get_simple_model(),
        mc_to_scan=get_mc_to_scan(),
        protocol=get_protocol(),
        time_points_per_step=1,
    )
    # args
    args = pd.DataFrame(
        {
            "S": {
                (0, 0.0): 0.0,
                (0, 1.0): 1.2642411153682063,
                (1, 0.0): 0.0,
                (1, 1.0): 1.2129616639267933,
                (2, 0.0): 0.0,
                (2, 1.0): 1.1646763116435936,
            },
            "P": {
                (0, 0.0): 0.0,
                (0, 1.0): 0.39957639706242154,
                (1, 0.0): 0.0,
                (1, 1.0): 0.41492646571750874,
                (2, 0.0): 0.0,
                (2, 1.0): 0.4275786502107995,
            },
            "v0": {
                (0, 0.0): 2.0,
                (0, 1.0): 2.0,
                (1, 0.0): 2.0,
                (1, 1.0): 2.0,
                (2, 0.0): 2.0,
                (2, 1.0): 2.0,
            },
            "v1": {
                (0, 0.0): 0.0,
                (0, 1.0): 1.2642411153682063,
                (1, 0.0): 0.0,
                (1, 1.0): 1.3342578303194728,
                (2, 0.0): 0.0,
                (2, 1.0): 1.3976115739723123,
            },
            "v2": {
                (0, 0.0): 0.0,
                (0, 1.0): 0.7991527941248431,
                (1, 0.0): 0.0,
                (1, 1.0): 0.8713455780067684,
                (2, 0.0): 0.0,
                (2, 1.0): 0.9406730304637589,
            },
        },
        dtype=float,
    )
    args.index.names = ["n", "time"]
    pd.testing.assert_frame_equal(
        res.get_args(),
        args,
        atol=1e-6,
        rtol=1e-6,
    )


def test_scan_steady_state() -> None:
    res = mc.scan_steady_state(
        get_simple_model(),
        to_scan=get_to_scan(),
        mc_to_scan=get_mc_to_scan(),
    )
    # args
    args = pd.DataFrame(
        {
            "S": {
                (0, 1.0): 1.0000000000015183,
                (0, 1.1): 1.1000000000014485,
                (0, 1.2): 1.2000000000018252,
                (1, 1.0): 0.9090909090916395,
                (1, 1.1): 1.0000000000013443,
                (1, 1.2): 1.0909090909101535,
                (2, 1.0): 0.8333333333342008,
                (2, 1.1): 0.9166666666675458,
                (2, 1.2): 1.0000000000012899,
            },
            "P": {
                (0, 1.0): 0.5000000000015187,
                (0, 1.1): 0.5500000000014477,
                (0, 1.2): 0.6000000000018254,
                (1, 1.0): 0.4761904761912797,
                (1, 1.1): 0.5238095238110023,
                (1, 1.2): 0.5714285714297405,
                (2, 1.0): 0.45454545454649536,
                (2, 1.1): 0.5000000000010547,
                (2, 1.2): 0.5454545454560934,
            },
        },
        dtype=float,
    )
    args.index.names = [None, "k0"]
    pd.testing.assert_frame_equal(
        res.variables,
        args,
        atol=1e-6,
        rtol=1e-6,
    )


def test_variable_elasticities() -> None:
    res = mc.variable_elasticities(
        get_simple_model(),
        mc_to_scan=get_mc_to_scan(),
        variables={"S": 1.0, "P": 0.5},
    )
    elasticities = pd.DataFrame(
        {
            "S": {
                (0, "v0"): 0.0,
                (0, "v1"): 0.9999999999998899,
                (0, "v2"): 0.0,
                (1, "v0"): 0.0,
                (1, "v1"): 1.0000000000004954,
                (1, "v2"): 0.0,
                (2, "v0"): 0.0,
                (2, "v1"): 0.9999999999991498,
                (2, "v2"): 0.0,
            },
            "P": {
                (0, "v0"): 0.0,
                (0, "v1"): 0.0,
                (0, "v2"): 0.9999999999998899,
                (1, "v0"): 0.0,
                (1, "v1"): 0.0,
                (1, "v2"): 1.000000000000207,
                (2, "v0"): 0.0,
                (2, "v1"): 0.0,
                (2, "v2"): 1.0000000000004954,
            },
        },
        dtype=float,
    )
    pd.testing.assert_frame_equal(res, elasticities)


def test_parameter_elasticities() -> None:
    res = mc.parameter_elasticities(
        get_simple_model(),
        mc_to_scan=get_mc_to_scan(),
        to_scan=["k0"],
        variables={"S": 1.0, "P": 0.5},
    )
    elasticities = pd.DataFrame(
        {
            "k0": {
                (0, "v0"): 0.9999999999998899,
                (0, "v1"): 0.0,
                (0, "v2"): 0.0,
                (1, "v0"): 0.9999999999998899,
                (1, "v1"): 0.0,
                (1, "v2"): 0.0,
                (2, "v0"): 0.9999999999998899,
                (2, "v1"): 0.0,
                (2, "v2"): 0.0,
            }
        },
        dtype=float,
    )
    pd.testing.assert_frame_equal(res, elasticities)


def test_response_coefficients() -> None:
    res = mc.response_coefficients(
        get_simple_model(),
        mc_to_scan=get_mc_to_scan(),
        to_scan=["k0"],
    )
    coefs = pd.DataFrame(
        {
            "k0": {
                (0, "S"): 1.000000000001147,
                (0, "P"): 1.0000000000029587,
                (1, "S"): 0.999999999984487,
                (1, "P"): 0.9999999999679767,
                (2, "S"): 1.0000000000048441,
                (2, "P"): 1.000000000009868,
            }
        },
        dtype=float,
    )
    pd.testing.assert_frame_equal(res.variables, coefs)
