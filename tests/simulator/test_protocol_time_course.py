import logging

import numpy as np
import pandas as pd
import pytest

from mxlpy import Model, Simulator, fns, make_protocol, unwrap


def get_model() -> Model:
    return (
        Model()
        .add_variables({"x1": 0.0})
        .add_parameters({"kf": 1.0})
        .add_reaction(
            "v1",
            fns.constant,
            args=["kf"],
            stoichiometry={"x1": 1.0},
        )
    )


def test_same() -> None:
    """Here we have the exact same return points for the protocol and
    the time course.

    Protocol:    [0, 1, 3, 6]
    Time Points: [0, 1, 3, 6]
    """
    protocol = make_protocol(
        [
            (1, {"kf": 1}),  # for one second value of 1
            (2, {"kf": 2}),  # for two seconds value of 2
            (3, {"kf": 1}),  # for three seconds value of 1
        ]
    )
    variables = unwrap(
        Simulator(get_model())
        .simulate_protocol_time_course(
            protocol,
            time_points=[0, 1, 3, 6],
        )
        .get_result()
    ).variables

    pd.testing.assert_frame_equal(
        variables,
        pd.DataFrame(
            {
                "x1": {
                    0.0: 0.0,
                    1.0: 1.0,
                    3.0: 5.0,
                    6.0: 8.0,
                }
            },
            dtype="float64",
        ),
        atol=1e-6,
        rtol=1e-6,
    )


def test_in_between() -> None:
    """Here the time points are all between the protocol points

    The expected behaviour here is to get both the protocol and the
    time points

    Protocol:    [0, 1, 3, 6]
    Time Points: [2, 4, 5]
    """
    protocol = make_protocol(
        [
            (1, {"kf": 1}),  # for one second value of 1
            (2, {"kf": 2}),  # for two seconds value of 2
            (3, {"kf": 1}),  # for three seconds value of 1
        ]
    )
    variables = unwrap(
        Simulator(get_model())
        .simulate_protocol_time_course(
            protocol,
            time_points=[2, 4, 5],
        )
        .get_result()
    ).variables

    pd.testing.assert_frame_equal(
        variables,
        pd.DataFrame(
            {
                "x1": {
                    0.0: 0.0,
                    1.0: 1.0,
                    2.0: 3.0,
                    3.0: 5.0,
                    4.0: 6.0,
                    5.0: 7.0,
                    6.0: 8.0,
                }
            },
            dtype="float64",
        ),
        atol=1e-6,
        rtol=1e-6,
    )


def test_both() -> None:
    """Here time points hit all protocol points and in between

    Protocol:    [0, 1, 3, 6]
    Time Points: [0, 1, 2, 3, 4, 5, 6]
    """
    protocol = make_protocol(
        [
            (1, {"kf": 1}),  # for one second value of 1
            (2, {"kf": 2}),  # for two seconds value of 2
            (3, {"kf": 1}),  # for three seconds value of 1
        ]
    )
    variables = unwrap(
        Simulator(get_model())
        .simulate_protocol_time_course(
            protocol,
            time_points=np.linspace(0, 6, 7),
        )
        .get_result()
    ).variables

    pd.testing.assert_frame_equal(
        variables,
        pd.DataFrame(
            {
                "x1": {
                    0.0: 0.0,
                    1.0: 1.0,
                    2.0: 3.0,
                    3.0: 5.0,
                    4.0: 6.0,
                    5.0: 7.0,
                    6.0: 8.0,
                }
            },
            dtype="float64",
        ),
        atol=1e-6,
        rtol=1e-6,
    )


def test_time_points_outside_of_protocol(caplog: pytest.LogCaptureFixture) -> None:
    """Here all time points are outside the simulation range

    Protocol:    [0, 1, 3, 6]
    Time Points: [7, 8, 9]
    """
    protocol = make_protocol(
        [
            (1, {"kf": 1}),  # for one second value of 1
            (2, {"kf": 2}),  # for two seconds value of 2
            (3, {"kf": 1}),  # for three seconds value of 1
        ]
    )

    with caplog.at_level(logging.WARNING):
        variables = unwrap(
            Simulator(get_model())
            .simulate_protocol_time_course(
                protocol,
                time_points=[7, 8, 9],
            )
            .get_result()
        ).variables

    pd.testing.assert_frame_equal(
        variables,
        pd.DataFrame(
            {
                "x1": {
                    0.0: 0.0,
                    1.0: 1.0,
                    3.0: 5.0,
                    6.0: 8.0,
                }
            },
            dtype="float64",
        ),
        atol=1e-6,
        rtol=1e-6,
    )
