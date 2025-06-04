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


def test_sim_con_sim_sim() -> None:
    variables = unwrap(
        Simulator(get_model())  # break for readability
        .simulate(3, steps=3)
        .simulate(4, steps=1)
        .get_result()
    ).variables

    pd.testing.assert_frame_equal(
        variables,
        pd.DataFrame(
            {
                "x1": {
                    0.0: 0.0,
                    1.0: 1.0,
                    2.0: 2.0,
                    3.0: 3.0,
                    4.0: 4.0,
                }
            },
            dtype="float64",
        ),
        atol=1e-6,
        rtol=1e-6,
    )


def test_con_sim_tc() -> None:
    expected = pd.DataFrame(
        {
            "x1": {
                0.0: 0.0,
                1.0: 1.0,
                2.0: 2.0,
                3.0: 3.0,
                4.0: 4.0,
            }
        },
        dtype="float64",
    )

    # Test both case with 3 and without
    variables = unwrap(
        Simulator(get_model())  # break for readability
        .simulate(3, steps=3)
        .simulate_time_course([4])
        .get_result()
    ).variables
    pd.testing.assert_frame_equal(
        variables,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )

    variables = unwrap(
        Simulator(get_model())
        .simulate(3, steps=3)
        .simulate_time_course([3, 4])
        .get_result()
    ).variables
    pd.testing.assert_frame_equal(
        variables,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


def test_con_tc_sim() -> None:
    expected = pd.DataFrame(
        {
            "x1": {
                0.0: 0.0,
                1.0: 1.0,
                2.0: 2.0,
                3.0: 3.0,
                4.0: 4.0,
            }
        },
        dtype="float64",
    )

    # Test both case with 3 and without
    variables = unwrap(
        Simulator(get_model())  # break for readability
        .simulate_time_course([0, 1, 2, 3])
        .simulate(4, steps=1)
        .get_result()
    ).variables
    pd.testing.assert_frame_equal(
        variables,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )
    variables = unwrap(
        Simulator(get_model())
        .simulate_time_course([0, 1, 2, 3])
        .simulate_time_course([4])
        .get_result()
    ).variables
    pd.testing.assert_frame_equal(
        variables,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


def test_con_tc_tc() -> None:
    expected = pd.DataFrame(
        {
            "x1": {
                0.0: 0.0,
                1.0: 1.0,
                2.0: 2.0,
                3.0: 3.0,
                4.0: 4.0,
            }
        },
        dtype="float64",
    )

    # Test both case with 3 and without
    variables = unwrap(
        Simulator(get_model())  # break for readability
        .simulate_time_course([0, 1, 2, 3])
        .simulate_time_course([3, 4])
        .get_result()
    ).variables
    pd.testing.assert_frame_equal(
        variables,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )
    variables = unwrap(
        Simulator(get_model())
        .simulate_time_course([0, 1, 2, 3])
        .simulate_time_course([4])
        .get_result()
    ).variables
    pd.testing.assert_frame_equal(
        variables,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


###############################################################################
# Include protocols
###############################################################################


def test_con_sim_proto() -> None:
    protocol = make_protocol(
        [
            (1, {"kf": 1}),  # for one second value of 1
            (2, {"kf": 2}),  # for two seconds value of 2
            (3, {"kf": 1}),  # for three seconds value of 1
        ]
    )
    expected = pd.DataFrame(
        {
            "x1": {
                0.0: 0.0,
                1.0: 1.0,
                2.0: 2.0,
                3.0: 3.0,
                4.0: 4.0,
                5.0: 5.0,
                6.0: 6.0,
                7.0: 7.0,
                9.0: 11.0,
                12.0: 14.0,
            }
        },
        dtype=float,
    )
    variables = unwrap(
        Simulator(get_model())
        .simulate(6, steps=6)
        .simulate_protocol(protocol, time_points_per_step=1)
        .get_result()
    ).variables
    pd.testing.assert_frame_equal(
        variables,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


def test_con_tc_proto() -> None:
    protocol = make_protocol(
        [
            (1, {"kf": 1}),  # for one second value of 1
            (2, {"kf": 2}),  # for two seconds value of 2
            (3, {"kf": 1}),  # for three seconds value of 1
        ]
    )
    expected = pd.DataFrame(
        {
            "x1": {
                0.0: 0.0,
                1.0: 1.0,
                2.0: 2.0,
                3.0: 3.0,
                4.0: 4.0,
                5.0: 5.0,
                6.0: 6.0,
                7.0: 7.0,
                9.0: 11.0,
                12.0: 14.0,
            }
        },
        dtype=float,
    )
    variables = unwrap(
        Simulator(get_model())
        .simulate_time_course(np.linspace(0, 6, 7))
        .simulate_protocol(protocol, time_points_per_step=1)
        .get_result()
    ).variables
    pd.testing.assert_frame_equal(
        variables,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


def test_con_proto_sim() -> None:
    protocol = make_protocol(
        [
            (1, {"kf": 1}),  # for one second value of 1
            (2, {"kf": 2}),  # for two seconds value of 2
            (3, {"kf": 1}),  # for three seconds value of 1
        ]
    )
    expected = pd.DataFrame(
        {
            "x1": {
                0.0: 0.0,
                1.0: 1.0,
                3.0: 5.0,
                6.0: 8.0,
                7.0: 9.0,
                8.0: 10.0,
                9.0: 11.0,
                10.0: 12.0,
                11.0: 13.0,
                12.0: 14.0,
            }
        },
        dtype=float,
    )
    variables = unwrap(
        Simulator(get_model())
        .simulate_protocol(protocol, time_points_per_step=1)
        .simulate(12, steps=6)
        .get_result()
    ).variables
    pd.testing.assert_frame_equal(
        variables,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


def test_con_proto_tc() -> None:
    protocol = make_protocol(
        [
            (1, {"kf": 1}),  # for one second value of 1
            (2, {"kf": 2}),  # for two seconds value of 2
            (3, {"kf": 1}),  # for three seconds value of 1
        ]
    )
    expected = pd.DataFrame(
        {
            "x1": {
                0.0: 0.0,
                1.0: 1.0,
                3.0: 5.0,
                6.0: 8.0,
                7.0: 9.0,
                8.0: 10.0,
                9.0: 11.0,
                10.0: 12.0,
                11.0: 13.0,
                12.0: 14.0,
            }
        },
        dtype=float,
    )
    variables = unwrap(
        Simulator(get_model())
        .simulate_protocol(protocol, time_points_per_step=1)
        .simulate_time_course(np.linspace(7, 12, 6))
        .get_result()
    ).variables
    pd.testing.assert_frame_equal(
        variables,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


def test_con_proto_proto() -> None:
    protocol = make_protocol(
        [
            (1, {"kf": 1}),  # for one second value of 1
            (2, {"kf": 2}),  # for two seconds value of 2
            (3, {"kf": 1}),  # for three seconds value of 1
        ]
    )
    expected = pd.DataFrame(
        {
            "x1": {
                0.0: 0.0,
                1.0: 1.0,
                3.0: 5.0,
                6.0: 8.0,
                7.0: 9.0,
                9.0: 13.0,
                12.0: 16.0,
            }
        },
        dtype=float,
    )
    variables = unwrap(
        Simulator(get_model())
        .simulate_protocol(protocol, time_points_per_step=1)
        .simulate_protocol(protocol, time_points_per_step=1)
        .get_result()
    ).variables
    pd.testing.assert_frame_equal(
        variables,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


###############################################################################
# Protocol time courses
###############################################################################


def test_con_sim_ptc() -> None:
    protocol = make_protocol(
        [
            (1, {"kf": 1}),  # for one second value of 1
            (2, {"kf": 2}),  # for two seconds value of 2
            (3, {"kf": 1}),  # for three seconds value of 1
        ]
    )
    expected = pd.DataFrame(
        {
            "x1": {
                0.0: 0.0,
                1.0: 1.0,
                2.0: 2.0,
                3.0: 3.0,
                4.0: 4.0,
                5.0: 5.0,
                6.0: 6.0,
                7.0: 7.0,
                8.0: 9.0,
                9.0: 11.0,
                10.0: 12.0,
                11.0: 13.0,
                12.0: 14.0,
            }
        },
        dtype=float,
    )
    variables = unwrap(
        Simulator(get_model())
        .simulate(6, steps=6)
        .simulate_protocol_time_course(
            protocol,
            time_points=np.linspace(7, 12, 6),
        )
        .get_result()
    ).variables
    pd.testing.assert_frame_equal(
        variables,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


def test_con_sim_ptc_rel() -> None:
    protocol = make_protocol(
        [
            (1, {"kf": 1}),  # for one second value of 1
            (2, {"kf": 2}),  # for two seconds value of 2
            (3, {"kf": 1}),  # for three seconds value of 1
        ]
    )
    expected = pd.DataFrame(
        {
            "x1": {
                0.0: 0.0,
                1.0: 1.0,
                2.0: 2.0,
                3.0: 3.0,
                4.0: 4.0,
                5.0: 5.0,
                6.0: 6.0,
                7.0: 7.0,
                8.0: 9.0,
                9.0: 11.0,
                10.0: 12.0,
                11.0: 13.0,
                12.0: 14.0,
            }
        },
        dtype=float,
    )
    variables = unwrap(
        Simulator(get_model())
        .simulate(6, steps=6)
        .simulate_protocol_time_course(
            protocol,
            time_points=np.linspace(0, 6, 7),
            time_points_as_relative=True,
        )
        .get_result()
    ).variables
    pd.testing.assert_frame_equal(
        variables,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


def test_con_ptc_sim() -> None:
    protocol = make_protocol(
        [
            (1, {"kf": 1}),  # for one second value of 1
            (2, {"kf": 2}),  # for two seconds value of 2
            (3, {"kf": 1}),  # for three seconds value of 1
        ]
    )
    expected = pd.DataFrame(
        {
            "x1": {
                0.0: 0.0,
                1.0: 1.0,
                2.0: 3.0,
                3.0: 5.0,
                4.0: 6.0,
                5.0: 7.0,
                6.0: 8.0,
                7.0: 9.0,
                8.0: 10.0,
                9.0: 11.0,
                10.0: 12.0,
                11.0: 13.0,
                12.0: 14.0,
            }
        },
        dtype=float,
    )
    variables = unwrap(
        Simulator(get_model())
        .simulate_protocol_time_course(
            protocol,
            time_points=np.linspace(0, 6, 7),
        )
        .simulate(12, steps=6)
        .get_result()
    ).variables
    pd.testing.assert_frame_equal(
        variables,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


def test_con_ptc_ptc() -> None:
    protocol = make_protocol(
        [
            (1, {"kf": 1}),  # for one second value of 1
            (2, {"kf": 2}),  # for two seconds value of 2
            (3, {"kf": 1}),  # for three seconds value of 1
        ]
    )
    expected = pd.DataFrame(
        {
            "x1": {
                0.0: 0.0,
                1.0: 1.0,
                2.0: 3.0,
                3.0: 5.0,
                4.0: 6.0,
                5.0: 7.0,
                6.0: 8.0,
                7.0: 9.0,
                8.0: 11.0,
                9.0: 13.0,
                10.0: 14.0,
                11.0: 15.0,
                12.0: 16.0,
            }
        },
        dtype=float,
    )
    variables = unwrap(
        Simulator(get_model())
        .simulate_protocol_time_course(
            protocol,
            time_points=np.linspace(0, 6, 7),
        )
        .simulate_protocol_time_course(
            protocol,
            time_points=np.linspace(7, 12, 6),
        )
        .get_result()
    ).variables
    pd.testing.assert_frame_equal(
        variables,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


###############################################################################
# Failure nicely on wrong user input
###############################################################################


def test_fail_con_sim_sim_same_end() -> None:
    with pytest.raises(ValueError):
        _ = (
            Simulator(get_model())  # fmt
            .simulate(3, steps=3)
            .simulate(3, steps=3)
        )


def test_fail_con_sim_tc_same_end() -> None:
    with pytest.raises(ValueError):
        _ = (
            Simulator(get_model())  # fmt
            .simulate(3, steps=3)
            .simulate_time_course([3])
        )


def test_fail_con_tc_sim_same_end() -> None:
    with pytest.raises(ValueError):
        _ = (
            Simulator(get_model())  # fmt
            .simulate_time_course([0, 1, 2, 3])
            .simulate(3)
        )


def test_fail_con_tc_tc_same_end() -> None:
    with pytest.raises(ValueError):
        _ = (
            Simulator(get_model())  # fmt
            .simulate_time_course([0, 1, 2, 3])
            .simulate_time_course([0, 1, 2, 3])
        )


def test_fail_con_sim_ptc_same_end() -> None:
    protocol = make_protocol(
        [
            (1, {"kf": 1}),  # for one second value of 1
            (2, {"kf": 2}),  # for two seconds value of 2
            (3, {"kf": 1}),  # for three seconds value of 1
        ]
    )

    with pytest.raises(ValueError):
        _ = (
            Simulator(get_model())
            .simulate(6, steps=6)
            .simulate_protocol_time_course(
                protocol,
                time_points=np.linspace(0, 6, 7),
            )
        )
