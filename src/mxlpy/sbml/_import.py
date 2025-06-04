from __future__ import annotations

import logging
import math  # noqa: F401  # models might need it
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self

import libsbml
import numpy as np  # noqa: F401  # models might need it
import sympy

from mxlpy.model import Dependency, Model, _sort_dependencies
from mxlpy.paths import default_tmp_dir
from mxlpy.sbml._data import (
    AtomicUnit,
    Compartment,
    CompositeUnit,
    Compound,
    Derived,
    Function,
    Parameter,
    Reaction,
)
from mxlpy.sbml._mathml import parse_sbml_math
from mxlpy.sbml._name_conversion import _name_to_py
from mxlpy.sbml._unit_conversion import get_operator_mappings, get_unit_conversion
from mxlpy.types import unwrap

if TYPE_CHECKING:
    from collections.abc import Callable

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "INDENT",
    "OPERATOR_MAPPINGS",
    "Parser",
    "UNIT_CONVERSION",
    "import_from_path",
    "read",
    "valid_filename",
]

UNIT_CONVERSION = get_unit_conversion()
OPERATOR_MAPPINGS = get_operator_mappings()
INDENT = "    "


def _nan_to_zero(value: float) -> float:
    return 0 if str(value) == "nan" else value


@dataclass(slots=True)
class Parser:
    # Collections
    boundary_species: set[str] = field(default_factory=set)
    # Parsed stuff
    atomic_units: dict[str, AtomicUnit] = field(default_factory=dict)
    composite_units: dict[str, CompositeUnit] = field(default_factory=dict)
    compartments: dict[str, Compartment] = field(default_factory=dict)
    parameters: dict[str, Parameter] = field(default_factory=dict)
    variables: dict[str, Compound] = field(default_factory=dict)
    derived: dict[str, Derived] = field(default_factory=dict)
    initial_assignment: dict[str, Derived] = field(default_factory=dict)
    functions: dict[str, Function] = field(default_factory=dict)
    reactions: dict[str, Reaction] = field(default_factory=dict)

    def parse(self, file: str | Path) -> Self:
        if not Path(file).exists():
            msg = "Model file does not exist"
            raise OSError(msg)
        doc = libsbml.readSBMLFromFile(str(file))

        # Check for unsupported packages
        for i in range(doc.num_plugins):
            if doc.getPlugin(i).getPackageName() == "comp":
                msg = "No support for comp package"
                raise NotImplementedError(msg)

        sbml_model = doc.getModel()
        if sbml_model is None:
            return self

        if bool(sbml_model.getConversionFactor()):
            msg = "Conversion factors are currently not supported"
            raise NotImplementedError(msg)

        self.parse_functions(sbml_model)
        self.parse_units(sbml_model)
        self.parse_compartments(sbml_model)
        self.parse_variables(sbml_model)
        self.parse_parameters(sbml_model)
        self.parse_initial_assignments(sbml_model)
        self.parse_rules(sbml_model)
        self.parse_constraints(sbml_model)
        self.parse_reactions(sbml_model)
        self.parse_events(sbml_model)

        # Modifications
        self._convert_substance_amount_to_concentration()
        # self._rename_species_references()
        return self

    ###############################################################################
    # PARSING STAGE
    ###############################################################################

    def parse_constraints(self, sbml_model: libsbml.Model) -> None:
        if len(sbml_model.getListOfConstraints()) > 0:
            msg = "mxlpy does not support model constraints. "
            raise NotImplementedError(msg)

    def parse_events(self, sbml_model: libsbml.Model) -> None:
        if len(sbml_model.getListOfEvents()) > 0:
            msg = (
                "mxlpy does not current support events. "
                "Check the file for how to integrate properly."
            )
            raise NotImplementedError(msg)

    def parse_units(self, sbml_model: libsbml.Model) -> None:
        for unit_definition in sbml_model.getListOfUnitDefinitions():
            composite_id = unit_definition.getId()
            local_units = []
            for unit in unit_definition.getListOfUnits():
                atomic_unit = AtomicUnit(
                    kind=UNIT_CONVERSION[unit.getKind()],
                    scale=unit.getScale(),
                    exponent=unit.getExponent(),
                    multiplier=unit.getMultiplier(),
                )
                local_units.append(atomic_unit.kind)
                self.atomic_units[atomic_unit.kind] = atomic_unit
            self.composite_units[composite_id] = CompositeUnit(
                sbml_id=composite_id,
                units=local_units,
            )

    def parse_compartments(self, sbml_model: libsbml.Model) -> None:
        for compartment in sbml_model.getListOfCompartments():
            sbml_id = _name_to_py(compartment.getId())
            size = compartment.getSize()
            if str(size) == "nan":
                size = 0
            self.compartments[sbml_id] = Compartment(
                name=compartment.getName(),
                dimensions=compartment.getSpatialDimensions(),
                size=size,
                units=compartment.getUnits(),
                is_constant=compartment.getConstant(),
            )

    def parse_parameters(self, sbml_model: libsbml.Model) -> None:
        for parameter in sbml_model.getListOfParameters():
            self.parameters[_name_to_py(parameter.getId())] = Parameter(
                value=_nan_to_zero(parameter.getValue()),
                is_constant=parameter.getConstant(),
            )

    def parse_variables(self, sbml_model: libsbml.Model) -> None:
        for compound in sbml_model.getListOfSpecies():
            compound_id = _name_to_py(compound.getId())
            if bool(compound.getConversionFactor()):
                msg = "Conversion factors are not implemented. "
                raise NotImplementedError(msg)

            # NOTE: What the shit is this?
            initial_amount = compound.getInitialAmount()
            if str(initial_amount) == "nan":
                initial_amount = compound.getInitialConcentration()
                is_concentration = str(initial_amount) != "nan"
            else:
                is_concentration = False

            has_boundary_condition = compound.getBoundaryCondition()
            if has_boundary_condition:
                self.boundary_species.add(compound_id)

            self.variables[compound_id] = Compound(
                compartment=compound.getCompartment(),
                initial_amount=_nan_to_zero(initial_amount),
                substance_units=compound.getSubstanceUnits(),
                has_only_substance_units=compound.getHasOnlySubstanceUnits(),
                has_boundary_condition=has_boundary_condition,
                is_constant=compound.getConstant(),
                is_concentration=is_concentration,
            )

    def parse_functions(self, sbml_model: libsbml.Model) -> None:
        for func in sbml_model.getListOfFunctionDefinitions():
            func_name = func.getName()
            sbml_id = func.getId()
            if sbml_id is None or sbml_id == "":
                sbml_id = func_name
            elif func_name is None or func_name == "":
                func_name = sbml_id
            func_name = _name_to_py(func_name)

            if (node := func.getMath()) is None:
                continue
            body, args = parse_sbml_math(node=node)

            self.functions[func_name] = Function(
                body=body,
                args=args,
            )

    ###############################################################################
    # Different kinds of derived values
    ###############################################################################

    def parse_initial_assignments(self, sbml_model: libsbml.Model) -> None:
        for assignment in sbml_model.getListOfInitialAssignments():
            name = _name_to_py(assignment.getSymbol())

            node = assignment.getMath()
            if node is None:
                msg = f"Unusable math for {name}"
                _LOGGER.warning(msg)
                continue

            body, args = parse_sbml_math(node)
            self.initial_assignment[name] = Derived(
                body=body,
                args=args,
            )

    def _parse_algebraic_rule(self, rule: libsbml.AlgebraicRule) -> None:
        msg = f"Algebraic rules are not implemented for {rule.getId()}"
        raise NotImplementedError(msg)

    def _parse_assignment_rule(self, rule: libsbml.AssignmentRule) -> None:
        if (node := rule.getMath()) is None:
            return

        name: str = _name_to_py(rule.getId())
        body, args = parse_sbml_math(node=node)

        self.derived[name] = Derived(
            body=body,
            args=args,
        )

    def _parse_rate_rule(self, rule: libsbml.RateRule) -> None:
        msg = f"Skipping rate rule {rule.getId()}"
        raise NotImplementedError(msg)

    def parse_rules(self, sbml_model: libsbml.Model) -> None:
        """Parse rules and separate them by type."""
        for rule in sbml_model.getListOfRules():
            if rule.element_name == "algebraicRule":
                self._parse_algebraic_rule(rule=rule)
            elif rule.element_name == "assignmentRule":
                self._parse_assignment_rule(rule=rule)
            elif rule.element_name == "rateRule":
                self._parse_rate_rule(rule=rule)
            else:
                msg = "Unknown rate type"
                raise ValueError(msg)

    def _parse_local_parameters(
        self, reaction_id: str, kinetic_law: libsbml.KineticLaw
    ) -> dict[str, str]:
        """Parse local parameters."""
        parameters_to_update = {}
        for parameter in kinetic_law.getListOfLocalParameters():
            old_id = _name_to_py(parameter.getId())
            if old_id in self.parameters:
                new_id = f"{reaction_id}__{old_id}"
                parameters_to_update[old_id] = new_id
            else:
                new_id = old_id
            self.parameters[new_id] = Parameter(
                value=_nan_to_zero(parameter.getValue()),
                is_constant=parameter.getConstant(),
            )
        # Some models apparently also write local parameters in this
        for parameter in kinetic_law.getListOfParameters():
            old_id = _name_to_py(parameter.getId())
            if old_id in self.parameters:
                new_id = f"{reaction_id}__{old_id}"
                parameters_to_update[old_id] = new_id
            else:
                new_id = old_id
            self.parameters[new_id] = Parameter(
                value=_nan_to_zero(parameter.getValue()),
                is_constant=parameter.getConstant(),
            )
        return parameters_to_update

    def parse_reactions(self, sbml_model: libsbml.Model) -> None:
        for reaction in sbml_model.getListOfReactions():
            sbml_id = _name_to_py(reaction.getId())
            kinetic_law = reaction.getKineticLaw()
            if kinetic_law is None:
                continue
            parameters_to_update = self._parse_local_parameters(
                reaction_id=sbml_id,
                kinetic_law=kinetic_law,
            )

            node = reaction.getKineticLaw().getMath()
            # FIXME: convert substance amount to concentration here
            body, args = parse_sbml_math(node=node)

            # Update parameter references
            for old, new in parameters_to_update.items():
                pat = re.compile(f"({old})" + r"\b")
                body = pat.sub(new, body)

            for i, arg in enumerate(args):
                args[i] = parameters_to_update.get(arg, arg)

            dynamic_stoichiometry: dict[str, str] = {}
            parsed_reactants: defaultdict[str, int] = defaultdict(int)
            for substrate in reaction.getListOfReactants():
                species = _name_to_py(substrate.getSpecies())
                if (ref := substrate.getId()) != "":
                    dynamic_stoichiometry[species] = ref
                    self.parameters[ref] = Parameter(0.0, is_constant=False)

                elif species not in self.boundary_species:
                    factor = substrate.getStoichiometry()
                    if str(factor) == "nan":
                        msg = f"Cannot parse stoichiometry: {factor}"
                        raise ValueError(msg)
                    parsed_reactants[species] -= factor

            parsed_products: defaultdict[str, int] = defaultdict(int)
            for product in reaction.getListOfProducts():
                species = _name_to_py(product.getSpecies())
                if (ref := product.getId()) != "":
                    dynamic_stoichiometry[species] = ref
                    self.parameters[ref] = Parameter(0.0, is_constant=False)
                elif species not in self.boundary_species:
                    factor = product.getStoichiometry()
                    if str(factor) == "nan":
                        msg = f"Cannot parse stoichiometry: {factor}"
                        raise ValueError(msg)
                    parsed_products[species] += factor

            # Combine stoichiometries
            # Hint: you can't just combine the dictionaries, as you have cases like
            # S1 + S2 -> 2S2, which have to be combined to S1 -> S2
            stoichiometries = dict(parsed_reactants)
            for species, value in parsed_products.items():
                if species in stoichiometries:
                    stoichiometries[species] = stoichiometries[species] + value
                else:
                    stoichiometries[species] = value

            self.reactions[sbml_id] = Reaction(
                body=body,
                stoichiometry=stoichiometries | dynamic_stoichiometry,
                args=args,
            )

    def _convert_substance_amount_to_concentration(
        self,
    ) -> None:
        """Convert substance amount to concentration if has_only_substance_units is false.

        The compounds in the test are supplied in mole if has_only_substance_units is false.
        In that case, the reaction equation has to be reformed like this:
        k1 * S1 * compartment -> k1 * S1
        or in other words the species have to be divided by the compartment to get
        concentration units.
        """
        for reaction in self.reactions.values():
            function_body = reaction.body
            removed_compartments = set()
            for arg in reaction.args:
                # the parsed species part is important to not
                # introduce conversion on things that aren't species
                if (species := self.variables.get(arg, None)) is None:
                    continue

                if not species.has_only_substance_units:
                    compartment = species.compartment
                    if compartment is not None:
                        if self.compartments[compartment].dimensions == 0:
                            continue
                        if species.is_concentration:
                            if compartment not in removed_compartments:
                                pattern = f"({compartment})" + r"\b"
                                repl = f"({compartment} / {compartment})"
                                function_body = re.sub(pattern, repl, function_body)
                                removed_compartments.add(compartment)
                        else:
                            # \b is word boundary
                            pattern = f"({arg})" + r"\b"
                            repl = f"({arg} / {compartment})"
                            function_body = re.sub(pattern, repl, function_body)
                        if compartment not in reaction.args:
                            reaction.args.append(compartment)

            # Simplify the function
            try:
                reaction.body = str(sympy.parse_expr(function_body))
            except AttributeError:
                # E.g. when math.factorial is called
                # FIXME: do the sympy conversion before?
                reaction.body = function_body


def _handle_fn(name: str, body: str, args: list[str]) -> Callable[..., float]:
    func_args = ", ".join(args)
    func_str = "\n".join(
        [
            f"def {name}({func_args}):",
            f"{INDENT}return {body}",
            "",
        ]
    )
    try:
        exec(func_str, globals(), None)  # noqa: S102
    except SyntaxError as e:
        msg = f"Invalid function definition: {func_str}"
        raise SyntaxError(msg) from e
    python_func = globals()[name]
    python_func.__source__ = func_str
    return python_func  # type: ignore


def _codegen_fn(name: str, body: str, args: list[str]) -> str:
    func_args = ", ".join(args)
    return "\n".join(
        [
            f"def {name}({func_args}):",
            f"{INDENT}return {body}",
            "",
        ]
    )


def _codegen_initial_assignment(
    k: str,
    sbml: Parser,
    parameters: dict[str, float],
) -> str:
    if k in parameters:
        return f"m.update_parameter('{k}', _init_{k}(*(args[i] for i in {sbml.initial_assignment[k].args})) )"

    ass = sbml.initial_assignment[k]
    fn = f"_init_{k}(*(args[i] for i in {ass.args}))"

    if (species := sbml.variables.get(k)) is not None:
        compartment = species.compartment
        if compartment is not None and (
            not species.is_concentration
            or (species.has_only_substance_units and species.is_concentration)
        ):
            size = 1 if (c := sbml.compartments.get(compartment)) is None else c.size
            if size != 0:
                fn = f"{fn} * {size}"
    elif k in sbml.compartments:
        pass
    else:
        msg = "Initial assignment targeting unknown type"
        raise NotImplementedError(msg)

    return f"m.update_variable('{k}', {fn})"


def _codgen(name: str, sbml: Parser) -> Path:
    import itertools as it

    functions = {
        k: _codegen_fn(k, body=v.body, args=v.args)
        for k, v in it.chain(
            sbml.functions.items(),
            sbml.derived.items(),
            sbml.reactions.items(),
            {f"_init_{k}": v for k, v in sbml.initial_assignment.items()}.items(),
        )
    }

    parameters = {
        k: v.value for k, v in sbml.parameters.items() if k not in sbml.derived
    }
    variables = {
        k: v.initial_amount for k, v in sbml.variables.items() if k not in sbml.derived
    }
    for k, v in sbml.compartments.items():
        if k in sbml.derived:
            continue
        if v.is_constant:
            parameters[k] = v.size
        else:
            variables[k] = v.size

    # Ensure non-zero value for initial assignments
    # EXPLAIN: we need to do this for the first round of get_args to work
    # otherwise we run into a ton of DivisionByZero errors.
    # Since the values are overwritte afterwards, it doesn't really matter anyways
    for k in sbml.initial_assignment:
        if k in parameters and parameters[k] == 0:
            parameters[k] = 1
        if k in variables and variables[k] == 0:
            variables[k] = 1

    derived_str = "\n    ".join(
        f"m.add_derived('{k}', fn={k}, args={v.args})" for k, v in sbml.derived.items()
    )
    rxn_str = "\n    ".join(
        f"m.add_reaction('{k}', fn={k}, args={rxn.args}, stoichiometry={rxn.stoichiometry})"
        for k, rxn in sbml.reactions.items()
    )

    functions_str = "\n\n".join(functions.values())

    parameters_str = f"m.add_parameters({parameters})" if len(parameters) > 0 else ""
    variables_str = f"m.add_variables({variables})" if len(variables) > 0 else ""

    # Initial assignments
    initial_assignment_order = _sort_dependencies(
        available=set(sbml.initial_assignment)
        ^ set(parameters)
        ^ set(variables)
        ^ set(sbml.derived)
        | {"time"},
        elements=[
            Dependency(name=k, required=set(v.args), provided={k})
            for k, v in sbml.initial_assignment.items()
        ],
    )

    if len(initial_assignment_order) > 0:
        initial_assignment_source = "\n    ".join(
            _codegen_initial_assignment(k, sbml, parameters=parameters)
            for k in initial_assignment_order
        )
    else:
        initial_assignment_source = ""

    file = f"""
import math

import numpy as np
import scipy

from mxlpy import Model

{functions_str}

def get_model() -> Model:
    m = Model()
    {parameters_str}
    {variables_str}
    {derived_str}
    {rxn_str}
    args = m.get_args()
    {initial_assignment_source}
    return m
"""
    path = default_tmp_dir(None, remove_old_cache=False) / f"{name}.py"
    with path.open("w+") as f:
        f.write(file)
    return path


def import_from_path(module_name: str, file_path: Path) -> Callable[[], Model]:
    import sys
    from importlib import util

    spec = unwrap(util.spec_from_file_location(module_name, file_path))
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    unwrap(spec.loader).exec_module(module)
    return module.get_model


def valid_filename(value: str) -> str:
    import re
    import unicodedata

    value = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    value = re.sub(r"[-\s]+", "_", value).strip("-_")
    return f"mb_{value}"


def read(file: Path) -> Model:
    """Import a metabolic model from an SBML file.

    Args:
        file: Path to the SBML file to import.

    Returns:
        Model: Imported model instance.

    """
    name = valid_filename(file.stem)
    sbml = Parser().parse(file=file)

    model_fn = import_from_path(name, _codgen(name, sbml))
    return model_fn()
