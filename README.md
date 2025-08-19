<p align="center">
    <img src="https://raw.githubusercontent.com/Computational-Biology-Aachen/MxlPy/refs/heads/main/docs/assets/logo-diagram.png" width="600px" alt='mxlpy-logo'>
</p>



# MxlPy

[![pypi](https://img.shields.io/pypi/v/mxlpy.svg)](https://pypi.python.org/pypi/mxlpy)
[![docs][docs-badge]][docs]
![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)
![Coverage](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fgist.github.com%2Fmarvinvanaalst%2F98ab3ce1db511de42f9871e91d85e4cd%2Fraw%2Fcoverage.json&query=%24.message&label=Coverage&color=%24.color&suffix=%20%25)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![PyPI Downloads](https://static.pepy.tech/badge/mxlpy)](https://pepy.tech/projects/mxlpy)

[docs-badge]: https://img.shields.io/badge/docs-main-green.svg?style=flat-square
[docs]: https://computational-biology-aachen.github.io/mxlpy/

MxlPy (pronounced "em axe el pie") is a Python package for mechanistic learning (Mxl) - the combination of mechanistic modeling and machine learning to deliver explainable, data-informed solutions.

## Installation

You can install mxlpy using pip: `pip install mxlpy`.

Due to their sizes, the machine learning packages are optional dependencies. You can install them using

```shell
# One of them respectively
pip install mxlpy[torch]
pip install mxlpy[tensorflow]
pip install mxlpy[keras]

# together
pip install mxlpy[torch, tensorflow, keras]
```

If you want access to the sundials solver suite via the [assimulo](https://jmodelica.org/assimulo/) package, we recommend setting up a virtual environment via [pixi](https://pixi.sh/) or [mamba / conda](https://mamba.readthedocs.io/en/latest/) using the [conda-forge](https://conda-forge.org/) channel.

```bash
pixi init
pixi add python assimulo
pixi add --pypi mxlpy
```

## How to cite

If you use this software in your scientific work, please cite [this article](https://doi.org/10.1101/2025.05.06.652335):

- [doi](https://doi.org/10.1101/2025.05.06.652335)
- [bibtex file](https://github.com/Computational-Biology-Aachen/MxlPy/citation.bibtex)


## Development setup

You have two choices here, using `uv` (pypi-only) or using `pixi` (conda-forge, including assimulo)

### uv

- Install `uv` as described in [the docs](https://docs.astral.sh/uv/getting-started/installation/).
- Run `uv sync --all-extras --all-groups` to install dependencies locally

### pixi

- Install `pixi` as described in [the docs](https://pixi.sh/latest/#installation)
- Run `pixi install --frozen`


## LLMs

We support the [llms.txt](https://llmstxt.org/) convention for making documentation available to large language models and the applications that make use of them. It is located at [docs/llms.txt](https://github.com/Computational-Biology-Aachen/MxlPy/tree/main/docs/llms.txt)

## Tool family 🏠

`MxlPy` is part of a larger family of tools that are designed with a similar set of abstractions. Check them out!

- [MxlBricks](https://github.com/Computational-Biology-Aachen/mxl-bricks) is built on top of `MxlPy` to build mechanistic models composed of pre-defined reactions (bricks)
- [MxlWeb](https://github.com/Computational-Biology-Aachen/mxl-web) brings simulation of mechanistic models to the browser!
