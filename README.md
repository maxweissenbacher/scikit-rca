# Reliability Component Analysis

[![tests](https://github.com/maxweissenbacher/scikit-rca/actions/workflows/python-app.yml/badge.svg)](https://github.com/maxweissenbacher/scikit-rca/actions/workflows/python-app.yml)

Implementation of Reliability Component Analysis (RCA) from the paper

(citation pending).

`scikit_rca` is a [scikit-learn](https://scikit-learn.org) compatible extension. The implementation is based on the [scikit-learn-contrib template](https://github.com/scikit-learn-contrib/project-template).

## Installation

The package is available on PyPI and can be installed with:

```bash
pip install scikit-rca
```

## Usage

Example usage is demonstrated in `examples/run_rca.py`. After installing, the script can be run from any environment that contains the installed package. A good starting point is calling

```bash
python examples/run_rca.py --help
```

The following shows some example usage of the script:

```bash
python examples/run_rca.py --data-dir /path/to/my/data --lr 0.005 --epochs 200 --dim 5 --out-dir /path/to/store/model --penalty-scale 0.1 --batch-size 4000 --weight-decay 0.001
```

## Authors

Anastasia Borovykh, Max Weissenbacher, Stephanie Noble, Maxwell Shinn.
