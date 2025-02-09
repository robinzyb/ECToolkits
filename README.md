# Introduction


ElectroChemical Toolkits(ECToolkits), the package to calculate electrochemical results from atomistic simulations.

![Python package](https://github.com/robinzyb/ectoolkits/actions/workflows/ci.yml/badge.svg)
![codecov](https://codecov.io/github/robinzyb/ECToolkits/graph/badge.svg?token=8M5ULYLP2U)
![pythonv](https://img.shields.io/pypi/pyversions/ectoolkits)
![pypiv](https://img.shields.io/pypi/v/ectoolkits)

# Installation
## From pip
```bash
pip install ectoolkits
```

## From source
One can download the source code of cp2kdata by
```bash
git clone https://github.com/robinzyb/ECToolkits.git ectoolkits
```
then use `pip` to install the module from source

```bash
cd ectoolkits
pip install .
```


# Analysis
- [Atomic/water density](./docs/analysis/atom_density.md)
- [Band alignment](./docs/analysis/band_align.md)
- [Acidity calculation](./docs/analysis/acidity.md)
- [Redox potential calculation](./docs/analysis/redox.md)
- [Finite size correction](./docs/analysis/finite_size_correction.md)

# Development

- [Development guide](./DEVEL.md)
- [To-do list](./TODO.md)
