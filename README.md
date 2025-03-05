# WUMS: Wremnants Utilities, Modules, and other Stuff

As the name suggests, this is a collection of different thins, all python based:
- Fitting with tensorflow or jax
- Custom pickling h5py objects 
- Plotting functionality

## Install

The `wums` package can be pip installed with minimal dependencies:
```bash
pip install wums
```
Different dependencies can be added with `plotting`, `fitting`, `pickling` to use the corresponding scripts.
For example, one can install with
```bash
pip install wums[plotting,fitting]
```
Or all dependencies with
```bash
pip install wums[all]
```