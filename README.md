## LowRankMDP

This directory contains the supplement code for the paper titled "Value Function Approximation via Low-Rank Models" by Hao Yi Ong.

Here you will find implementations of the following:
* Classic mountain car and inverted pendulum MDPs and solutions obtained by value iteration
* Low-rank + sparsification of state-action value function using external MATLAB library for Robust PCA (Lin et al., 2009)

### Dependencies

The software is implemented in Julia, with calls from Julia to an external MATLAB library. (So the user must have a local version of MATLAB.) For the best results, the user should use a notebook. Example notebooks are shown in both the `mdps` and `lrm` subdirectories. The following Julia packages are required for running all code. 
* PGFPlots
* GridInterpolations

### Layout
```
data/

lrm/
    PROPACK/
    LowRankModel.jl
    LowRankModeling.ipynb
    choosvd.m
    exact\_alm\_rpca.m

mdps/
    InvertedPendulum.ipynb
    InvertedPendulum.jl
    MDPs.jl
    MountainCar.ipynb
    MountainCar.jl
    SPDot.jl

README.md
```