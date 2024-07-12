# JAX Installation Troubleshooting
The instructions for installing the LINX dependencies, including JAX, diffrax, and equinox, should work well on Intel-based PCs, Linux PCs, and Linux clusters.  
Some Apple Silicon users report difficulty installing JAX.  
If the above installation instructions return errors related to JAX/jaxlib, we have compiled an unscientific set of troubleshooting suggestions.  
If these also fail, we suggest you explore the [JAX GitHub](https://github.com/google/jax) for more community tips on installing JAX.

Please be sure to upgrade pip before proceeding!

**M2 Macs**

Follow the instructions above, but install JAX with conda instead of pip.  
This will entail uninstalling the `jax` version that is installed with `diffrax`, and then install `jax` using conda, specifying version 0.4.26.  
`jaxlib` should automatically install.

In summary, you'll want to run
```
conda create --name linx
conda activate linx
conda install scipy matplotlib numpy jupyter
pip install diffrax==0.4.1
pip uninstall jax jaxlib
conda install jax==0.4.26
```

**M1 Macs**

Follow the M2 Mac instructions above, but first create a conda environment with python 3.11.  

Run
```
conda create --name linx python=3.11
conda activate linx
conda install scipy matplotlib numpy jupyter
pip install diffrax==0.4.1
pip uninstall jax jaxlib
conda install jax==0.4.26
```

**M3 Macs**

Anecdotally, we have found M3 Mac users report a variety of different issues installing JAX.  
We suspect individual python ecosystems and compilers are more responsible for these issues than hardware---
without access to a steady supply of M3 Macs it is difficult for our team to uncover the root of the problems that have been reported to us.

First, try the M1 Mac instructions above.  
If this fails, try to `pip install` JAX instead of installing with conda, specifying version 0.4.26.  
This will require either uninstalling the JAX version installed by `diffrax`, or installing `diffrax` with the `--no-deps` flag and manually installing the dependencies:

```
conda create --name linx python=3.11
conda activate linx
conda install scipy matplotlib numpy jupyter
pip install diffrax==0.4.1 --no-deps
pip install equinox --no-deps
pip install jaxtyping --no-deps
pip install typeguard --no-deps
pip install lineax --no-deps
pip install optimistix --no-deps
pip install jax==0.4.26
pip install jaxlib=0.4.26
```
These instructions did not work for everyone who tested our code on an M3 Mac---
if you continue to struggle with the JAX installation, please check out the [JAX GitHub](https://github.com/google/jax).
