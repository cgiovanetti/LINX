# LINX
 - Installation
     - Troubleshooting
     - CLASS and plik
 - Repository Structure
 - Citation
 - References

Light Isotope Nucleosynthesis with JAX (LINX) is a fast, differentiable, extensible code that can be used to predict primordial light element abundances during Big Bang Nucleosynthesis (BBN).  LINX is written in python using JAX, and is readable, accessible, and well-documented.

[![Run Tests](https://github.com/cgiovanetti/LINX/actions/workflows/test.yml/badge.svg)](https://github.com/cgiovanetti/LINX/actions/workflows/test.yml)


## Installation
It is strongly recommended that you create a clean conda environment with the latest python. All of the requirements are included in `requirements.txt`.  In summary, run
```
conda create --name linx
conda activate linx
pip install requirements.txt
```
(Include `path/to/requirements.txt` if you're in a different directory.)  Verify your JAX installation by opening up python in terminal and attempting to `import jax`. 

If you'd like to install a minimal set of packages just to run LINX without the SVI examples, you can run
```
conda create --name linx
conda activate linx
conda install --yes scipy matplotlib numpy jupyter
pip install diffrax==0.4.1 jax==0.4.28 jaxlib==0.4.28
```
(pip or conda can be used for `scipy`, `matplotlib`, `numpy`, and `jupyter`.)

### Troubleshooting
The above instructions should work well on Intel-based PCs, Linux PCs, and Linux clusters.  Some Apple Silicon users report difficulty installing JAX, frequently receiving an error `This version of jaxlib was built using AVX instructions, which your CPU and/or operating system do not support. You may be able work around this issue by building jaxlib from source.`.  If the above installation instructions return errors related to JAX/jaxlib, we have compiled an unscientific set of troubleshooting suggestions.  If these also fail, we suggest you explore the [JAX GitHub](https://github.com/google/jax) for more community tips on installing JAX.

Please be sure to upgrade pip before proceeding!

**M2 Macs**

Follow the instructions above, but install JAX with conda instead of pip.  This will entail uninstalling the `jax` version that is installed with `diffrax`, and then install `jax` using conda, specifying version 0.4.26.  `jaxlib` should automatically install.

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

Anecdotally, we have found M3 Mac users report a variety of different issues installing JAX.  We suspect individual python ecosystems and compilers are more responsible for these issues than hardware---without access to a steady supply of M3 Macs it is difficult for our team to uncover the root of the problems that have been reported to us.

First, try the M1 Mac instructions above.  If this fails, try to `pip install` JAX instead of installing with conda, specifying version 0.4.26.  This will require either uninstalling the JAX version installed by `diffrax`, or installing `diffrax` with the `--no-deps` flag and manually installing the dependencies:

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
These instructions did not work for everyone who tested our code on an M3 Mac--if you continue to struggle with the JAX installation, please check out the [JAX GitHub](https://github.com/google/jax).

### CLASS and plik
Some analysis files in this repository depend on the user having installed CLASS and clik.  Installation instructions are available for each of these codes: [CLASS](https://lesgourg.github.io/class_public/class.html), [clik](https://github.com/brinckmann/montepython_public#the-planck-likelihood-part).  Note that some users have reported issues using clik with python 3.12--we will not maintain this note to check if this compatibility issue is resolved, but if you have difficulty installing clik you might try to install with python 3.11.

## Repository structure
Modules of the code are contained in the `linx` directory, along with the various sets of BBN reaction rates that ship with LINX.  

`test_scripts` contains a suite of scripts intended for testing and validating LINX.  The `test_SBBN` scripts test "standard BBN", or BBN in LCDM cosmology, using a truncated reaction network.  This reaction network is the minimal network required for accurate deuterium and helium-4 predictions, and these two test scripts compare the LINX results for this network against results from PRIMAT and PRyMordial.  The `test_full_network` scripts are similar, but test the full network instead of the truncated network.

Inside `test_scripts`, `pytest` contains brief pytest scripts for continuous checking of this repository.  You can run them yourself by navigating into the directory and running `pytest .` (after pip installing pytest).

`example_notebooks` contains a set of Jupyter noteboooks intended to be a pedagogical reference for users familiarizing themselves with LINX.  
* background_evolution explores the thermodynamic calculation performed in LINX to determine quantities like energy densities and Hubble.  It includes a comparison of the LINX Neff results with other results in the literature.
* NuisanceParametersImpact explores the impact of the uncertainties of BBN rates on final predicted abundances.  The ability to sample these nuisance parameters is a key feature of LINX.
* weak_rates illustrates the impact of different settings for computing the rate of proton-neutron interconversion on the prediction for the primordial helium-4 abundance.
* Schramm includes an example of a Schramm plot, or the primordial abundances as a function of the baryon density.  It includes examples with the truncated and full networks, as well as with nuclear rate uncertainties incorporated into the results.
* nuclear_evolution examines the evolution of primordial abundances as a function of time during BBN.  It also includes a cool movie!
* inference_omega_b provides a toy example of parameter inference with LINX, with a profile likelihood test.
* scan_over_Omega_b_tau_n includes a scan over two parameters, both the baryon density and the neutron lifetime.

`analysis_scripts` contains a set of python scripts and batch scripts used for the analyses in our analysis paper (coming soon!).  If you plan to run these scripts on a cluster, the batch scripts may need to be tweaked for your cluster's configuration (e.g. the NYU Greene cluster, on which these analyses were run, requires the use of singularity containers, and so the syntax for calling the script may look strange).  The CMB_BBN script also requires the user to have installed CLASS and clik.

This directory also contains a subdirectory called `samples` and a jupyter notebook called `example_analysis`.  These include our data in our analysis paper (coming soon!) and brief examples of how to unpack and view the data (requires the user to have `corner` installed).

`SVI_scripts` contains the scripts we used to perform a Stochastic Variational Inference (SVI) analysis with LINX and [CosmoPower](https://arxiv.org/abs/2106.03846) in our formalism paper (coming soon!).  Running these scripts requires the user to have installed CosmoPower.


## Citation
If you use LINX, please cite both LINX and the other public codes upon which LINX depends.  This includes nudec_BSM (https://arxiv.org/abs/1812.05605, https://arxiv.org/pdf/2001.04466), PRIMAT (http://www2.iap.fr/users/pitrou/primat.htm), and PRyMordial (https://arxiv.org/abs/2307.07061).  

The suggested language for citation is "LINX [linx], which depends on methods, tables, and frameworks from Refs. [nudecBSM1,nudecBSM2,PRIMAT,PRyM]".  All of these citations are included below.

```
@article{linx,
    title = {{LINX}: A Fast, Differentiable, and Extensible Big Bang Nucleosynthesis Package},
    author = {Giovanetti, Cara and Lisanti, Mariangela and Liu, Hongwan and Mishra-Sharma, Siddharth and Ruderman, Joshua T.},
    year={2024},    
    note={To appear},
}

@article{nudecBSM1,
   title={Neutrino decoupling beyond the {S}tandard {M}odel: {CMB} constraints on the Dark Matter mass with a fast and precise {N}eff evaluation},
   volume={2019},
   ISSN={1475-7516},
   url={http://dx.doi.org/10.1088/1475-7516/2019/02/007},
   DOI={10.1088/1475-7516/2019/02/007},
   number={02},
   journal={Journal of Cosmology and Astroparticle Physics},
   publisher={IOP Publishing},
   author={Escudero, Miguel},
   year={2019},
   month=feb, pages={007–007} }

@article{nudecBSM2,
   title={Precision early universe thermodynamics made simple:{N}eff and neutrino decoupling in the {S}tandard {M}odel and beyond},
   volume={2020},
   ISSN={1475-7516},
   url={http://dx.doi.org/10.1088/1475-7516/2020/05/048},
   DOI={10.1088/1475-7516/2020/05/048},
   number={05},
   journal={Journal of Cosmology and Astroparticle Physics},
   publisher={IOP Publishing},
   author={Abenza, Miguel Escudero},
   year={2020},
   month=may, pages={048–048} }

@article{PRIMAT,
   title={Precision big bang nucleosynthesis with improved {H}elium-4 predictions},
   volume={754},
   ISSN={0370-1573},
   url={http://dx.doi.org/10.1016/j.physrep.2018.04.005},
   DOI={10.1016/j.physrep.2018.04.005},
   journal={Physics Reports},
   publisher={Elsevier BV},
   author={Pitrou, Cyril and Coc, Alain and Uzan, Jean-Philippe and Vangioni, Elisabeth},
   year={2018},
   month=sep, pages={1–66} }

@misc{PRyM,
      title={{PRyMordial}: The First Three Minutes, Within and Beyond the {S}tandard {M}odel}, 
      author={Anne-Katherine Burns and Tim M. P. Tait and Mauro Valli},
      year={2023},
      eprint={2307.07061},
      archivePrefix={arXiv},
      primaryClass={hep-ph}
      url={https://arxiv.org/abs/2307.07061}, 
}
```

## References
Two reference papers detailing LINX and the analyses we have performed with it will appear on the arXiv soon!
