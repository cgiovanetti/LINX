<h1 align="center">
LINX<!-- omit from toc -->
</h1>

<h4 align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-240x.xxxxx%20-green.svg)](https://arxiv.org/abs/240x.xxxxx)
[![Run Tests](https://github.com/cgiovanetti/LINX/actions/workflows/test.yml/badge.svg)](https://github.com/cgiovanetti/LINX/actions/workflows/test.yml)
</h4>

Light Isotope Nucleosynthesis with JAX (LINX) is a fast, differentiable, extensible code that can be used to predict primordial light element abundances during Big Bang Nucleosynthesis (BBN).  LINX is written in python using JAX, and is readable, accessible, and well-documented.

## Contents<!-- omit from toc -->

- [Installation](#installation)
  - [CLASS and plik](#class-and-plik)
- [Repository structure](#repository-structure)
- [Citation](#citation)
- [References](#references)
</center>

## Installation
It is strongly recommended that you create a clean conda environment with the latest python. All of the requirements are included in `requirements.txt`.  In summary, run
```
conda create --name linx
conda activate linx
pip install -r requirements.txt
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

If you have trouble installing JAX, we have a [troubleshooting document](https://github.com/cgiovanetti/LINX/blob/main/TROUBLESHOOTING.md) with some tips.

### CLASS and plik
Some analysis files in this repository depend on the user having installed CLASS and clik.  Installation instructions are available for each of these codes: [CLASS](https://lesgourg.github.io/class_public/class.html), [clik](https://github.com/brinckmann/montepython_public#the-planck-likelihood-part).  Note that some users have reported issues using clik with python 3.12--we will not maintain this note to check if this compatibility issue is resolved, but if you have difficulty installing clik you might try to install with python 3.11.

## Repository structure

Modules of the code are contained in the `linx` directory, along with the various sets of BBN reaction rates that ship with LINX.  

`example_notebooks` contains a set of pedagogical Jupyter noteboooks for users familiarizing themselves with LINX.  
* [background_evolution](https://github.com/cgiovanetti/LINX/blob/main/example_notebooks/background_evolution.ipynb) explores the thermodynamic calculation performed in LINX to determine quantities like energy densities and Hubble.  It includes a comparison of the LINX Neff results with other results in the literature.
* [NuisanceParametersImpact](https://github.com/cgiovanetti/LINX/blob/main/example_notebooks/NuisanceParametersImpact.ipynb) explores the impact of the uncertainties of BBN rates on final predicted abundances.  The ability to sample these nuisance parameters is a key feature of LINX.
* [weak_rates](https://github.com/cgiovanetti/LINX/blob/main/example_notebooks/weak_rates.ipynb) illustrates the impact of different settings for computing the rate of proton-neutron interconversion on the prediction for the primordial helium-4 abundance.
* [Schramm](https://github.com/cgiovanetti/LINX/blob/main/example_notebooks/Schramm.ipynb) includes an example of a Schramm plot, or the primordial abundances as a function of the baryon density.  It includes examples with the truncated and full networks, as well as with nuclear rate uncertainties incorporated into the results.
* [nuclear_evolution](https://github.com/cgiovanetti/LINX/blob/main/example_notebooks/nuclear_evolution.ipynb) examines the evolution of primordial abundances as a function of time during BBN.  It also includes a cool movie!
* [inference_omega_b](https://github.com/cgiovanetti/LINX/blob/main/example_notebooks/inference_Omega_b.ipynb) provides a toy example of parameter inference with LINX, with a profile likelihood test.
* [scan_over_Omega_b_tau_n](https://github.com/cgiovanetti/LINX/blob/main/example_notebooks/scan_over_Omega_b_tau_n.ipynb) includes a scan over two parameters, both the baryon density and the neutron lifetime.

`scripts` contains a suite of scripts for testing and using LINX.

* test scripts: The `test_SBBN` scripts test "standard BBN", or BBN in LCDM cosmology, using a truncated reaction network.  This reaction network is the minimal network required for accurate deuterium and helium-4 predictions, and these two test scripts compare the LINX results for this network against results from PRIMAT and PRyMordial.  The `test_full_network` scripts are similar, but test the full network instead of the truncated network.
* Analysis scripts: A set of python scripts and batch scripts used for the analyses in our analysis paper (coming soon!).  These are the scripts that begin with `BBN_only...` or `CMB_BBN...`.  The CMB_BBN script requires the user to have installed CLASS and clik.
* `run_numpyro.py`: a script used to perform a Stochastic Variational Inference (SVI) analysis with LINX and [CosmoPower](https://arxiv.org/abs/2106.03846) in our formalism paper (coming soon!).  Running this scripts requires the user to have installed CosmoPower.

This directory also contains two subdirectories: `samples`, which includes the data in our analysis paper (coming soon!), and `related_notebooks`, which contain jupyter notebooks that outline how to analyze the data generated by these scripts and stored in `samples` (requires the user to have `corner` installed).  **Important:** The samples are somewhat large, taking up ~72MB of space, so they are not included in our `main` branch.  If you'd like to see the samples, pull the `samples` branch instead.

Finally, `pytest` contains brief pytest scripts for continuous checking of this repository.  You can run them yourself by navigating into the directory and running `pytest .` (after pip installing pytest).


## Citation
If you use LINX, please cite both LINX and the other public codes with components utilized by LINX.  This includes nudec_BSM (https://arxiv.org/abs/1812.05605, https://arxiv.org/pdf/2001.04466), PRIMAT (http://www2.iap.fr/users/pitrou/primat.htm), and PRyMordial (https://arxiv.org/abs/2307.07061).  

The suggested language for citation is "LINX [linx], which uses methods, tables, and frameworks from Refs. [nudecBSM1,nudecBSM2,PRIMAT,PRyM]".  All of these citations are included below.

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
