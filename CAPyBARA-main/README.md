# CAPyBARA

The **CAPyBARA** repository contains simulation and post-processing tools related to the Hybrid Lyot Coronagraph (HLC) modes of the Coronagraphic Instrument (CGI) for the Roman Space Telescope.

This package aims to simulate observing sequences with CGI, from Dark Zone digging using high-order wavefront sensing and control (HOWFSC) algorithms on a reference star, to science target acquisitions. It is designed to allow the injection of custom wavefront maps onto the deformable mirrors in addition to the HOWFSC solutions during reference star acquisitions, in order to simulate alternative observing strategies. Post-processing tools are also included to evaluate the performance of these strategies based on the final detection limits. It is developed as part of the European Research Council project **ESCAPE** (ERC Grant No. 101044152).

Please refer to "ESCAPE project CAPyBARA: a Roman Coronagraph simulator for post-processing methods development," Proc. SPIE 13092, Space Telescopes and Instrumentation 2024: Optical, Infrared, and Millimeter Wave, 1309258 (23 August 2024); https://doi.org/10.1117/12.3019211 for more details on the code. This research made use of HCIPy, an open-source object-oriented framework written in Python for performing end-to-end simulations of high-contrast imaging instruments (Por et al. 2018).

## Environment setup

It is recommended to create a dedicated virtual environment using the provided `environment.yml` file. This ensures consistency across dependencies.

```bash
conda env create -f environment.yml
conda activate capybara
```

## Installation

To install the package in **editable/development** mode:

```bash
pip install -e .
```

This allows you to modify the source code without needing to reinstall the package.


## Getting started

Once installed, edit the configuration `.ini` files (e.g. `capy-pup-900.ini`) to match your setup and data paths.

Then, you may run one of the main scripts, such as:

```bash
python run_CAPyBARA.py --config capy-pup-900.ini
```

Results will be saved in the directory specified under `data_path`.

## ⚠️ Important Note

The Jupyter notebook `capy-ground-broadband.ipynb` is currently the most up-to-date and validated version of the CAPyBARA workflow.

The `run_CAPyBARA.py` script may not reflect the latest tested structure or logic. It might fail depending on changes in dependencies or internal interfaces. Use the notebook as the primary reference for running simulations. `run_CAPyBARA.py` will be updated in the near future. 

## Repository structure

* `efc.py`: Implements electric field conjugation (EFC) loop logic.
* `aberration.py`: Handles static and chromatic aberration models.
* `post_processing.py`: Post-EFC diagnostics and analysis tools.
* `utils.py`: Shared utility functions.
* `capy-pup-900.ini`: Configuration file with optical and loop parameters.


## Requirements

The environment includes the following core dependencies:

* `numpy`
* `scipy`
* `matplotlib`
* `astropy`
* `configparser`


## Support and collaboration

For issues or feature requests, please open an issue on the GitLab project page.

Collaborators interested in contributing are welcome. Please open a merge request and follow the code structure and conventions.



## Authors and acknowledgment

Developed by:
**Alexis Lau**, **Lisa Altinier**, **Damien Camugli**, **Elodie Choquet**

This project is funded by the **European Union** (ERC, ESCAPE, project No 101044152).
Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency.
Neither the European Union nor the granting authority can be held responsible for them.


## Licence

This code and all associated files are released under a custom **Academic Use Only** licence.
You are free to use, modify, and distribute the materials **for non-commercial academic and research purposes**, as long as **proper credit is given**.

Commercial use is **strictly prohibited** without prior written permission.

This work also conforms to the terms of the [Creative Commons Attribution-NonCommercial 4.0 International Licence](https://creativecommons.org/licenses/by-nc/4.0/).
