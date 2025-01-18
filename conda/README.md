
# Description of the conda environment being used.

*(now using instead [Mamba](https://mamba.readthedocs.io/en/latest/), the reimplementation of the conda package manager in C++.)*

Files generated with:

	mamba list --explicit > environment_explicit.txt

	mamba env export > environment.yml

	mamba env export --no-builds > environment_nobuilds.yml

	mamba env export --from-history > environment_fromhistory.yml

	mamba list > environment_list.txt

	pip freeze > pip_requirements.txt


Usage instructions (anaconda):

* https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#sharing-an-environment
* https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
