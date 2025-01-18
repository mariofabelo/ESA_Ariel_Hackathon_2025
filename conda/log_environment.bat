call mamba list --explicit > environment_explicit.txt
call mamba env export > environment.yml
call mamba env export --no-builds > environment_nobuilds.yml
call mamba env export --from-history > environment_fromhistory.yml
call mamba list > environment_list.txt
pip freeze > pip_requirements.txt