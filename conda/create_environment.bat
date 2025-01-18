mamba update -n base mamba
mamba create -n mla_202411 python=3.11.10
mamba activate mla_202411
@REM mamba env update -n mla_202411 --file environment_fromhistory.yml
mamba env update -n mla_202411 --file steps/environment_fromhistory_v2_postinstall.yml
