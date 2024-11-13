::
:: Start compiled api code with environment from Virtual Environment
::

@echo off
setlocal enabledelayedexpansion

set PREFECT_ENV_NAME=automl_py310

:: Get args =========
set argC=0
for %%x in (%*) do Set /A argC+=1
if %argC%==0 (
    echo [USAGE] start_svc.bat ^<cmd^>
    goto :eof
)
set cmd=%*

:: Execute the code inside the virtual env =====
call %USERPROFILE%\Miniforge3\Scripts\activate.bat %PREFECT_ENV_NAME%
call %cmd%
conda deactivate