#!/usr/bin/env python


from subprocess import run


# - Ensure Python 3.10 or later is installed first before running this script
#   - E.g. $ conda install "python=3.10"
run('pip install -r requirements.txt', shell=True, check=True)
run('pip install -e lib/timeVAE', shell=True, check=True)