# Dialogue and Narrative Coursework 2021


## Environment Setup

Assuming you're using conda you can run the following commands:

1. To install a new environment named nlp-2021

    `conda env create -f environment.yml`

    `conda activate nlp-2021`

2. To update the environment, e.g. when a new dependency is needed

    `conda env update -n nlp-2021 --file environment.yml --prune`

3. To create a new environment.yml file

    `conda env export -n nlp-2021 -f environment.yml --from-history --no-builds`

*check flags as they are supposed to get rid of architecture dependent dependencies and unnecessary version forcing*

## Run a wand experiment

In root directory run ` python -m src.run_experiment.py`