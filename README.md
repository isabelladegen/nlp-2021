# Dialogue and Narrative Coursework 2021

This project was build for the Dialogue and Narrative coursework at Bristol University. The aim was to play with NLP
techniques learned during the semester. This coursework is a simple solution for subtask 1 from the
[DialDoc21 competition](https://github.com/doc2dial/sharedtask-dialdoc2021). My personal aim was to learn how a simple
embeddings based solutions would work and to learn more python and useful tooling around ML. There's a report in this
repository with more details.

## Project structure

```
├── LICENSE
├── README.md
├── conda.yml
├── notebooks
│   ├── DataInvestigationDoc2Dial.ipynb
│   ├── Doc2DialSubtask1.ipynb
│   ├── Doc2VecSimple.ipynb
│   ├── Doc2VecSolutionForSectionsForSubtask1.ipynb
│   ├── Doc2VecSolutionForSpansForSubtask1.ipynb
│   ├── RCDataInvestigation.ipynb
├── report
│   ├── Makefile
│   ├── README.md
│   ├── bibliography.bib
│   ├── coling.bst
│   ├── coling2020.sty
│   ├── images
│   │   ├── number_of_dialogues_and_turns.png
│   │   ├── parallel-coordinates-F1.png
│   │   ├── parallel-coordinates-exact-match.png
│   │   ├── span_counts.png
│   │   ├── sparsity_of_spans.png
│   │   └── sub_section_usage_of_spans.png
│   ├── nlp-report-isabella.tex
├── src
│   ├── __init__.py
│   ├── configurations.py
│   ├── evaluate.py
│   ├── preprocessing_documents.py
│   ├── preprocessing_rc.py
│   ├── run_experiment.py
│   ├── sweep.py
│   ├── train_and_predict.py
│   ├── tune_model_parameters_experiment.py
├── tests
│   ├── configurations_test.py
│   ├── doc2vec_test.py
│   ├── evaluate_test.py
│   ├── preprocessing_documents_test.py
│   ├── preprocessing_rc_test.py
│   ├── train_and_predict_test.py
│   └── utils
│       └── test_utils.py
```

## Environment Setup

Assuming you're using conda you can run the following commands:

1. Create Conda env
   ```
   conda env create -f conda.yml
   conda activate nlp-2021
   ```

2. Update Conda env

   ```
   conda env update -n nlp-2021 --file conda.yml --prune
   ```

## Tracking Experiments

Started using Weights and Biases to track experiments once I switched from notebooks to Python scripts.

The runs and all the parameters can be found here [NLP Experiments](https://wandb.ai/idegen/test?workspace=user-idegen)

The Configuration.py dataclass is used to store all the default configurations including the model's hyperparameters. It
is also used as the `wand.config`.

### Running the experiments

**Sweep**

Run the `sweep.py`. There's a configuration at the top of the files that will be used for different hyper parameters to
be tried. Any parameter not set there will be taken from the default Configuration. Again this will not run on the
comandline as I run it from my IDE [DataSpell](https://www.jetbrains.com/dataspell/).

**Experiment**

Run the `run_experiment.py`. Configurations used are in Configuration. However do change the notes and tags for `wand`

## Testing

If started testing after the notebooks code became unmanageable. The tests can be found in the tests folder. They might
or might not run out of the box on the commandline, I run them using the
IDE [DataSpell](https://www.jetbrains.com/dataspell/).

For a new project I definitely would do more TDD. The existing tests still give quite a good safety net and found many
bugs.