# PyALCS Examples

## Development mode
In order to work both on developing PyALCS code and optionally OpenAI gym environments you can apply the following instructions.
 
Create base Conda environment from environment file and activate it

    conda env create --file environment-base.yml
    conda activate pyalcs-experiments
    conda env update --file environment-base.yml --prune

Install development versions of PyALCS and OpenAI Gym

    cd PATH_TO_PYALCS/
    python setup.py develop
    
    cd ../pyalcs && python setup.py develop && cd ../pyalcs-experiments
    
    cd PATH_TO_OPENAIGYM_ENVS/
    python setup.py develop
    cd ../openai-envs && python setup.py develop && cd ../pyalcs-experiments

By doing so the PYTHONPATH from the `pyalcs-experiments` environment will point to local directories.

### Investigating Exploration Techniques for Anticipatory Classifier System in Real-Valued Environments

Online reproduction notebooks

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ParrotPrediction/pyalcs-experiments/a2c8d40e7426a5dce899e3ebb323431f0a57c6f3)
