# NearPy

## Installation

Managing dependencies for Python across operating systems is annoying and can be a big waste of time. Below are steps which should work in a platform agnostic way for any operating system. Ensure that you have ```pipx``` installed. Additionally, we recommend using ```virtualenv``` to manage your environment.

```zsh
pipx install virtualenv poetry
git clone https://github.com/meowkash/nearpy.git
cd nearpy 
poetry install 
```

That's it. You should be able to use NearPy on all operating systems from now on

## Directory Structure

```zsh
ai>datamodules: Each project gets its own datamodule to manage its unique dataset 
ai>models: Each unique model is implemented here
```

### References

* [DataModule - PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)
