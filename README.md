# Pytorch connector for Google Cloud Bigtable

This is the repository holding the code for the plugin.

## User guide

For instructions on how to use this plugin please refer to
this [readme](plugin/README.md)

## Building the project

Because for some reason pytorch is compiled by default with a different version
of ABI, all the dependencies had to be compiled with the same ABI version. 
For that reason, building the project on your local machine might be a 
little tricky. To make the installation of all the dependencies a bit easier,
we provide a docker file with all the necessary steps.

Make sure you have docker installed and then just run the `run_devel.sh` 
script in the project's dir. It will build the container and then run it 
and mount the `plugin` folder. From there you can just run:
```python
python setup.py develop
```

## Releasing
The releases are maintained in the following way:
TODO