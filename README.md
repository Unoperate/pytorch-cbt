# Pytorch connector for Google Cloud Bigtable

This is the repository holding the code for the plugin.

## User guide

For instructions on how to use this plugin please refer to
this [readme](plugin/README.md).

## Building the project

Building the project on your local machine might be a 
little tricky. To make the installation of all the dependencies a bit easier,
we provide a docker file with all the necessary steps.

Make sure you have docker installed and then just run the `run_devel.sh` 
script in the project's dir. It will build the container and then run it 
and mount the `plugin` folder. From there you can just run:
```bash
python setup.py develop
```

To build the wheel we use the manylinux2014 dockerfile supplied by
[manywheels](https://github.com/pypa/manylinux). 

Our extension has dynamic dependencies to some libraries as well as to libraries 
from pytorch. Normally this would be solved when running auditwheel, which would 
take all dependencies and include them in our wheel. 

This creates a problem because it includes pytorch libs which are loaded first
when you import pytorch itself. Afterwards, when our extension is 
loaded, all the dependencies would be loaded as well, including 
pytorch libraries for the second time, which causes problems.

What we do instead, is look for the 
dynamic dependencies not from pytorch manually and then include them as part
of our wheel. The code for that is already included in `setup.py`.

To build the wheels, run docker image `centos7.Dockerfile` and execute
the `plugin/build_wheels.sh` script which will execute
`python setup.py bdist_wheel` for each python version.

### Note on glibc++ ABI 

PyTorch is compiled using the new libstdc++ ABI. When C++11 was introduced,
libstdc++ maintainers had to rewrite parts of the library. In order not to
break binary compatibility with previously compiled programs, they decided
that new, C++11-conformant library implementation will be in an inline
namespace (__cxx11), which means that symbols for that implementation are
different (contain the __cxx11 namespace name in them). They also allowed the
users to change this behavior (i.e. break binary compatibility but link
against symbols without __cxx11 substring in them) if they set the flag
`_GLIBCXX_USE_CXX11_ABI` to `0`. Ubuntu (and Fedora) compile libraries with
this definition set to `1`. PyTorch, however, is compiled with it set to `0`,
which means that our C++ Bigtable plugin and all its dependencies have to be
compiled that way too.

In order to make it easy for the users, we compile the plugin and its
dependecies statically. This is especially helpful given the
`_GLIBCXX_USE_CXX11_ABI` incompatibility. However, the end result is a library
loaded by python, so we need to compile the code as position independent
(hence CMAKE_POSITION_INDEPENDENT_CODE is set to ON).

## Releases
The releases are maintained in the following way:
Every major release gets its own branch and then all smaller releases get 
their own tags. To publish the release, first the `Publish` workflow is run, 
compiling the wheels, uploading them to pypi and saving them as workflow 
artifacts. Then, these wheels are also described and attached to the release 
note on github.