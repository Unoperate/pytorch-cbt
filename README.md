<!-- 
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 
-->

# PyTorch connector for Google Cloud Bigtable

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

To build the wheels, run docker image `centos7.Dockerfile` and execute
the `plugin/build_wheels.sh` script which will execute
`python setup.py bdist_wheel` for each python version.


### Note on linking dependencies

This library has a link dependency on the PyTorch library. An easy way of 
distributing it would be to link PyTorch statically. This, however, creates 
a problem because PyTorch python module would load its own copy of the same 
library, yielding two copies loaded simultaneously, which leads to problems.

Therefore, we cannot link the PyTorch library statically. A natural approach 
would be to use auditwheel, which would include all dynamic libraries in the 
wheel, but it doesn't work because it would also include PyTorch leading to 
the same problem - loading the library twice.

The proper solution is to rely on the PyTorch wheel to contain the relevant 
library and its dependencies. This library has some dependencies, too, which 
means that they need to be included in the wheel. Unfortunately, auditwheel 
cannot be used for reasons already described (and there is no way to instruct 
auditwheel to not include PyTorch libraries and their dependencies), so we 
need to manually include the relevant libraries, which is what we do in 
`setup.py`.


### Note on glibc++ ABI 

When C++11 was introduced, libstdc++ maintainers had to rewrite parts of the 
library. In order not to break binary compatibility with previously compiled 
programs, they decided that the C++11-conformant library implementation will 
be in an inline namespace (`__cxx11`), which means that symbols for that 
implementation are different (contain the `__cxx11` namespace name in them). 
They also allowed the users to change this behavior (i.e. break binary 
compatibility but link against symbols without `__cxx11` substring in them) 
if they set the flag `_GLIBCXX_USE_CXX11_ABI` to `0`. Ubuntu (and Fedora) 
compile libraries with this definition set to `1`. PyTorch, however, is 
compiled with it set to `0`, which means that our C++ Bigtable plugin and all 
its dependencies have to be compiled that way too. 

In order to make it easy for the users, we compile the plugin and its
dependecies statically. This is especially helpful given the
`_GLIBCXX_USE_CXX11_ABI` incompatibility. However, the end result is a library
loaded by python, so we need to compile the code as position independent
(hence `CMAKE_POSITION_INDEPENDENT_CODE` is set to `ON`).

## Releases
The releases are maintained in the following way:
Every major release gets its own branch and then all smaller releases get 
their own tags. To publish the release, first the `Publish` workflow is run, 
compiling the wheels, uploading them to pypi and saving them as workflow 
artifacts. Then, these wheels are also described and attached to the release 
note on github.