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

import shutil
from subprocess import Popen, PIPE
from setuptools import setup, Extension, find_packages
import setuptools.command.egg_info
import setuptools.command.build_py
from torch.utils import cpp_extension
import os
import sys
import glob

extra_compile_args = os.popen(
  "pkg-config bigtable_client --cflags").read().split() + ["-std=c++17"]
extra_link_args = os.popen(
  "pkg-config bigtable_client --libs").read().split() + [
                    "-Wl,-rpath=$ORIGIN:$ORIGIN/lib"]

os.environ["CXX"] = "g++"

INCLUDE_LIBS = ["libssl", "libcrypto"]

with open("README.md", "r", encoding="utf-8") as fh:
  long_description = fh.read()

with open("version.txt", "r", encoding="utf-8") as fh:
  version = fh.read()


class BuildExtCommand(cpp_extension.BuildExtension):
  """Extension for including .so files in the wheel.
  
  The reason we need to do this, is our extension has dynamic 
  dependencies to some libraries as well as to libraries from pytorch. For 
  that reason, we can"t use auditwheel which does not accept such 
  dependencies to another package. What we do instead, is look for the 
  dynamic dependencies not from pytorch manually and then include them as part
  of our wheel.
  """

  def run(self):
    cpp_extension.BuildExtension.run(self)

    print("looking for libs")

    file_list = glob.glob(
      os.path.join("build", f"lib.linux-x86_64-3.{sys.version_info.minor}",
                   "pytorch_bigtable", "pbt_C.cpython*"))
    if len(file_list) != 1:
      raise RuntimeError(
        "Error when looking for C extension .so file. Expected to find 1 "
        f"file, found {len(file_list)}.")
    so_file_loc = file_list[0]
    lib_dir = os.path.join(os.path.dirname(so_file_loc), "lib")

    if not os.path.exists(lib_dir):
      os.makedirs(lib_dir)

    process = Popen(["ldd", so_file_loc], stdout=PIPE)
    (output, err) = process.communicate()
    process.wait()
    assert (err is None)

    print("patching libs")

    for line in output.decode().split("\n"):
      for lib in INCLUDE_LIBS:
        if lib in line:
          lib_loc = line.split()[2]
          try:
            shutil.copyfile(lib_loc,
                            os.path.join(lib_dir, os.path.basename(lib_loc)))
          except shutil.SameFileError:
            pass


setup(name="pytorch_bigtable", version=version, author="Google",
      author_email="info@unoperate.com",
      description="Pytorch Extension for BigTable",
      long_description=long_description,
      url="https://github.com/Unoperate/pytorch-cbt",
      install_requires=["torch>=1.9.0"], ext_modules=[
    cpp_extension.CppExtension("pytorch_bigtable.pbt_C",
                               ["csrc/bigtable_dataset.cc"],
                               extra_compile_args=extra_compile_args,
                               extra_link_args=extra_link_args, )],
      package_dir={"": "."}, packages=["pytorch_bigtable"],
      cmdclass={"build_ext": BuildExtCommand, }, python_requires=">=3.6", )
