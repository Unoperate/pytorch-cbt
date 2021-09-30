from subprocess import Popen, PIPE
from setuptools import setup, Extension, find_packages
import setuptools.command.egg_info
import setuptools.command.build_py
from torch.utils import cpp_extension
import os
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


class BuildExtCommand(cpp_extension.BuildExtension):
  """Extension for patching .so files to wheel
  
  The reason we need to do this, is our extension has dynamic 
  dependencies to some libraries as well as to libraries from pytorch. For 
  that reason, we can"t use auditwheel which does not accept such 
  dependencies to another package. What we do instead, is look for the 
  dynamic dependencies not from pytorch manually and then patch them as part 
  of our wheel.
  """

  def run(self):
    cpp_extension.BuildExtension.run(self)

    print("looking for libs")

    file_list = glob.glob("build/lib*/pytorch_bigtable/pbt_C.cpython*")
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
          os.system(f"cp {lib_loc} {lib_dir}")


setup(name="pytorch_bigtable", version="0.0.1", author="Google",
      author_email="support@gmail.com",
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
