from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension
import os

extra_compile_args = os.popen('pkg-config bigtable_client --cflags').read().split()
extra_link_args = os.popen('pkg-config bigtable_client --libs').read().split()

setup(name='pytorch_bigtable',
      ext_modules=[cpp_extension.CppExtension(
            'pbt_C', 
            ['csrc/dataset.cpp'], 
            include_dirs=['/usr/local/include/'],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            )],
      package_dir={"": "."},
      packages=['pytorch_bigtable'],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
