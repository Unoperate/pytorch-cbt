from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension
import os

extra_compile_args = os.popen('pkg-config bigtable_client --cflags').read().split()
extra_link_args = os.popen('pkg-config bigtable_client --libs').read().split()

setup(name='dataset_cpp',
      ext_modules=[cpp_extension.CppExtension(
            'dataset_cpp.code', 
            ['src/dataset_cpp/dataset.cpp'], 
            include_dirs=['/usr/local/include/'],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            )],
      package_dir={"": "src"},
      packages=['dataset_cpp'],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
