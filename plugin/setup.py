from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='dataset_cpp',
      ext_modules=[cpp_extension.CppExtension('dataset_cpp', ['dataset.cpp'], include_dirs=['/usr/local/include/'])], # include_dirs=['/home/kajetan/git/google-cloud-cpp']
      cmdclass={'build_ext': cpp_extension.BuildExtension})
