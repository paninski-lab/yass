import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

#ENV = os.environ["CONDA_PREFIX"]  # Absolute path of active conda env root
#LIBRARY_DIRS = [os.path.join(ENV, "lib")]  # .[so,dylib]
#INCLUDE_DIRS = [os.path.join(ENV, "include")]
#LIBRARIES = ["magma"]

setup(name='inplace',
      ext_modules=[
          CUDAExtension('rowshift',
                        ['rowshift.cpp',
                         'rowshift_kernels.cu'])
      ],
      cmdclass={
          'build_ext': BuildExtension
      }
)