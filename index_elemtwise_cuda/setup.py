import os
import glob

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.0.1"

def get_extension():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    csrc_dir = os.path.join(root_dir, 'csrc')
    cuda_dir = os.path.join(csrc_dir, 'cuda')

    cpp_files = glob.glob(os.path.join(csrc_dir, "*.cpp"))
    cu_files = glob.glob(os.path.join(cuda_dir, "*.cu"))
    
    extension = CUDAExtension(
        "IndexElemtwiseCUDA",
        cpp_files + cu_files
    )
    return [extension]

setup(
    name="idxelemwise_cuda",
    version=__version__,
    python_requires=">=3.7",
    install_requires=[],
    ext_modules=get_extension(),
    cmdclass={
        "build_ext":
        BuildExtension
    },
    packages=find_packages()
)