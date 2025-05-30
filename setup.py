from setuptools import setup, find_packages
import re

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

    has_dev_pytorch = "dev" in torch.__version__
    has_cuda = torch.cuda.is_available()
except ImportError:
    has_dev_pytorch = False
    has_cuda = False

# Base requirements
install_requires = [
    "torch>=2.4.0",
]
if has_dev_pytorch:  # Remove the PyTorch requirement
    install_requires = [
        install_require for install_require in install_requires if "torch" != re.split(r"(=|<|>)", install_require)[0]
    ]

# Define extension modules
ext_modules = []
cmdclass = {}

if has_cuda:
    # Add CUDA extension if CUDA is available
    try:
        ext_modules.append(
            CUDAExtension(
                name='qtorch.quant_cuda',
                sources=[
                    'qtorch/quant/quant_cuda/quant_cuda.cpp',
                    'qtorch/quant/quant_cuda/bit_helper.cu',
                    'qtorch/quant/quant_cuda/sim_helper.cu',
                    'qtorch/quant/quant_cuda/block_kernel.cu',
                    'qtorch/quant/quant_cuda/float_kernel.cu',
                    'qtorch/quant/quant_cuda/fixed_point_kernel.cu',
                    'qtorch/quant/quant_cuda/quant.cu',
                    'qtorch/quant/quant_cuda/posit_kernel.cu',
                ],
                extra_compile_args={
                    'cxx': ['-O3'],
                    'nvcc': ['-O3', '--use_fast_math']
                }
            )
        )
    except Exception:
        print("Warning: CUDA extension compilation failed, falling back to CPU only")

# Add CPU extension
try:
    ext_modules.append(
        CppExtension(
            name='qtorch.quant_cpu',
            sources=[
                'qtorch/quant/quant_cpu/quant_cpu.cpp',
                'qtorch/quant/quant_cpu/bit_helper.cpp',
                'qtorch/quant/quant_cpu/sim_helper.cpp',
            ],
            extra_compile_args={'cxx': ['-O3']}
        )
    )
    cmdclass['build_ext'] = BuildExtension
except Exception:
    print("Warning: C++ extension compilation failed")

setup(
    name="qtorch",
    version="2.0.0",
    description="Low-Precision Arithmetic Simulation in Pytorch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Tianyi Zhang, Zhiqiu Lin, Guandao Yang, Christopher De Sa",
    author_email="tz58@cornell.edu",
    project_urls={
        "Documentation": "https://qpytorch.readthedocs.io",
        "Source": "https://github.com/Tiiiger/QPyTorch/graphs/contributors",
    },
    packages=find_packages(),
    include_package_data=True,
    license="MIT",
    python_requires=">=3.8",
    install_requires=install_requires,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
)
