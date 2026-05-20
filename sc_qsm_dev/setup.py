from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "lbv_2D",
        ["lbv_2D.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["/std:c++17"],  # Windows MSVC
    ),
]

setup(
    name="lbv_2D",
    version="0.1",
    ext_modules=ext_modules,
)