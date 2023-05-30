from setuptools import setup, find_packages, Extension, Command
#from skbuild import setup
from setuptools.command.build_ext import build_ext
import subprocess

import re
import os

class CMakeBuildExt(build_ext):
    """Custom command to build Submodules packages during installation."""
    
    def run(self):
        
        package_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir = package_dir + "/build"
        config_settings = getattr(self, 'config_settings', {})
        cmake_args = []
        if 'build_subm' in config_settings.keys() and config_settings['build_subm']=='no':
            pass
        else:
            if not os.path.isdir(build_dir):
                os.makedirs(build_dir)
            subprocess.check_call(
                ["cmake", ".."] + cmake_args, cwd=build_dir
            )
            subprocess.check_call(
                ["make", "install", "-j4"], cwd=build_dir
            )

        super().run()

class BuildCommand(Command):
    """Custom command to build Submodules packages without installation."""

    description = 'Build Submodules in lib packages'
    user_options = [
        ('cmake-args=', None, 'Additional CMake arguments'),
    ]

    def initialize_options(self):
        self.cmake_args = None

    def finalize_options(self):
        pass

    def run(self):
        # Run the CMake build step with additional cmake_args
        package_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir = package_dir + "/build"
        if not os.path.isdir(build_dir):
            os.makedirs(build_dir)
        if self.cmake_args is not None:
            subprocess.check_call(
                ["cmake", f"{self.cmake_args}", ".."], cwd=build_dir
            )
        else:
            subprocess.check_call(
                ["cmake", ".."], cwd=build_dir
            )
            
        subprocess.check_call(
            ["make", "install", "-j4"], cwd=build_dir
        )

ext_modules = [
    Extension('lib', []),
    # Add more Extension instances for additional extension modules
]


__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open(os.path.join(this_directory, "sharpy/version.py")).read(),
)[0]

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="sharpy",
    version=__version__,
    description="""SHARPy is a nonlinear aeroelastic analysis package developed
    at the Department of Aeronautics, Imperial College London. It can be used
    for the structural, aerodynamic and aeroelastic analysis of flexible
    aircraft, flying wings and wind turbines.""",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="nonlinear aeroelastic structural aerodynamic analysis",
    author="",
    author_email="",
    url="https://github.com/ImperialCollegeLondon/sharpy",
    license="BSD 3-Clause License",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuildExt,
              "build_subm": BuildCommand},
    packages=find_packages(
        where='./',
        include=['sharpy*'],
        exclude=['tests']
        ),
    install_requires=[
    ],
    classifiers=[
        "Operating System :: Linux, Mac OS",
        "Programming Language :: Python, C++",
        ],

    entry_points={
        'console_scripts': ['sharpy=sharpy.sharpy_main:sharpy_run'],
        }
)
