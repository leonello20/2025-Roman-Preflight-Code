import sys
import platform

from setuptools import find_packages, setup, Extension

copy_args = sys.argv[1:]
#copy_args.append('--user')
 
ext_modules = []

setup(
      name="cgisim",
      version = "4.0",
      packages=find_packages(),
      zip_safe=False,
      install_requires = ['numpy>=1.8', 'scipy>=0.19', 'astropy>=1.3', 'emccd_detect>=2.2.5', 'PyPROPER3>=3.3', 'roman_preflight_proper>=2.0'],

      package_data = {
        '': ['*.*']
      },

      script_args = copy_args,

      # Metadata for upload to PyPI
      author="John Krist",
      author_email = "john.krist@jpl.nasa.gov",
      description="Roman Space Telescope coronagraph simulator",
      license = "BSD",
      platforms=["any"],
      url="",
      ext_modules = ext_modules
)
