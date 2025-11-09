
import os
import sys
import platform

from setuptools import find_packages, setup, Extension

copy_args = sys.argv[1:]
#copy_args.append('--user')
 
ext_modules = []

setup(
      name="roman_preflight_proper",
      version = "2.0.1",
      packages=find_packages(),

      install_requires = ['numpy>=1.8', 'scipy>=0.19', 'astropy>=1.3', 'PyPROPER3>=3.3'],

      package_data = {
        '': ['*.*']
      },

      script_args = copy_args,

      zip_safe = False, 

      # Metadata for upload to PyPI
      author="John Krist",
      author_email = "john.krist@jpl.nasa.gov",
      description="Roman Space Telescope coronagraph pre-flight PROPER prescription",
      license = "BSD",
      platforms=["any"],
      url="",
      ext_modules = ext_modules
)
