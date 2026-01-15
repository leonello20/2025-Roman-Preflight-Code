from setuptools import setup, find_packages

setup(
    name='CAPyBARA',
    version='0.1.0',  # Changed from '0.1' to '0.1.0'
    description='Python Distribution Utilities',
    author='Alexis Y. S. Lau, Lisa Altinier, Damien Camugli, Elodie Choquet',
    author_email='alexis.lau@lam.fr',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'CAPyBARA': ['show_capybara.txt'],
    },
)