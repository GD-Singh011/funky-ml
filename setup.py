from setuptools import setup, find_packages
from codecs import open
from os import path
here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='funky-ml',
    packages=['funkyml'],
    version='0.1.2',
    license='MIT',
    description="Automated ML by GD-Singh",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Gagandeep Singh',
    author_email="gdswasir@gmail.com", 
    url="https://github.com/GD-Singh011/funky-ml",
    keywords=['funkyml', 'AutoML', 'Python'],
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "seaborn",
        "matplotlib",
        "scikit-learn",
        "imblearn",
        "xgboost",
        "lightgbm",
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
