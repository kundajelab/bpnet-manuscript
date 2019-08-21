#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

requirements = [
    # "cyvcf2",
    # "openpyxl",
    "xgboost",
    "joblib",
    "deepdish",
    "dask",
    "toolz",
    "cloudpickle",
    "kipoi",
    "kipoiseq",
    "argh",
    "papermill",
    "nbconvert",
    "python-dotenv",
    "comet_ml",  # deprecate ?
    "vdom",
    "gin-config",
    "gin-train",
    "pyarrow",
]

test_requirements = [
    "pytest",
    "virtualenv",
]


setup(
    name='basepair',
    version='0.0.1',
    description="basepair: toolkit for seq2seq models in genomics",
    author="Ziga Avsec",
    author_email='avsec@in.tum.de',
    url='https://github.com/kundajelab/basepair',
    long_description="basepair: toolkit for seq2seq models in genomics",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "develop": ["bumpversion",
                    "wheel",
                    "jedi",
                    "epc",
                    "pytest",
                    "pytest-pep8",
                    "pytest-cov"],
    },
    entry_points={'console_scripts': ['basepair = basepair.__main__:main']},
    license="MIT license",
    zip_safe=False,
    keywords=["model zoo", "deep learning",
              "computational biology", "bioinformatics", "genomics"],
    test_suite='tests',
    package_data={'basepair': ['logging.conf']},
    tests_require=test_requirements
)
