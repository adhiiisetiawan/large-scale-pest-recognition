#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="pest_rec",
    version="0.0.1",
    description="Implementation of an advanced Convolutional Neural Network (CNN) for large-scale pest recognition, incorporating augmentation techniques and regularizers for improved accuracy and generalization.",
    author="Adhi Setiawan",
    author_email="adhisetiawan518@gmail.com",
    url="https://github.com/adhiiisetiawan/large-scale-pest-recognition",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = pest_rec.train:main",
            "eval_command = pest_rec.eval:main",
        ]
    },
)