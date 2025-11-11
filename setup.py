from setuptools import setup, find_packages

setup(
    name='bert4torch',
    version='0.1.0',
    description='Reimplement bert4keras with PyTorch',
    author='kelvin',
    packages=find_packages(),
    install_requires=[
        'torch>=1.10.0',
    ],
    python_requires='>=3.7',
)
