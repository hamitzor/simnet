from setuptools import setup, find_packages

with open('README') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='simnet',
    version='0.1.0',
    description='A simple neural network libray',
    long_description=readme,
    author='Hamit Zor',
    author_email='thenrerise@gmail.com',
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
