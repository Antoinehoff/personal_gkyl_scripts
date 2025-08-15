from setuptools import setup, find_packages

def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, 'r') as file:
        return [line.strip() for line in file if line.strip() and not line.startswith('#')]

setup(
    name='pygkyl',
    version='0.3.0',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    author='Antoine C.D. Hoffmann',
    author_email='ahoffman@pppl.gov',
    description='A collection of python utilities for Gkeyll simulations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Antoinehoff/personal_gkyl_scripts/tree/main/pygkyl',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)


