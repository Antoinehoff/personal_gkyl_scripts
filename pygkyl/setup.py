from setuptools import setup, find_packages

setup(
    name='pygkyl',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
    ],
    author='Antoine C.D. Hoffmann',
    author_email='your.email@example.com',
    description='A collection of Python utilities for Gkeyll simulations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/python_utilities',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
