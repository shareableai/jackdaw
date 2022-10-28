from setuptools import find_packages, setup
from jackdaw import __name__, __version__


try:
    requirements = open("requirements.txt").readlines()
except FileNotFoundError:
    requirements = []

try:
    dev_requirements = open("dev_requirements.txt").readlines()
except FileNotFoundError:
    dev_requirements = []

LIBRARIES = [*requirements, *dev_requirements]

setup(
    name=__name__,
    version=__version__,
    packages=find_packages(),
    install_requires=requirements,
    extras_require={"dev": dev_requirements + requirements, "all": LIBRARIES},
    license='LGPL-3.0-or-later',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)
