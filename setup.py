from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'D&D dungeon generator'
LONG_DESCRIPTION = 'Package to generate Dungeons and populate them with monsters and traps'

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="pydungeon",
    version=VERSION,
    author="Rex Boyce",
    author_email="rexboyce@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],  # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'

    # keywords=['python', 'first package'],
)
