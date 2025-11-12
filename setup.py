from setuptools import setup, find_packages

install_requires = [
    "numpy",
    "pandas",
    "matplotlib",
    "cartopy",
    "geopy",
    "geopandas",
    "statsmodels",
    "h5py",
    "hapiclient",
    "spacepy",
    "datetick @ git+https://github.com/rweigel/datetick",
    "utilrsw @ git+https://github.com/rweigel/utilrsw"
]

setup(
    name='swerve',
    version='0.0.3',
    author='Lucy Wilkerson, Bob Weigel, Dean Thomas',
    author_email='rweigel@gmu.edu',
    packages=find_packages(),
    description='Process data for the May, 2024 geomagnetic storm',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    include_package_data=True,
)
