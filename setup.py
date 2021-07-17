from setuptools import setup, find_packages

pkgs = find_packages()

setup(
    name="cglb",
    author="Artem Artemev, David Burt",
    author_email="a.artemev20@imperial.ac.uk, drb62@cam.ac.uk",
    version="0.0.1",
    packages=pkgs,
    install_requires=["numpy", "scipy"],
    dependency_links=[],
)
