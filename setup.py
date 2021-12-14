from setuptools import setup, find_packages

setup(
    name="src",
    version="0.0.1",
    description="To predict whether a failure of a Scania Truck component is related to the air pressure system (APS) or not.",
    url="https://github.com/ankitsharma6652/Scania-Truck-Failures",
    author="Devashri Chaudhari and Ankit Sharma", 
    packages=["src"],
    python_requires=">=3.7",
    license="MIT"
)