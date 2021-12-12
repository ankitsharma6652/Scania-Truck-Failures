from setuptools import setup, find_packages

setup(
    name="Scania-Truck-Failures",
    version="0.0.1",
    description="To predict whether a failure of a Scania Truck component is related to the air pressure system (APS) or not.",
    author="Devashri Chaudhari and Ankit Sharma", 
    packages=find_packages(),
    python_requires=">=3.7",
    license="MIT"
)