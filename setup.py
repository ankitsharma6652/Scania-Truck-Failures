from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="src",
    version="0.0.1",
    description="To predict whether a failure of a Scania Truck component is related to the air pressure system (APS) or not.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="git@github.com:ankitsharma6652/Scania-Truck-Failures.git",
    author="Devashri Chaudhari and Ankit Sharma",
    author_email="devashrichaudhari@gmail.com,ankitcoolji@gmail.com" ,
    packages=["src"],
    python_requires=">=3.7",
    license="MIT"
)