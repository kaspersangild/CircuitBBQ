from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="circuitbbq",  # Required
    version="0.0.2",  # Required
    description="Package for analysis of superconducting circuits",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  # Optional (see note above)
    url="https://github.com/kaspersangild/CircuitBBQ",  # Optional
    author="Kasper Sangild",  # Optional
    author_email="kaspersangild@phys.au.dk",  # Optional
    packages=find_packages(),  # Required
    python_requires=">3",
    install_requires=["numpy", "sympy", "scipy", "networkx"],  # Optional
)
