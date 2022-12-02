# Template package


<!-- # Installation
To install this package using pip, you must have access to this repository through your GitLab account. 
If you have set up an ssh key with your account (https://docs.gitlab.com/ee/ssh/).

    pip install git+ssh://git@gitlab.au.dk/ksc/pkgs/package_name.git

Alternatively, you can install using https.

    pip install git+https://gitlab.au.dk/ksc/pkgs/package_name.git

This method requires setting up a personal access token (https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html), if you haven't already.

Lastly, you can also download the repository, unpack it and install using 

    pip install path/to/package_name/root -->

# Development
## Setup
Fork this repository and set up a development environment using

    conda env create

or

    make env

and activate it using

    conda activate package_name

The environment name is changed in environment.yml file

## Running the test suite
Run all tests

    python -m unittest discover -v -s tests

or 

    make test

Run a single test file

    python -m unittest tests/test_file.py

## Building the documentation
To build the documentation run 

    make docs