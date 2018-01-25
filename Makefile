export SHELL := /bin/bash

doctest:
	pytest --doctest-modules xenonpy

unittest:
	pytest tests

lint:
	pylint xenonpy
