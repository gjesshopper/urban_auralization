ifeq ($(OS), Windows_NT)
run:
	python urban_auralization/main.py

install:
	pip install -r requirements.txt

build: setup.py
	python setup.py build bdist_wheel

clean:
	if exists "./build" rd /s /q build
	if exists ".dist" rd /s /q dist

else
run:
	python3 urban_auralization/main.py

req:
	pip3 freeze > requirements.txt

install: requirements.txt
	pip3 install -r requirements.txt

build: setup.py
	python3 setup.py install

clean:
	rm -rf build
	rm -rf dist
	rm -rf urban_auralization.egg-info
endif