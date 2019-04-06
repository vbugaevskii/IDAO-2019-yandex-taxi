all: build

build:
	cat /proc/cpuinfo | grep processor -c >&2

run: predict.py
	bash ./run.sh
