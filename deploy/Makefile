all:
	@echo Prepares CI docker image

1-build:
	docker build -t ubuntu-xenial-fiontb .

2-test:
	docker run ubuntu-xenial-fiontb

3-tag:
	docker tag ubuntu-xenial-rflow otaviog/ubuntu-xenial-fiontb

4-push:
	docker push otaviog/ubuntu-xenial-fiontb

