.PHONY: docs
experiments_image = acs:4

test:
	py.test

notebook:
	jupyter lab --notebook-dir .

publish_experiments_docker_image:
	docker build -f Dockerfile -t $(experiments_image) .
	docker tag $(experiments_image) khozzy/$(experiments_image)
	docker push khozzy/$(experiments_image)

verify_publication_notebooks:
	(cd notebooks/publications; python verify.py)
