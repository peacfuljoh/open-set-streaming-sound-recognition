SHELL = /bin/bash
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

REPO_ROOT_ = /home/nuc/web-apps
BB_ROOT_ = $(REPO_ROOT_)/buddybork
REPO_ROOT_GHA_ = /home/runner/work/web-apps/web-apps
BB_ROOT_GHA_ = $(REPO_ROOT_GHA_)/buddybork


test:
	npm test
	cd $(BB_ROOT_)
	$(CONDA_ACTIVATE) buddybork && \
	pytest

test-gha:
	npm test
	cd $(BB_ROOT_GHA_)
	$(CONDA_ACTIVATE) buddybork && \
	pytest

build:
	echo 'export BB_PATHS=$(BB_ROOT_)/paths.json' >> ~/.bashrc
	echo 'export PYTHONPATH="${PYTHONPATH}:$(BB_ROOT_)"' >> ~/.bashrc
	echo 'export BB_ROOT=$(BB_ROOT_)' >> ~/.bashrc

build-gha:
	echo 'BB_PATHS=$(BB_ROOT_GHA_)/paths.json' >> $(GITHUB_ENV)
	echo 'PYTHONPATH=$(BB_ROOT_GHA_)' >> $(GITHUB_ENV)
	echo 'BB_ROOT=$(BB_ROOT_GHA_)' >> $(GITHUB_ENV)

app:
	cd '$(BB_ROOT_)/node' && \
	npx nodemon app.js

stream:
	$(CONDA_ACTIVATE) buddybork && \
	python $(BB_ROOT_)/src/main_stream.py

stream-flask:
	$(CONDA_ACTIVATE) buddybork && \
	python $(BB_ROOT_)/src/data_capture_flask_app.py

cleanup:
	$(CONDA_ACTIVATE) buddybork && \
	python $(BB_ROOT_)/src/main_asset_cleanup.py

precompute:
	$(CONDA_ACTIVATE) buddybork && \
	python $(BB_ROOT_)/src/main_feature_precompute.py

