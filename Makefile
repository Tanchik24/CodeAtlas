FOLDER = src/
VENV = .venv


set_config:
	@echo "setting the config and creating poetry.lock ..."
	poetry config virtualenvs.in-project true
	poetry lock


.PHONY: install
install:
	@echo "installing dependencies..."
	poetry install --no-root


.PHONY: setup
setup: set_config install


.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".idea" -delete
	rm -rf .mypy_cache
	rm -rf .ruff_cash
	rm -rf $(VENV)


.PHONY: lint
lint:
	poetry run ruff check $(FOLDER)
	poetry run mypy $(FOLDER)


.PHONY: format
format:
	poetry run ruff format $(FOLDER)
	poetry run isort $(FOLDER)
