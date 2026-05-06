format:
	uv run black .
	uv run isort .
	uv run mypy sara_thermal_reading utils_cli.py main.py
