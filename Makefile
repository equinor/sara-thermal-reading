format:
	black .
	isort .
	mypy sara_thermal_reading utils_cli.py main.py
