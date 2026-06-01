format:
	uv run black .
	uv run isort .
	uv run mypy sara_thermal_reading utils_cli.py main.py

run-example:
	uv run utils_cli.py run-fff-workflow \
		--polygon-path example-data/asset-example/polygon.json \
		--reference-image-path example-data/asset-example/thermal_image.tiff
