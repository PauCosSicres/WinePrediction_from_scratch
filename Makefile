install:
	conda install -r requirements.txt

scratch:
	python src/main_scratch.py

lib:
	python src/main_lib.py

clean:
	@if exist src\__pycache__ rmdir /s /q src\__pycache__
	@if exist .ipynb_checkpoints rmdir /s /q .ipynb_checkpoints