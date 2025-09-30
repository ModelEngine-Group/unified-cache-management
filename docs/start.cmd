pip install -r requirements-docs.txt
start "" /wait cmd /c .\make.bat clean
start "" /wait cmd /c .\make.bat html
start python -m http.server -d build/html/
start http://localhost:8000
