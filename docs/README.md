# Unified Cache Manager documents

Live doc: Coming soon

## Design documents

- [UCM Connector v2 详细设计](connector-v2-detailed-design.md)：当前实现的数据流、布局寻址、二维 Transfer 接口、生命周期和验证边界。

## Build the docs

```bash
# Install dependencies.
pip install -r requirements-docs.txt

# Build the docs.
make clean
make html


# Open the docs with your browser
python3 -m http.server -d build/html/
```

Launch your browser and open:
- English version: http://localhost:8000
