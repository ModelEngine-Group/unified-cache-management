# How to contribute
## Building and testing
It’s recommended to set up a local development environment to build and test before you submit a PR.
### Run lint locally
Run following commands to format your code before submit:
```bash
# Choose a base dir (~/vllm-project/) and set up venv
cd ~/vllm-project/
python3 -m venv .venv
source ./.venv/bin/activate

# Clone UCM and install
git clone https://github.com/ModelEngine-Group/unified-cache-management.git 
cd unified-cache-management

# Install lint requirement and enable pre-commit hook
pip install -r requirements-lint.txt

# Run lint (You need install pre-commits deps via proxy network at first time)
bash format.sh
```
### Run unit test locally
Run unit test locally with following command:
```bash
python3 -m unittest discover -s test
```
