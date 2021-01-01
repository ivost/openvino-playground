# openvino-playground
learning openvino

needs python 3.6 or 3.7
install pyenv

in .bashrc
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"


pyenv --version
pyenv install 3.7.9
pyenv virtualenv 3.7.9 venv-3.7

pyenv local venv-3.7 && python3 -V

~/.pyenv/versions/3.7.9/envs/venv-3.7/bin/python3.7 -m pip install --upgrade pip

Successfully installed pip-20.3.3

pip install openvino

Successfully installed numpy-1.19.4 openvino-2021.2

?? on mac ??
export DYLD_LIBRARY_PATH=$HOME/.pyenv/versions/3.7.9/lib:${DYLD_LIBRARY_PATH}


pyenv local venv-3.7

python3 -c "import openvino"

---

sudo cp -rv /opt/intel/openvino_2021/deployment_tools/inference_engine/lib/intel64/* /lib/x86_64-linux-gnu/

Get imagenet samples

git clone https://github.com/ajschumacher/imagen

assuming they are in $HOME/data

Classification example

./scripts/classify.sh


Performance:

on fast i-9 CPU (12 cores): 
2.6 ms 

on Vizi: Atom(TM) Processor E3940 @ 1.60GHz 4 cores
27 ms ~10X slower

with HDDL
9 ms - ~3X faster than CPU





