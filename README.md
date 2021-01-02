# openvino-playground
learning openvino

needs python 3.7
sudo apt install python3.7

install pyenv
https://github.com/pyenv/pyenv-installer

curl https://pyenv.run | bash


echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc

in .bashrc
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
---
pyenv --version
# pyenv install 3.7.9
pyenv virtualenv 3.7.9 venv-3.7

pyenv local venv-3.7 && python3 -V

~/.pyenv/versions/3.7.9/envs/venv-3.7/bin/python3.7 -m pip install --upgrade pip

Successfully installed pip-20.3.3

pip install openvino

pyenv local venv-3.7

python3 -c "import openvino"

open -na "PyCharm.app"

Make sure to change working dir in run config to openvino-playground

---

sudo cp -rv /opt/intel/openvino_2021/deployment_tools/inference_engine/lib/intel64/* /lib/x86_64-linux-gnu/

Get imagenet samples

MD ~/datap
git clone https://github.com/ajschumacher/imagen
cd imagen/
mv imagen/* .
rmdir imagen/

assuming jpg files are in $HOME/data/imagen

Classification example

./scripts/classify.sh

Performance:

on fast i7 3 GHz
2.6 ms 

on Mac - Core i7 2.6 GHz
2.7 ms

on Vizi: Atom(TM) Processor E3940 @ 1.60GHz 4 cores
27 ms ~10X slower

with HDDL / Myriad
9 ms - ~3X faster than Atom - still 3X slower than i7






