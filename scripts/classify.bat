
PUSHD "C:\Program Files (x86)\Intel\openvino_2021\bin"
call setupvars.cmd
POPD

python .\py\classify.py ^
  -n 1 ^
  --device MYRIAD

