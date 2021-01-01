# openvino-playground
learning openvino


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





