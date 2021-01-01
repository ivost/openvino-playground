#! /bin/bash

SCRIPT=`realpath $0`
DIR=`dirname $SCRIPT`
D=$DIR/..

INP=/home/ivo/data/imagen
#INP=$HOME/data/imagen/n01443537_5048_goldfish.jpg
#INP=$D/images/duo01.jpeg
#INP=$D/images/cat01.jpeg
#INP=$D/images/dog03.jpeg

MODEL=$D/models/ir/public/squeezenet1.1/FP16/squeezenet1.1


#ls -al "${INP}"
#  --quiet True \

python3 ${D}/inference_engine/samples/python/hello_classification/hello_classification.py \
  --input $INP \
  --model $MODEL.xml \
  --labels $MODEL.labels \
  --device HDDL \
  -n 1000

# max batch size with HDDL is 100  