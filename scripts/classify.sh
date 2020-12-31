#! /bin/bash

SCRIPT=`realpath $0`
DIR=`dirname $SCRIPT`
D=$DIR/..

INP=$HOME/data/imagen
#INP=$HOME/data/imagen/n01443537_5048_goldfish.jpg
#INP=$D/images/duo01.jpeg
#INP=$D/images/cat01.jpeg
#INP=$D/images/dog03.jpeg

MODEL=$D/models/ir/public/squeezenet1.1/FP16/squeezenet1.1


#ls -al "${INP}"

python3 ${D}/inference_engine/samples/python/hello_classification/hello_classification.py \
  --input $INP \
  --model $MODEL.xml \
  --labels $MODEL.labels \
  -n 100

# not sure why python3 blocks from bash?


