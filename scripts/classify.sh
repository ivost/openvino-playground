#! /bin/bash

SCRIPT=`realpath $0`
DIR=`dirname $SCRIPT`
D=$DIR/..

echo $D

#pip install -r requirements.txt

export PYTHONPATH=$D:$PYTHONPATH
export INPUT=$HOME/data/imagen
# interesting case
#export INPUT=$D/images/918-02.jpg
export MODEL=$D/models/squeezenet1.1/FP16/squeezenet1.1.xml
export LABELS=$D/models/squeezenet1.1/FP16/squeezenet1.1.labels

if [[ ! -d  $INPUT ]]; then 
    echo -e "Input dir $INPUT not found"
    exit 1
fi
if [[ ! -f  $MODEL ]]; then 
    echo -e "Model $MODEL not found"
    exit 1
fi

# python3 $D/py/classify/classify.py -h


python3 $D/insg/ncs2/classify.py \
  -n 100 \
  --device MYRIAD
#  --quiet \

# max batch size with HDDL is 100
