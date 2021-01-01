#! /bin/bash

SCRIPT=`realpath $0`
DIR=`dirname $SCRIPT`
D=$DIR/..

# INP=$D/images/duo.jpg
# INP=$D/images/duo01.jpg
# INP=$D/images/duo02.jpg
# INP=$D/images/cat01.jpg
# INP=$D/images/dog03.jpeg


MODEL=$D/models/ir/public/ssdlite_mobilenet_v2/ssdlite_mobilenet_v2.xml

if [[ ! -f  $MODEL ]]; then 
    echo -e "Model $MODEL not found"
    exit 1
fi

for file in "$D/images"/car*.jpg; do
  echo "$file"
  python3 ${D}/inference_engine/samples/python/object_detection_ssd/detect.py \
    --model $MODEL \
    --device CPU \
    --input $file
done



