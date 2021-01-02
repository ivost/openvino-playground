#! /bin/bash

SCRIPT=`realpath $0`
DIR=`dirname $SCRIPT`
D=$DIR/..

MODEL=$D/models/person-vehicle-bike-detection-crossroad-1016/FP16/person-vehicle-bike-detection-crossroad-1016.xml

echo -e "Model $MODEL"

if [[ ! -f  $MODEL ]]; then 
    echo -e "Model $MODEL not found"
    exit 1
fi

P=$D/python/detect/detect.py 

for file in "$D/images"/car-ped-bike-01.jpg; do
  echo "$file"
  python3 $P \
    --model $MODEL \
    --device CPU \
    --input $file
done



