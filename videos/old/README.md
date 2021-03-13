# openvino-playground


CLASSIFICATION 100 images 
Squeezenet

| Device | time |
| -------------------------- | ----: | 
| Intel i7 3GHz  | 2.6 ms |
| Mac Core i7 2.6GHz  | 2.7 ms |
| Adlink Atom E3940 1.6 GHz | 27 ms |
| Adlink MyriadX / HDDL | 9 ms |
| RPi4 MyriadX / NCS2 | 11.3 ms |

```
Full device name: Intel Movidius Myriad X VPU
Count:      1000 iterations
Duration:   3554.43 ms
Latency:    14.16 ms
Throughput: 281.34 FPS

python   3.6
openvino 2021.2
numpy    1.16.3
opencv-python 3.4.4.19
```




