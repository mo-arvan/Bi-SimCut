
# Reproduction Steps


## Setup
```bash
cd Bi-SimCut

docker build -t bisimcut-rep:latest -f artifacts/dockerfile .

```


```bash

docker run -it --rm \
    --gpus all \
    --shm-size=1g \
     --ulimit stack=67108864 \
    -v $PWD:/workspace \
    bisimcut-rep:latest \
    bash

```



## Training and Evaluation

You need to set the `MODEL_DIR` and `SEED` environment variables for each run.
```bash
# inside the container
export SEED=1400 # I've trained on 1313 and 1314 
export EXP=iwslt14_de_en_simcut_alpha3_p005_${SEED}

. ./artifacts/train_and_eval.sh
```
