Using pytorch to implement [Deep Deterministic Policy Gradient(DDPG)](https://arxiv.org/abs/1509.02971).

## Denpendency
- python 3.6
- pytorch 0.4+
- tensorboard
- gym

## Train
```
main.py --train --env MountainCarContinuous-v0 --cuda
```
Parameters:
|Parameters | description|
|-|-|
|  --train|  train model |
|  --test|  test model |
|  --retrain |      retrain model
|  --retrain_model |   retrain model path|
|  --env|        gym environment name|
|  --episodes| train episodes |
|  --eps_decay| noise epsilon decay |
|  --cuda|                use cuda |
|  --model_path|    if test mode, import the model |
|  --record       |   record the video |
|  --record_ep_interval   | record episodes interval |
|  --checkpoint        |  use model checkpoint |
|  --checkpoint_interval |    checkpoint interval |

(more parameters see the file)


You can use the tensorboard to see the training.
```
tensorboard --logdir=out/MountainCarContinuous-v0
```

## Test
You can test your model with `--test` like this:
```
main.py --test --env MountainCarContinuous-v0 --model_path out/MountainCarContinuous-v0-run0
```
It will render graphical interface.

## Result
It turns out that tuning parameters are very important, especially `eps_decay`.  I use the simple linear noise decay such as `epsilon -= eps_decay` every episode.

- Pendulum-v0

```
main.py --train --env Pendulum-v0 --cuda --eps_decay 0.01
```

<img src="https://i.loli.net/2018/08/30/5b87ed3d3de32.png" width="700px">

- MountainCarContinuous-v0

```
main.py --train --env MountainCarContinuous-v0 --cuda --eps_decay 0.001
```

<img src="https://i.loli.net/2018/08/30/5b87ee3d97439.png" width="700px">

## Reference
- paper [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)



