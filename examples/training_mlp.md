# Training a Neural Network to Estimate Sensitivity

[![Open in GitHub](https://img.shields.io/badge/Open-GitHub-black?logo=github)][REPRODUCIBILITY_LINK]

GWKokab has easy to build and train APIs to train a Multilayer Perceptron (MLP) to estimate the probability of detection or sensitive spacetime volume. This notebook demonstrates how to use these APIs on toy data.

Let's first create some toy data where the target is simply the sum of the input features.

```python
import h5py
import numpy as np
from gwkokab.utils.train import train_regressor

data = np.random.normal(size=(10_000, 5))
target = np.sum(data, axis=1, keepdims=True)

data_path = "data.hdf5"
with h5py.File(data_path, "w") as f:
    f.create_dataset("x0", data=data[:, 0])
    f.create_dataset("x1", data=data[:, 1])
    f.create_dataset("x2", data=data[:, 2])
    f.create_dataset("x3", data=data[:, 3])
    f.create_dataset("x4", data=data[:, 4])
    f.create_dataset("y", data=target)
```

Now we can use the [`train_regressor`](https://gwkokab.readthedocs.io/en/latest/autoapi/gwkokab/utils/train/index.html#gwkokab.utils.train.train_regressor) function to train an MLP on this data. We will specify the input and output keys, the architecture of the MLP (width and depth), batch size, data path, checkpoint path, number of epochs, validation split, and learning rate.

```python
input_keys = ["x" + str(i) for i in range(5)]
output_keys = ["y"]

checkpoint_path = "model_checkpoint.hdf5"

train_regressor(
    input_keys=input_keys,
    output_keys=output_keys,
    width_size=32,
    depth=2,
    batch_size=64,
    data_path=data_path,
    checkpoint_path=checkpoint_path,
    epochs=500,
    validation_split=0.2,  # 20% of data used for validation
    learning_rate=1e-3,
)
```

```{toggle}
```md
Mon Nov  3 02:33:45 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.57.08              Driver Version: 575.57.08      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4050 ...    On  |   00000000:01:00.0 Off |                  N/A |
| N/A   37C    P0             12W /   30W |      15MiB /   6141MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1745      G   /usr/lib/xorg/Xorg                        4MiB |
+-----------------------------------------------------------------------------------------+
```

```{toggle}
```md
2025-11-03 01:41:40.050 | INFO     | gwkokab.utils.train:train_regressor:251 - Input Keys: x0, x1, x2, x3, x4
2025-11-03 01:41:40.050 | INFO     | gwkokab.utils.train:train_regressor:252 - Output Keys: y
2025-11-03 01:41:40.050 | INFO     | gwkokab.utils.train:train_regressor:253 - Width Size: 32
2025-11-03 01:41:40.050 | INFO     | gwkokab.utils.train:train_regressor:254 - Depth: 2
2025-11-03 01:41:40.050 | INFO     | gwkokab.utils.train:train_regressor:255 - Data Path: data.hdf5
2025-11-03 01:41:40.050 | INFO     | gwkokab.utils.train:train_regressor:256 - Checkpoint Path: model_checkpoint.hdf5
2025-11-03 01:41:40.050 | INFO     | gwkokab.utils.train:train_regressor:257 - Train Size: 8000
2025-11-03 01:41:40.050 | INFO     | gwkokab.utils.train:train_regressor:258 - Test Size: 2000
2025-11-03 01:41:40.050 | INFO     | gwkokab.utils.train:train_regressor:259 - Validation Split: 0.2
2025-11-03 01:41:40.050 | INFO     | gwkokab.utils.train:train_regressor:260 - Batch Size: 64
2025-11-03 01:41:40.050 | INFO     | gwkokab.utils.train:train_regressor:261 - Epochs: 500
2025-11-03 01:41:40.050 | INFO     | gwkokab.utils.train:train_regressor:262 - Learning Rate (peak): 0.001
2025-11-03 01:41:40.050 | INFO     | gwkokab.utils.train:train_regressor:263 - Loss Type: mse
2025-11-03 01:41:40.050 | INFO     | gwkokab.utils.train:train_regressor:264 - Scheduler: cosine+warmup
2025-11-03 01:41:40.050 | INFO     | gwkokab.utils.train:train_regressor:265 - Weight Decay: 0.0001, Grad Clip: 1.0, Seed: 42
Epoch 1/500, Loss: 4.78830E+00, Val Loss: 4.41401E+00:   0%|                                   | 0/500 [00:06<?, ?epochs/s, epoch=1, loss=4.54130E+00]2025-11-03 01:41:47.527
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 1/500, Loss: 4.78830E+00, Val Loss: 4.41401E+00
Epoch 11/500, Loss: 1.62058E-03, Val Loss: 1.52827E-03:   2%|▍                       | 10/500 [00:07<02:00,  4.06epochs/s, epoch=11, loss=1.45763E-03]2025-11-03 01:41:49.226
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 11/500, Loss: 1.62058E-03, Val Loss: 1.52827E-03
Epoch 21/500, Loss: 2.82888E-04, Val Loss: 2.86742E-04:   4%|▉                       | 20/500 [00:09<01:18,  6.14epochs/s, epoch=21, loss=2.25036E-04]2025-11-03 01:41:50.806
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 21/500, Loss: 2.82888E-04, Val Loss: 2.86742E-04
Epoch 31/500, Loss: 7.98825E-05, Val Loss: 8.94090E-05:   6%|█▍                      | 30/500 [00:11<01:22,  5.71epochs/s, epoch=31, loss=7.22907E-05]2025-11-03 01:41:52.984
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 31/500, Loss: 7.98825E-05, Val Loss: 8.94090E-05
Epoch 41/500, Loss: 4.05051E-05, Val Loss: 3.69315E-05:   8%|█▉                      | 40/500 [00:13<01:25,  5.40epochs/s, epoch=41, loss=4.12687E-05]2025-11-03 01:41:54.849
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 41/500, Loss: 4.05051E-05, Val Loss: 3.69315E-05
Epoch 51/500, Loss: 3.85009E-05, Val Loss: 3.01194E-05:  10%|██▍                     | 50/500 [00:15<01:20,  5.56epochs/s, epoch=51, loss=4.79204E-05]2025-11-03 01:41:56.760
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 51/500, Loss: 3.85009E-05, Val Loss: 3.01194E-05
Epoch 61/500, Loss: 3.33267E-05, Val Loss: 2.00760E-05:  12%|██▉                     | 60/500 [00:18<02:16,  3.23epochs/s, epoch=61, loss=3.28595E-05]2025-11-03 01:41:59.494
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 61/500, Loss: 3.33267E-05, Val Loss: 2.00760E-05
Epoch 71/500, Loss: 4.54250E-05, Val Loss: 5.21082E-05:  14%|███▎                    | 70/500 [00:21<02:27,  2.92epochs/s, epoch=71, loss=4.27637E-05]2025-11-03 01:42:03.218
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 71/500, Loss: 4.54250E-05, Val Loss: 5.21082E-05
Epoch 81/500, Loss: 3.08365E-05, Val Loss: 2.94331E-05:  16%|███▊                    | 80/500 [00:26<03:08,  2.22epochs/s, epoch=81, loss=1.21774E-05]2025-11-03 01:42:07.528
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 81/500, Loss: 3.08365E-05, Val Loss: 2.94331E-05
Epoch 91/500, Loss: 2.55722E-05, Val Loss: 2.86532E-05:  18%|████▎                   | 90/500 [00:29<02:34,  2.66epochs/s, epoch=91, loss=9.23580E-06]2025-11-03 01:42:11.333
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 91/500, Loss: 2.55722E-05, Val Loss: 2.86532E-05
Epoch 101/500, Loss: 1.42712E-05, Val Loss: 1.23330E-05:  20%|████▏                | 100/500 [00:33<02:28,  2.70epochs/s, epoch=101, loss=5.57162E-06]2025-11-03 01:42:15.175
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 101/500, Loss: 1.42712E-05, Val Loss: 1.23330E-05
Epoch 111/500, Loss: 2.22185E-05, Val Loss: 1.74799E-05:  22%|████▌                | 110/500 [00:37<02:45,  2.36epochs/s, epoch=111, loss=1.58493E-05]2025-11-03 01:42:19.382
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 111/500, Loss: 2.22185E-05, Val Loss: 1.74799E-05
Epoch 121/500, Loss: 3.08569E-05, Val Loss: 3.87379E-05:  24%|█████                | 120/500 [00:41<02:30,  2.53epochs/s, epoch=121, loss=2.94480E-05]2025-11-03 01:42:23.213
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 121/500, Loss: 3.08569E-05, Val Loss: 3.87379E-05
Epoch 131/500, Loss: 2.09844E-05, Val Loss: 2.31755E-05:  26%|█████▍               | 130/500 [00:45<02:18,  2.67epochs/s, epoch=131, loss=1.09586E-05]2025-11-03 01:42:27.198
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 131/500, Loss: 2.09844E-05, Val Loss: 2.31755E-05
Epoch 141/500, Loss: 1.33393E-05, Val Loss: 9.01372E-06:  28%|█████▉               | 140/500 [00:49<01:57,  3.05epochs/s, epoch=141, loss=5.29127E-06]2025-11-03 01:42:31.373
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 141/500, Loss: 1.33393E-05, Val Loss: 9.01372E-06
Epoch 151/500, Loss: 3.00024E-05, Val Loss: 7.14355E-06:  30%|██████▎              | 150/500 [00:53<01:41,  3.43epochs/s, epoch=151, loss=9.90104E-06]2025-11-03 01:42:34.573
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 151/500, Loss: 3.00024E-05, Val Loss: 7.14355E-06
Epoch 161/500, Loss: 8.71151E-06, Val Loss: 1.17521E-05:  32%|██████▋              | 160/500 [00:56<01:48,  3.13epochs/s, epoch=161, loss=1.36664E-05]2025-11-03 01:42:38.021
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 161/500, Loss: 8.71151E-06, Val Loss: 1.17521E-05
Epoch 171/500, Loss: 2.08338E-05, Val Loss: 1.53395E-05:  34%|███████▏             | 170/500 [01:00<01:54,  2.88epochs/s, epoch=171, loss=5.63181E-06]2025-11-03 01:42:41.765
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 171/500, Loss: 2.08338E-05, Val Loss: 1.53395E-05
Epoch 181/500, Loss: 8.61565E-06, Val Loss: 1.71178E-06:  36%|███████▌             | 180/500 [01:04<01:58,  2.70epochs/s, epoch=181, loss=2.72948E-06]2025-11-03 01:42:45.839
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 181/500, Loss: 8.61565E-06, Val Loss: 1.71178E-06
Epoch 191/500, Loss: 2.97204E-05, Val Loss: 1.60814E-05:  38%|███████▉             | 190/500 [01:08<02:02,  2.52epochs/s, epoch=191, loss=2.96285E-05]2025-11-03 01:42:49.722
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 191/500, Loss: 2.97204E-05, Val Loss: 1.60814E-05
Epoch 201/500, Loss: 3.89809E-06, Val Loss: 4.39903E-06:  40%|████████▍            | 200/500 [01:12<02:02,  2.46epochs/s, epoch=201, loss=7.11772E-06]2025-11-03 01:42:53.602
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 201/500, Loss: 3.89809E-06, Val Loss: 4.39903E-06
Epoch 211/500, Loss: 5.43906E-06, Val Loss: 4.67936E-06:  42%|████████▊            | 210/500 [01:15<01:48,  2.66epochs/s, epoch=211, loss=4.39174E-06]2025-11-03 01:42:57.396
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 211/500, Loss: 5.43906E-06, Val Loss: 4.67936E-06
Epoch 221/500, Loss: 3.35200E-06, Val Loss: 1.21764E-06:  44%|█████████▏           | 220/500 [01:19<01:43,  2.72epochs/s, epoch=221, loss=1.63585E-06]2025-11-03 01:43:01.131
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 221/500, Loss: 3.35200E-06, Val Loss: 1.21764E-06
Epoch 231/500, Loss: 4.48657E-06, Val Loss: 4.46005E-06:  46%|█████████▋           | 230/500 [01:23<01:50,  2.45epochs/s, epoch=231, loss=6.53376E-06]2025-11-03 01:43:05.072
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 231/500, Loss: 4.48657E-06, Val Loss: 4.46005E-06
Epoch 241/500, Loss: 7.98740E-06, Val Loss: 4.98521E-06:  48%|██████████           | 240/500 [01:27<01:42,  2.55epochs/s, epoch=241, loss=9.98378E-06]2025-11-03 01:43:08.986
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 241/500, Loss: 7.98740E-06, Val Loss: 4.98521E-06
Epoch 251/500, Loss: 2.69785E-06, Val Loss: 4.61949E-06:  50%|██████████▌          | 250/500 [01:31<01:34,  2.65epochs/s, epoch=251, loss=1.44903E-06]2025-11-03 01:43:12.890
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 251/500, Loss: 2.69785E-06, Val Loss: 4.61949E-06
Epoch 261/500, Loss: 3.11129E-06, Val Loss: 2.57439E-06:  52%|██████████▉          | 260/500 [01:35<01:34,  2.55epochs/s, epoch=261, loss=2.32726E-06]2025-11-03 01:43:16.926
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 261/500, Loss: 3.11129E-06, Val Loss: 2.57439E-06
Epoch 271/500, Loss: 1.37926E-06, Val Loss: 1.12710E-06:  54%|███████████▎         | 270/500 [01:39<01:26,  2.67epochs/s, epoch=271, loss=8.07112E-07]2025-11-03 01:43:21.027
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 271/500, Loss: 1.37926E-06, Val Loss: 1.12710E-06
Epoch 281/500, Loss: 1.23831E-06, Val Loss: 1.08944E-06:  56%|███████████▊         | 280/500 [01:43<01:28,  2.48epochs/s, epoch=281, loss=5.06915E-07]2025-11-03 01:43:25.012
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 281/500, Loss: 1.23831E-06, Val Loss: 1.08944E-06
Epoch 291/500, Loss: 7.17176E-06, Val Loss: 1.19274E-05:  58%|████████████▏        | 290/500 [01:47<01:25,  2.45epochs/s, epoch=291, loss=3.06516E-05]2025-11-03 01:43:29.087
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 291/500, Loss: 7.17176E-06, Val Loss: 1.19274E-05
Epoch 301/500, Loss: 3.02643E-06, Val Loss: 1.87289E-06:  60%|████████████▌        | 300/500 [01:51<01:17,  2.58epochs/s, epoch=301, loss=8.53610E-06]2025-11-03 01:43:33.132
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 301/500, Loss: 3.02643E-06, Val Loss: 1.87289E-06
Epoch 311/500, Loss: 1.55948E-06, Val Loss: 1.21821E-06:  62%|█████████████        | 310/500 [01:55<01:13,  2.58epochs/s, epoch=311, loss=9.66504E-07]2025-11-03 01:43:37.183
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 311/500, Loss: 1.55948E-06, Val Loss: 1.21821E-06
Epoch 321/500, Loss: 2.38016E-06, Val Loss: 1.84904E-06:  64%|█████████████▍       | 320/500 [01:58<00:55,  3.24epochs/s, epoch=321, loss=8.94015E-06]2025-11-03 01:43:40.390
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 321/500, Loss: 2.38016E-06, Val Loss: 1.84904E-06
Epoch 331/500, Loss: 2.08734E-06, Val Loss: 2.54965E-06:  66%|█████████████▊       | 330/500 [02:03<01:10,  2.40epochs/s, epoch=331, loss=7.39505E-06]2025-11-03 01:43:44.522
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 331/500, Loss: 2.08734E-06, Val Loss: 2.54965E-06
Epoch 341/500, Loss: 1.93411E-06, Val Loss: 1.42584E-06:  68%|██████████████▎      | 340/500 [02:06<00:54,  2.92epochs/s, epoch=341, loss=1.00988E-06]2025-11-03 01:43:48.288
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 341/500, Loss: 1.93411E-06, Val Loss: 1.42584E-06
Epoch 351/500, Loss: 8.60778E-07, Val Loss: 9.74585E-07:  70%|██████████████▋      | 350/500 [02:10<01:06,  2.24epochs/s, epoch=351, loss=1.38937E-06]2025-11-03 01:43:52.370
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 351/500, Loss: 8.60778E-07, Val Loss: 9.74585E-07
Epoch 361/500, Loss: 8.15149E-07, Val Loss: 8.34922E-07:  72%|███████████████      | 360/500 [02:14<00:53,  2.62epochs/s, epoch=361, loss=9.72131E-07]2025-11-03 01:43:55.443
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 361/500, Loss: 8.15149E-07, Val Loss: 8.34922E-07
Epoch 371/500, Loss: 1.27564E-06, Val Loss: 8.75924E-07:  74%|███████████████▌     | 370/500 [02:18<00:54,  2.39epochs/s, epoch=371, loss=8.02762E-07]2025-11-03 01:43:59.507
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 371/500, Loss: 1.27564E-06, Val Loss: 8.75924E-07
Epoch 381/500, Loss: 8.08968E-07, Val Loss: 6.15586E-07:  76%|███████████████▉     | 380/500 [02:23<00:58,  2.07epochs/s, epoch=381, loss=1.50614E-06]2025-11-03 01:44:04.445
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 381/500, Loss: 8.08968E-07, Val Loss: 6.15586E-07
Epoch 391/500, Loss: 6.28568E-07, Val Loss: 8.40945E-07:  78%|████████████████▍    | 390/500 [02:27<00:46,  2.37epochs/s, epoch=391, loss=7.01196E-07]2025-11-03 01:44:08.717
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 391/500, Loss: 6.28568E-07, Val Loss: 8.40945E-07
Epoch 401/500, Loss: 7.39137E-07, Val Loss: 6.74655E-07:  80%|████████████████▊    | 400/500 [02:31<00:40,  2.48epochs/s, epoch=401, loss=9.20119E-07]2025-11-03 01:44:12.947
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 401/500, Loss: 7.39137E-07, Val Loss: 6.74655E-07
Epoch 411/500, Loss: 5.59700E-07, Val Loss: 6.08583E-07:  82%|█████████████████▏   | 410/500 [02:35<00:35,  2.53epochs/s, epoch=411, loss=4.71163E-07]2025-11-03 01:44:16.977
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 411/500, Loss: 5.59700E-07, Val Loss: 6.08583E-07
Epoch 421/500, Loss: 5.42947E-07, Val Loss: 4.19553E-07:  84%|█████████████████▋   | 420/500 [02:39<00:33,  2.37epochs/s, epoch=421, loss=7.23629E-07]2025-11-03 01:44:21.250
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 421/500, Loss: 5.42947E-07, Val Loss: 4.19553E-07
Epoch 431/500, Loss: 5.07290E-07, Val Loss: 8.21947E-07:  86%|██████████████████   | 430/500 [02:44<00:30,  2.31epochs/s, epoch=431, loss=8.69467E-07]2025-11-03 01:44:25.475
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 431/500, Loss: 5.07290E-07, Val Loss: 8.21947E-07
Epoch 441/500, Loss: 4.70043E-07, Val Loss: 6.36099E-07:  88%|██████████████████▍  | 440/500 [02:48<00:24,  2.47epochs/s, epoch=441, loss=4.67106E-07]2025-11-03 01:44:29.407
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 441/500, Loss: 4.70043E-07, Val Loss: 6.36099E-07
Epoch 451/500, Loss: 4.50860E-07, Val Loss: 4.55556E-07:  90%|██████████████████▉  | 450/500 [02:51<00:18,  2.68epochs/s, epoch=451, loss=4.78485E-07]2025-11-03 01:44:33.149
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 451/500, Loss: 4.50860E-07, Val Loss: 4.55556E-07
Epoch 461/500, Loss: 4.12607E-07, Val Loss: 5.57647E-07:  92%|███████████████████▎ | 460/500 [02:55<00:15,  2.59epochs/s, epoch=461, loss=4.35658E-07]2025-11-03 01:44:37.067
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 461/500, Loss: 4.12607E-07, Val Loss: 5.57647E-07
Epoch 471/500, Loss: 3.84479E-07, Val Loss: 4.26796E-07:  94%|███████████████████▋ | 470/500 [02:59<00:11,  2.54epochs/s, epoch=471, loss=4.29766E-07]2025-11-03 01:44:41.085
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 471/500, Loss: 3.84479E-07, Val Loss: 4.26796E-07
Epoch 481/500, Loss: 3.67495E-07, Val Loss: 4.56033E-07:  96%|████████████████████▏| 480/500 [03:04<00:08,  2.29epochs/s, epoch=481, loss=3.51827E-07]2025-11-03 01:44:45.425
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 481/500, Loss: 3.67495E-07, Val Loss: 4.56033E-07
Epoch 491/500, Loss: 3.54512E-07, Val Loss: 4.04510E-07:  98%|████████████████████▌| 490/500 [03:07<00:03,  2.67epochs/s, epoch=491, loss=4.88164E-07]2025-11-03 01:44:49.118
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 491/500, Loss: 3.54512E-07, Val Loss: 4.04510E-07
Epoch 500/500, Loss: 3.58511E-07, Val Loss: 4.70954E-07: 100%|████████████████████▉| 499/500 [03:11<00:00,  2.74epochs/s, epoch=500, loss=3.21778E-07]2025-11-03 01:44:52.463
| INFO     | gwkokab.utils.train:train_regressor:355 - Epoch 500/500, Loss: 3.58511E-07, Val Loss: 4.70954E-07
Epoch 500/500, Loss: 3.58511E-07, Val Loss: 4.70954E-07: 100%|█████████████████████| 500/500 [03:11<00:00,  2.62epochs/s, epoch=500, loss=3.21778E-07]
```

<img src="https://raw.githubusercontent.com/gwkokab/hello-gwkokab/refs/heads/main/training_mlp/model_checkpoint.hdf5_loss.png"/>
<img src="https://raw.githubusercontent.com/gwkokab/hello-gwkokab/refs/heads/main/training_mlp/model_checkpoint.hdf5_total_loss.png"/>

All the code and files used in this tutorial can be found in
[hello-gwkokab/training_mlp][REPRODUCIBILITY_LINK].

[REPRODUCIBILITY_LINK]: https://github.com/gwkokab/hello-gwkokab/tree/main/training_mlp
