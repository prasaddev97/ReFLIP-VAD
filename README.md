# ReFLIP-VAD
This is the official Pytorch implementation of our paper:
**"ReFLIP-VAD: Towards Weakly Supervised Video Anomaly Detection via Vision-Language Model"** in **TCSVT 2024.**  

![framework](data/framework.png)

## Highlights
- A novel framework i.e. ReFLIP-VAD is developed that employs a prompt encoder to generate reparameterized learnable prompt templates instead of hand-crafted templates. These templates are contextually rich, enhancing interpretability and providing a deeper understanding of the specific semantics associated with anomalies. 

- The proposed approach comprises a classification block and a video-text alignment block. The former leverages visual features for binary classification while the latter leverages both textual and visual features for language-vision alignment. Consequently, this dual block based proposed approach is able to detect video anomalies at both coarse and fine-grained levels.

- A Glimpse-Emphasize network is developed that effectively captures both the global and local temporal dependencies across time. The MIL-Align mechanism is also developed to optimize visual-language alignment under weak supervision. 

- The effectiveness of ReFLIP-VAD is demonstrated on two large-scale benchmarks. ReFLIP-VAD achieves state-of-the-art performance, including 86.29\% AP on XD-Violence and 89.14\% AUC on UCF-Crime, surpassing existing state-of-the-art methods by a large margin.

The following files need to be adapted in order to run the code on your own machine:
- Change the file paths to the download datasets above in `list/xd_ReFLIP_rgb.csv` and `list/xd_ReFLIP_rgbtest.csv`. 
- Feel free to change the hyperparameters in `xd_option.py` and `ucf_option.py`

### Train and Test
After the setup, simply run the following command: 

Traing and infer for XD-Violence dataset
```
python xd_train.py
python xd_test.py
```
Traing and infer for UCF-Crime dataset
```
python ucf_train.py
python ucf_test.py
```

## References
We referenced the repos below for the code.
* [XDVioDet](https://github.com/Roc-Ng/XDVioDet)
* [DeepMIL](https://github.com/Roc-Ng/DeepMIL)
* [VadCLIP] (https://github.com/nwpu-zxr/VadCLIP)

```

