# DenseGaussian
gaussian_renderer/__init__ 和 DenseGaussian/scene/__init__ 不额外实现功能，且同文件夹新实现类分开py文件
重构scene class，把可拆开的功能解耦成函数，scene本身作用变为仅挂载配置信息
diff-gaussian-rasterization的新功能实现，直接挂载到文件夹rasterization_kernels

# train_with_dataset.py v.s. train.py
On scene garden, we get:
train_with_dataset.py:
[ITER 30000] Evaluating train: L1 0.013847948797047139 PSNR 31.884959030151368 
train.py:
[ITER 30000] Evaluating train: L1 0.015808938443660738 PSNR 31.569805908203126 
