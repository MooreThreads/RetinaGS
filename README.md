# Organizational Reformations

1. gaussian_renderer/__init__.py and scene/__init__.py have no additional implementation functionality. New implementation class should be a separate py file. 

2. Reformat the scene class, decouple functions that can be broken down, the role of the scene itself is changed to just mount configuration information.

3. The implementation of the new feature of diff-gaussian-rasterization, directly mounted to the rasterization_kernels folder.

# Precision Validation

Shold be done after new implementation of diff-gaussian-rasterization and py file of train.

On scene garden, no eval, default hyper-parameters (only -m + -s):

1. train.py: [ITER 30000] Evaluating train: L1 0.015808938443660738 PSNR 31.569805908203126 

2. train_with_dataset.py: [ITER 30000] Evaluating train: L1 0.013847948797047139 PSNR 31.884959030151368 