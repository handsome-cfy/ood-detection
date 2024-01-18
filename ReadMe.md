#OOD 
requirements.txt
## Dataset
使用了三种不同的Dataset设置方式：  
- 对于DomainNet，将所下载的各个domain的数据放在data/domainnet/{domain}之下，以real为例：` data/domainnet/real/real/{the picture}`是一个正确的放置方式。（可能需要修改root_path参数，dataset的相关参数都写了相应的注释）
- 对于CiFar10_SVHN
- 对于CIFAR100

## Model
model放在my_model.py之中，`continual\convit.py`是与locality注意力相关的代码部分，取自LCA文章的github仓库。  
具体的模型只有两个改进部分，一个是Semantic Block，一个是Domain Block：
- Semantic Block已经跑通了
- Domain Block部分由于Domain token的加入，使得他们所提出的注意力机制跑不通（报错了，我一时间还没解决），具体的改进方法我认为在于`使用convit的方式重新写整个模型，更合理的设计Domain Block的位置，要么产生“即插即用”的效果，要么合理设置该block的参数，使得其能够跑通 `