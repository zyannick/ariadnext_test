# Biometrie



## Prerequisites

Mainly:

- Anaconda
- Pytorch
- OpenCV

To run the program, you should install all the libraries in the file ariad.yml as follows:

```bash
conda env create -f ariad.yml
conda activate ariad
# you can run the code
```

## How to run

When you launch the code, the dataset is partitionned between test and train through the file 'datacreator.py'. 
Your files must be in the directory 'Datasets'.

### Train

By default, you'll run Resnet18 model with 20 epochs

```bash
 python3 train.py  -backbone='resnet18'  -init_lr=0.0001 -step_size=5 -nums_epochs=20 -phase='train'
```



### Testing

To test your model, you can specify the backbone

```bash
 python3 train.py  -backbone='resnet18'  phase='test'
```

### Demo



```bash
python3 train.py  -backbone='resnet'  -phase='demo'  -video='the_office.mp4' -list_imgs='mickael_scott.jpg,dwight.jpg'
```

