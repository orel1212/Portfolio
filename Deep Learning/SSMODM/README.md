# Detecting Attacks on Semantic Segmentation Models (SSM) via Object Detection Model (ODM)
## Attacks
### PGD (projected gradient descent)
### Gradient Weighted Average ( more powerful than pgd)

### alpha-> foreground transparency hyperparameter

## Dataset
Cityscapes dataset
<a> https://www.cityscapes-dataset.com </a>
## Models
### SSM
SCNN <a>https://github.com/Tramac/Fast-SCNN-pytorch</a>
### ODM
Yolov5 <a>https://github.com/ultralytics/yolov5</a>

## Files
### attack.ipynb
running notebook of the code of the attack (pgd/weighted)
### defense.ipynb
running notebook of the code of the defense (odm)

## Method - ODM
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{equation}&space;ODM_{SSM}&space;=&space;ODM(\alpha&space;\cdot&space;IMG_{input}&space;&plus;&space;(1-\alpha)&space;\cdot&space;SSM_{output})&space;\end{equation}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\begin{equation}&space;ODM_{SSM}&space;=&space;ODM(\alpha&space;\cdot&space;IMG_{input}&space;&plus;&space;(1-\alpha)&space;\cdot&space;SSM_{output})&space;\end{equation}" title="\begin{equation} ODM_{SSM} = ODM(\alpha \cdot IMG_{input} + (1-\alpha) \cdot SSM_{output}) \end{equation}" /></a>

![alt tag](https://github.com/orel1212/MyWorks/blob/main/Deep%20Learning/SSMODM/%E2%80%8F%E2%80%8Falpha.PNG)
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Deep%20Learning/SSMODM/%E2%80%8F%E2%80%8Fod_on_image.PNG)
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Deep%20Learning/SSMODM/%E2%80%8F%E2%80%8Fexamples.PNG)
