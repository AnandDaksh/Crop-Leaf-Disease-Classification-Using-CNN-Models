**Crop Leaf Disease Classification using CNN Models**
I have  three Models of CNN which are ResNet-50, VGG-16 and CD-CNN for training the model. Each datatsets were trained on 3 models and in final i have selected one model from the three for Each dataset besed on their performance. SO you might think that why in some there are ResNet , CD-CNN and VGG-16 in others. 

Dataset:
Plant Village Dataset is used. In some Models i  have merged some class of database. That has been done to reduce underfitting.
Link for PlantVillage Dataset: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

Static:
It has the Css and Javascript files.

Templates:
As named it  has the templates for  the flask app. No need to change unless required by your side.

Uploads:
It stores the images uploaded to the flask app for disease detection.

File with .py  extension:
You just have to run all the files individually without editing the code till when not needed. (**Primary Task** after downloading the whole file and Adjustment with Datasets)

After running each .py file you will get another files with the  extension .h5 saved in the same directory with .py file. No need to displace them. Just come to app.py file and run it.
.h5 files are the weights of the models which will help your model to predict the disease. It is a HDF5 class file. Usually it stores weights and bias of the models

app.py:
It has the basic on run interface code for the whole model. 
