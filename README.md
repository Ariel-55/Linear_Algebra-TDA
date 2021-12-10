# Linear_Algebra-TDA


# This github repository consists of two files

# 1. The first file is plot-homologyMNIST.py-----
As its name says, this file is based on this post http://outlace.com/TDApart2.html, we used this code to plot the simplical complex using images from MNIST Dataset and to calculate the homology, fot calculating homology we used ripser: https://ripser.scikit-tda.org/en/latest/

All in alll, we read the images fromMNIST dataset, change pixels to cartisian coordiantes and apply the algorithm to plot the simplicail comples, the algorithm is explained with more details here: http://outlace.com/TDApart2.html

#To run the code we need to:
1. Install all the python libraries with the pip command
2. run the program using python  "python plot-homologyMNIST.py"


# 2. The second file is python MNIST-classifierAccuracy.py. 
For a better understanding of the code read https://giotto-ai.github.io/gtda-docs/0.3.1/notebooks/MNIST_classification.html. Also, VietorisRips can be used with the following link: https://giotto-ai.github.io/gtda-docs/0.3.0/notebooks/vietoris_rips_quickstart.html but the MNIST datset has to be changed to 3D point clouds. So in the code it is used CubicalPersistence which is an analogue of vietoris ripspersistence. 

#To run the code we need to:
1. Install all the python libraries with the pip command
2. run the program using python  "python MNIST-classifierAccuracy.py"






