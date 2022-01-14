COVID-Rate
===

##COVID-Rate: An Automated Framework for Segmentation of COVID-19 Lesions from Chest CT

This research work introduces an open-access COVID-19 CT segmentation dataset containing 
433 CT images from 82 patients that have been annotated by an expert radiologist. 
Second, a Deep Neural Network (DNN)-based framework is proposed, referred to as the COVID-Rate, that
autonomously segments lung abnormalities associated with COVID-19 from chest CT images.
Besides, it introduces an unsupervised enhancement approach that can mitigate the gap between 
the training set and test set and improve model generalization on CT images obtained by a different scanner, 
addressing a critical challenge in applying AI in medical imaging. A synthetic data generation (augmentation) 
method is applied that generates synthetic pairs of CT images and infection masks by inserting the infectious 
regions from COVID-19 CT images into healthy CT images, which improves the model performance by introducing 
more variability to the training set.

Instead of taking the original chest CT images, COVID-Rate takes the segmented lung area as 
the input. A [U-Net based segmentation model](https://github.com/JoHof/lungmask) is used to extract lung region from each CT image.

The detailed COVID-Facts's structure and methodology is explained in detail 
at https://arxiv.org/pdf/2107.01527.

##Dataset: COVID-CT-Rate
COVID-CT-Rate is a dataset including 433 CT images from 82 COVID-19 patients 
and their associated infection masks. It can be used for training AI models 
to segment COVID-19 lesions from chest CT images. For the annotation process, 
first, infection masks were generated using a standard U-Net pre-trained on 
[a public COVID-19 dataset](https://arxiv.org/abs/2004.12537). Then, a thoracic radiologist with 20 years of experience in lung imaging carefully modified and verified the generated infection masks. All CT images have been obtained without contrast enhancement and saved in the Digital Imaging and Communications in Medicine (DICOM) format and the Hounsfield Unit.
CT images have been selected from diffident parts of the lung (top, middle, and bottom) with different infection rates to help the AI model better predict the infection regions on unseen CT images from the whole lung volume.

##Lung Segmentation
We utilize the lungmask module from <a href="https://github.com/JoHof/lungmask">here</a> 
to segmment lung region from CT images, which can be installed through the following line of code:
```
pip install git+https://github.com/JoHof/lungmask
```
To use the lungmask module, you need to have torch installed in your system. 
<a href = "https://pytorch.org">https://pytorch.org</a>

##Code
The available code contains:

* De-identification of DICOM files
* Synthetic data generation method
* Segmentation Network
* Certainty index for Unsupervise Enhancement approach 
