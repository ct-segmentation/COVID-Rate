COVID-Rate
===

COVID-Rate: An Automated Framework for Segmentation of COVID-19 Lesions from Chest CT Images
---

This research first introduces an open-access COVID-19 CT segmentation dataset containing 433 CT images from 82 patients annotated by an expert radiologist. Second, a deep learning framework referred to as the COVID-Rate is proposed that autonomously segments lung abnormalities associated with COVID-19 from chest CT images. It also introduces an unsupervised enhancement approach that can mitigate the gap between the training set and test set and improve model generalization on CT images obtained by a different scanner, addressing a critical challenge in applying AI in medical imaging. A synthetic data generation (augmentation) method generates synthetic pairs of CT images and infection masks by inserting the infectious regions from COVID-19 CT images into healthy CT images, which improves the model performance by introducing more variability to the training set.

Instead of taking the original chest CT images, COVID-Rate takes the segmented lung area as the input. A U-Net-based segmentation model is used to extract the lung region from each CT image.

The detailed description of COVID-Rate's structure and implementation is explained in https://arxiv.org/pdf/2107.01527.

Dataset: COVID-CT-Rate
---
[COVID-CT-Rate](https://figshare.com/articles/dataset/COVID-CT-Rate_zip/18339416) is a dataset including 433 CT images from 82 COVID-19 patients and their associated infection masks. It can be used for training AI models to segment COVID-19 lesions from chest CT images. For the annotation process, first, infection masks were generated using a standard U-Net pre-trained on a public [COVID-19 segmentation datase](https://arxiv.org/abs/2004.12537). Then, a thoracic radiologist with 20 years of experience in lung imaging carefully modified and verified the generated infection masks. All CT images have been obtained without contrast enhancement and saved in the Digital Imaging and Communications in Medicine (DICOM) format and the Hounsfield Unit. CT images have been selected from diffident parts of the lung (top, middle, and bottom) with different infection rates to help the AI model identify the infection regions on unseen CT images from the whole lung volume.

Samples of CT images and their infection masks are shown in the following image.
<img src="https://github.com/ct-segmentation/COVID-Rate/blob/main/Figures/CT_Masks_Sample.PNG" width="750" height="300"/>

Lung Segmentation
---
We utilize the lungmask module from <a href="https://github.com/JoHof/lungmask">here</a> to segmment lung region from CT images, which can be installed through the following line of code:
```
pip install git+https://github.com/JoHof/lungmask
```
To use the lungmask module, you need to have torch installed in your system. <a href = "https://pytorch.org">https://pytorch.org</a>

Code
---
The available code contains:

* De-identification of DICOM files
* Synthetic data generation method
* Segmentation Network
* Certainty index for Unsupervised Enhancement approach

## Citation
If you found this dataset and the related paper helpful for your research, please consider citing:

```
Enshaei, N., Oikonomou, A., Rafiee, M.J., Afshar, P., Heidarian, S., Mohammadi, A., Plataniotis, K.N. and Naderkhani, F., 2021. COVID-Rate: 
An Automated Framework for Segmentation of COVID-19 Lesions from Chest CT Scans. arXiv preprint arXiv:2107.01527.
```
