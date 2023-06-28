# GlaucomaEyeDiseases-Detection

# Description

The Glaucoma Eye Diseases Prediction project utilizes the InspectionV2 architecture to develop a deep learning model capable of predicting the presence of glaucoma in eye images. Glaucoma is a serious eye condition that can lead to vision loss and blindness if not detected and treated early. This project aims to leverage the power of deep learning algorithms to assist in the early detection and diagnosis of glaucoma, enabling timely intervention and better patient outcomes.Glaucoma is referred as an eye disease that damage optic nerve and cause vision loss. Optica nerve carries information that we can see through eye to brain.Optic nerve head is called optic disc, it connects retina and optic nerve. Thecenter of optic disc is called optic cup. When the optic cup enlarges and occupymore area of optical disc then the cup to disc ratio (CDR) increases. When the cup to disc ratio is greater than normal range, the patient’s eye is suspected asGlaucomatous eye. Doctors need to perform many tests such as: Ophthalmic Test, Tonometry, Ophthalmoscopy, Perimetry, Pachymetry, Gonioscopy. After gettingresults from different test, doctor have to decide whether it is a Glaucomatous eye or not. Some of the methods used to detect Glaucoma include the Topcon image net method, optical coherence tomography, and the retinal nerve fibre layer analyser. However due to high cost and lack of research in this field . 


Most of the algorithm for automatic Glaucoma assesement using fundus images rely on handcrafted featires based on segmentation, which are affected by the performance of the chosen segmentaion method and the extracted features. Hence automatic Glaucoma detecting algorithm is developed Careful evolution is important to detect Glaucoma and there is a highchance of not getting accurate result due to lack of skill. This work proposes an Glaucoma Detection Using  Convolutional Neural Network  efficient method for detecting Glaucoma which will lessen time and costs at the same time in order to facilitate ophthalmologists and optometrists which is known for its ability to learn highly discriminative features from raw pixel intensities.

# Dataset

Dataset that relies in this project is from kaggle where there are three different datasets and i have used a all together. The Paper of the dataset are:

ORIGA: https://pubmed.ncbi.nlm.nih.gov/21095735/
REFUGE: https://ieee-dataport.org/documents/refuge-retinal-fundus-glaucoma-challenge
G1020: https://arxiv.org/abs/2006.09158

Total of 745 images where 486 images are Glaucoma negative image and 259 Glaucoma positive images.

# Installation

1. Clone the repository : git clone https://github.com/RajKumarBiswokarma/GlaucomaEyeDiseases-Detection.git
2. Navigate to the project directory: cd GlaucomaEyeDiseases
3. Install the requirements : pip install -r requirements.txt
4. Run the  interference code: gradio gradio_interface.py

   

