# MFI-FFP: Prediction of Lymph Node Metastasis in Colorectal Cancer using Intraoperative Fluorescence Multi-modal Imaging

## Abstract

The diagnosis of lymph node metastasis (LNM) is essential for colorectal cancer (CRC) treatment. The primary method of identifying LNM is to perform frozen sections and pathologic analysis, but this method is labor-intensive and time-consuming. Therefore, combining intraoperative fluorescence imaging with deep learning (DL) methods can improve efficiency. The majority of recent studies only analyze uni-modal fluorescence imaging, which provides less semantic information. In this work, we mainly established a multi-modal fluorescence imaging feature fusion prediction (MFI-FFP) model combining white light, fluorescence, and pseudo-color imaging of lymph nodes for LNM prediction. Firstly, based on the properties of various modal imaging, distinct feature extraction networks are chosen for feature extraction, which could significantly enhance the complementarity of various modal information. Secondly, the multi-modal feature fusion (MFF) module, which combines global and local information, is designed to fuse the extracted features. Furthermore, a novel loss function is formulated to tackle the issue of imbalanced samples, challenges in differentiating samples, and enhancing sample variety. Lastly, the experiments show that the model has a higher area under the receiver operating characteristic (ROC) curve (AUC), accuracy (ACC), and F1 score than the uni-modal and bi-modal models and has a better performance compared to other efficient image classification networks. Our study demonstrates that the MFI-FFP model has the potential to help doctors predict LNM and shows its promise in medical image analysis.
## Framework
![framework_revised](https://github.com/user-attachments/assets/ae1b49ab-4e49-452e-b858-09616f3c0aff)
## Multi-modal feature fusion (MFF) module
![MFF](https://github.com/user-attachments/assets/12741fa3-aa3a-437c-9fd2-fe05704e3e69)
## Results
![ROC_1](https://github.com/user-attachments/assets/83604643-b4d9-44a2-840a-cc2a9dbe88b5)
![ROC_2](https://github.com/user-attachments/assets/963fe305-f2c7-4206-abf6-f70bf4fb3ecf)
