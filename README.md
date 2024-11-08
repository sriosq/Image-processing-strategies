# Image-processing-strategies
Repository with usefull code for processing CT and MRI data

Inside every folder you will find examples of (I hope) interesting image processing questions.  </br>
This is a learning repository and I am open to chat about improvements for the code and new ideas/perspectives to the solution of the example problems.

**MR sim** </BR>
The objective of this folder is to show the workflow to simulte MR signal acquisition based on a reduced interpretion of signal equation using only T2 star and proton density values. It has integrated 2 git-submodules: a converter to create volumes with MR property values and a susceptibility to fieldmap calculator based on susceptibility volume created with the first submodule. The codelines are meant to show what needs to be done. But it is not recommended to run on a cloned jupyter notebook due to memory limitations. The data used for this simulations is from: Selfridge, A. R., Spencer, B., Shiyam Sundar, L. K., Abdelhafez, Y., Nardo, L., Cherry, S. R., & Badawi, R. D. (2023). Low-Dose CT Images of Healthy Cohort (Healthy-Total-Body-CTs) (Version 1) [Subject1]. The Cancer Imaging Archive. https://doi.org/10.7937/NC7Z-4F76.

**Compare FM** </br>
Inside folder you'll find usefull code to simulate B0 fields from segmentations that have susceptibility ($\chi$) values. It goes through the conversion of a susceptibility distribution, conversion to Hz, demodulating and metric extraction through a label spinal cord mask. </br>
Made for ISMRM 2025 abstract - *work in progress*

**QSM testing**
To go from segmentations to realistic simulation of MR images through calculation of sequences like recalled-spoiled gradient echo. </br>
Used understanding the role of different post processing algorithms and their effectiveness to be applied outside of, typically, brain ROI. </br>
Made for MSc. Project - *work in progress*

**Label creation** </br>
Inside this folder you will find the strategy for adding labels to segmented images. In the example data set we use a data set from Gatidis et al [1], an open source whole-body. A powerful tool for segmenting CT images, Total Segmentator by  Wasserthal et al. [2], is used to acquire a whole body total segmentation.  </br>
In this exploration example I will show how to create a new label with steps and thought process that lets you edit segmented images for interesting tests and research.

**References**
[1] Gatidis, S., Hepp, T., Früh, M. et al. A whole-body FDG-PET/CT Dataset with manually annotated Tumor Lesions. Sci Data 9, 601 (2022). https://doi.org/10.1038/s41597-022-01718-3 || Attribution to the data set link : https://creativecommons.org/licenses/by/4.0/ </br>
[2] Wasserthal, J., Breit, H.-C., Meyer, M.T., Pradella, M., Hinck, D., Sauter, A.W., Heye, T., Boll, D., Cyriac, J., Yang, S., Bach, M., Segeroth, M., 2023. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiology: Artificial Intelligence. https://doi.org/10.1148/ryai.230024 <bre>
