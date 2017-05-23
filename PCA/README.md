# Face Recognition using PCA

Here I present an implementation of the classical face recognition algorithm using Principal Component Analysis (PCA) and 1-Nearest Neighbour approach for classification. The database used for the task is the "Database of Faces" (formerly known as the ORL database).

### Getting Started

The training and testing datasets are provided in the "Train" and "Test" folders. FaceRecog.m implements the feature extraction process, does the classification using 1-NN approach and evaluates classification accuracy over the test database.

### About the Dataset

The database is freely available for download from the link below
```
http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
```

There are 10 different images of each of 40 distinct subjects. For some subjects, the images were taken at different times, varying the lighting, facial expressions (open / closed eyes, smiling / not smiling) and facial details (glasses / no glasses). All the images were taken against a dark homogeneous background with the subjects in an upright, frontal position (with tolerance for some side movement). A preview image of the Database of Faces is available.

The files are in PGM format, and can conveniently be viewed on UNIX (TM) systems using the 'xv' program. The size of each image is 92x112 pixels, with 256 grey levels per pixel. The images are organised in 40 directories (one for each subject), which have names of the form sX, where X indicates the subject number (between 1 and 40). In each of these directories, there are ten different images of that subject, which have names of the form Y.pgm, where Y is the image number for that subject (between 1 and 10).

Since earlier versions of MATLAB did not support .pgm format, I have also uploaded the converted .jpg version of the images.

### Methodology and Results

For each of the 40 subjects, 5 images out of 10 were randomly chosen to form the training dataset. The remaining 200 images were used for testing purpose. An accuracy of **90%** is obtained when the number of chosen eigenvectors is 10 and the distribution of test and train datasets is what we see uploaded.

### Ideas for experimentation

One might try different train-test proportions to see how robust the algorithm is. Another idea might be to vary the number of eigenvectors and see the variation in performance. You can also see how to create eigenfaces out of the eigenvectors and use them for reconstruction of the face images. 

## Author

* **[Pranav Sodhani](https://sites.google.com/site/sodhanipranav)**



