# Semi-Supervised Learning for Image Classification using Compact Networks in the Medical Context
The development of mobile applications that embed deep convolutional neural models
has the potential to revolutionise healthcare. However, most deep learning models
require computational resources that are not available in smartphones or edge devices;
an issue that can be faced by means of compact models. The problem with such models is that
they are, at least usually, less accurate than bigger models.
In this work, we address this limitation of compact networks with the application of
semi-supervised learning techniques, which take advantage of unlabelled data.
Using this combination, we have shown that it is possible to construct compact
models as accurate as bigger models in two widely employed datasets for [melanoma
classification](https://www.kaggle.com/c/siim-isic-melanoma-classification)
and [diabetic retinopathy detection](https://journals.sagepub.com/doi/10.1177/193229680900300315).
Finally, to facilitate the application of the methods studied in this work, we have developed a
library that simplifies the construction of compact models using semi-supervised learning methods.