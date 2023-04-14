# Modules-for-implementation-of-Devanagari-Document-Restoration
This repository includes various modules demonstrating the various processes in our novel Devanagari Document Restoration paper that is currently being reviewed. Citation for the same will be included here when paper is published.

## To implement modules:

Word Segmentation - Set variables in word-segmentation.py (OUTPUT_PATH and INPUT_IMAGE_PATH) as per your requirements and run the script
Character Segmentation

Character Segmentation - Set variables in character-segmentation.py (OUTPUT_PATH same as OUTPUT_PATH of Word Segmentation) and run script after running word-segmentation.py

MLM and RegEx restoration-In file fill-mask-code.py, set variable files to list of strings containing files of damaged document text ("<blank>" denoting damaged characters). 

OCR - In file OCR1.py, set the test_image in the last cell with the image path of your input image and run the script 

## Methodology Overview:
  
The document image is first preprocessed to prepare for word and subsequent character segmentation. This consists of gray scaling, thresholding and image dilation. The preprocessed document is segmented into words using contouring. Each word is then further segmented into characters after Shirorekha erasure and contouring. 

Each character is passed to an OCR model for classification. Any character which has an OCR confidence level below a defined threshold is labelled with “<blank>” token. This prepares our data for the core of the restoration model: the masked language modelling and regex pattern matching. 

Each token with a “<blank>” symbol (damaged word) is masked for the Masked Language Modelling, which generates k predictions for each masked token. We then iterate through each prediction and apply pattern matching to select the prediction which matched the remaining visible characters and restore the document with matched word. 
