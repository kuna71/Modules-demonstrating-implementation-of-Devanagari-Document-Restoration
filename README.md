# Modules-for-implementation-of-Devanagari-Document-Restoration
This repository includes various modules demonstrating the various processes in our novel Devanagari Document Restoration paper that is currently being reviewed. Citation for the same will be included here when paper is published.

To implement modules:
Word Segmentation - Set variables in word-segmentation.py (OUTPUT_PATH and INPUT_IMAGE_PATH) as per your requirements and run the script
Character Segmentation

Character Segmentation - Set variables in character-segmentation.py (OUTPUT_PATH same as OUTPUT_PATH of Word Segmentation) and run script after running word-segmentation.py

MLM and RegEx restoration-In file fill-mask-code.py, set variable files to list of strings containing files of damaged document text ("<blank>" denoting damaged characters). 
