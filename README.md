# Obscured-Wild-Fire-Detection
Welcome to the GitHub repository for our work on wildfire management using Deep Learning! 
In this project, we present a novel approach to detect both visible and obscured fires by harnessing the power of Deep Learning, 
specifically focusing on joint spatial and temporal analysis of RGB videos. Unlike conventional methods that independently analyze 
images and individual video frames, our proposed frame-wise transformer architecture incorporates an attention mechanism between consecutive frames while preserving spatial information within each frame. This innovative approach allows us to capture temporal patterns of smoke motions, indicating obscured fire flames. We optimize temporal pattern lengths under various spatial resolutions to achieve superior performance while minimizing computational costs. To validate the applicability of our method in aerial image processing, we applied it to a curated version of the [FLAME2](http://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset) dataset with dual-mode RGB/IR videos. Notably, our model is trained solely on RGB videos to cater to low-cost commercial drones equipped with regular cameras. 

<!-- The proposed method demonstrates remarkable results with a Segmentation Foreground Dice score of 92.61% and Object Detection Precision and Recall rates of 93.21% and 91.73%, respectively. The repository includes the implementation of our frame-wise transformer architecture, dataset preparation scripts, and model evaluation tools. Join us in revolutionizing wildfire detection with cutting-edge Deep Learning techniques! -->

## Usage

