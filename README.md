# Joint Fully Convolutional and Graph Convolutional Networks for Weakly-Supervised Segmentation of Pathology Images

# Instructions
A trained checkpoint numbered 1000 is provided with 9 HER2 pathology images for use in inference.  
This checkpoint is trained with 226 HER2 pathology images from a private dataset  

## To run inference:  
`python3 finaledgegcncopy.py --inference-path full_path_to/inference --checkpoint xxxx`
for example, with provided images and state dict, run like:  
`python3 finaledgegcncopy.py --inference-path full_path_to/inference --checkpoint 1000`
  
## To run Train:  
`python3 finaledgegcncopy.py` 
  
## To resume Train:  
`python3 finaledgegcncopy.py --checkpoint xxxx`
  
## Flags and folders:  
  
`--train-path` or `./train_process_files`: a folder which the pipeline saves training visualization files to  
  
`--input-path` or `./input_data`: images used for inference or training. For our weakly supervised loss to work, the training images should be named as: AreaRatio_Uncertainty_*.png.   
For example, 0.4_0.05_*.png means the target region occupies (40+/-5)% of the image.  

`--inference-path` or `full_path_to/inference`: an argument that defines a folder for the pipeline to output inferenced mask to. 
Setting this argument will switch on inference mode. This argument must be used with `--checkpoint`.  
