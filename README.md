# Final Results

The final results that is the average image and the average gradcam image for each cohort in MORPH is shown here: [val/test txt](https://github.com/Czajka-Teaching/semester-project-abhatta1234/tree/main/finalresults.md)


# Preprocessing

The txtfiles that are required to for train,test and validation can be created as such:

~~~bash
python3 utils/folder_to_list.py -s path_to_metadata_txt -d save_destination -l 0 -n name_to_save --path sourcepath -opt 0 
~~~

There are several options available, such as a folder with all images of same class, multiple directories with same class or different. Refer to utils/folder_to_list.py to see several options available as fit. The choice of -l and -opt are dependent on case basis.

All the metadata for males and females are uploaded here: [metadata txt](https://drive.google.com/drive/folders/1zp7BsRb7M42PRj6EoyHbqZIeQAlV8oc4)

All the txt files needed for train/test/val and gradcam generation should be in this format below:
~~~bash
imgpath label
imgpath label
imgpath label
...
...
~~~

# Training

~~~bash
python3 codes/final_main_2.0.py -n num_resnet_layer -t path_to_train_txt -v path_to_val_txt -d path_to_save_trained_models
~~~

All the txt files use to train are here: [train txt](https://drive.google.com/drive/folders/1zp7BsRb7M42PRj6EoyHbqZIeQAlV8oc4) and validation/test txt are here: [val/test txt](https://github.com/Czajka-Teaching/semester-project-abhatta1234/tree/main/txtfiles)

# Testing

~~~bash
python codes/final_validate.py -t path_to_val_txt -p path_to_trained_model
~~~


# Gradcam generation

~~~bash
python3 codes/final_cam_run.py --use-cuda -imgpath path_to_img_files --method gradcam \ 
-g group_label -path path_to_trained_resnet_model -d img_save_destination
~~~
