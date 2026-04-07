-----Alzheimers Disease Images Preprocessing Pipeline----

Aim of this module is to convert the images from 3d into 2d images and greyscale.
Also convert them from .nii images to .png images 

------Final Metadata------------
A CSV file with combined information of all the csv present in various folders along with sliced image ids

------Uploading images in raw_data folder-----
 * Folder Structure:-
    raw_data
        ADNI1_ Complete 1.5Yr 3T
            ADNI
                002_S_0413
                    ... .nii image

            .csv file
    

Sample folders are available in the raw_data folder

----Steps to run:---------------
1) cd ADNI_PRE
2) pip install -r requirements.txt
3) python run_pipeline.py

------Expected output-------------
1) Image slices present in processed_data/slices
2) Final CSV files in processed_data/final_metadata


--------Expected Error/Challenges------------

1) If you have csv file opened somewhere it will not allow to run the program to run further

2) Too long file path
    Windows allows only 279 characters at max, but we have file names exceeding that limit
    
    Possible solutions:-
    1) Shorten the folder name ,ex- ADNI1_Complete 3Yr 3T ====> 3Yr 3T
