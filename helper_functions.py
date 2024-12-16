import os
import pandas as pd 
def find_mammogram(id,dataset_dir):
    mammogram_path = os.path.join(dataset_dir,id)
    mammogram_name = "Mammogram/mammo_slice_1.png"
    full_mammogram_path = os.path.join(mammogram_path,mammogram_name)
    return full_mammogram_path

def find_ultrasound(id,dataset_dir):
    ultrasound_path = os.path.join(dataset_dir,id)
    ultrasound_name = "Ultrasound/ultrasound_slice_no_top1.png"
    full_ultrasound_path = os.path.join(ultrasound_path,ultrasound_name)
    return full_ultrasound_path
def generate_dataset(data_dir):
        Full_Dataset = pd.read_csv("dataset.tsv",sep='\t')
        # Full_Dataset = Full_Dataset[Full_Dataset["UltraSound"]=="Yes"]
        Full_Dataset = Full_Dataset[Full_Dataset["Mammogram"]=="Yes"]
        Full_Dataset = Full_Dataset[Full_Dataset["UltraSound"]=="Yes"]
        Full_Dataset = Full_Dataset[Full_Dataset["Report"]=="Yes"]
        Full_Dataset = Full_Dataset[Full_Dataset["Classification"]!="CLIPINSERTION"]
        print(Full_Dataset["Classification"].value_counts())
        Full_Dataset["UltraSound_Path"] = Full_Dataset["ID"].apply(lambda x:find_ultrasound(x,data_dir))
        Full_Dataset["Mammogram_Path"] = Full_Dataset["ID"].apply(lambda x:find_mammogram(x,data_dir))
        return Full_Dataset
 




