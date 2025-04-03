import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import os
import shutil
import random
import itertools
from PIL import Image, ImageTk
import numpy as np

mobile = tf.keras.applications.MobileNetV3Large()
from tensorflow.keras.models import load_model
new_model = load_model('models/medicinal_herb_model.keras')

def predict():
    import numpy as np
    file_path = filedialog.askopenfilename()   
    if file_path:
        def prepare_image(file_path,np):
            import numpy as np
            img = image.load_img(file_path,target_size=(224,224))
            img_array = image.img_to_array(img)
            img_array_expanded_dims = np.expand_dims(img_array,axis=0)
            return tf.keras.applications.mobilenet_v3.preprocess_input(img_array_expanded_dims)
        preprocessed_image=prepare_image(file_path,np)
        predictions=new_model.predict(preprocessed_image)
        cm_plot_labels=['Mustard', 'Jackfruit', 'Cherry', 'Jamun', 'Jasmine', 'Karanda', 'Lemon', 'Mango', 'Mexican_Mint', 'Mint']
        import numpy as np
        
        top3_indices = np.argsort(predictions, axis=-1)[:, -3:]
        reversed_top3_indices = np.flip(top3_indices, axis=-1)

        # Initialize an array to store class names
        top3_class_names = np.empty_like(reversed_top3_indices, dtype=object)
        top3_probabilities = np.empty_like(reversed_top3_indices, dtype=float)

        # Iterate through each row of reversed_top3_indices
        for i, row in enumerate(reversed_top3_indices):
            for j, idx in enumerate(row):
                # Assign class names
                top3_class_names[i, j] = cm_plot_labels[idx]
                # Assign probabilities
                top3_probabilities[i, j] = predictions[i, idx]

        if(top3_probabilities[0][0]<0.95):
            #print("Not a Medicinal Plant")
            result_label.config(text=f'Not in our categories ')
            #result_label.config(text=f'Not in our categories {top3_probabilities[0][0]} ')
            #result_label.config(text=f'Predicted Species: {top3_class_names[0][0]}')
        else:
           # print(top3_class_names[0][0])
            result_label.config(text=f'Predicted Species: {top3_class_names[0][0]}')
        
        # top3_indices = np.argsort(predictions, axis =- 1) [:, -3:]
        # reversed_top3_indices = np.flip(top3_indices, axis =- 1)
        # top3_class_names = np.empty_like(reversed_top3_indices, dtype=object)
        # for i, row in enumerate(reversed_top3_indices):
        #     for j, idx in enumerate(row):
        #         top3_class_names[i, j] = cm_plot_labels[idx]
       # result_label.config(text=f'Predicted Species: {top3_class_names[0][0]}')
        for widget in root.winfo_children():
            if isinstance(widget, tk.Label) and widget != title_label and widget != result_label:
                widget.destroy()
        img = Image.open(file_path)
        resampling_method = Image.BICUBIC if hasattr(Image, 'BICUBIC') else Image.BILINEAR 
        img = img.resize((300, 300), resampling_method)

        img = ImageTk.PhotoImage(img)
        panel = tk.Label(root, image=img,bg="white")
        panel.configure(image=img)
        panel.image = img
        panel.pack(side="top", fill="both", expand=True, padx=20, pady=15)

#GUI
root = tk.Tk()
root.geometry("600x600")
root.title('Medicinal Herb Predictor')
root.configure(bg="white")
upload_button = tk.Button(root, text='Upload Image',bg="green",fg="white",width=12,height=2,bd=0, command=predict)
title_label = tk.Label(root,text='Medicinal Herb Predictor',font=("Helvetica", 28,"bold"),bg="white")
title_label.pack(side='top',expand=True, padx=20, pady=10)
upload_button.pack(side="top", expand=True,)
result_label = tk.Label(root, text='',bg="white",font=("Helvetica", 18,"bold"))
result_label.pack(side="top", expand=True, padx=20, pady=10)


root.mainloop()

