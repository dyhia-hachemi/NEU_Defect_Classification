import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Chargement du modèle
model = load_model('best_tl_model.keras')

class_names = ['crazing', 'inclusion', 'patches',
               'pitted_surface', 'rolled-in_scale', 'scratches']

def predict_defect(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    probas = model.predict(img_array, verbose=0)[0]
    return {class_names[i]: float(probas[i]) for i in range(len(class_names))}

demo = gr.Interface(
    fn=predict_defect,
    inputs=gr.Image(type="pil", label="Image de défaut (acier)"),
    outputs=gr.Label(num_top_classes=6, label="Probabilités par classe"),
    title="Détecteur de Défauts de Surface",
    description="MobileNetV2 Transfer Learning"
)

demo.launch(share=True)