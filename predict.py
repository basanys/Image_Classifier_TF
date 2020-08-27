import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
from utils import get_class_names, load_model, process_image
import argparse
import json


def predict(image_path, model_path, top_k, all_class_names):
    top_k = int(top_k)
    print(top_k, type(top_k))
    model = load_model(model_path)
    
    image = Image.open(image_path)
    image = np.asarray(image)
    
    processed_image = process_image(image)
    
    prob_preds = model.predict(np.expand_dims(processed_image, axis=0))
    prob_preds = prob_preds[0].tolist()
    
    values, indices = tf.math.top_k(prob_preds, k=top_k)
    
    probs_topk = values.numpy().tolist()
    classes_topk = indices.numpy().tolist()
    print('top k probs: ', probs_topk)
    print('top k classes: ', classes_topk)
    class_labels = [all_class_names[str(i)] for i in classes_topk]
    print('top k classes labels: ', class_labels)
    class_probs_dict = dict(zip(class_labels, probs_topk))
    print('\n Top k classes with their probabilities:\n', class_probs_dict)
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description= 'Description of Parser')
    parser.add_argument('image_path', help='Path of Image File', default='')
    parser.add_argument('model_path', help='Saved Path of Trained Model', default='')
    parser.add_argument('--top_k', help='Top k Predictions', required=False, default=3)
    parser.add_argument('--category_names', help='Class Map JSON File', required=False, default='label_map.json')
    args = parser.parse_args()
    
    all_class_names = get_class_names(args.category_names)
    
    predict(args.image_path, args.model_path, args.top_k, all_class_names)
    