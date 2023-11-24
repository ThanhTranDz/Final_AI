import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model
import requests as rq


model = load_model('FruitModel2.h5')
labels = {
    0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot', 7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger', 14: 'grapes', 15: 'jalepeño', 16: 'kiwi', 17: 'lemon', 18: 'lettuce', 19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika',
    23: 'pear', 24: 'peas', 25: 'pineapple', 26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy bean', 30: 'spinach', 
    31: 'sweetcorn', 32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'
}
fruits = ['banana', 'apple', 'pear', 'grapes', 'orange', 'kiwi', 'watermelon', 'pomegranate', 'pineapple', 'mango']
vegetables = ['cucumber', 'carrot', 'capsicum', 'onion', 'potato', 'lemon', 'tomato', 'raddish', 'beetroot', 'cabbage', 'lettuce', 'spinach', 'soy bean', 'cauliflower', 'bell pepper', 'chilli pepper', 'turnip', 'corn', 'sweetcorn', 'sweet potato', 'paprika', 'jalepeño', 'ginger', 'garlic', 'peas', 'eggplant']

def fetch_calories(prediction):
    url = 'https://api.edamam.com/api/nutrition-data'
    option = {'app_id': '5db67232', 'app_key': '6ca927083659549ec48c3b8d758e8e6c', 'ingr': prediction}
    req = rq.get(url, option).json()
    calories = round(req['calories'], 2)
    return calories
def processed_img(img_path):
    img = load_img(img_path, target_size = (224,224,3))
    img = img_to_array(img)
    img = img/255
    img = np.expand_dims(img,[0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = ' '.join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    return res.capitalize()

def run():
    st.title("Fruits - Vegetable Classification")
    img_file = st.file_uploader("Choose an image",type=["jpg", "png","jpeg"])
    if img_file is not None:
        img = Image.open(img_file).resize((250,250))
        st.image(img)
        save_image_path = './upload_image' + img_file.name
        with open(save_image_path,"wb") as f:
            f.write(img_file.getbuffer())

        if img_file is not None:
            result = processed_img(save_image_path)
            if result in vegetables:
                st.info('Category : Vegetable')
            else:
                st.info('Category : Fruit')
            st.success('Prediction: ' + result )
            countResult = '1 ' + str(result)
            cal = fetch_calories(countResult)

            if cal:
                st.warning(f'1 {countResult} contains: {cal} kcal')
run()
