import numpy as np
import tensorflow as tf
import urllib.request as urlrequest
import scipy.misc as sm
# scipy 1.1.0 should be installed - pip3 install scipy==1.1.0

from django.http import HttpResponse
from django.shortcuts import render
from skimage.transform import resize
from urllib.request import urlopen

# Imporing model form
from .forms import PredictForm

import imageio
import cv2
from PIL import Image
import requests
from io import BytesIO

# final V
import os, re
import random, math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings 
import PIL

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model

warnings.filterwarnings(action='ignore')


def home(request):
    html = "<html><body>Tensorflow version is %s.</body></html>" % tf.__version__
    return HttpResponse(html)

def food(request):
	form = PredictForm(request.POST or None)
	result = None
	if form.is_valid():
		url = form.cleaned_data['url']
		
		response = requests.get(url)
		image = Image.open(BytesIO(response.content))
		image = np.asarray(image, dtype=np.uint8)
		image = cv2.resize(image, dsize=(180, 180)).astype(np.uint8).copy()

		result = predict_image(image)
		
		form.save()

	return render(request, 'predict/food.html', {'form':form, 'prediction':result})

def predict_image(img):
	classes = ['가지볶음', '간장게장', '갈비구이', '갈비탕', '갈치구이', '갈치조림', '감자조림', '감자채볶음', '감자탕', '갓김치', '건새우볶음', '경단', '계란국', '고등어구이', '고등어조림', '고사리나물', '고추장진미채볶음', '고추튀김', '곰탕_설렁탕', '곱창구이', '곱창전골', '과메기', '김치찌개', '깍두기', '깻잎장아찌', '꽁치조림', '꽈리고추무침', '꿀떡', '나박김치', '닭갈비', '닭계장', '더덕구이', '도라지무침', '도토리묵', '동태찌개', '된장찌개', '두부김치', '두부조림', '땅콩조림', '떡갈비', '떡국_만두국', '떡볶이', '라볶이', '만두', '매운탕', '메추리알장조림', '멸치볶음', '무국', '무생채', '물회', '미역국', '미역줄기볶음', '배추김치', '백김치', '보쌈', '부추김치', '북엇국', '불고기', '삼겹살', '삼계탕', '새우튀김', '소세지볶음', '송편', '수정과', '숙주나물', '순두부찌개', '시금치나물', '시래기국', '식혜', '애호박볶음', '양념게장', '양념치킨', '어묵볶음', '연근조림', '열무김치', '오이소박이', '오징어채볶음', '오징어튀김', '우엉조림', '육개장', '육회', '잡채', '장어구이', '장조림', '전복죽', '젓갈', '제육볶음', '조개구이', '조기구이', '주꾸미볶음', '총각김치', '추어탕', '코다리조림', '콩나물국', '콩나물무침', '콩자반', '파김치', '편육', '피자', '호박죽', '홍어무침', '황태구이', '회무침', '후라이드치킨', '훈제오리']

	num_classes = len(classes)

	data_augmentation = keras.Sequential(
		[
		layers.RandomFlip("horizontal", input_shape=(180,180,3)),
		layers.RandomRotation(0.1),
		layers.RandomZoom(0.1),
		]
	)

	model = Sequential([
		data_augmentation,
		layers.Rescaling(1./255),
		layers.Conv2D(16, 3, padding='same', activation='relu'),
		layers.MaxPooling2D(),
		layers.Conv2D(32, 3, padding='same', activation='relu'),
		layers.MaxPooling2D(),
		layers.Conv2D(64, 3, padding='same', activation='relu'),
		layers.MaxPooling2D(),
		layers.Dropout(0.2),
		layers.Flatten(),
		layers.Dense(128, activation='relu'),
		layers.Dense(num_classes)
	])

	model.compile(optimizer='adam',
					loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
					metrics=['accuracy'])

	model.load_weights("predict/tensormodel/model.h5")

	img_array = tf.keras.utils.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0) # Create a batch

	predictions = model.predict(img_array)
	score = tf.nn.softmax(predictions[0])

	return classes[np.argmax(predictions[0])]









