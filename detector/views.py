from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views import View
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the model
model = tf.keras.models.load_model('file://model.h5')

class IndexView(View):
    template_name = 'detector/index.html'
    dic = {'Dacrymyces Palmatus': 0, 'Ganoderma Lucidum': 1, 'Gomphus Floccossus': 2, 'Lactarius_deliciosus': 3, 'Matsutake': 4, 'Russulla': 5, 'Shiitake ': 6, 'bitter bolete': 7, 'cantharellus cibarius': 8, 'lyophyllum aggregatum': 9}
    
    def get(self, request):
        return render(request, self.template_name, {'prediction': None})

    @csrf_exempt
    def post(self, request):
        # Get the uploaded image
        image = request.FILES['image']

        # Preprocess the image
        # Convert the image to PIL format
        image = Image.open(image)

        # Resize the image to the expected input size of the model
        image = image.resize((256, 256))

        # Convert the image to a NumPy array
        image = np.array(image)

        # Expand the dimensions of the image to match the input shape of the model
        image = np.expand_dims(image, axis=0)

        # Normalize the image
        image = image / 255.0

        # Make a prediction
        prediction = model.predict(image)

        # Get the predicted class label
        class_label = np.argmax(prediction[0])

        # Return the prediction to the client
        return JsonResponse({'prediction': self.dic[class_label]})
