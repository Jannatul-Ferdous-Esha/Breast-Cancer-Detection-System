from django.shortcuts import render
from django.shortcuts import HttpResponse
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import render
from .forms import CancerPredictionForm
import requests

from App_post.models import Project 
@login_required
def new(request):
    projects = Project.objects.all()
    return render(request,'App_post/new.html',context={'title': 'Home Page','projects':projects})


# COLAB_URL = "YOUR_COLAB_ENDPOINT_HERE"  # Replace with your actual Colab URL

# # @login_required
# # def input_page(request):
# #     if request.method == "POST":
# #         form = CancerPredictionForm(request.POST)
# #         if form.is_valid():
# #             input_data = form.cleaned_data
            
# #             # Send data to your Colab model
# #             response = requests.post(COLAB_URL, json=input_data)
# #             prediction = response.json().get("prediction", "Error")

# #             return render(request, "input_page.html", {"form": form, "prediction": prediction})
# #     else:
# #         form = CancerPredictionForm()

    
# #     return render(request, "App_post/input_page.html", {"form": form})

# # # def picture(request):
# # #     projects = Project.objects.all()
# # #     return render(request, 'App_post/new.html', {'projects':projects})

# # from django.shortcuts import render, redirect
# # from django.contrib.auth.decorators import login_required
# # import numpy as np
# # import joblib  # For loading the ML model
# # from .forms import CancerPredictionForm
# # from django.shortcuts import HttpResponse

# # # Load the pre-trained ML model
# # MODEL_PATH = "App_post/model.pkl"
# # model = joblib.load(MODEL_PATH)

# # from django.shortcuts import redirect

# # @login_required
# # def input_page(request):
# #     if request.method == "POST":
# #         form = CancerPredictionForm(request.POST)
# #         if form.is_valid():
# #             # Prepare the input data for prediction
# #             input_data = np.array([
# #                 form.cleaned_data.get(field) or 0 for field in form.fields
# #             ]).reshape(1, -1)

# #             # Make prediction (0 = benign, 1 = malignant)
# #             pred = model.predict(input_data)[0]
# #             prediction = "Malignant" if pred == 1 else "Benign"

# #             # Redirect to the results page with the prediction value
# #             return redirect('App_post:results_page', prediction=prediction)

# #     else:
# #         form = CancerPredictionForm()

# #     return render(request, "App_post/input_page.html", {"form": form})

# @login_required
# def results_page(request, prediction):
#     return render(request, "App_post/results_page.html", {"prediction": prediction})
# from django.shortcuts import render, redirect
# from django.contrib.auth.decorators import login_required
# import numpy as np
# import joblib
# from .forms import CancerPredictionForm
# import os
# import pickle
# import numpy as np
# from django.conf import settings
# from django.shortcuts import render
# from .forms import InputForm  # Create this form
# from django.http import HttpResponse
# # # Load once at module level
# # MODEL_PATH = os.path.join(settings.BASE_DIR, 'predictor', 'model')

# # with open(os.path.join(MODEL_PATH, 'scaler.pkl'), 'rb') as f:
# #     scaler = pickle.load(f)

# # with open(os.path.join(MODEL_PATH, 'model.pkl'), 'rb') as f:
# #     model = pickle.load(f)
# # MODEL_PATH = "App_post/model.pkl"
# # SCALER_PATH = "App_post/scaler.pkl"

# # model = joblib.load(MODEL_PATH)
# # scaler = joblib.load(SCALER_PATH)



# # def predict_view(request):
# #     prediction = None
# #     if request.method == 'POST':
# #         form = InputForm(request.POST)
# #         if form.is_valid():
# #             input_data = np.array([[form.cleaned_data[field] for field in form.fields]])
# #             scaled_data = scaler.transform(input_data)
# #             result = model.predict(scaled_data)[0]
# #             prediction = "Malignant" if result == 1 else "Benign"
# #     else:
# #         form = InputForm()

# #     return render(request, 'predictor/form.html', {'form': form, 'prediction': prediction})

# # @login_required
# # def input_page(request):
# #     if request.method == "POST":
# #         form = CancerPredictionForm(request.POST)
# #         if form.is_valid():
# #             # Prepare input
# #             input_data = np.array([
# #                 form.cleaned_data.get(field) or 0 for field in form.fields
# #             ]).reshape(1, -1)

# #             # 🔍 Debug print
# #             print("Input shape:", input_data.shape)  # Should be (1, 30)

# #             # Normalize input
# #             input_scaled = scaler.transform(input_data)

# #             # Predict
# #             prediction = model.predict(input_scaled)[0]
# #             result = "Malignant" if prediction == 1 else "Benign"

# #             return redirect('App_post:results_page', prediction=result)
# #     else:
# #         form = CancerPredictionForm()

# #     return render(request, "App_post/input_page.html", {"form": form})
# from django.shortcuts import render, redirect
# from django.contrib.auth.decorators import login_required
# import numpy as np
# import joblib
# from .forms import CancerPredictionForm

# # Load the new model and scaler
# MODEL_PATH = "App_post/new_model.pkl"  # Update with the correct path
# SCALER_PATH = "App_post/new_scaler.pkl"  # Update with the correct path

# model = joblib.load(MODEL_PATH)
# scaler = joblib.load(SCALER_PATH)

# @login_required
# def input_page(request):
#     if request.method == "POST":
#         form = CancerPredictionForm(request.POST)
#         if form.is_valid():
#             # Prepare input
#             input_data = np.array([
#                 form.cleaned_data.get(field) or 0 for field in form.fields
#             ]).reshape(1, -1)

#             # Normalize input using the new scaler
#             input_scaled = scaler.transform(input_data)

#             # Make prediction using the newly trained model
#             prediction = model.predict(input_scaled)[0]
#             result = "Malignant" if prediction == 1 else "Benign"

#             return redirect('App_post:results_page', prediction=result)
#     else:
#         form = CancerPredictionForm()

#     return render(request, "App_post/input_page.html", {"form": form})

# @login_required
# def results_page(request, prediction):
#     return render(request, "App_post/results_page.html", {"prediction": prediction})
import numpy as np
import joblib
from django.shortcuts import render, redirect
from .forms import CancerPredictionForm
from django.shortcuts import render
from django.shortcuts import HttpResponse
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import render
from .forms import CancerPredictionForm
import requests

# # Load the new model and scaler
# MODEL_PATH = "App_post/new_model.pkl"  # Update with the correct path
# SCALER_PATH = "App_post/new_scaler.pkl"  # Update with the correct path

# model = joblib.load(MODEL_PATH)
# scaler = joblib.load(SCALER_PATH)

# @login_required
# def input_page(request):
#     if request.method == "POST":
#         form = CancerPredictionForm(request.POST)
#         if form.is_valid():
#             # Prepare input data
#             input_data = np.array([
#                 form.cleaned_data.get(field) or 0 for field in form.fields
#             ]).reshape(1, -1)

#             # Debugging: Print the input data
#             print("Input Data:", input_data)

#             # Normalize input using the new scaler
#             input_scaled = scaler.transform(input_data)

#             # Debugging: Print the scaled input data
#             print("Scaled Input Data:", input_scaled)

#             # Make prediction using the newly trained model
#             prediction = model.predict(input_scaled)[0]

#             # Debugging: Print the prediction result
#             print("Prediction:", prediction)

#             result = "Malignant" if prediction == 1 else "Benign"

#             return redirect('App_post:results_page', prediction=result)
#     else:
#         form = CancerPredictionForm()

#     return render(request, "App_post/input_page.html", {"form": form})

# @login_required
# def results_page(request, prediction):
#     return render(request, "App_post/results_page.html", {"prediction": prediction})
import numpy as np
import joblib
from django.shortcuts import render, redirect
from .forms import CancerPredictionForm
from django.conf import settings

# # Load the new model and scaler
# MODEL_PATH = "App_post/final_model.pkl"
# SCALER_PATH = "App_post/final_scaler.pkl"

# model = joblib.load(MODEL_PATH)
# scaler = joblib.load(SCALER_PATH)

# @login_required
# def input_page(request):
#     if request.method == "POST":
#         form = CancerPredictionForm(request.POST)
#         if form.is_valid():
#             # Prepare input
#             input_data = np.array([
#                 form.cleaned_data.get(field) or 0 for field in form.fields
#             ]).reshape(1, -1)

#             # Normalize input using the new scaler
#             input_scaled = scaler.transform(input_data)

#             # Make prediction using the newly trained model
#             prediction = model.predict(input_scaled)[0]
#             result = "Malignant" if prediction == 1 else "Benign"

#             return redirect('App_post:results_page', prediction=result)
#     else:
#         form = CancerPredictionForm()

#     return render(request, "App_post/input_page.html", {"form": form})
import numpy as np
import joblib
from django.shortcuts import render, redirect
from .forms import CancerPredictionForm
from django.contrib.auth.decorators import login_required
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, BatchNormalization, Flatten, Dense

# Load model and scaler once
MODEL_PATH = "App_post/final_model.pkl"
SCALER_PATH = "App_post/final_scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# CNN feature extractor (same architecture as in training)
def create_cnn_feature_extractor(input_shape=(30, 1)):
    return Sequential([
        Conv1D(128, 3, activation='relu', input_shape=input_shape),
        BatchNormalization(), Dropout(0.5),
        Conv1D(64, 3, activation='relu'),
        BatchNormalization(), Dropout(0.5),
        Conv1D(32, 3, activation='relu'),
        BatchNormalization(), Dropout(0.5),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(), Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5)
    ])

feature_extractor = create_cnn_feature_extractor()

@login_required
def input_page(request):
    if request.method == "POST":
        form = CancerPredictionForm(request.POST)
        if form.is_valid():
            # Step 1: Extract raw form data
            input_data = np.array([form.cleaned_data[field] or 0 for field in form.fields]).reshape(1, -1)  # (1, 30)

            # Step 2: Scale the input
            input_scaled = scaler.transform(input_data)  # (1, 30)

            # Step 3: Reshape for CNN
            input_reshaped = input_scaled.reshape((1, 30, 1))  # (1, 30, 1)

            # Step 4: Extract CNN features
            extracted_features = feature_extractor.predict(input_reshaped)  # (1, 64)

            # Step 5: Predict
            prediction = model.predict(extracted_features)[0]
            result = "Malignant" if prediction == 1 else "Benign"

            return redirect('App_post:results_page', prediction=result)
    else:
        form = CancerPredictionForm()

    return render(request, "App_post/input_page.html", {"form": form})
@login_required
def results_page(request, prediction):
    return render(request, "App_post/results_page.html", {"prediction": prediction})
@login_required
def new(request):
    projects = Project.objects.all()
    return render(request,'App_post/new.html',context={'title': 'Home Page','projects':projects})