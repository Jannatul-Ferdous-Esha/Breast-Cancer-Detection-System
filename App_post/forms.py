# import os
# import pandas as pd
# from django import forms
# from django import forms
# from .models import PatientData

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# csv_path = os.path.join(BASE_DIR, "App_post", "data.csv")  # Updated path

# df = pd.read_csv(csv_path)
# input_columns = [col for col in df.columns if col.lower() != 'diagnosis']

# class CancerPredictionForm(forms.Form):
#     for col in input_columns:
#         locals()[col] = forms.FloatField(label=col.replace("_", " ").title(), required=True)


# class CancerPredictionForm(forms.ModelForm):
#     class Meta:
#         model = PatientData
#         fields = "__all__"  # All fields will be included, but optional

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         for field in self.fields.values():
#             field.required = False  # Make all fields optional
#             field.widget.attrs.update({'class': 'form-control'})

# import os
# import pandas as pd
# from django import forms

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# csv_path = os.path.join(BASE_DIR, "App_post", "data.csv")  # Updated path

# df = pd.read_csv(csv_path)
# input_columns = [col for col in df.columns if col.lower() != 'diagnosis']

# class CancerPredictionForm(forms.Form):
#     # Dynamically create form fields based on CSV columns
#     for col in input_columns:
#         locals()[col] = forms.FloatField(label=col.replace("_", " ").title(), required=True)
# import os
# import pandas as pd
# from django import forms

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# csv_path = os.path.join(BASE_DIR, "App_post", "data.csv")

# df = pd.read_csv(csv_path)
# input_columns = [col for col in df.columns if col.lower() != 'diagnosis']

# class CancerPredictionForm(forms.Form):
#     for col in input_columns:
#         locals()[col] = forms.FloatField(label=col.replace("_", " ").title(), required=True)
import os
import pandas as pd
from django import forms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "App_post", "data.csv")

# Load and clean only the correct features
df = pd.read_csv(csv_path)
df.drop(columns=['id', 'Unnamed: 32'], errors='ignore', inplace=True)
input_columns = [col for col in df.columns if col != 'diagnosis']  

class CancerPredictionForm(forms.Form):
    for col in input_columns:
        locals()[col] = forms.FloatField(
            label=col.replace("_", " ").title(),
            required=True
        )