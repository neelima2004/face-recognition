from django.shortcuts import get_object_or_404, render, redirect
from django.core.files.storage import FileSystemStorage
import os
import uuid
import cv2
import numpy as np
import tkinter as tk
import face_recognition
import pickle
from .models import FaceDetect
# Create your views here.

face_encodings = []
labels = []
label_dict = {}
label_idx = 0
output_dir = "output_faces"
os.makedirs(output_dir, exist_ok=True)

model_path = "output/cowrec_knn_model.xml"
label_dict_path = "output/label_dict.npy"
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

if not (os.path.exists(model_path) and os.path.exists(label_dict_path)):
    raise FileNotFoundError("Model or label dictionary not found. Please train first.")

knn = cv2.ml.KNearest_create()
knn = knn.load(model_path)
with open("output_faces/label_dict.pkl", "rb") as f:
        label_dict = pickle.load(f)

print(label_dict)

orb = cv2.ORB_create()
nose_cascade = cv2.CascadeClassifier(cascade_path)


def get_face_encoding(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_image)
    return encodings[0] if encodings else None

import hashlib

def train_on_image(file_path, name, location):
    global face_encodings, labels, label_dict

    label_dict_path = os.path.join('output_faces', 'label_dict.pkl')
    if os.path.exists(label_dict_path):
        with open(label_dict_path, "rb") as f:
            saved_dict = pickle.load(f)
        label_dict.update(saved_dict)

    label_idx = max(label_dict.keys(), default=-1) + 1

    img = cv2.imread(file_path)
    if img is None:
        return None, {}, "Could not read image"

    face_encoding = get_face_encoding(img)
    if face_encoding is None:
        return None, {}, "No face detected in the image"

    face_encodings.append(face_encoding)

    with open(file_path, 'rb') as f:
        image_bytes = f.read()
    unique_code = hashlib.md5(image_bytes).hexdigest()[:8]
    
    for i in label_dict:
        if label_dict[i]['unique_id'] == unique_code:
            return None,{},"Already exists"

    label_dict[label_idx] = {
        'unique_id': unique_code,
        'name': name,
        'location': location
    }

    labels.append(label_idx)

    relative_image_path = os.path.relpath(file_path, start='media')
    FaceDetect.objects.create(
        unique_id=unique_code,
        name=name,
        location=location,
        upload=relative_image_path
    )

    return unique_code, label_dict, "Image Trained Successfully."


def finalize_training():
    global face_encodings, labels, label_dict, label_idx
    if not face_encodings:
        return "No training data to save."

    encodings_array = np.array(face_encodings)
    labels_array = np.array(labels)
    
    print('Final Labels Dict - ', label_dict)
    np.save('output_faces/face_encodings.npy', np.array(encodings_array, dtype=object))
    np.save(os.path.join(output_dir, "face_labels.npy"), labels_array)
    with open(os.path.join(output_dir, "label_dict.pkl"), "wb") as f:
        pickle.dump(label_dict, f)

    return "Training data saved successfully."


def predict_id(image_path):
    known_face_encodings = np.load("output_faces/face_encodings.npy", allow_pickle=True)
    known_face_encodings = np.vstack(known_face_encodings).astype('float64')  # FIXED

    known_face_labels = np.load("output_faces/face_labels.npy", allow_pickle=True)

    with open("output_faces/label_dict.pkl", "rb") as f:
        label_dict = pickle.load(f)

    img = cv2.imread(image_path)
    if img is None:
        return None, "Could not load image"

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    if not face_encodings:
        return None, "No face detected"

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        if len(face_distances) > 0 and matches:
            best_match_index = np.argmin(face_distances)
            label = known_face_labels[best_match_index]
            predicted_details = label_dict.get(label, "Unknown")
            return predicted_details, "Prediction successful"

    return None, "Face not recognized"

def home(request):
    return render(request, 'index.html')

# def home(request):
#     global label_dict

#     # Paths to saved files
#     encodings_path = 'output_faces/face_encodings.npy'
#     labels_path = 'output_faces/face_labels.npy'
#     label_dict_path = 'output_faces/label_dict.pkl'

#     # 1. Clear in-memory label_dict
#     label_dict = {}

#     # 2. Overwrite saved files with empty arrays/dict
#     np.save(encodings_path, np.empty((0, 128)))  # 128 is face_encoding size
#     np.save(labels_path, np.empty((0,), dtype=int))
#     with open(label_dict_path, 'wb') as f:
#         pickle.dump(label_dict, f)

#     # 3. Delete all FaceDetect entries from DB
#     FaceDetect.objects.all().delete()

#     return render(request, 'index.html')

def training(request):
    objs = FaceDetect.objects.all()
    return render(request, 'Train_cow.html',context={'objs':objs})
def testing(request):
    return render(request, 'test.html')

def train(request):
    message = finalize_training()
    objs = FaceDetect.objects.all()
    print(message)
    return render(request, 'Train_cow.html',{
                    'message': "Training done successfully",
                    'objs' : objs
                })


def upload(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        location = request.POST.get('location')
        uploaded_file = request.FILES.get('image')

        if uploaded_file:
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            file_path = fs.path(filename)

            unique_id,label_dict, message = train_on_image(file_path, name, location)
            
            print(label_dict)
            objs = FaceDetect.objects.all()
            if unique_id:
                return render(request, 'Train_cow.html', {
                    'message': f"Training successful. Unique ID: {unique_id}",
                    'name': name,
                    'location': location,
                    'file_url': fs.url(filename),
                    'objs' : objs
                })
            else:
                return render(request, 'Train_cow.html', {
                    'error': message,
                    'name': name,
                    'location': location,
                    'objs' : objs
                }) 
        else:
            print('Failed to upload')

    return render(request, 'index.html')


def test_image(request):
    if request.method == 'POST' and request.FILES.get('testfile'):
        uploaded_file = request.FILES['testfile']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)

        predicted_details, message = predict_id(file_path)

        if predicted_details:
            return render(request, 'test.html', {
                'file_url': fs.url(filename),
                'name': predicted_details['name'],
                'unique_id': predicted_details['unique_id'],
                'location': predicted_details['location'],
                'message': message
            })
        else:
            return render(request, 'test.html', {
                'file_url': fs.url(filename),
                'error': message
            })

    return render(request, 'test.html')

# from django.shortcuts import get_object_or_404, redirect
# def delete_cow(request, id):
#     import os
#     import pickle
#     import numpy as np
#     from django.shortcuts import get_object_or_404, redirect
#     from .models import FaceDetect

#     label_dict_path = 'output_faces/label_dict.pkl'
#     encodings_path = 'output_faces/face_encodings.npy'
#     labels_path = 'output_faces/face_labels.npy'

#     if request.method == 'POST':
#         obj = get_object_or_404(FaceDetect, id=id)
#         unique_id = obj.unique_id
#         obj.delete()

#         # Load existing label dict
#         with open(label_dict_path, 'rb') as f:
#             label_dict = pickle.load(f)

#         # Find labels to remove
#         labels_to_remove = [label for label, info in label_dict.items() if info['unique_id'] == unique_id]

#         if not labels_to_remove:
#             return redirect('training')

#         # Load existing encodings and labels
#         all_encodings = np.load(encodings_path, allow_pickle=True)
#         all_labels = np.load(labels_path, allow_pickle=True)

#         # Filter out deleted labels
#         filtered_encodings = []
#         filtered_labels = []

#         for enc, label in zip(all_encodings, all_labels):
#             if label not in labels_to_remove:
#                 filtered_encodings.append(enc)
#                 filtered_labels.append(label)

#         # Remove from label_dict
#         for label in labels_to_remove:
#             label_dict.pop(label, None)

#         # Reindex label_dict
#         sorted_old_labels = sorted(label_dict.keys())
#         old_to_new = {old: new for new, old in enumerate(sorted_old_labels)}
#         new_label_dict = {new: label_dict[old] for old, new in old_to_new.items()}

#         # Update filtered_labels to reflect new indices
#         updated_labels = [old_to_new[label] for label in filtered_labels]
        
#         print('New Labe Dict - ', new_label_dict)
#         print('New LAbels' , updated_labels)

#         # Save the updated data
#         np.save(encodings_path, np.array(filtered_encodings, dtype=object))
#         np.save(labels_path, np.array(updated_labels, dtype=object))
#         with open(label_dict_path, 'wb') as f:
#             pickle.dump(new_label_dict, f)
            
#         label_dict = new_label_dict

#         print("Successfully deleted cow and updated encodings, labels, and label_dict.")

#     return redirect('training')
