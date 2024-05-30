from flask import Flask, render_template, Response, url_for, request
import cv2
import math
from ultralytics import YOLO
import torch
import os
from torchvision import transforms, models
import PIL.Image as Image



app = Flask(__name__)

# Define your options list
samples = [
    'Vanilla Planifolia(Flat-leaved vanilla)', 
    'Alpinia galanga (L.) Willd(Blue ginger)',
    'Brucea mollis Wall. Ex Kurz(Kunain)', 
    'Belamcanda chinensis (L.) Redouté(Blackberry Lily)',
    'Chamaecostus cuspidatus (Nees & Mart.)(Fiery costus)', 
    'Cinnamomum tamala T.Nees & Eberm.(Malabar leaf)',
    'Citrus aurantiifolia (Christm.) Swingle(Bitter orange)', 
    'Dendrobium nobile(Noble dendrobium)',
    'Eclipta prostrata(False daisy)', 
    'Flemingia strobilifera (L.) W.T.Aiton(Wild hops )',
    'Eryngium foetidum L.(Culantro)', 
    'Hibiscus rosasinensis(Red Hibiscus)', 
    'Jatropha curcas L(Physic nut)',
    'Kalanchoe pinnata (Lam.) Pers(Miracle leaf)', 
    'Leucas aspera link(Thumba)', 
    'Mentha arvensis L(Corn Mint)',
    'Ocimum tenuiflorum(White holy basil)', 
    'Opuntia vulgaris Mill(Prickly pear )', 
    'paederia foetida l.',
    'Passiflora edulis sims(Passion fruit)', 
    'Piper longum L.(Indian Long Pepper)',
    'Plectranthus amboinicus (Lour.) Spreng.(Indian Borage)', 
    'Zingiber officinale Rosc.(Ginger rhizome)',
    'Streblus Asper Lour.(Bar-inka)'
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load YOLO model
model = YOLO("best.pt",task ='classify')

transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

fo_model_path = 'resnet18n_model.pth'  # Path to the saved model
fo_class_names = [ 'foreign-object','fresh']  # Update with your actual class names
num_classes = len(fo_class_names)  # Ensure this matches the number of classes in your dataset
fo_model = models.resnet18(pretrained=False)
fo_model.fc = torch.nn.Linear(fo_model.fc.in_features, num_classes)
fo_model.load_state_dict(torch.load(fo_model_path))
fo_model = fo_model.to(device)

# Object classes
classNames = ['Vanilla Planifolia(Flat-leaved vanilla)', 'Alpinia galanga (L.) Willd(Blue ginger)',
              'Brucea mollis Wall. Ex Kurz(Kunain)', 'Belamcanda chinensis (L.) Redouté(Blackberry Lily)',
              'Chamaecostus cuspidatus (Nees & Mart.)(Fiery costus)', 'Cinnamomum tamala T.Nees & Eberm.(Malabar leaf)',
              'Citrus aurantiifolia (Christm.) Swingle(Bitter orange)', 'Dendrobium nobile(Noble dendrobium)',
              'Eclipta prostrata(False daisy)', 'Flemingia strobilifera (L.) W.T.Aiton(Wild hops )',
              'Eryngium foetidum L.(Culantro)', 'Hibiscus rosasinensis(Red Hibiscus)', 'Jatropha curcas L(Physic nut)',
              'Kalanchoe pinnata (Lam.) Pers(Miracle leaf)', 'Leucas aspera link(Thumba)', 'Mentha arvensis L(Corn Mint)',
              'Ocimum tenuiflorum(White holy basil)', 'Opuntia vulgaris Mill(Prickly pear )', 'paederia foetida l.',
              'Passiflora edulis sims(Passion fruit)', 'Piper longum L.(Indian Long Pepper)',
              'Plectranthus amboinicus (Lour.) Spreng.(Indian Borage)', 'Zingiber officinale Rosc.(Ginger rhizome)',
              'Streblus Asper Lour.(Bar-inka)']

# Define a variable to keep track of whether the camera is currently in use
camera = None

# Function to start the camera
def start_camera():
    global camera
    if camera is None:
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        camera = cap

# Function to stop the camera
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

# Function to check if the camera is running
def is_camera_running():
    return camera is not None

# Function to perform object detection
def detect_objects_rl():
    global camera
    start_camera()  # Start the camera before detection
    while True:
        success, img = camera.read()
        results = model(img, stream=True)

        for r in results:
            
            boxes = r.boxes

            for box in boxes:
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    if confidence > 0.80:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                        # Draw bounding box
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        # Add spacing between the bounding box and image boundary
                        padding = 10  # Adjust this value to decrease the size of the bounding box
                        x1 += padding
                        y1 += padding
                        x2 -= padding
                        y2 -= padding

                        # Display class name
                        org = [x1, y1]  # Adjusted position for displaying text below the box

                        # Confidence
                        confidence = math.ceil((box.conf[0] * 100)) / 100
                        print(confidence)
                        # Class name
                        cls = int(box.cls[0])
                        class_name = classNames[cls]

                        # Display class name
                        padding_top = 10
                        org = [x1, y1 + padding_top]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2

                        cv2.putText(img, class_name, org, font, fontScale, color, thickness)

                    ret, jpeg = cv2.imencode('.jpg', img)
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        # Release the camera when the function ends
        if camera is None:
            break

def predict_image(frame, model, class_names):
    try:
        # Ensure the model is in evaluation mode
        model.eval()
        
        # Open and transform the image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Apply the transformations
        image_tensor = transform(pil_image).unsqueeze(0).to(device)

        # Make the prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            print(outputs)
            _, preds = torch.max(outputs, 1)
            predicted_class = class_names[preds[0]]

        return predicted_class
    except Exception as e:
        
        return None, None


def detect_objects_ql(input_sample):
    global camera
    start_camera()  # Start the camera before detection
    while True:
        success, img = camera.read()
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                if confidence > 0.80 and class_name != input_sample:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                        # Draw bounding box
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        # Add spacing between the bounding box and image boundary
                        padding = 10  # Adjust this value to decrease the size of the bounding box
                        x1 += padding
                        y1 += padding
                        x2 -= padding
                        y2 -= padding

                        # Display class name
                        org = [x1, y1]  # Adjusted position for displaying text below the box

                        # Confidence
                        confidence = math.ceil((box.conf[0] * 100)) / 100
                        print(confidence)
                        # Class name
                        cls = int(box.cls[0])
                        class_name = classNames[cls]

                        # Display class name
                        padding_top = 10
                        org = [x1, y1 + padding_top]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2

                        cv2.putText(img, class_name, org, font, fontScale, color, thickness)
                elif confidence > 0.80:
                    cropped_frame = img[y1:y2, x1:x2]
                    predicted_class = " "
                    predicted_class += predict_image(cropped_frame, fo_model, fo_class_names)
                    predicted_class += "-healthy"

                    padding_top = 10
                    org = [x1, y1 + padding_top]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(img, predicted_class, org, font, fontScale, color, thickness)
                    

        


                ret, jpeg = cv2.imencode('.jpg', img)
                frame = jpeg.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        # Release the camera when the function ends
        if camera is None:
            break




@app.route('/')
def backgroundImage():
    hero_bg_url = url_for('static', filename='images/hero-bg.jpg')
    return render_template('sign-in.html', hero_bg_url=hero_bg_url)

@app.route('/index-particles.html')
def index_particles():
    button_url_1 = url_for('realtime_classification')  # Generate URL for realtime_classification route
    button_url_2 = url_for('quality_checking')  # Generate URL for quality_checking route
    return render_template('index-particles.html', button_url_1=button_url_1, button_url_2=button_url_2)

@app.route('/realtime_classification')
def realtime_classification():
    stop_camera()  # Stop the camera before rendering the template
    # Call the detect_objects function and pass its output to Response
    return Response(detect_objects_rl(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/quality_checking', methods=['GET', 'POST'])
def quality_checking():
    stop_camera()  # Stop the camera before rendering the template
    return render_template('quality_checking.html', samples=samples)

# Route for triggering object detection
@app.route('/object_detection')
def object_detection(): 
    # Get the selected sample from the query parameters
    selected_sample = request.args.get('sample')
    # Stop the camera before triggering object detection
    stop_camera()
    # Call the detect_objects function and pass its output to Response
    return Response(detect_objects_ql(selected_sample), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
