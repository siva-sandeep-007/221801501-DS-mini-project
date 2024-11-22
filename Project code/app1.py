import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import onnxruntime as ort
import logging
from multiprocessing import Pool
import matplotlib
from tqdm import tqdm
# Use non-GUI backend for Matplotlib
matplotlib.use('Agg')

# Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.secret_key = 'your_secure_secret_key_here'

# Folder configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = r'E:\Vs codes\project\Web app1\Web app\static\processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the ONNX model
onnx_model_path = r'E:\Vs codes\project\Web app1\Web app\faster_rcnn_model.onnx'  # Update with your model path
onnx_session = ort.InferenceSession(onnx_model_path)

# Action labels
action_labels = ['hand_raising', 'leaning', 'texting', 'sleeping', 'clapping', 'laughing', 'fighting', 'using_laptop']
positive_actions = ['hand_raising', 'clapping', 'laughing']
negative_actions = ['leaning', 'texting', 'sleeping', 'fighting', 'using_laptop']

# Dummy user database
users_db = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_frame(frame, model_width=640, model_height=640):
    original_height, original_width = frame.shape[:2]
    scale = min(model_width/original_width, model_height/original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    resized_frame = cv2.resize(frame, (new_width, new_height))
    canvas = np.zeros((model_height, model_width, 3), dtype=np.uint8)
    y_offset = (model_height - new_height) // 2
    x_offset = (model_width - new_width) // 2
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
    
    processed = canvas.transpose(2, 0, 1)  # HWC to CHW format
    processed = processed.astype(np.float32) / 255.0
    return processed

def generate_suggestions(action_counts):
    suggestion = "Classroom Engagement Suggestions:\n"
    
    if action_counts.get('hand_raising', 0) > 5:
        suggestion += "- Good engagement with frequent hand-raising.\n"
    elif action_counts.get('hand_raising', 0) < 3:
        suggestion += "- Consider encouraging more participation.\n"

    if action_counts.get('leaning', 0) > 5:
        suggestion += "- Multiple students showing signs of disengagement.\n"

    if action_counts.get('texting', 0) > 3:
        suggestion += "- Consider addressing phone usage in class.\n"

    positive_count = sum(action_counts.get(action, 0) for action in positive_actions)
    negative_count = sum(action_counts.get(action, 0) for action in negative_actions)

    if negative_count > positive_count:
        suggestion += "- Overall engagement needs improvement.\n"
    else:
        suggestion += "- Overall positive engagement observed.\n"

    return suggestion

def create_pie_chart(action_counts):
    plt.figure(figsize=(6, 6))
    labels = list(action_counts.keys())
    sizes = list(action_counts.values())
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    chart_path = os.path.join(PROCESSED_FOLDER, 'action_pie_chart.png')
    plt.savefig(chart_path)
    plt.close()
    return 'action_pie_chart.png'  # Return relative path

def process_and_analyze_video(uploads, frame_skip=1):

    detected_actions = []
    cap = cv2.VideoCapture(uploads)
    
    if not cap.isOpened():
        raise Exception("Error: Could not open video file")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'processed_video_{timestamp}.mp4'
    processed_video_path = os.path.join(PROCESSED_FOLDER, output_filename)
    
    try:
        # Try different codecs if mp4v fails
        codecs = ['mp4v', 'avc1', 'H264']
        out = None
        
        for codec in codecs:
            try:
                out = cv2.VideoWriter(processed_video_path, 
                                    cv2.VideoWriter_fourcc(*codec), 
                                    max(1, fps // frame_skip), 
                                    (frame_width, frame_height))
                if out.isOpened():
                    break
            except Exception as e:
                logging.warning(f"Codec {codec} failed: {str(e)}")
                continue
        
        if out is None or not out.isOpened():
            raise Exception("Could not initialize video writer with any available codec")

        input_name = onnx_session.get_inputs()[0].name
        model_height, model_width = 640, 640

        frame_count = 0
        processed_frames = 0

        with tqdm(total=total_frames // frame_skip, desc="Processing Video", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                try:
                    processed_frame = preprocess_frame(frame, model_width, model_height)
                    model_input = np.expand_dims(processed_frame, axis=0)

                    predictions = onnx_session.run(None, {input_name: model_input})
                    pred_idx = np.argmax(predictions[0], axis=1)[0]
                    
                    if 0 <= pred_idx < len(action_labels):
                        predicted_class = action_labels[pred_idx]
                        detected_actions.append(predicted_class)

                        # Add more visible label with background
                        label = f"Action: {predicted_class}"
                        # Create background rectangle for text
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        cv2.rectangle(frame, (5, 5), (text_width + 15, text_height + 15), (0, 0, 0), -1)
                        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        processed_frames += 1
                    else:
                        logging.warning(f"Invalid prediction index: {pred_idx}")
                        cv2.putText(frame, "Unknown Action", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    out.write(frame)

                except Exception as e:
                    logging.error(f"Error during frame processing: {str(e)}")
                    cv2.putText(frame, "Error processing frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    out.write(frame)

                pbar.update(1)

        if processed_frames == 0:
            raise Exception("No frames were successfully processed")

    except Exception as e:
        logging.error(f"Error during video processing: {str(e)}")
        raise

    finally:
        cap.release()
        if 'out' in locals() and out is not None:
            out.release()

    # Calculate action counts and create visualizations
    action_counts = {action: detected_actions.count(action) for action in set(detected_actions)}
    chart_path = create_pie_chart(action_counts)
    suggestions = generate_suggestions(action_counts)

    return output_filename, suggestions, chart_path

@app.route('/')
def index():
    if 'user' in session:
        return render_template('index.html', user=session['user'])
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        if email in users_db and check_password_hash(users_db[email]['password'], password):
            session['user'] = email
            flash('Login successful', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirmPassword']

        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register'))

        if email in users_db:
            flash('Email already registered', 'error')
            return redirect(url_for('register'))

        users_db[email] = {
            'name': name,
            'password': generate_password_hash(password)
        }
        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # Process the video and get results
                video_path, suggestions, chart_path = process_and_analyze_video(file_path)
                
                # Clean up the upload file
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                return render_template('result.html', 
                                    video_path=video_path,
                                    chart_path=chart_path,
                                    suggestions=suggestions)

            except Exception as e:
                logging.error(f"Error processing video: {str(e)}", exc_info=True)
                if os.path.exists(file_path):
                    os.remove(file_path)
                flash(f"Error processing video: {str(e)}", 'error')
                return redirect(url_for('upload'))
        else:
            flash('Invalid file type. Allowed types are: ' + ', '.join(ALLOWED_EXTENSIONS), 'error')
            return redirect(url_for('upload'))
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)