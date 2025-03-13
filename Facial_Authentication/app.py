import os
import cv2
import numpy as np
import pickle
from pathlib import Path
import time
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, redirect, url_for, session, Response, jsonify
import base64
from werkzeug.utils import secure_filename
import hashlib
import json

app = Flask(__name__)
app.secret_key = 'facial_auth_secret_key'  # Change this to a strong secret key
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# File to store user credentials
USERS_FILE = 'users.json'


class FacialAuthSystem:
    def __init__(self):
        self.embedding_file = 'face_embeddings2.pkl'
        self.embeddings_dict = self.load_embeddings()
        self.model_name = "Facenet"  # Using Facenet for better embeddings
        self.detector_backend = "opencv"  # Using OpenCV for face detection
        self.capture_device = 0  # Default webcam
        self.similarity_threshold = 0.6  # Adjust based on testing
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def load_embeddings(self):
        """Load existing embeddings from pickle file"""
        if os.path.exists(self.embedding_file):
            with open(self.embedding_file, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_embeddings(self):
        """Save embeddings to pickle file"""
        with open(self.embedding_file, 'wb') as f:
            pickle.dump(self.embeddings_dict, f)
        print(f"Embeddings saved to {self.embedding_file}")

    def generate_embedding(self, img):
        """Generate face embedding from an image"""
        try:
            # Use DeepFace to extract embedding
            embedding_objs = DeepFace.represent(
                img,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True
            )

            if embedding_objs and len(embedding_objs) > 0:
                # Return first face embedding
                return embedding_objs[0]["embedding"]
            return None
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return None

    def register_user(self, user_id, img):
        """Register a new user with an image"""
        embedding = self.generate_embedding(img)
        if embedding is not None:
            self.embeddings_dict[user_id] = embedding
            self.save_embeddings()
            return True
        return False

    def authenticate_user(self, img):
        """Authenticate a user by comparing with stored embeddings"""
        if not self.embeddings_dict:
            print("No registered users found.")
            return None, -1

        test_embedding = self.generate_embedding(img)
        if test_embedding is None:
            return None, -1

        max_similarity = -1
        best_match = None

        for user_id, stored_embedding in self.embeddings_dict.items():
            # Calculate cosine similarity
            similarity = cosine_similarity(
                np.array(test_embedding).reshape(1, -1),
                np.array(stored_embedding).reshape(1, -1)
            )[0][0]

            if similarity > max_similarity:
                max_similarity = similarity
                best_match = user_id

        if max_similarity >= self.similarity_threshold:
            return best_match, max_similarity
        else:
            return None, max_similarity

    def detect_faces(self, frame):
        """Detect faces in the frame and return coordinates"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces


# User management functions
def get_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_users(users_dict):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users_dict, f)


def hash_password(password):
    """Simple password hashing using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(user_id, password):
    """Verify password for a user"""
    users = get_users()
    if user_id in users and users[user_id] == hash_password(password):
        return True
    return False


def register_user_with_password(user_id, password):
    """Register a new user with password"""
    users = get_users()
    if user_id in users:
        return False  # User already exists

    users[user_id] = hash_password(password)
    save_users(users)
    return True


# Initialize the facial authentication system
auth_system = FacialAuthSystem()


# Flask routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        password = request.form.get('password')

        if not user_id or not password:
            return render_template('register.html', error="User ID and Password are required")

        # Register password
        if not register_user_with_password(user_id, password):
            return render_template('register.html', error="User ID already exists")

        # Check if image was uploaded or captured
        facial_registered = False

        if 'image_upload' in request.files and request.files['image_upload'].filename:
            # Handle uploaded image
            file = request.files['image_upload']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the image
            img = cv2.imread(filepath)
            facial_registered = auth_system.register_user(user_id, img)

        elif request.form.get('image_data'):
            # Handle captured image from webcam
            image_data = request.form.get('image_data').split(',')[1]
            img_bytes = base64.b64decode(image_data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Save the captured image
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{user_id}_capture.jpg")
            cv2.imwrite(filepath, img)

            facial_registered = auth_system.register_user(user_id, img)

        session['registration_success'] = True
        if facial_registered:
            session['facial_registered'] = True

        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    message = None
    if session.pop('registration_success', False):
        message = "Registration successful! Please login."
        facial_registered = session.pop('facial_registered', False)
        if facial_registered:
            message += " Facial recognition is enabled for your account."

    if request.method == 'POST':
        # Check login method
        login_method = request.form.get('login_method', 'facial')

        if login_method == 'password':
            # Handle password login
            user_id = request.form.get('user_id')
            password = request.form.get('password')

            if not user_id or not password:
                return render_template('login.html', error="User ID and Password are required")

            if verify_password(user_id, password):
                session['user_id'] = user_id
                return redirect(url_for('dashboard'))
            else:
                return render_template('login.html', error="Invalid credentials")
        else:
            # Handle facial recognition login
            if request.form.get('image_data'):
                image_data = request.form.get('image_data').split(',')[1]
                img_bytes = base64.b64decode(image_data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                # Authenticate the user
                user_id, similarity = auth_system.authenticate_user(img)

                if user_id:
                    session['user_id'] = user_id
                    return redirect(url_for('dashboard'))
                else:
                    return render_template('login.html', error=f"Authentication failed. Similarity: {similarity:.2f}")
            else:
                return render_template('login.html', error="No image provided")

    return render_template('login.html', message=message)


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    return render_template('dashboard.html', user_id=session['user_id'])


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)