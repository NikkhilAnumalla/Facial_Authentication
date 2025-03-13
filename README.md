# Facial Authentication System

A web-based facial recognition authentication system built with Flask and DeepFace. This application allows users to register with both traditional password authentication and facial recognition, providing a dual-factor authentication option.

## Features

- User registration with password and facial recognition
- Login with either password or facial recognition
- Secure password hashing using SHA-256
- Face detection using OpenCV
- Face embedding generation using DeepFace with FaceNet model
- Cosine similarity matching for facial authentication
- Webcam integration for live facial capture
- Upload option for profile photos
- User session management

## Technologies Used

- Python 3.x
- Flask - Web framework
- OpenCV - Computer vision and face detection
- DeepFace - Facial recognition and embedding generation
- NumPy - Numerical operations
- scikit-learn - Machine learning utilities and cosine similarity
- Werkzeug - Secure filename handling
- HTML/CSS/JavaScript - Frontend interface

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/facial-authentication-system.git
cd facial-authentication-system
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the application:
```
python app.py
```

4. Navigate to `http://localhost:5000` in your browser

## Project Structure

```
facial-authentication-system/
├── app.py                 # Main application file
├── requirements.txt       # Project dependencies
├── face_embeddings2.pkl   # Stored facial embeddings (generated at runtime)
├── users.json             # User credentials (generated at runtime)
├── static/                # Static files
│   └── uploads/           # User uploaded images
└── templates/             # HTML templates
    ├── index.html         # Home page
    ├── register.html      # Registration page
    ├── login.html         # Login page
    └── dashboard.html     # User dashboard
```

## Usage

### Registration
1. Navigate to the registration page
2. Enter a unique user ID and password
3. Either upload a profile photo or capture one using your webcam
4. Submit the form to register

### Login
1. Navigate to the login page
2. Choose between password or facial recognition login
3. For password login: Enter your user ID and password
4. For facial recognition: Allow webcam access and position your face in the frame
5. Submit to authenticate

## Security Considerations

- Passwords are hashed using SHA-256 before storage
- Facial embeddings are stored in a pickle file
- Adjust the `similarity_threshold` in the code for stricter or more lenient facial matching
- This is a demonstration project and may need additional security measures for production use

## Future Improvements

- Implement HTTPS for secure communication
- Add more robust password hashing (e.g., bcrypt)
- Implement rate limiting to prevent brute force attacks
- Add liveness detection to prevent spoofing with photos
- Implement multi-factor authentication combining face and password
