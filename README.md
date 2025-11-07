# Face Recognition Web System

## Overview

A comprehensive web-based face recognition system leveraging deep learning for biometric authentication, face detection, and smart photo organization.

## Features

- 🎯 Real-time face detection and recognition
- 📸 Multiple enrollment methods:
  - Camera-based enrollment
  - Folder-based batch enrollment
- 👥 User management system
- ✅ Human-in-the-Loop verification
- 🔒 Privacy-focused design
- 📱 Cross-platform support

## Tech Stack

- **Backend**: Python Flask
- **Frontend**: HTML5, JavaScript
- **AI Models**:
  - MTCNN (Face Detection)
  - FaceNet/InceptionResnetV1 (Face Embeddings)
- **Database**: Pickle-based storage

## Installation

### Prerequisites

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Dependencies

```txt
torch
torchvision
facenet-pytorch
opencv-python
pillow
numpy
flask
flask-cors
pyopenssl
```

## Quick Start

1. **Clone repository and setup**

```powershell
git clone <repository-url>
cd mtcnn
pip install -r requirements.txt
```

2. **Run application**

```powershell
python soccer.py
```

3. **Access web interface**

- Local: `https://localhost:5000`
- Network: `https://<your-ip>:5000`

## Project Structure

```
mtcnn/
│
├── app.py              # Main application file
├── face_database_web.pkl   # Database file
├── requirements.txt    # Dependencies
├── faces/             # Enrollment images
├── pending_faces/     # Verification queue
└── upload-foto/       # Uploaded images
```

## Usage Guide

### Face Recognition

1. Open "Recognize" tab
2. Select camera or upload mode
3. System detects and identifies faces in real-time

### User Enrollment

Choose between:

**Camera Method**

1. Go to "Camera Enroll"
2. Follow face positioning prompts
3. Complete enrollment process

**Folder Method**

1. Place photos in `faces/` directory
2. Use "Folder Enroll" tab
3. Submit enrollment

### Verification

1. Access "Review" tab
2. Review pending entries
3. Verify or assign users

## Configuration

Default recognition thresholds:

- Strict: 0.5
- Normal: 0.7
- Loose: 0.9

## Troubleshooting

Common solutions:

- iOS devices require HTTPS
- Check camera permissions
- Ensure proper lighting
- Verify file formats (jpg/png)

## Security Notes

- Development use only
- Implements self-signed SSL
- Local storage for face data
- Basic database security

## License

MIT License

## Support

Open an issue for support requests
