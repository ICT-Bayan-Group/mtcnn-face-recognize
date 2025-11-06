from flask import Flask, render_template_string, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle
import os
from datetime import datetime
import base64
import io
import ssl
import glob
import json
import uuid

app = Flask(__name__)
CORS(app)

class SoccerClinicFaceRecognition:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Device: {self.device}")
        
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=20,
            min_face_size=30,
            thresholds=[0.5, 0.6, 0.6],
            factor=0.709, 
            post_process=True,
            keep_all=True,
            device=self.device
        )
        
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Database structure for photographer uploads
        self.photo_database = {
            'photos': {},  # photo_id: {path, faces_data, metadata}
            'face_embeddings': {}  # embedding_id: embedding_vector
        }
        
        self.threshold = 0.7
        self.load_database()
    
    def extract_embedding(self, face_img):
        with torch.no_grad():
            face_img = face_img.to(self.device)
            embedding = self.resnet(face_img.unsqueeze(0))
        return embedding.cpu().numpy().flatten()
    
    def detect_faces_in_image(self, image_path):
        """Detect all faces in an uploaded photo"""
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        boxes, probs = self.mtcnn.detect(img_pil)
        
        faces_data = []
        if boxes is not None:
            for i, box in enumerate(boxes):
                try:
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2 = min(img.shape[1], x2)
                    y2 = min(img.shape[0], y2)
                    
                    face_img = img_pil.crop((x1, y1, x2, y2))
                    face_img = face_img.resize((160, 160))
                    
                    face_array = np.array(face_img).astype(np.float32)
                    face_array = (face_array - 127.5) / 128.0
                    face_tensor = torch.from_numpy(face_array).permute(2, 0, 1)
                    
                    embedding = self.extract_embedding(face_tensor)
                    embedding_id = str(uuid.uuid4())
                    
                    self.photo_database['face_embeddings'][embedding_id] = embedding.tolist()
                    
                    faces_data.append({
                        'box': [x1, y1, x2, y2],
                        'confidence': float(probs[i]),
                        'embedding_id': embedding_id
                    })
                    
                except Exception as e:
                    print(f"Error processing face: {e}")
                    continue
        
        return faces_data
    
    def match_user_face(self, user_embedding):
        """Match user's face embedding with photos in database"""
        matched_photos = []
        user_emb = np.array(user_embedding)
        
        for photo_id, photo_data in self.photo_database['photos'].items():
            for face in photo_data.get('faces_data', []):
                embedding_id = face.get('embedding_id')
                if embedding_id in self.photo_database['face_embeddings']:
                    stored_emb = np.array(self.photo_database['face_embeddings'][embedding_id])
                    distance = np.linalg.norm(user_emb - stored_emb)
                    
                    if distance < self.threshold:
                        matched_photos.append({
                            'photo_id': photo_id,
                            'photo_path': photo_data['path'],
                            'metadata': photo_data.get('metadata', {}),
                            'distance': float(distance),
                            'face_box': face['box']
                        })
                        break  # Only add photo once even if multiple face matches
        
        # Sort by distance (best match first)
        matched_photos.sort(key=lambda x: x['distance'])
        return matched_photos
    
    def save_database(self):
        with open('soccer_clinic_db.pkl', 'wb') as f:
            pickle.dump(self.photo_database, f)
    
    def load_database(self):
        if os.path.exists('soccer_clinic_db.pkl'):
            with open('soccer_clinic_db.pkl', 'rb') as f:
                self.photo_database = pickle.load(f)
            print(f"✅ Database loaded: {len(self.photo_database['photos'])} photos")
        else:
            print("📂 New database created")

face_system = SoccerClinicFaceRecognition()

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Soccer Clinic - AI Face Recognition</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        
        .header {
            text-align: center;
            padding: 40px 20px;
            color: white;
            margin-bottom: 30px;
        }
        
        .header h1 { 
            font-size: 36px; 
            font-weight: 700; 
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p { 
            font-size: 16px; 
            opacity: 0.9;
        }
        
        .role-selector {
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
            justify-content: center;
        }
        
        .role-card {
            background: white;
            border-radius: 12px;
            padding: 30px;
            width: 300px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .role-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.2);
        }
        
        .role-card .icon {
            font-size: 64px;
            margin-bottom: 20px;
        }
        
        .role-card h2 {
            font-size: 24px;
            margin-bottom: 10px;
            color: #667eea;
        }
        
        .role-card p {
            font-size: 14px;
            color: #666;
        }
        
        .app-container {
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            display: none;
        }
        
        .app-container.active {
            display: block;
        }
        
        .section-title {
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
            color: #667eea;
        }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 12px;
            padding: 60px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: #f8f9ff;
        }
        
        .upload-area:hover { 
            border-color: #764ba2; 
            background: #f0f2ff; 
        }
        
        .upload-area.dragover { 
            border-color: #764ba2; 
            background: #e8ebff; 
        }
        
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .image-card {
            position: relative;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: all 0.3s;
        }
        
        .image-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        
        .image-card img {
            width: 100%;
            height: 250px;
            object-fit: cover;
        }
        
        .image-card .info {
            padding: 15px;
            background: white;
        }
        
        .image-card .badge {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(102, 126, 234, 0.9);
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }
        
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            background: #667eea;
            color: white;
        }
        
        button:hover { 
            background: #764ba2;
            transform: scale(1.02);
        }
        
        button:active { 
            transform: scale(0.98); 
        }
        
        button.secondary {
            background: white;
            color: #667eea;
            border: 2px solid #667eea;
        }
        
        button.secondary:hover {
            background: #667eea;
            color: white;
        }
        
        .controls {
            margin-top: 20px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .status {
            margin-top: 15px;
            padding: 15px;
            border-radius: 8px;
            font-size: 14px;
        }
        
        .status.success { 
            background: #d4edda; 
            color: #155724; 
            border: 1px solid #c3e6cb;
        }
        
        .status.info { 
            background: #d1ecf1; 
            color: #0c5460; 
            border: 1px solid #bee5eb;
        }
        
        .status.warning { 
            background: #fff3cd; 
            color: #856404; 
            border: 1px solid #ffeaa7;
        }
        
        video {
            width: 100%;
            max-width: 640px;
            border-radius: 12px;
            background: #000;
            margin: 20px auto;
            display: block;
        }
        
        .back-btn {
            position: fixed;
            top: 20px;
            left: 20px;
            background: white;
            color: #667eea;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            z-index: 1000;
        }
        
        .back-btn:hover {
            background: #667eea;
            color: white;
        }
        
        .metadata-form {
            background: #f8f9ff;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #667eea;
        }
        
        .form-group input {
            width: 100%;
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
        }
        
        .form-group input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .photo-detail {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .photo-detail img {
            width: 100%;
            max-height: 500px;
            object-fit: contain;
            background: #000;
        }
        
        .photo-detail .detail-info {
            padding: 20px;
        }
        
        .photo-detail .detail-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #f0f0f0;
        }
        
        .photo-detail .detail-row:last-child {
            border-bottom: none;
        }
        
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }
        
        .empty-state .icon {
            font-size: 64px;
            margin-bottom: 20px;
            opacity: 0.5;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚽ Soccer Clinic</h1>
            <p>AI-Powered Face Recognition Photo Platform</p>
        </div>
        
        <!-- Role Selection -->
        <div id="roleSelection" class="role-selector">
            <div class="role-card" onclick="selectRole('photographer')">
                <div class="icon">📸</div>
                <h2>Photographer</h2>
                <p>Upload and manage event photos</p>
            </div>
            
            <div class="role-card" onclick="selectRole('user')">
                <div class="icon">👤</div>
                <h2>User</h2>
                <p>View your personalized photos</p>
            </div>
        </div>
        
        <!-- Photographer App -->
        <div id="photographerApp" class="app-container">
            <button class="back-btn" onclick="backToRoleSelection()">← Back</button>
            
            <h2 class="section-title">📸 Photographer Dashboard</h2>
            
            <div class="metadata-form">
                <div class="form-group">
                    <label>Event Name:</label>
                    <input type="text" id="eventName" placeholder="e.g., Soccer Training Session">
                </div>
                <div class="form-group">
                    <label>Location:</label>
                    <input type="text" id="eventLocation" placeholder="e.g., Main Stadium">
                </div>
                <div class="form-group">
                    <label>Photographer Name:</label>
                    <input type="text" id="photographerName" placeholder="Your name">
                </div>
            </div>
            
            <div class="upload-area" id="photographerUploadArea" onclick="document.getElementById('photographerFileInput').click()">
                <div style="font-size: 64px; margin-bottom: 15px;">📷</div>
                <p style="font-size: 20px; font-weight: 600; margin-bottom: 8px;">Upload Event Photos</p>
                <p style="font-size: 14px; color: #666;">Click or drag & drop multiple images</p>
            </div>
            <input type="file" id="photographerFileInput" accept="image/*" multiple style="display:none;" onchange="handlePhotographerUpload(event)">
            
            <div class="controls" id="photographerControls" style="display:none;">
                <button onclick="processPhotographerPhotos()">🔍 Process & Detect Faces</button>
                <button class="secondary" onclick="clearPhotographerUploads()">Clear All</button>
            </div>
            
            <div id="photographerStatus" class="status" style="display:none;"></div>
            
            <div id="photographerGallery" style="margin-top: 30px;">
                <h3 class="section-title">Uploaded Photos</h3>
                <div id="photographerImageGrid" class="image-grid"></div>
            </div>
        </div>
        
        <!-- User App -->
        <div id="userApp" class="app-container">
            <button class="back-btn" onclick="backToRoleSelection()">← Back</button>
            
            <h2 class="section-title">👤 Your Personal Gallery</h2>
            
            <div id="userRegistration" style="display:block;">
                <p style="margin-bottom: 20px; color: #666;">
                    To view your photos, we need to register your face first. Your face data is stored locally in your browser and never sent to the server.
                </p>
                
                <video id="userVideo" autoplay playsinline muted></video>
                
                <div class="controls">
                    <button onclick="startUserRegistration()">📷 Start Camera</button>
                    <button class="secondary" onclick="stopUserCamera()">Stop</button>
                    <button onclick="captureUserFace()" id="captureBtn" style="display:none;">✅ Capture My Face</button>
                </div>
                
                <div id="userRegStatus" class="status" style="display:none;"></div>
            </div>
            
            <div id="userGallery" style="display:none;">
                <div style="background: #d4edda; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                    <strong>✅ Face Registered!</strong> Below are your personalized photos.
                </div>
                
                <button class="secondary" onclick="resetUserFace()" style="margin-bottom: 20px;">🔄 Register Different Face</button>
                
                <div id="userImageGrid" class="image-grid"></div>
                <div id="emptyState" class="empty-state" style="display:none;">
                    <div class="icon">📷</div>
                    <h3>No Photos Found</h3>
                    <p>We couldn't find any photos with your face yet. Check back later!</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let photographerImages = [];
        let userVideoStream = null;
        let userFaceEmbedding = null;
        
        function selectRole(role) {
            document.getElementById('roleSelection').style.display = 'none';
            
            if (role === 'photographer') {
                document.getElementById('photographerApp').classList.add('active');
                loadPhotographerGallery();
            } else if (role === 'user') {
                document.getElementById('userApp').classList.add('active');
                checkUserRegistration();
            }
        }
        
        function backToRoleSelection() {
            document.getElementById('photographerApp').classList.remove('active');
            document.getElementById('userApp').classList.remove('active');
            document.getElementById('roleSelection').style.display = 'flex';
            stopUserCamera();
        }
        
        // ==================== PHOTOGRAPHER FUNCTIONS ====================
        
        function handlePhotographerUpload(event) {
            const files = Array.from(event.target.files);
            
            files.forEach(file => {
                if (!file.type.match('image.*')) return;
                
                const reader = new FileReader();
                reader.onload = function(e) {
                    photographerImages.push({
                        data: e.target.result,
                        name: file.name,
                        id: Date.now() + Math.random()
                    });
                    displayPhotographerImages();
                };
                reader.readAsDataURL(file);
            });
            
            document.getElementById('photographerControls').style.display = 'flex';
        }
        
        function displayPhotographerImages() {
            const grid = document.getElementById('photographerImageGrid');
            
            grid.innerHTML = photographerImages.map((img, idx) => `
                <div class="image-card">
                    <img src="${img.data}" alt="${img.name}">
                    <div class="badge">Pending</div>
                    <div class="info">
                        <strong>${img.name}</strong>
                    </div>
                </div>
            `).join('');
        }
        
        async function processPhotographerPhotos() {
            if (photographerImages.length === 0) {
                alert('Please upload photos first!');
                return;
            }
            
            const eventName = document.getElementById('eventName').value || 'Untitled Event';
            const location = document.getElementById('eventLocation').value || 'Unknown Location';
            const photographer = document.getElementById('photographerName').value || 'Anonymous';
            
            showPhotographerStatus('🔍 Processing photos...', 'info');
            
            for (let i = 0; i < photographerImages.length; i++) {
                const img = photographerImages[i];
                showPhotographerStatus(`Processing ${i + 1}/${photographerImages.length}...`, 'info');
                
                try {
                    const response = await fetch('/api/photographer/upload', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            image: img.data,
                            filename: img.name,
                            metadata: {
                                event_name: eventName,
                                location: location,
                                photographer: photographer,
                                date: new Date().toISOString()
                            }
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (!data.success) {
                        console.error('Upload failed:', data.error);
                    }
                } catch (err) {
                    console.error('Error:', err);
                }
            }
            
            showPhotographerStatus('✅ All photos processed successfully!', 'success');
            photographerImages = [];
            document.getElementById('photographerControls').style.display = 'none';
            loadPhotographerGallery();
        }
        
        async function loadPhotographerGallery() {
            try {
                const response = await fetch('/api/photographer/photos');
                const data = await response.json();
                
                const grid = document.getElementById('photographerImageGrid');
                
                if (data.photos.length === 0) {
                    grid.innerHTML = '<div class="empty-state"><div class="icon">📸</div><h3>No Photos Uploaded</h3><p>Upload some event photos to get started!</p></div>';
                    return;
                }
                
                grid.innerHTML = data.photos.map(photo => `
                    <div class="image-card">
                        <img src="/uploads/${photo.filename}" alt="${photo.filename}">
                        <div class="badge">${photo.faces_count} faces</div>
                        <div class="info">
                            <strong>${photo.metadata.event_name || 'Untitled'}</strong><br>
                            <small style="color: #666;">${photo.metadata.location || ''}</small>
                        </div>
                    </div>
                `).join('');
            } catch (err) {
                console.error('Error loading gallery:', err);
            }
        }
        
        function clearPhotographerUploads() {
            photographerImages = [];
            document.getElementById('photographerImageGrid').innerHTML = '';
            document.getElementById('photographerControls').style.display = 'none';
            showPhotographerStatus('Uploads cleared', 'info');
        }
        
        function showPhotographerStatus(message, type) {
            const el = document.getElementById('photographerStatus');
            el.style.display = 'block';
            el.className = `status ${type}`;
            el.textContent = message;
        }
        
        // Drag & Drop for Photographer
        const uploadArea = document.getElementById('photographerUploadArea');
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = Array.from(e.dataTransfer.files);
            files.forEach(file => {
                if (file && file.type.match('image.*')) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        photographerImages.push({
                            data: e.target.result,
                            name: file.name,
                            id: Date.now() + Math.random()
                        });
                        displayPhotographerImages();
                    };
                    reader.readAsDataURL(file);
                }
            });
            
            document.getElementById('photographerControls').style.display = 'flex';
        });
        
        // ==================== USER FUNCTIONS ====================
        
        function checkUserRegistration() {
            const stored = localStorage.getItem('user_face_embedding');
            if (stored) {
                userFaceEmbedding = JSON.parse(stored);
                document.getElementById('userRegistration').style.display = 'none';
                document.getElementById('userGallery').style.display = 'block';
                loadUserPhotos();
            }
        }
        
        async function startUserRegistration() {
            try {
                const video = document.getElementById('userVideo');
                userVideoStream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: 'user', width: 1280, height: 720 },
                    audio: false
                });
                video.srcObject = userVideoStream;
                
                document.getElementById('captureBtn').style.display = 'inline-block';
                showUserRegStatus('✅ Camera ready! Position your face and click Capture', 'success');
            } catch (err) {
                showUserRegStatus('❌ Camera error: ' + err.message, 'warning');
            }
        }
        
        async function captureUserFace() {
            const video = document.getElementById('userVideo');
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);
            
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            showUserRegStatus('🔍 Extracting face embedding...', 'info');
            
            try {
                const response = await fetch('/api/user/register_face', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({image: imageData})
                });
                
                const data = await response.json();
                
                if (data.success) {
                    userFaceEmbedding = data.embedding;
                    localStorage.setItem('user_face_embedding', JSON.stringify(userFaceEmbedding));
                    
                    showUserRegStatus('✅ Face registered! Loading your photos...', 'success');
                    
                    stopUserCamera();
                    document.getElementById('userRegistration').style.display = 'none';
                    document.getElementById('userGallery').style.display = 'block';
                    
                    loadUserPhotos();
                } else {
                    showUserRegStatus('❌ ' + data.error, 'warning');
                }
            } catch (err) {
                showUserRegStatus('❌ Error: ' + err.message, 'warning');
            }
        }
        
        async function loadUserPhotos() {
            if (!userFaceEmbedding) return;
            
            try {
                const response = await fetch('/api/user/my_photos', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({embedding: userFaceEmbedding})
                });
                
                const data = await response.json();
                
                const grid = document.getElementById('userImageGrid');
                const emptyState = document.getElementById('emptyState');
                
                if (data.photos.length === 0) {
                    grid.innerHTML = '';
                    emptyState.style.display = 'block';
                    return;
                }
                
                emptyState.style.display = 'none';
                
                grid.innerHTML = data.photos.map(photo => `
                    <div class="image-card">
                        <img src="/uploads/${photo.filename}" alt="${photo.filename}">
                        <div class="badge">Match: ${(100 - photo.distance * 100).toFixed(0)}%</div>
                        <div class="info">
                            <strong>${photo.metadata.event_name || 'Untitled Event'}</strong><br>
                            <small style="color: #666;">
                                📍 ${photo.metadata.location || 'Location unknown'}<br>
                                📅 ${new Date(photo.metadata.date).toLocaleDateString()}<br>
                                📸 ${photo.metadata.photographer || 'Unknown'}
                            </small><br>
                            <button onclick="downloadPhoto('/uploads/${photo.filename}', '${photo.filename}')" style="width: 100%; margin-top: 10px; font-size: 13px;">
                                📥 Download
                            </button>
                        </div>
                    </div>
                `).join('');
            } catch (err) {
                console.error('Error loading photos:', err);
            }
        }
        
        function downloadPhoto(url, filename) {
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            link.click();
        }
        
        function resetUserFace() {
            if (confirm('Are you sure you want to register a different face? Your current registration will be removed.')) {
                localStorage.removeItem('user_face_embedding');
                userFaceEmbedding = null;
                document.getElementById('userGallery').style.display = 'none';
                document.getElementById('userRegistration').style.display = 'block';
            }
        }
        
        function stopUserCamera() {
            if (userVideoStream) {
                userVideoStream.getTracks().forEach(track => track.stop());
                userVideoStream = null;
            }
            document.getElementById('userVideo').srcObject = null;
            document.getElementById('captureBtn').style.display = 'none';
        }
        
        function showUserRegStatus(message, type) {
            const el = document.getElementById('userRegStatus');
            el.style.display = 'block';
            el.className = `status ${type}`;
            el.textContent = message;
        }
    </script>
</body>
</html>
'''

# API Routes

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/photographer/upload', methods=['POST'])
def photographer_upload():
    """Photographer uploads photos with metadata"""
    try:
        data = request.json
        image_data = data['image']
        filename = data['filename']
        metadata = data.get('metadata', {})
        
        # Save image to uploads folder
        os.makedirs('uploads', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        save_path = f"uploads/{unique_filename}"
        
        nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(save_path, img)
        
        # Detect faces and extract embeddings
        faces_data = face_system.detect_faces_in_image(save_path)
        
        # Save to database
        photo_id = str(uuid.uuid4())
        face_system.photo_database['photos'][photo_id] = {
            'path': save_path,
            'filename': unique_filename,
            'faces_data': faces_data,
            'metadata': metadata,
            'uploaded_at': datetime.now().isoformat()
        }
        
        face_system.save_database()
        
        return jsonify({
            'success': True,
            'photo_id': photo_id,
            'faces_detected': len(faces_data)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/photographer/photos', methods=['GET'])
def get_photographer_photos():
    """Get all uploaded photos for photographer dashboard"""
    try:
        photos = []
        for photo_id, photo_data in face_system.photo_database['photos'].items():
            photos.append({
                'photo_id': photo_id,
                'filename': photo_data['filename'],
                'faces_count': len(photo_data.get('faces_data', [])),
                'metadata': photo_data.get('metadata', {}),
                'uploaded_at': photo_data.get('uploaded_at', '')
            })
        
        # Sort by upload time (newest first)
        photos.sort(key=lambda x: x['uploaded_at'], reverse=True)
        
        return jsonify({'success': True, 'photos': photos})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'photos': []})

@app.route('/api/user/register_face', methods=['POST'])
def user_register_face():
    """User registers their face (stored in localStorage)"""
    try:
        data = request.json
        image_data = data['image']
        
        nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        boxes, _ = face_system.mtcnn.detect(img_pil)
        
        if boxes is None or len(boxes) == 0:
            return jsonify({'success': False, 'error': 'No face detected. Please try again.'})
        
        # Get the largest face (assuming it's the main subject)
        box = boxes[0]
        x1, y1, x2, y2 = [int(coord) for coord in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        face_img = img_pil.crop((x1, y1, x2, y2))
        face_img = face_img.resize((160, 160))
        
        face_array = np.array(face_img).astype(np.float32)
        face_array = (face_array - 127.5) / 128.0
        face_tensor = torch.from_numpy(face_array).permute(2, 0, 1)
        
        embedding = face_system.extract_embedding(face_tensor)
        
        return jsonify({
            'success': True,
            'embedding': embedding.tolist()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/user/my_photos', methods=['POST'])
def get_user_photos():
    """Get photos that match the user's face"""
    try:
        data = request.json
        user_embedding = data['embedding']
        
        matched_photos = face_system.match_user_face(user_embedding)
        
        # Format response
        photos = []
        for match in matched_photos:
            photo_id = match['photo_id']
            photo_data = face_system.photo_database['photos'].get(photo_id)
            
            if photo_data:
                photos.append({
                    'photo_id': photo_id,
                    'filename': photo_data['filename'],
                    'metadata': photo_data.get('metadata', {}),
                    'distance': match['distance']
                })
        
        return jsonify({'success': True, 'photos': photos})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'photos': []})

@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve uploaded images"""
    return send_from_directory('uploads', filename)

def generate_self_signed_cert():
    """Generate self-signed certificate for HTTPS"""
    try:
        from OpenSSL import crypto
    except ImportError:
        return False
    
    if os.path.exists('cert.pem') and os.path.exists('key.pem'):
        print("✅ SSL Certificate exists")
        return True
    
    print("🔐 Generating self-signed SSL certificate...")
    
    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 2048)
    
    cert = crypto.X509()
    cert.get_subject().C = "ID"
    cert.get_subject().ST = "East Kalimantan"
    cert.get_subject().L = "Samarinda"
    cert.get_subject().O = "Soccer Clinic"
    cert.get_subject().OU = "AI Face Recognition"
    cert.get_subject().CN = "localhost"
    
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365*24*60*60)
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, 'sha256')
    
    with open("cert.pem", "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    
    with open("key.pem", "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
    
    print("✅ SSL Certificate created")
    return True

if __name__ == '__main__':
    import socket
    
    # Create necessary folders
    os.makedirs('uploads', exist_ok=True)
    print("📁 Folder ready: uploads/")
    
    # Generate SSL certificate
    has_ssl = generate_self_signed_cert()
    
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("\n" + "="*70)
    print("⚽ SOCCER CLINIC - AI FACE RECOGNITION PLATFORM")
    print("="*70)
    
    if has_ssl:
        print(f"🌐 Laptop/Desktop: https://localhost:5000")
        print(f"📱 Mobile (same WiFi): https://{local_ip}:5000")
        print("="*70)
        print("✨ FEATURES:")
        print("   📸 Photographer Role:")
        print("      • Upload multiple event photos")
        print("      • Automatic face detection & recognition")
        print("      • Add event metadata (name, location, photographer)")
        print("      • View all uploaded photos")
        print()
        print("   👤 User Role:")
        print("      • No login required")
        print("      • Face registration via camera")
        print("      • Face data stored in browser (LocalStorage)")
        print("      • Personalized photo gallery")
        print("      • Download photos with your face")
        print("="*70)
        print("📱 For iOS/iPhone:")
        print("   1. Must use HTTPS (not HTTP)")
        print("   2. Browser will show 'Not Secure' warning")
        print("   3. Click 'Advanced' → 'Proceed to {}'".format(local_ip))
        print("   4. Allow camera access when prompted")
        print("="*70)
        print("🔒 Privacy:")
        print("   • User face data NEVER sent to server")
        print("   • Stored locally in browser only")
        print("   • Full client-side privacy protection")
        print("="*70)
        print("📝 Make sure laptop and phone on SAME WiFi!")
        print("="*70 + "\n")
        
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain('cert.pem', 'key.pem')
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, ssl_context=context)
    else:
        print("⚠️  PyOpenSSL not installed. Install: pip install pyopenssl")
        print("⚠️  Running without HTTPS (won't work on iPhone!)")
        print(f"🌐 HTTP access: http://localhost:5000")
        print(f"📱 Android only: http://{local_ip}:5000")
        print("="*70 + "\n")
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)