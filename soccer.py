from flask import Flask, render_template_string, request, jsonify, send_file, send_from_directory, Response
from flask_cors import CORS, cross_origin
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
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

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
        
        self.photo_database = {
            'photos': {},
            'face_embeddings': {}
        }
        
        self.threshold = 0.7
        self.load_database()
    
    def extract_embedding(self, face_img):
        with torch.no_grad():
            face_img = face_img.to(self.device)
            embedding = self.resnet(face_img.unsqueeze(0))
        return embedding.cpu().numpy().flatten()
    
    def detect_faces_in_image(self, image_path):
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
                        break
        
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

# ─────────────────────────────────────────────
# HTML TEMPLATE
# ─────────────────────────────────────────────
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
            background: #ffffff;
            min-height: 100vh;
            color: #333;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        @media (max-width: 768px) { .container { padding: 15px; } }
        .header {
            text-align: center; padding: 40px 20px; color: #2563eb;
            margin-bottom: 30px;
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            border-radius: 16px; box-shadow: 0 4px 12px rgba(37,99,235,0.3);
        }
        .header h1 { font-size: 36px; font-weight: 700; margin-bottom: 10px; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }
        .header p { font-size: 16px; color: white; opacity: 0.95; }
        @media (max-width: 768px) { .header { padding: 30px 15px; border-radius: 12px; } .header h1 { font-size: 28px; } .header p { font-size: 14px; } }
        .role-selector { display: flex; gap: 20px; margin-bottom: 30px; justify-content: center; flex-wrap: wrap; }
        @media (max-width: 768px) { .role-selector { flex-direction: column; gap: 15px; } }
        .role-card { background: white; border-radius: 12px; padding: 30px; width: 300px; text-align: center; cursor: pointer; transition: all 0.3s; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .role-card:hover { transform: translateY(-5px); box-shadow: 0 8px 12px rgba(0,0,0,0.2); }
        @media (max-width: 768px) { .role-card { width: 100%; padding: 25px; } }
        .role-card .icon { font-size: 64px; margin-bottom: 20px; }
        .role-card h2 { font-size: 24px; margin-bottom: 10px; color: #2563eb; }
        .role-card p { font-size: 14px; color: #666; }
        .app-container { background: white; border-radius: 12px; padding: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: none; }
        .app-container.active { display: block; }
        .section-title { font-size: 24px; font-weight: 700; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 3px solid #2563eb; color: #2563eb; }
        .upload-area { border: 3px dashed #2563eb; border-radius: 12px; padding: 60px 20px; text-align: center; cursor: pointer; transition: all 0.3s; background: #eff6ff; }
        .upload-area:hover { border-color: #1d4ed8; background: #dbeafe; }
        .upload-area.dragover { border-color: #1d4ed8; background: #bfdbfe; }
        .image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; margin-top: 20px; }
        @media (max-width: 768px) { .image-grid { grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 15px; } }
        @media (max-width: 480px) { .image-grid { grid-template-columns: 1fr; gap: 15px; } }
        .image-card { position: relative; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); transition: all 0.3s; cursor: pointer; }
        .image-card:hover { transform: translateY(-5px); box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
        .image-card img { width: 100%; height: 250px; object-fit: cover; }
        .image-card .info { padding: 15px; background: white; }
        .image-card .badge { position: absolute; top: 10px; right: 10px; background: rgba(37,99,235,0.95); color: white; padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
        button { padding: 12px 24px; border: none; border-radius: 8px; font-size: 15px; font-weight: 600; cursor: pointer; transition: all 0.2s; background: #2563eb; color: white; }
        button:hover { background: #1d4ed8; transform: scale(1.02); }
        button:active { transform: scale(0.98); }
        button.secondary { background: #fb923c; color: white; border: none; }
        button.secondary:hover { background: #f97316; }
        button:disabled { opacity: 0.6; cursor: not-allowed; transform: none !important; }
        .controls { margin-top: 20px; display: flex; gap: 10px; flex-wrap: wrap; }
        .status { margin-top: 15px; padding: 15px; border-radius: 8px; font-size: 14px; }
        .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status.info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        .status.warning { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
        video { width: 100%; max-width: 640px; border-radius: 12px; background: #000; margin: 20px auto; display: block; }
        .back-btn { position: fixed; top: 20px; left: 20px; background: white; color: #2563eb; padding: 10px 20px; border-radius: 8px; font-weight: 600; cursor: pointer; box-shadow: 0 2px 8px rgba(0,0,0,0.1); z-index: 1000; border: 2px solid #2563eb; }
        .back-btn:hover { background: #2563eb; color: white; }
        .metadata-form { background: #eff6ff; padding: 20px; border-radius: 12px; margin-bottom: 20px; border: 2px solid #bfdbfe; }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: 600; color: #2563eb; font-size: 14px; }
        .form-group input { width: 100%; padding: 10px; border: 2px solid #bfdbfe; border-radius: 8px; font-size: 14px; }
        .form-group input:focus { outline: none; border-color: #2563eb; }
        .password-modal { display: none; position: fixed; z-index: 3000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.85); align-items: center; justify-content: center; }
        .password-modal.active { display: flex; }
        .password-modal-content { background: white; padding: 40px; border-radius: 16px; max-width: 400px; width: 90%; box-shadow: 0 8px 32px rgba(0,0,0,0.3); animation: modalSlideIn 0.3s ease-out; }
        .password-modal h2 { color: #2563eb; margin-bottom: 10px; font-size: 24px; }
        .password-modal p { color: #666; margin-bottom: 25px; font-size: 14px; }
        .password-input-group { margin-bottom: 20px; }
        .password-input-group input { width: 100%; padding: 12px; border: 2px solid #bfdbfe; border-radius: 8px; font-size: 16px; }
        .password-input-group input:focus { outline: none; border-color: #2563eb; }
        .password-error { color: #dc2626; font-size: 13px; margin-top: 8px; display: none; }
        .password-buttons { display: flex; gap: 10px; }
        .password-buttons button { flex: 1; }
        .empty-state { text-align: center; padding: 60px 20px; color: #999; }
        .empty-state .icon { font-size: 64px; margin-bottom: 20px; opacity: 0.5; }
        .modal { display: none; position: fixed; z-index: 2000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.9); overflow: auto; }
        .modal.active { display: block; }
        .modal-content { position: relative; margin: 2% auto; padding: 0; max-width: 1000px; background-color: white; border-radius: 16px; box-shadow: 0 8px 32px rgba(0,0,0,0.3); animation: modalSlideIn 0.3s ease-out; }
        @keyframes modalSlideIn { from { transform: translateY(-50px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
        @keyframes shake { 0%,100%{transform:translateX(0)} 10%,30%,50%,70%,90%{transform:translateX(-10px)} 20%,40%,60%,80%{transform:translateX(10px)} }
        @keyframes spin { to { transform: rotate(360deg); } }
        .spin { animation: spin 0.8s linear infinite; display: inline-block; }
        .modal-header { padding: 20px 30px; background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white; border-radius: 16px 16px 0 0; display: flex; justify-content: space-between; align-items: center; }
        .modal-header h2 { margin: 0; font-size: 24px; }
        .modal-close { background: rgba(255,255,255,0.2); color: white; border: none; font-size: 28px; font-weight: bold; cursor: pointer; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; transition: all 0.2s; padding: 0; }
        .modal-close:hover { background: rgba(255,255,255,0.3); transform: rotate(90deg); }
        .modal-body { padding: 0; }
        .modal-image { width: 100%; max-height: 600px; object-fit: contain; background: #000; }
        .modal-info { padding: 30px; }
        .modal-detail-row { display: flex; padding: 15px 0; border-bottom: 1px solid #e5e7eb; }
        .modal-detail-row:last-child { border-bottom: none; }
        .modal-detail-label { font-weight: 600; color: #2563eb; min-width: 150px; font-size: 15px; }
        .modal-detail-value { color: #374151; flex: 1; font-size: 15px; }
        .modal-actions { padding: 20px 30px; background: #f9fafb; border-radius: 0 0 16px 16px; display: flex; gap: 10px; }
        .modal-actions button { flex: 1; }
        .btn-download { background: linear-gradient(135deg, #16a34a, #15803d); color: white; display: inline-flex; align-items: center; justify-content: center; gap: 8px; }
        .btn-download:hover { background: linear-gradient(135deg, #15803d, #166534); }
        @media (max-width: 768px) {
            .modal-content { margin: 5% 10px; max-width: 100%; }
            .modal-header { padding: 15px 20px; } .modal-header h2 { font-size: 20px; }
            .modal-info { padding: 20px; }
            .modal-detail-row { flex-direction: column; gap: 5px; }
            .modal-detail-label { min-width: auto; }
            .modal-actions { flex-direction: column; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚽ Soccer Clinic</h1>
            <p>AI-Powered Face Recognition Photo Platform</p>
        </div>
        
        <div id="roleSelection" class="role-selector">
            <div class="role-card" onclick="showPasswordModal()">
                <div class="icon">📸</div>
                <h2>Photographer</h2>
                <p>Upload and manage event photos</p>
            </div>
            <div class="role-card" onclick="selectRole(\'user\')">
                <div class="icon">👤</div>
                <h2>User</h2>
                <p>View your personalized photos</p>
            </div>
        </div>
        
        <div id="passwordModal" class="password-modal">
            <div class="password-modal-content">
                <h2>🔐 Photographer Login</h2>
                <p>Enter password to access photographer dashboard</p>
                <div class="password-input-group">
                    <input type="password" id="photographerPassword" placeholder="Enter password" onkeypress="handlePasswordKeypress(event)">
                    <div id="passwordError" class="password-error">❌ Incorrect password!</div>
                </div>
                <div class="password-buttons">
                    <button onclick="verifyPassword()">Login</button>
                    <button class="secondary" onclick="closePasswordModal()">Cancel</button>
                </div>
            </div>
        </div>
        
        <div id="photographerApp" class="app-container">
            <button class="back-btn" onclick="backToRoleSelection()">← Back</button>
            <h2 class="section-title">📸 Photographer Dashboard</h2>
            <div class="metadata-form">
                <div class="form-group"><label>Event Name:</label><input type="text" id="eventName" placeholder="e.g., Soccer Training Session"></div>
                <div class="form-group"><label>Location:</label><input type="text" id="eventLocation" placeholder="e.g., Main Stadium"></div>
                <div class="form-group"><label>Photographer Name:</label><input type="text" id="photographerName" placeholder="Your name"></div>
            </div>
            <div class="upload-area" id="photographerUploadArea" onclick="document.getElementById(\'photographerFileInput\').click()">
                <div style="font-size:64px;margin-bottom:15px">📷</div>
                <p style="font-size:20px;font-weight:600;margin-bottom:8px">Upload Event Photos</p>
                <p style="font-size:14px;color:#666">Click or drag & drop multiple images</p>
            </div>
            <input type="file" id="photographerFileInput" accept="image/*" multiple style="display:none" onchange="handlePhotographerUpload(event)">
            <div class="controls" id="photographerControls" style="display:none">
                <button onclick="processPhotographerPhotos()">🔍 Process & Detect Faces</button>
                <button class="secondary" onclick="clearPhotographerUploads()">Clear All</button>
            </div>
            <div id="photographerStatus" class="status" style="display:none"></div>
            <div id="photographerGallery" style="margin-top:30px">
                <h3 class="section-title">Uploaded Photos</h3>
                <div id="photographerImageGrid" class="image-grid"></div>
            </div>
        </div>
        
        <div id="userApp" class="app-container">
            <button class="back-btn" onclick="backToRoleSelection()">← Back</button>
            <h2 class="section-title">👤 Your Personal Gallery</h2>
            <div id="userRegistration" style="display:block">
                <p style="margin-bottom:20px;color:#666">To view your photos, we need to register your face first. Your face data is stored locally in your browser and never sent to the server.</p>
                <video id="userVideo" autoplay playsinline muted></video>
                <div class="controls">
                    <button onclick="startUserRegistration()">📷 Start Camera</button>
                    <button class="secondary" onclick="stopUserCamera()">Stop</button>
                    <button onclick="captureUserFace()" id="captureBtn" style="display:none">✅ Capture My Face</button>
                </div>
                <div id="userRegStatus" class="status" style="display:none"></div>
            </div>
            <div id="userGallery" style="display:none">
                <div style="background:#d4edda;padding:15px;border-radius:8px;margin-bottom:20px"><strong>✅ Face Registered!</strong> Below are your personalized photos.</div>
                <button class="secondary" onclick="resetUserFace()" style="margin-bottom:20px">🔄 Register Different Face</button>
                <div id="userImageGrid" class="image-grid"></div>
                <div id="emptyState" class="empty-state" style="display:none">
                    <div class="icon">📷</div>
                    <h3>No Photos Found</h3>
                    <p>We couldn\'t find any photos with your face yet. Check back later!</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let photographerImages = [];
        let userVideoStream = null;
        let userFaceEmbedding = null;
        const PHOTOGRAPHER_PASSWORD = "Bayansoccer123!";

        // ── Password Modal ──
        function showPasswordModal() {
            document.getElementById('passwordModal').classList.add('active');
            document.getElementById('photographerPassword').value = '';
            document.getElementById('passwordError').style.display = 'none';
            document.body.style.overflow = 'hidden';
            setTimeout(() => document.getElementById('photographerPassword').focus(), 100);
        }
        function closePasswordModal() {
            document.getElementById('passwordModal').classList.remove('active');
            document.body.style.overflow = 'auto';
        }
        function handlePasswordKeypress(event) { if (event.key === 'Enter') verifyPassword(); }
        function verifyPassword() {
            const password = document.getElementById('photographerPassword').value;
            const errorEl = document.getElementById('passwordError');
            if (password === PHOTOGRAPHER_PASSWORD) {
                closePasswordModal();
                selectRole('photographer');
            } else {
                errorEl.style.display = 'block';
                document.getElementById('photographerPassword').value = '';
                document.getElementById('photographerPassword').focus();
                const modal = document.querySelector('.password-modal-content');
                modal.style.animation = 'shake 0.5s';
                setTimeout(() => { modal.style.animation = ''; }, 500);
            }
        }

        // ── Role Selection ──
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

        // ── Photographer ──
        function handlePhotographerUpload(event) {
            const files = Array.from(event.target.files);
            files.forEach(file => {
                if (!file.type.match('image.*')) return;
                const reader = new FileReader();
                reader.onload = function(e) {
                    photographerImages.push({ data: e.target.result, name: file.name, id: Date.now() + Math.random() });
                    displayPhotographerImages();
                };
                reader.readAsDataURL(file);
            });
            document.getElementById('photographerControls').style.display = 'flex';
        }
        function displayPhotographerImages() {
            document.getElementById('photographerImageGrid').innerHTML = photographerImages.map(img => `
                <div class="image-card">
                    <img src="${img.data}" alt="${img.name}">
                    <div class="badge">Pending</div>
                    <div class="info"><strong>${img.name}</strong></div>
                </div>`).join('');
        }
        async function processPhotographerPhotos() {
            if (photographerImages.length === 0) { alert('Please upload photos first!'); return; }
            const eventName    = document.getElementById('eventName').value || 'Untitled Event';
            const location     = document.getElementById('eventLocation').value || 'Unknown Location';
            const photographer = document.getElementById('photographerName').value || 'Anonymous';
            showPhotographerStatus('🔍 Processing photos...', 'info');
            for (let i = 0; i < photographerImages.length; i++) {
                const img = photographerImages[i];
                showPhotographerStatus(`Processing ${i + 1}/${photographerImages.length}...`, 'info');
                try {
                    const response = await fetch('/api/photographer/upload', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: img.data, filename: img.name, metadata: { event_name: eventName, location, photographer, date: new Date().toISOString() } })
                    });
                    const data = await response.json();
                    if (!data.success) console.error('Upload failed:', data.error);
                } catch (err) { console.error('Error:', err); }
            }
            showPhotographerStatus('✅ All photos processed successfully!', 'success');
            photographerImages = [];
            document.getElementById('photographerControls').style.display = 'none';
            loadPhotographerGallery();
        }
        async function loadPhotographerGallery() {
            try {
                const data = await (await fetch('/api/photographer/photos')).json();
                const grid = document.getElementById('photographerImageGrid');
                if (data.photos.length === 0) {
                    grid.innerHTML = '<div class="empty-state"><div class="icon">📸</div><h3>No Photos Uploaded</h3><p>Upload some event photos to get started!</p></div>';
                    return;
                }
                grid.innerHTML = data.photos.map(photo => `
                    <div class="image-card">
                        <img src="/api/preview/${photo.filename}" alt="${photo.filename}">
                        <div class="badge">${photo.faces_count} faces</div>
                        <div class="info"><strong>${photo.metadata.event_name || 'Untitled'}</strong><br><small style="color:#666">${photo.metadata.location || ''}</small></div>
                    </div>`).join('');
            } catch (err) { console.error('Error loading gallery:', err); }
        }
        function clearPhotographerUploads() {
            photographerImages = [];
            document.getElementById('photographerImageGrid').innerHTML = '';
            document.getElementById('photographerControls').style.display = 'none';
            showPhotographerStatus('Uploads cleared', 'info');
        }
        function showPhotographerStatus(message, type) {
            const el = document.getElementById('photographerStatus');
            el.style.display = 'block'; el.className = `status ${type}`; el.textContent = message;
        }

        // Drag & Drop
        const uploadArea = document.getElementById('photographerUploadArea');
        uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
        uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault(); uploadArea.classList.remove('dragover');
            Array.from(e.dataTransfer.files).forEach(file => {
                if (file && file.type.match('image.*')) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        photographerImages.push({ data: e.target.result, name: file.name, id: Date.now() + Math.random() });
                        displayPhotographerImages();
                    };
                    reader.readAsDataURL(file);
                }
            });
            document.getElementById('photographerControls').style.display = 'flex';
        });

        // ── User ──
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
                userVideoStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 1280, height: 720 }, audio: false });
                video.srcObject = userVideoStream;
                document.getElementById('captureBtn').style.display = 'inline-block';
                showUserRegStatus('✅ Camera ready! Position your face and click Capture', 'success');
            } catch (err) { showUserRegStatus('❌ Camera error: ' + err.message, 'warning'); }
        }
        async function captureUserFace() {
            const video = document.getElementById('userVideo');
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth; canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            showUserRegStatus('🔍 Extracting face embedding...', 'info');
            try {
                const data = await (await fetch('/api/user/register_face', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imageData })
                })).json();
                if (data.success) {
                    userFaceEmbedding = data.embedding;
                    localStorage.setItem('user_face_embedding', JSON.stringify(userFaceEmbedding));
                    showUserRegStatus('✅ Face registered! Loading your photos...', 'success');
                    stopUserCamera();
                    document.getElementById('userRegistration').style.display = 'none';
                    document.getElementById('userGallery').style.display = 'block';
                    loadUserPhotos();
                } else { showUserRegStatus('❌ ' + data.error, 'warning'); }
            } catch (err) { showUserRegStatus('❌ Error: ' + err.message, 'warning'); }
        }
        async function loadUserPhotos() {
            if (!userFaceEmbedding) return;
            try {
                const data = await (await fetch('/api/user/my_photos', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ embedding: userFaceEmbedding })
                })).json();
                const grid = document.getElementById('userImageGrid');
                const emptyState = document.getElementById('emptyState');
                if (data.photos.length === 0) { grid.innerHTML = ''; emptyState.style.display = 'block'; return; }
                emptyState.style.display = 'none';
                grid.innerHTML = data.photos.map(photo => `
                    <div class="image-card" onclick="showPhotoDetail('${photo.photo_id}','${photo.filename}',${photo.distance},'${(photo.metadata.event_name||'Untitled Event').replace(/'/g,"\\'")}','${(photo.metadata.location||'Location unknown').replace(/'/g,"\\'")}','${(photo.metadata.photographer||'Unknown').replace(/'/g,"\\'")}','${photo.metadata.date}')">
                        <img src="/api/preview/${photo.filename}" alt="${photo.filename}">
                        <div class="badge">Match: ${(100 - photo.distance * 100).toFixed(0)}%</div>
                        <div class="info">
                            <strong>${photo.metadata.event_name || 'Untitled Event'}</strong><br>
                            <small style="color:#666">
                                📍 ${photo.metadata.location || 'Location unknown'}<br>
                                📅 ${new Date(photo.metadata.date).toLocaleDateString()}<br>
                                📸 ${photo.metadata.photographer || 'Unknown'}
                            </small>
                        </div>
                    </div>`).join('');
            } catch (err) { console.error('Error loading photos:', err); }
        }
        function resetUserFace() {
            if (confirm('Are you sure you want to register a different face?')) {
                localStorage.removeItem('user_face_embedding');
                userFaceEmbedding = null;
                document.getElementById('userGallery').style.display = 'none';
                document.getElementById('userRegistration').style.display = 'block';
            }
        }
        function stopUserCamera() {
            if (userVideoStream) { userVideoStream.getTracks().forEach(t => t.stop()); userVideoStream = null; }
            document.getElementById('userVideo').srcObject = null;
            document.getElementById('captureBtn').style.display = 'none';
        }
        function showUserRegStatus(message, type) {
            const el = document.getElementById('userRegStatus');
            el.style.display = 'block'; el.className = `status ${type}`; el.textContent = message;
        }

        // ── Photo Detail Modal ──
        function showPhotoDetail(photoId, filename, distance, eventName, location, photographer, date) {
            let modal = document.getElementById('photoDetailModal');
            if (!modal) { modal = document.createElement('div'); modal.id = 'photoDetailModal'; modal.className = 'modal'; document.body.appendChild(modal); }
            const matchPct      = (100 - distance * 100).toFixed(0);
            const formattedDate = new Date(date).toLocaleDateString('id-ID', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
            const formattedTime = new Date(date).toLocaleTimeString('id-ID');
            modal.innerHTML = `
                <div class="modal-content">
                    <div class="modal-header">
                        <h2>📸 Photo Details</h2>
                        <button class="modal-close" onclick="closePhotoDetail()">×</button>
                    </div>
                    <div class="modal-body">
                        <img src="/api/preview/${filename}" class="modal-image" alt="${filename}">
                        <div class="modal-info">
                            <div class="modal-detail-row"><div class="modal-detail-label">🎯 Match Score</div><div class="modal-detail-value"><strong style="color:#10b981;font-size:18px">${matchPct}%</strong></div></div>
                            <div class="modal-detail-row"><div class="modal-detail-label">⚽ Event Name</div><div class="modal-detail-value">${eventName}</div></div>
                            <div class="modal-detail-row"><div class="modal-detail-label">📍 Location</div><div class="modal-detail-value">${location}</div></div>
                            <div class="modal-detail-row"><div class="modal-detail-label">📅 Date</div><div class="modal-detail-value">${formattedDate}</div></div>
                            <div class="modal-detail-row"><div class="modal-detail-label">🕐 Time</div><div class="modal-detail-value">${formattedTime}</div></div>
                            <div class="modal-detail-row"><div class="modal-detail-label">📸 Photographer</div><div class="modal-detail-value">${photographer}</div></div>
                            <div class="modal-detail-row"><div class="modal-detail-label">📁 Filename</div><div class="modal-detail-value" style="font-family:monospace;font-size:13px">${filename}</div></div>
                        </div>
                    </div>
                    <div class="modal-actions">
                        <button id="downloadBtn" class="btn-download" onclick="downloadPhoto('${filename}')">
                            📥 Download HD
                        </button>
                        <button class="secondary" onclick="closePhotoDetail()">✕ Close</button>
                    </div>
                </div>`;
            modal.classList.add('active');
            document.body.style.overflow = 'hidden';
        }

        // ── FIX: Download via Blob ──────────────────────────────────────────
        // Fetch file sebagai blob terlebih dahulu, lalu buat object URL lokal.
        // Ini wajib karena browser TIDAK akan menghormati atribut `download`
        // untuk URL cross-origin — hasilnya selalu buka tab baru, bukan download.
        // Dengan blob URL (same-origin), browser langsung simpan ke file.
        async function downloadPhoto(filename) {
            const btn = document.getElementById('downloadBtn');
            if (!btn || btn.disabled) return;

            // Tampilkan loading state
            btn.disabled = true;
            btn.innerHTML = '<span class="spin">⏳</span> Mengunduh...';

            try {
                // Ambil file dari route /api/download/ yang pakai as_attachment=True
                const response = await fetch(`/api/download/${filename}`);
                if (!response.ok) throw new Error(`Server error: ${response.status}`);

                const blob = await response.blob();

                // Buat URL sementara dari blob (same-origin → browser mau download)
                const blobUrl = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href     = blobUrl;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);

                // Bebaskan memory setelah download terpicu
                setTimeout(() => URL.revokeObjectURL(blobUrl), 10000);

                btn.innerHTML = '✅ Download Berhasil!';
                setTimeout(() => {
                    if (btn) { btn.disabled = false; btn.innerHTML = '📥 Download HD'; }
                }, 2500);

            } catch (err) {
                console.error('Download error:', err);
                btn.innerHTML = '❌ Gagal, coba lagi';
                setTimeout(() => {
                    if (btn) { btn.disabled = false; btn.innerHTML = '📥 Download HD'; }
                }, 2500);
            }
        }
        // ───────────────────────────────────────────────────────────────────

        function closePhotoDetail() {
            const modal = document.getElementById('photoDetailModal');
            if (modal) { modal.classList.remove('active'); document.body.style.overflow = 'auto'; }
        }
        window.onclick = function(event) {
            const modal = document.getElementById('photoDetailModal');
            if (event.target === modal) closePhotoDetail();
        };
        document.addEventListener('keydown', function(event) { if (event.key === 'Escape') closePhotoDetail(); });
    </script>
</body>
</html>
'''

# ─────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/photographer/upload', methods=['POST'])
def photographer_upload():
    try:
        data          = request.json
        image_data    = data['image']
        filename      = data['filename']
        metadata      = data.get('metadata', {})

        os.makedirs('uploads', exist_ok=True)
        timestamp       = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        save_path       = f"uploads/{unique_filename}"

        nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(save_path, img)

        faces_data = face_system.detect_faces_in_image(save_path)
        photo_id   = str(uuid.uuid4())

        face_system.photo_database['photos'][photo_id] = {
            'path':        save_path,
            'filename':    unique_filename,
            'faces_data':  faces_data,
            'metadata':    metadata,
            'uploaded_at': datetime.now().isoformat()
        }
        face_system.save_database()

        # Generate preview terkompresi langsung saat upload
        # sehingga saat user request /api/preview/ sudah tersedia (tidak perlu generate on-the-fly)
        ensure_preview_exists(unique_filename)

        return jsonify({'success': True, 'photo_id': photo_id, 'faces_detected': len(faces_data)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/photographer/photos', methods=['GET'])
def get_photographer_photos():
    try:
        photos = []
        for photo_id, photo_data in face_system.photo_database['photos'].items():
            photos.append({
                'photo_id':    photo_id,
                'filename':    photo_data['filename'],
                'faces_count': len(photo_data.get('faces_data', [])),
                'metadata':    photo_data.get('metadata', {}),
                'uploaded_at': photo_data.get('uploaded_at', '')
            })
        photos.sort(key=lambda x: x['uploaded_at'], reverse=True)
        return jsonify({'success': True, 'photos': photos})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'photos': []})


@app.route('/api/user/register_face', methods=['POST', 'OPTIONS'])
@cross_origin()
def user_register_face():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data       = request.json
        image_data = data['image']

        nparr  = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
        frame  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        boxes, _ = face_system.mtcnn.detect(img_pil)
        if boxes is None or len(boxes) == 0:
            return jsonify({'success': False, 'error': 'No face detected. Please try again.'})

        box             = boxes[0]
        x1, y1, x2, y2 = [int(c) for c in box]
        x1, y1         = max(0, x1), max(0, y1)
        x2              = min(frame.shape[1], x2)
        y2              = min(frame.shape[0], y2)

        face_img   = img_pil.crop((x1, y1, x2, y2)).resize((160, 160))
        face_array = np.array(face_img).astype(np.float32)
        face_array = (face_array - 127.5) / 128.0
        face_tensor = torch.from_numpy(face_array).permute(2, 0, 1)

        embedding = face_system.extract_embedding(face_tensor)
        return jsonify({'success': True, 'embedding': embedding.tolist()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/user/my_photos', methods=['POST', 'OPTIONS'])
@cross_origin()
def get_user_photos():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        user_embedding  = request.json['embedding']
        matched_photos  = face_system.match_user_face(user_embedding)
        base_url        = request.host_url.rstrip('/')

        photos = []
        for match in matched_photos:
            photo_id   = match['photo_id']
            photo_data = face_system.photo_database['photos'].get(photo_id)
            if photo_data:
                photos.append({
                    'photo_id': photo_id,
                    'filename': photo_data['filename'],
                    'url':      f"{base_url}/uploads/{photo_data['filename']}",
                    'metadata': photo_data.get('metadata', {}),
                    'distance': match['distance']
                })
        return jsonify({'success': True, 'photos': photos})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'photos': []})


@app.route('/api/health', methods=['GET'])
@cross_origin()
def health_check():
    return jsonify({
        'status':       'ok',
        'service':      'Soccer Clinic Face Recognition',
        'total_photos': len(face_system.photo_database['photos']),
        'total_faces':  len(face_system.photo_database['face_embeddings'])
    })


@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve foto HD asli untuk ditampilkan di browser."""
    return send_from_directory('uploads', filename)


# ─────────────────────────────────────────────────────────────────────────────
# COMPRESSION UTILS
# ─────────────────────────────────────────────────────────────────────────────

def compress_image_to_target(image_path: str, target_kb: int = 80) -> bytes:
    """
    Kompres gambar secara agresif hingga di bawah target_kb kilobytes.

    Strategi bertahap:
      1. Resize bertahap jika resolusi terlalu besar (max 1280px wide)
      2. Loop turunkan JPEG quality dari 85 → 10 sampai ukuran < target
      3. Jika masih belum cukup kecil, resize lagi sampai 50% dan ulangi
    Hasil dijamin < target_kb dalam kondisi normal.
    """
    target_bytes = target_kb * 1024

    img = Image.open(image_path).convert('RGB')

    # Langkah 1: resize awal — max lebar/tinggi 1280px
    max_dim = 1280
    w, h = img.size
    if w > max_dim or h > max_dim:
        ratio = min(max_dim / w, max_dim / h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    # Langkah 2: coba compress dengan quality turun bertahap
    for quality in range(85, 5, -5):
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality, optimize=True)
        if buf.tell() <= target_bytes:
            buf.seek(0)
            return buf.read()

    # Langkah 3: masih belum cukup kecil → resize paksa 50% lagi lalu compress
    w, h = img.size
    img = img.resize((max(w // 2, 100), max(h // 2, 100)), Image.LANCZOS)
    for quality in range(60, 5, -5):
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality, optimize=True)
        if buf.tell() <= target_bytes:
            buf.seek(0)
            return buf.read()

    # Fallback terakhir — kembalikan apa adanya
    buf.seek(0)
    return buf.read()


def get_preview_path(filename: str) -> str:
    """Kembalikan path file preview di folder uploads/previews/."""
    preview_dir = os.path.join('uploads', 'previews')
    os.makedirs(preview_dir, exist_ok=True)
    name, _ = os.path.splitext(filename)
    return os.path.join(preview_dir, f"{name}_preview.jpg")


def ensure_preview_exists(filename: str) -> str:
    """
    Pastikan file preview sudah ada.
    Jika belum, generate dari file HD dan simpan.
    Kembalikan path preview.
    """
    hd_path      = os.path.join('uploads', filename)
    preview_path = get_preview_path(filename)

    if not os.path.exists(preview_path):
        if not os.path.exists(hd_path):
            return None
        data = compress_image_to_target(hd_path, target_kb=80)
        with open(preview_path, 'wb') as f:
            f.write(data)
        size_kb = len(data) / 1024
        print(f"🗜️  Preview generated: {os.path.basename(preview_path)} ({size_kb:.1f} KB)")

    return preview_path


# ── Route: Preview terkompresi (< 80 KB) ─────────────────────────────────────
@app.route('/api/preview/<filename>')
@cross_origin()
def serve_preview(filename):
    """
    Serve versi terkompresi foto untuk preview di grid/modal.
    Ukuran target < 80 KB — cocok untuk load cepat di mobile.
    File preview di-cache di uploads/previews/ agar tidak di-generate ulang.
    """
    preview_path = ensure_preview_exists(filename)
    if not preview_path or not os.path.exists(preview_path):
        # Fallback ke file asli jika preview gagal di-generate
        return send_from_directory('uploads', filename)

    return send_file(
        preview_path,
        mimetype='image/jpeg',
        max_age=86400  # cache 1 hari di browser
    )


# ── Route: Download HD (file asli, paksa download) ───────────────────────────
@app.route('/api/download/<filename>')
@cross_origin()
def download_file(filename):
    """
    Download file HD asli dengan Content-Disposition: attachment.
    Browser akan langsung menyimpan file, bukan membukanya.
    """
    return send_from_directory(
        'uploads',
        filename,
        as_attachment=True
    )
# ─────────────────────────────────────────────────────────────────────────────


def generate_self_signed_cert():
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
    cert.get_subject().C  = "ID"
    cert.get_subject().ST = "East Kalimantan"
    cert.get_subject().L  = "Samarinda"
    cert.get_subject().O  = "Soccer Clinic"
    cert.get_subject().OU = "AI Face Recognition"
    cert.get_subject().CN = "localhost"
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365 * 24 * 60 * 60)
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

    os.makedirs('uploads', exist_ok=True)
    print("📁 Folder ready: uploads/")

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    print("\n" + "=" * 70)
    print("⚽ SOCCER CLINIC - AI FACE RECOGNITION PLATFORM")
    print("=" * 70)

    use_https = input("Use HTTPS? (y/n) [recommended: n for local network]: ").lower() == 'y'

    if use_https:
        has_ssl = generate_self_signed_cert()
        if has_ssl:
            print(f"🌐 HTTPS - Laptop/Desktop: https://localhost:5000")
            print(f"📱 HTTPS - Mobile (same WiFi): https://{local_ip}:5000")
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain('cert.pem', 'key.pem')
            app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, ssl_context=context)
        else:
            print("⚠️  PyOpenSSL not installed. Falling back to HTTP...")
            use_https = False

    if not use_https:
        print(f"🌐 HTTP - Laptop/Desktop: http://localhost:5000")
        print(f"📱 HTTP - Mobile (same WiFi): http://{local_ip}:5000")
        print("=" * 70)
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)