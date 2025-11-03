"""
Face Recognition Web App dengan HTTPS Support untuk iOS/iPhone
Kompatibel dengan Safari dan semua browser mobile

INSTALASI:
pip install flask flask-cors torch torchvision facenet-pytorch opencv-python pillow numpy pyopenssl

CARA PAKAI:
1. Jalankan: python app_https.py
2. Install sertifikat SSL (sekali saja)
3. Akses dari iPhone: https://192.168.x.x:5000

CATATAN PENTING:
- iPhone/Safari WAJIB pakai HTTPS untuk akses kamera
- Akan ada warning "Not Secure", klik "Advanced" → "Proceed"
"""

from flask import Flask, render_template_string, request, jsonify, Response
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

app = Flask(__name__)
CORS(app)

# ... [Class FaceRecognitionWeb sama seperti sebelumnya] ...
class FaceRecognitionWeb:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Device: {self.device}")
        
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=0, 
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709, 
            post_process=True,
            device=self.device
        )
        
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        self.database = {
            'users': {},
            'clusters': {},
            'pending_clusters': {},
            'training_log': []
        }
        
        self.next_cluster_id = 1
        self.threshold = 0.6
        self.load_database()
    
    def extract_embedding(self, face_img):
        with torch.no_grad():
            face_img = face_img.to(self.device)
            embedding = self.resnet(face_img.unsqueeze(0))
        return embedding.cpu().numpy().flatten()
    
    def process_frame(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        boxes, probs = self.mtcnn.detect(img_pil)
        
        results = []
        if boxes is not None:
            for i, box in enumerate(boxes):
                try:
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    
                    face_img = img_pil.crop((x1, y1, x2, y2))
                    face_img = face_img.resize((160, 160))
                    
                    face_array = np.array(face_img).astype(np.float32)
                    face_array = (face_array - 127.5) / 128.0
                    face_tensor = torch.from_numpy(face_array).permute(2, 0, 1)
                    
                    embedding = self.extract_embedding(face_tensor)
                    best_match = self._find_best_match(embedding)
                    
                    result = {
                        'box': [x1, y1, x2, y2],
                        'confidence': float(probs[i]),
                        'embedding': embedding.tolist()
                    }
                    
                    if best_match:
                        result['name'] = best_match['name']
                        result['user_id'] = best_match['user_id']
                        result['distance'] = float(best_match['distance'])
                        result['status'] = 'recognized'
                        color = [0, 255, 0]
                    else:
                        result['name'] = 'Unknown'
                        result['status'] = 'unknown'
                        color = [255, 165, 0]
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    label = result['name']
                    if 'distance' in result:
                        label += f" ({result['distance']:.2f})"
                    
                    cv2.putText(frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error processing face: {e}")
                    continue
        
        return frame, results
    
    def _find_best_match(self, embedding):
        best_match = None
        best_distance = float('inf')
        
        for cluster_id, cluster_data in self.database['clusters'].items():
            if not cluster_data.get('verified', False):
                continue
            
            mean_emb = cluster_data['mean_embedding']
            distance = np.linalg.norm(embedding - mean_emb)
            
            if distance < best_distance and distance < self.threshold:
                best_distance = distance
                user_id = cluster_data.get('user_id')
                user_name = self.database['users'].get(user_id, {}).get('name', 'Unknown')
                
                best_match = {
                    'cluster_id': cluster_id,
                    'user_id': user_id,
                    'name': user_name,
                    'distance': distance
                }
        
        return best_match
    
    def save_pending_face(self, embedding, image_data):
        cluster_id = f"pending_{self.next_cluster_id}"
        self.next_cluster_id += 1
        
        img_path = f"pending_faces/pending_{self.next_cluster_id}.jpg"
        os.makedirs('pending_faces', exist_ok=True)
        
        nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(img_path, img)
        
        self.database['pending_clusters'][cluster_id] = {
            'embeddings': [embedding],
            'mean_embedding': embedding,
            'image_path': img_path,
            'detected_at': datetime.now().isoformat()
        }
        
        self.save_database()
        return cluster_id
    
    def register_user_biometric(self, user_id, user_name, embeddings_list):
        cluster_ids = []
        
        for embedding in embeddings_list:
            cluster_id = self.next_cluster_id
            self.next_cluster_id += 1
            
            self.database['clusters'][cluster_id] = {
                'embeddings': [embedding],
                'mean_embedding': embedding,
                'verified': True,
                'user_id': user_id,
                'created_at': datetime.now().isoformat()
            }
            cluster_ids.append(cluster_id)
        
        self.database['users'][user_id] = {
            'name': user_name,
            'cluster_ids': cluster_ids,
            'created_at': datetime.now().isoformat()
        }
        
        self.save_database()
        return len(cluster_ids)
    
    def verify_pending_cluster(self, cluster_id, user_id, user_name=None):
        if cluster_id not in self.database['pending_clusters']:
            return False
        
        pending_data = self.database['pending_clusters'][cluster_id]
        
        if user_id not in self.database['users']:
            if not user_name:
                return False
            self.database['users'][user_id] = {
                'name': user_name,
                'cluster_ids': [],
                'created_at': datetime.now().isoformat()
            }
        
        new_cluster_id = self.next_cluster_id
        self.next_cluster_id += 1
        
        self.database['clusters'][new_cluster_id] = {
            'embeddings': pending_data['embeddings'],
            'mean_embedding': pending_data['mean_embedding'],
            'verified': True,
            'user_id': user_id,
            'verified_at': datetime.now().isoformat()
        }
        
        self.database['users'][user_id]['cluster_ids'].append(new_cluster_id)
        del self.database['pending_clusters'][cluster_id]
        
        self._retrain_user(user_id)
        self.save_database()
        return True
    
    def _retrain_user(self, user_id):
        user_data = self.database['users'].get(user_id)
        if not user_data:
            return
        
        cluster_ids = user_data['cluster_ids']
        all_embeddings = []
        
        for cid in cluster_ids:
            cluster = self.database['clusters'].get(cid)
            if cluster:
                all_embeddings.extend(cluster['embeddings'])
        
        if all_embeddings:
            mean_embedding = np.mean(all_embeddings, axis=0)
            
            for cid in cluster_ids:
                if cid in self.database['clusters']:
                    self.database['clusters'][cid]['mean_embedding'] = mean_embedding
    
    def save_database(self):
        with open('face_database_web.pkl', 'wb') as f:
            pickle.dump(self.database, f)
    
    def load_database(self):
        if os.path.exists('face_database_web.pkl'):
            with open('face_database_web.pkl', 'rb') as f:
                self.database = pickle.load(f)
            
            all_ids = list(self.database['clusters'].keys())
            if all_ids:
                numeric_ids = [int(str(x).replace('pending_', '')) 
                              for x in all_ids if str(x).replace('pending_', '').isdigit()]
                if numeric_ids:
                    self.next_cluster_id = max(numeric_ids) + 1
            
            print(f"✅ Database loaded: {len(self.database['users'])} users")
        else:
            print("📂 New database created")

face_system = FaceRecognitionWeb()

# HTML Template dengan iOS Camera Fallback
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <title>Face Recognition System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 10px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }
        
        .header p {
            font-size: 14px;
            opacity: 0.9;
        }
        
        .ios-warning {
            background: #fff3cd;
            color: #856404;
            padding: 15px;
            margin: 15px;
            border-radius: 10px;
            border-left: 4px solid #ffc107;
            display: none;
        }
        
        .tabs {
            display: flex;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
        }
        
        .tab {
            flex: 1;
            padding: 15px;
            text-align: center;
            cursor: pointer;
            background: none;
            border: none;
            font-size: 14px;
            font-weight: 600;
            color: #6c757d;
            transition: all 0.3s;
        }
        
        .tab.active {
            color: #667eea;
            background: white;
            border-bottom: 3px solid #667eea;
        }
        
        .content {
            padding: 20px;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        #video, #enrollVideo {
            width: 100%;
            max-width: 100%;
            border-radius: 15px;
            background: #000;
            aspect-ratio: 4/3;
            object-fit: cover;
        }
        
        /* iOS Safari specific fixes */
        video::-webkit-media-controls-play-button {
            display: none !important;
        }
        
        video::-webkit-media-controls-start-playback-button {
            display: none !important;
        }
        
        .controls {
            margin-top: 15px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        button {
            flex: 1;
            min-width: 120px;
            padding: 15px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            -webkit-tap-highlight-color: transparent;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-success {
            background: #28a745;
            color: white;
        }
        
        .btn-warning {
            background: #ffc107;
            color: #333;
        }
        
        .btn-danger {
            background: #dc3545;
            color: white;
        }
        
        button:active {
            transform: scale(0.95);
        }
        
        .status {
            margin-top: 15px;
            padding: 15px;
            border-radius: 10px;
            font-size: 14px;
        }
        
        .status.info {
            background: #d1ecf1;
            color: #0c5460;
        }
        
        .status.success {
            background: #d4edda;
            color: #155724;
        }
        
        .status.warning {
            background: #fff3cd;
            color: #856404;
        }
        
        .instruction {
            margin-top: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        
        .instruction h3 {
            font-size: 18px;
            margin-bottom: 10px;
            color: #667eea;
        }
        
        .instruction-step {
            padding: 10px;
            margin: 5px 0;
            background: white;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            text-align: center;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #333;
        }
        
        input, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            font-size: 16px;
        }
        
        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .pending-list {
            margin-top: 15px;
        }
        
        .pending-item {
            background: #f8f9fa;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 10px;
            border-left: 4px solid #ffc107;
        }
        
        .pending-item img {
            width: 100%;
            border-radius: 8px;
            margin: 10px 0;
        }
        
        .user-list {
            margin-top: 15px;
        }
        
        .user-item {
            background: #f8f9fa;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .badge {
            background: #667eea;
            color: white;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 12px;
        }
        
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Face Recognition System</h1>
            <p>iOS & Android Compatible</p>
        </div>
        
        <div class="ios-warning" id="iosWarning">
            <strong>⚠️ iOS/Safari Users:</strong> Pastikan Anda mengakses via <strong>HTTPS</strong> bukan HTTP. Jika ada warning keamanan, klik "Advanced" → "Proceed".
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('recognize')">🎥 Recognize</button>
            <button class="tab" onclick="showTab('enroll')">📝 Enroll</button>
            <button class="tab" onclick="showTab('review')">✅ Review</button>
            <button class="tab" onclick="showTab('users')">👥 Users</button>
        </div>
        
        <div class="content">
            <!-- TAB 1: RECOGNIZE -->
            <div id="recognize" class="tab-content active">
                <video id="video" autoplay playsinline muted></video>
                <div class="controls">
                    <button class="btn-primary" onclick="startCamera()">▶️ Start Camera</button>
                    <button class="btn-warning" onclick="savePendingFace()">💾 Save Unknown</button>
                    <button class="btn-danger" onclick="stopCamera()">⏹️ Stop</button>
                </div>
                <div id="recognizeStatus" class="status info" style="display:none;"></div>
            </div>
            
            <!-- TAB 2: ENROLL -->
            <div id="enroll" class="tab-content">
                <div class="form-group">
                    <label>User ID:</label>
                    <input type="text" id="enrollUserId" placeholder="Contoh: user001">
                </div>
                <div class="form-group">
                    <label>Nama:</label>
                    <input type="text" id="enrollUserName" placeholder="Contoh: John Doe">
                </div>
                
                <div class="instruction">
                    <h3>📋 Instruksi Enrollment</h3>
                    <p style="margin-bottom: 10px;">Ikuti 8 gerakan berikut:</p>
                    <div id="enrollInstruction" class="instruction-step">Tekan Start untuk mulai</div>
                    <div class="progress-bar">
                        <div id="enrollProgress" class="progress-fill" style="width: 0%;">0%</div>
                    </div>
                </div>
                
                <video id="enrollVideo" autoplay playsinline muted style="display:none;"></video>
                
                <div class="controls">
                    <button class="btn-success" onclick="startEnrollment()">🎬 Start Enrollment</button>
                    <button class="btn-danger" onclick="stopEnrollment()">⏹️ Stop</button>
                </div>
                <div id="enrollStatus" class="status info" style="display:none;"></div>
            </div>
            
            <!-- TAB 3: REVIEW -->
            <div id="review" class="tab-content">
                <h3>📋 Pending Verifikasi</h3>
                <button class="btn-primary" onclick="loadPendingClusters()" style="width:100%; margin-bottom:15px;">🔄 Refresh</button>
                <div id="pendingList" class="pending-list"></div>
            </div>
            
            <!-- TAB 4: USERS -->
            <div id="users" class="tab-content">
                <h3>👥 Registered Users</h3>
                <button class="btn-primary" onclick="loadUsers()" style="width:100%; margin-bottom:15px;">🔄 Refresh</button>
                <div id="userList" class="user-list"></div>
            </div>
        </div>
    </div>
    
    <script>
        // Detect iOS
        const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
        if (isIOS) {
            document.getElementById('iosWarning').style.display = 'block';
        }
        
        let videoStream = null;
        let recognizeInterval = null;
        let enrollmentActive = false;
        let enrollmentData = {
            embeddings: [],
            currentStep: 0,
            instructions: [
                {text: "Lihat lurus ke kamera", frames: 15},
                {text: "Putar kepala ke KIRI", frames: 10},
                {text: "Putar kepala ke KANAN", frames: 10},
                {text: "Putar kepala ke ATAS", frames: 10},
                {text: "Putar kepala ke BAWAH", frames: 10},
                {text: "Kedipkan mata 3x", frames: 10},
                {text: "Senyum 😊", frames: 10},
                {text: "Ekspresi normal", frames: 10}
            ],
            frameCount: 0
        };
        
        function showTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
            
            if (tabName === 'review') loadPendingClusters();
            if (tabName === 'users') loadUsers();
        }
        
        async function startCamera() {
            try {
                const video = document.getElementById('video');
                
                // iOS-friendly constraints
                const constraints = {
                    video: { 
                        facingMode: 'user',
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                    },
                    audio: false
                };
                
                videoStream = await navigator.mediaDevices.getUserMedia(constraints);
                video.srcObject = videoStream;
                
                // Force play on iOS
                video.play().catch(e => {
                    console.log('Autoplay prevented:', e);
                    showStatus('recognizeStatus', 'Tap video to start', 'warning');
                });
                
                recognizeInterval = setInterval(recognizeFace, 1500); // Slower for iOS
                showStatus('recognizeStatus', '✅ Camera started!', 'success');
            } catch (err) {
                console.error('Camera error:', err);
                showStatus('recognizeStatus', '❌ Error: ' + err.message + ' (Pastikan pakai HTTPS!)', 'warning');
            }
        }
        
        function stopCamera() {
            if (videoStream) {
                videoStream.getTracks().forEach(track => track.stop());
                videoStream = null;
            }
            if (recognizeInterval) {
                clearInterval(recognizeInterval);
            }
            document.getElementById('video').srcObject = null;
            showStatus('recognizeStatus', 'Camera stopped', 'info');
        }
        
        async function recognizeFace() {
            const video = document.getElementById('video');
            if (!video.videoWidth) return; // Video not ready
            
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);
            
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            try {
                const response = await fetch('/api/recognize', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({image: imageData})
                });
                
                const data = await response.json();
                
                if (data.faces && data.faces.length > 0) {
                    const face = data.faces[0];
                    if (face.status === 'recognized') {
                        showStatus('recognizeStatus', `✅ ${face.name} (${face.distance.toFixed(2)})`, 'success');
                    } else {
                        showStatus('recognizeStatus', '⚠️ Unknown face detected', 'warning');
                    }
                } else {
                    showStatus('recognizeStatus', 'No face detected', 'info');
                }
            } catch (err) {
                console.error('Error:', err);
            }
        }
        
        async function savePendingFace() {
            const video = document.getElementById('video');
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);
            
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            try {
                const response = await fetch('/api/save_pending', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({image: imageData})
                });
                
                const data = await response.json();
                
                if (data.success) {
                    showStatus('recognizeStatus', `✅ Saved as ${data.cluster_id}`, 'success');
                } else {
                    showStatus('recognizeStatus', '❌ Failed to save', 'warning');
                }
            } catch (err) {
                showStatus('recognizeStatus', 'Error: ' + err.message, 'warning');
            }
        }
        
        async function startEnrollment() {
            const userId = document.getElementById('enrollUserId').value;
            const userName = document.getElementById('enrollUserName').value;
            
            if (!userId || !userName) {
                showStatus('enrollStatus', 'Please fill User ID and Name!', 'warning');
                return;
            }
            
            enrollmentData.embeddings = [];
            enrollmentData.currentStep = 0;
            enrollmentData.frameCount = 0;
            enrollmentActive = true;
            
            try {
                const video = document.getElementById('enrollVideo');
                video.style.display = 'block';
                
                const constraints = {
                    video: { 
                        facingMode: 'user',
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                    },
                    audio: false
                };
                
                videoStream = await navigator.mediaDevices.getUserMedia(constraints);
                video.srcObject = videoStream;
                
                video.play().catch(e => console.log('Autoplay prevented:', e));
                
                updateEnrollmentUI();
                captureEnrollmentFrames();
            } catch (err) {
                showStatus('enrollStatus', '❌ Error: ' + err.message + ' (Pastikan pakai HTTPS!)', 'warning');
            }
        }
        
        async function captureEnrollmentFrames() {
            if (!enrollmentActive) return;
            
            const video = document.getElementById('enrollVideo');
            if (!video.videoWidth) {
                setTimeout(captureEnrollmentFrames, 100);
                return;
            }
            
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);
            
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            if (enrollmentData.frameCount % 3 === 0) {
                try {
                    const response = await fetch('/api/extract_embedding', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({image: imageData})
                    });
                    
                    const data = await response.json();
                    if (data.embedding) {
                        enrollmentData.embeddings.push(data.embedding);
                    }
                } catch (err) {
                    console.error('Error:', err);
                }
            }
            
            enrollmentData.frameCount++;
            
            const currentInstruction = enrollmentData.instructions[enrollmentData.currentStep];
            if (enrollmentData.frameCount >= currentInstruction.frames) {
                enrollmentData.currentStep++;
                enrollmentData.frameCount = 0;
                
                if (enrollmentData.currentStep >= enrollmentData.instructions.length) {
                    await completeEnrollment();
                    return;
                }
            }
            
            updateEnrollmentUI();
            setTimeout(captureEnrollmentFrames, 100);
        }
        
        function updateEnrollmentUI() {
            const instruction = enrollmentData.instructions[enrollmentData.currentStep];
            document.getElementById('enrollInstruction').textContent = 
                `${enrollmentData.currentStep + 1}/8: ${instruction.text}`;
            
            const progress = ((enrollmentData.currentStep / enrollmentData.instructions.length) * 100).toFixed(0);
            const progressBar = document.getElementById('enrollProgress');
            progressBar.style.width = progress + '%';
            progressBar.textContent = progress + '%';
        }
        
        async function completeEnrollment() {
            enrollmentActive = false;
            stopEnrollment();
            
            const userId = document.getElementById('enrollUserId').value;
            const userName = document.getElementById('enrollUserName').value;
            
            try {
                const response = await fetch('/api/register_user', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        user_id: userId,
                        user_name: userName,
                        embeddings: enrollmentData.embeddings
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    showStatus('enrollStatus', 
                        `✅ User ${userName} registered with ${data.clusters} clusters!`, 'success');
                } else {
                    showStatus('enrollStatus', '❌ Registration failed', 'warning');
                }
            } catch (err) {
                showStatus('enrollStatus', 'Error: ' + err.message, 'warning');
            }
        }
        
        function stopEnrollment() {
            enrollmentActive = false;
            if (videoStream) {
                videoStream.getTracks().forEach(track => track.stop());
                videoStream = null;
            }
            document.getElementById('enrollVideo').style.display = 'none';
            document.getElementById('enrollVideo').srcObject = null;
        }
        
        async function loadPendingClusters() {
            try {
                const response = await fetch('/api/pending_clusters');
                const data = await response.json();
                
                const pendingList = document.getElementById('pendingList');
                
                if (data.pending.length === 0) {
                    pendingList.innerHTML = '<p style="text-align:center; color:#6c757d;">No pending clusters</p>';
                    return;
                }
                
                pendingList.innerHTML = data.pending.map(item => `
                    <div class="pending-item">
                        <strong>Cluster ID:</strong> ${item.cluster_id}<br>
                        <strong>Detected:</strong> ${new Date(item.detected_at).toLocaleString()}<br>
                        <img src="${item.image_url}" alt="Face">
                        
                        <div class="form-group" style="margin-top:10px;">
                            <label>Assign to User:</label>
                            <select id="user_${item.cluster_id}" style="width:100%; padding:10px; border-radius:8px; border:2px solid #e9ecef;">
                                <option value="">-- Select User --</option>
                                ${data.users.map(u => `<option value="${u.id}">${u.name} (${u.id})</option>`).join('')}
                                <option value="NEW_USER">➕ New User</option>
                            </select>
                        </div>
                        
                        <div id="newUserForm_${item.cluster_id}" style="display:none; margin-top:10px;">
                            <input type="text" id="newUserId_${item.cluster_id}" placeholder="User ID" style="width:100%; margin-bottom:5px; padding:10px; border-radius:8px;">
                            <input type="text" id="newUserName_${item.cluster_id}" placeholder="Name" style="width:100%; padding:10px; border-radius:8px;">
                        </div>
                        
                        <button onclick="verifyCluster('${item.cluster_id}')" class="btn-success" style="width:100%; margin-top:10px;">
                            ✅ Verify
                        </button>
                    </div>
                `).join('');
                
                data.pending.forEach(item => {
                    const select = document.getElementById(`user_${item.cluster_id}`);
                    select.addEventListener('change', (e) => {
                        const form = document.getElementById(`newUserForm_${item.cluster_id}`);
                        form.style.display = e.target.value === 'NEW_USER' ? 'block' : 'none';
                    });
                });
                
            } catch (err) {
                console.error('Error:', err);
            }
        }
        
        async function verifyCluster(clusterId) {
            const userSelect = document.getElementById(`user_${clusterId}`);
            let userId = userSelect.value;
            let userName = null;
            
            if (!userId) {
                alert('Please select a user!');
                return;
            }
            
            if (userId === 'NEW_USER') {
                userId = document.getElementById(`newUserId_${clusterId}`).value;
                userName = document.getElementById(`newUserName_${clusterId}`).value;
                
                if (!userId || !userName) {
                    alert('Please fill User ID and Name!');
                    return;
                }
            }
            
            try {
                const response = await fetch('/api/verify_cluster', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        cluster_id: clusterId,
                        user_id: userId,
                        user_name: userName
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    alert('✅ Cluster verified successfully!');
                    loadPendingClusters();
                } else {
                    alert('❌ Verification failed');
                }
            } catch (err) {
                alert('Error: ' + err.message);
            }
        }
        
        async function loadUsers() {
            try {
                const response = await fetch('/api/users');
                const data = await response.json();
                
                const userList = document.getElementById('userList');
                
                if (data.users.length === 0) {
                    userList.innerHTML = '<p style="text-align:center; color:#6c757d;">No users registered</p>';
                    return;
                }
                
                userList.innerHTML = data.users.map(user => `
                    <div class="user-item">
                        <div>
                            <strong>${user.name}</strong><br>
                            <small style="color:#6c757d;">ID: ${user.id}</small>
                        </div>
                        <span class="badge">${user.clusters} clusters</span>
                    </div>
                `).join('');
                
            } catch (err) {
                console.error('Error:', err);
            }
        }
        
        function showStatus(elementId, message, type) {
            const el = document.getElementById(elementId);
            el.style.display = 'block';
            el.className = `status ${type}`;
            el.textContent = message;
        }
    </script>
</body>
</html>
'''

# Flask Routes (sama seperti sebelumnya)
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    try:
        data = request.json
        image_data = data['image']
        nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        processed_frame, results = face_system.process_frame(frame)
        return jsonify({'success': True, 'faces': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/extract_embedding', methods=['POST'])
def api_extract_embedding():
    try:
        data = request.json
        image_data = data['image']
        nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        boxes, _ = face_system.mtcnn.detect(img_pil)
        
        if boxes is not None and len(boxes) > 0:
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
            return jsonify({'success': True, 'embedding': embedding.tolist()})
        else:
            return jsonify({'success': False, 'error': 'No face detected'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/save_pending', methods=['POST'])
def api_save_pending():
    try:
        data = request.json
        image_data = data['image']
        nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        boxes, _ = face_system.mtcnn.detect(img_pil)
        
        if boxes is not None and len(boxes) > 0:
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
            cluster_id = face_system.save_pending_face(embedding, image_data)
            return jsonify({'success': True, 'cluster_id': cluster_id})
        else:
            return jsonify({'success': False, 'error': 'No face detected'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/register_user', methods=['POST'])
def api_register_user():
    try:
        data = request.json
        user_id = data['user_id']
        user_name = data['user_name']
        embeddings = [np.array(emb) for emb in data['embeddings']]
        clusters_count = face_system.register_user_biometric(user_id, user_name, embeddings)
        return jsonify({'success': True, 'clusters': clusters_count})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/pending_clusters')
def api_pending_clusters():
    try:
        pending = []
        for cluster_id, cluster_data in face_system.database['pending_clusters'].items():
            try:
                with open(cluster_data['image_path'], 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode()
                    image_url = f"data:image/jpeg;base64,{img_data}"
            except:
                image_url = ""
            
            pending.append({
                'cluster_id': cluster_id,
                'detected_at': cluster_data['detected_at'],
                'image_url': image_url
            })
        
        users = [
            {'id': uid, 'name': udata['name']} 
            for uid, udata in face_system.database['users'].items()
        ]
        
        return jsonify({'pending': pending, 'users': users})
    except Exception as e:
        return jsonify({'pending': [], 'users': [], 'error': str(e)})

@app.route('/api/verify_cluster', methods=['POST'])
def api_verify_cluster():
    try:
        data = request.json
        cluster_id = data['cluster_id']
        user_id = data['user_id']
        user_name = data.get('user_name')
        success = face_system.verify_pending_cluster(cluster_id, user_id, user_name)
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/users')
def api_users():
    try:
        users = [
            {
                'id': uid,
                'name': udata['name'],
                'clusters': len(udata['cluster_ids'])
            }
            for uid, udata in face_system.database['users'].items()
        ]
        return jsonify({'users': users})
    except Exception as e:
        return jsonify({'users': [], 'error': str(e)})

def generate_self_signed_cert():
    """Generate self-signed certificate untuk HTTPS"""
    from OpenSSL import crypto
    
    if os.path.exists('cert.pem') and os.path.exists('key.pem'):
        print("✅ SSL Certificate sudah ada")
        return
    
    print("🔐 Generating self-signed SSL certificate...")
    
    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 2048)
    
    cert = crypto.X509()
    cert.get_subject().C = "ID"
    cert.get_subject().ST = "East Java"
    cert.get_subject().L = "Surabaya"
    cert.get_subject().O = "Face Recognition"
    cert.get_subject().OU = "Face Recognition System"
    cert.get_subject().CN = "localhost"
    
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365*24*60*60)  # Valid 1 tahun
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, 'sha256')
    
    with open("cert.pem", "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    
    with open("key.pem", "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
    
    print("✅ SSL Certificate created: cert.pem & key.pem")

if __name__ == '__main__':
    import socket
    
    # Generate SSL certificate
    try:
        generate_self_signed_cert()
    except ImportError:
        print("⚠️  PyOpenSSL tidak terinstall. Install dengan: pip install pyopenssl")
        print("⚠️  Running tanpa HTTPS (tidak akan jalan di iPhone!)")
        
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        print("\n" + "="*60)
        print("🚀 FACE RECOGNITION WEB APP (HTTP ONLY)")
        print("="*60)
        print(f"🌐 Akses dari laptop: http://localhost:5000")
        print(f"📱 Akses dari Android: http://{local_ip}:5000")
        print(f"⚠️  iPhone TIDAK AKAN JALAN tanpa HTTPS!")
        print("="*60 + "\n")
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        exit()
    
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("\n" + "="*60)
    print("🚀 FACE RECOGNITION WEB APP (HTTPS)")
    print("="*60)
    print(f"🌐 Akses dari laptop: https://localhost:5000")
    print(f"📱 Akses dari iPhone: https://{local_ip}:5000")
    print("="*60)
    print("💡 PENTING untuk iPhone/iOS:")
    print("   1. WAJIB pakai HTTPS (bukan HTTP)")
    print("   2. Browser akan warning 'Not Secure'")
    print("   3. Klik 'Advanced' → 'Proceed to {local_ip}'")
    print("   4. Izinkan akses kamera saat diminta")
    print("="*60)
    print("📱 Pastikan laptop dan HP di WiFi yang SAMA!")
    print("="*60 + "\n")
    
    # Create SSL context
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain('cert.pem', 'key.pem')
    
    # Run Flask dengan HTTPS
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, ssl_context=context)