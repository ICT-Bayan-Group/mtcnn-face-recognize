from flask import Flask, render_template_string, request, jsonify, send_file
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

app = Flask(__name__)
CORS(app)

class FaceRecognitionWeb:
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
        
        self.database = {
            'users': {},
            'clusters': {},
            'pending_clusters': {},
            'training_log': []
        }
        
        self.next_cluster_id = 1
        self.threshold = 0.7
        self.strict_threshold = 0.5
        self.loose_threshold = 0.9
        
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
                    else:
                        result['name'] = 'Unknown'
                        result['status'] = 'unknown'
                    
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
    
    def enroll_from_folder(self, user_id, user_name, folder_path='faces'):
        if not os.path.exists(folder_path):
            return {'success': False, 'error': f'Folder {folder_path} tidak ditemukan'}
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        if not image_files:
            return {'success': False, 'error': f'Tidak ada gambar di folder {folder_path}'}
        
        embeddings_list = []
        processed_count = 0
        
        for img_path in image_files:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                
                boxes, _ = self.mtcnn.detect(img_pil)
                
                if boxes is not None:
                    for box in boxes:
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
                        embeddings_list.append(embedding)
                        processed_count += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        if not embeddings_list:
            return {'success': False, 'error': 'Tidak ada wajah terdeteksi di gambar'}
        
        clusters_count = self.register_user_biometric(user_id, user_name, embeddings_list)
        
        return {
            'success': True,
            'clusters': clusters_count,
            'images_processed': processed_count,
            'total_images': len(image_files)
        }
    
    def save_uploaded_image(self, image_data, filename):
        """Save uploaded image to upload-foto folder"""
        os.makedirs('upload-foto', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f"upload-foto/{timestamp}_{filename}"
        
        nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(save_path, img)
        
        return save_path
    
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

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <title>Face Recognition System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #ffffff;
            min-height: 100vh;
            color: #333;
        }
        
        .container { max-width: 1000px; margin: 0 auto; padding: 20px; }
        
        .header {
            text-align: center;
            padding: 30px 20px;
            border-bottom: 2px solid #f0f0f0;
            margin-bottom: 30px;
        }
        
        .header h1 { font-size: 28px; font-weight: 700; color: #333; margin-bottom: 8px; }
        .header p { font-size: 14px; color: #666; }
        
        .tabs {
            display: flex;
            gap: 10px;
            border-bottom: 2px solid #f0f0f0;
            margin-bottom: 30px;
            overflow-x: auto;
        }
        
        .tab {
            padding: 12px 24px;
            background: none;
            border: none;
            font-size: 15px;
            font-weight: 600;
            color: #666;
            cursor: pointer;
            transition: all 0.3s;
            border-bottom: 3px solid transparent;
            white-space: nowrap;
        }
        
        .tab.active { color: #333; border-bottom-color: #333; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        video, canvas, .canvas-container {
            width: 100%;
            max-width: 100%;
            border-radius: 8px;
            background: #000;
            border: 1px solid #e0e0e0;
        }
        
        .canvas-container {
            position: relative;
            display: inline-block;
            width: 100%;
        }
        
        .canvas-container canvas {
            display: block;
            width: 100%;
            height: auto;
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
            padding: 12px 20px;
            border: 2px solid #333;
            border-radius: 6px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            background: #fff;
            color: #333;
        }
        
        button:hover { background: #333; color: #fff; }
        button:active { transform: scale(0.98); }
        .btn-primary { background: #333; color: #fff; }
        .btn-primary:hover { background: #555; }
        
        .status {
            margin-top: 15px;
            padding: 15px;
            border-radius: 6px;
            font-size: 14px;
            border: 1px solid #e0e0e0;
        }
        
        .status.info { background: #f8f9fa; color: #495057; }
        .status.success { background: #e8f5e9; color: #2e7d32; border-color: #a5d6a7; }
        .status.warning { background: #fff3e0; color: #e65100; border-color: #ffcc80; }
        
        .form-group { margin-bottom: 20px; }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
            font-size: 14px;
        }
        
        input, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            font-size: 15px;
            transition: border-color 0.2s;
        }
        
        input:focus, select:focus { outline: none; border-color: #333; }
        
        .upload-area {
            border: 3px dashed #ccc;
            border-radius: 8px;
            padding: 60px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: #fafafa;
        }
        
        .upload-area:hover { border-color: #333; background: #f5f5f5; }
        .upload-area.dragover { border-color: #333; background: #f0f0f0; }
        
        .face-result {
            background: #fafafa;
            padding: 15px;
            margin: 10px 0;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
        }
        
        .face-result.recognized { border-left: 4px solid #4caf50; }
        .face-result.unknown { border-left: 4px solid #ff9800; }
        
        .progress-bar {
            width: 100%;
            height: 40px;
            background: #f0f0f0;
            border-radius: 6px;
            overflow: hidden;
            margin: 15px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: #333;
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
        }
        
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .image-preview {
            position: relative;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            overflow: hidden;
            background: #fafafa;
        }
        
        .image-preview img {
            width: 100%;
            height: 200px;
            object-fit: cover;
        }
        
        .image-preview .remove-btn {
            position: absolute;
            top: 8px;
            right: 8px;
            background: #fff;
            border: 2px solid #333;
            border-radius: 50%;
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 18px;
            font-weight: bold;
        }
        
        .badge {
            display: inline-block;
            background: #333;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        
        .user-item, .pending-item {
            background: #fafafa;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
        }
        
        .section-title {
            font-size: 18px;
            font-weight: 700;
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }
        
        .image-with-detection {
            position: relative;
            margin-bottom: 15px;
        }
        
        .download-btn {
            margin-top: 10px;
            min-width: 150px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Face Recognition System</h1>
            <p>With LocalStorage & Download Features</p>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('recognize')">Recognize</button>
            <button class="tab" onclick="showTab('enroll')">Camera Enroll</button>
            <button class="tab" onclick="showTab('enrollFolder')">Folder Enroll</button>
            <button class="tab" onclick="showTab('review')">Review</button>
            <button class="tab" onclick="showTab('users')">Users</button>
        </div>
        
        <div class="content">
            <!-- TAB 1: RECOGNIZE -->
            <div id="recognize" class="tab-content active">
                <div style="margin-bottom: 20px;">
                    <button class="btn-primary" onclick="toggleRecognizeMode('camera')" style="flex:1; margin-right:5px;">
                        Camera Mode
                    </button>
                    <button onclick="toggleRecognizeMode('upload')" style="flex:1; margin-left:5px;">
                        Upload Mode
                    </button>
                </div>
                
                <!-- CAMERA MODE -->
                <div id="cameraMode" style="display:block;">
                    <video id="video" autoplay playsinline muted></video>
                    <div class="controls">
                        <button class="btn-primary" onclick="startCamera()">Start Camera</button>
                        <button onclick="savePendingFace()">Save Unknown</button>
                        <button onclick="stopCamera()">Stop</button>
                    </div>
                </div>
                
                <!-- UPLOAD MODE -->
                <div id="uploadMode" style="display:none;">
                    <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
                        <div style="font-size: 48px; margin-bottom: 15px;">📸</div>
                        <p style="font-size: 18px; font-weight: 600; margin-bottom: 8px;">Upload Images</p>
                        <p style="font-size: 14px; color: #666;">Click or drag & drop multiple images</p>
                        <p style="font-size: 12px; color: #999; margin-top: 10px;">Saved to upload-foto folder</p>
                    </div>
                    <input type="file" id="fileInput" accept="image/*" multiple style="display:none;" onchange="handleFileUpload(event)">
                    
                    <div id="imageGrid" class="image-grid" style="display:none;"></div>
                    
                    <div class="controls" id="uploadControls" style="display:none; margin-top:20px;">
                        <button class="btn-primary" onclick="processAllImages()">Detect All Faces</button>
                        <button onclick="clearAllUploads()">Clear All</button>
                    </div>
                    
                    <div id="uploadResults" style="margin-top:20px; display:none;"></div>
                </div>
                
                <div id="recognizeStatus" class="status info" style="display:none;"></div>
            </div>
            
            <!-- TAB 2: CAMERA ENROLL -->
            <div id="enroll" class="tab-content">
                <h3 class="section-title">Camera Enrollment (Saved to LocalStorage)</h3>
                <div class="form-group">
                    <label>User ID:</label>
                    <input type="text" id="enrollUserId" placeholder="e.g., user001">
                </div>
                <div class="form-group">
                    <label>Name:</label>
                    <input type="text" id="enrollUserName" placeholder="e.g., John Doe">
                </div>
                
                <div style="background: #fafafa; padding: 20px; border-radius: 6px; margin-bottom: 20px; border: 1px solid #e0e0e0;">
                    <p style="font-weight: 600; margin-bottom: 10px;">Instructions:</p>
                    <div id="enrollInstruction" style="padding: 15px; background: #fff; border-radius: 6px; text-align: center; font-weight: 600; margin-bottom: 15px;">
                        Press Start to begin
                    </div>
                    <div class="progress-bar">
                        <div id="enrollProgress" class="progress-fill" style="width: 0%;">0%</div>
                    </div>
                </div>
                
                <video id="enrollVideo" autoplay playsinline muted style="display:none;"></video>
                
                <div class="controls">
                    <button class="btn-primary" onclick="startEnrollment()">Start Enrollment</button>
                    <button onclick="stopEnrollment()">Stop</button>
                </div>
                <div id="enrollStatus" class="status info" style="display:none;"></div>
                
                <div style="margin-top: 20px; padding: 15px; background: #fafafa; border-radius: 6px; border: 1px solid #e0e0e0;">
                    <strong>ℹ️ Info:</strong> Enrollment data will be saved in browser's localStorage
                </div>
            </div>
            
            <!-- TAB 3: FOLDER ENROLL -->
            <div id="enrollFolder" class="tab-content">
                <h3 class="section-title">Enroll from Folder</h3>
                <div class="form-group">
                    <label>User ID:</label>
                    <input type="text" id="folderUserId" placeholder="e.g., user001">
                </div>
                <div class="form-group">
                    <label>Name:</label>
                    <input type="text" id="folderUserName" placeholder="e.g., John Doe">
                </div>
                <div class="form-group">
                    <label>Folder Path:</label>
                    <input type="text" id="folderPath" placeholder="faces" value="faces">
                    <p style="font-size: 12px; color: #666; margin-top: 5px;">
                        Place images in the 'faces' folder on server
                    </p>
                </div>
                
                <button class="btn-primary" onclick="enrollFromFolder()" style="width: 100%;">
                    Enroll from Folder
                </button>
                
                <div id="folderEnrollStatus" class="status info" style="display:none;"></div>
            </div>
            
            <!-- TAB 4: REVIEW -->
            <div id="review" class="tab-content">
                <h3 class="section-title">Pending Verification</h3>
                <button class="btn-primary" onclick="loadPendingClusters()" style="width:100%; margin-bottom:20px;">
                    Refresh
                </button>
                <div id="pendingList"></div>
            </div>
            
            <!-- TAB 5: USERS -->
            <div id="users" class="tab-content">
                <h3 class="section-title">Registered Users</h3>
                <button class="btn-primary" onclick="loadUsers()" style="width:100%; margin-bottom:20px;">
                    Refresh
                </button>
                <div id="userList"></div>
            </div>
        </div>
    </div>
    
    <script>
        let videoStream = null;
        let recognizeInterval = null;
        let enrollmentActive = false;
        let uploadedImages = [];
        let processedResults = [];
        let enrollmentData = {
            embeddings: [],
            currentStep: 0,
            instructions: [
                {text: "Look straight at camera", frames: 15},
                {text: "Turn head LEFT", frames: 10},
                {text: "Turn head RIGHT", frames: 10},
                {text: "Turn head UP", frames: 10},
                {text: "Turn head DOWN", frames: 10},
                {text: "Blink 3 times", frames: 10},
                {text: "Smile 😊", frames: 10},
                {text: "Normal expression", frames: 10}
            ],
            frameCount: 0
        };
        
        // LocalStorage functions
        function saveEnrollmentToLocalStorage(userId, userName, embeddings) {
            const enrollmentData = {
                user_id: userId,
                user_name: userName,
                embeddings: embeddings,
                enrolled_at: new Date().toISOString()
            };
            
            // Get existing enrollments
            let enrollments = JSON.parse(localStorage.getItem('face_enrollments') || '[]');
            
            // Check if user already exists
            const existingIndex = enrollments.findIndex(e => e.user_id === userId);
            if (existingIndex >= 0) {
                enrollments[existingIndex] = enrollmentData;
            } else {
                enrollments.push(enrollmentData);
            }
            
            localStorage.setItem('face_enrollments', JSON.stringify(enrollments));
            console.log('✅ Enrollment saved to localStorage:', userId);
            return true;
        }
        
        function getEnrollmentsFromLocalStorage() {
            return JSON.parse(localStorage.getItem('face_enrollments') || '[]');
        }
        
        function clearLocalStorageEnrollments() {
            if (confirm('Clear all enrollments from localStorage?')) {
                localStorage.removeItem('face_enrollments');
                alert('✅ LocalStorage cleared!');
            }
        }
        
        // Draw detection boxes on canvas
        function drawDetectionBoxes(canvas, faces) {
            const ctx = canvas.getContext('2d');
            
            faces.forEach(face => {
                const [x1, y1, x2, y2] = face.box;
                
                // Set box color
                const color = face.status === 'recognized' ? '#00ff00' : '#ffa500';
                
                // Draw rectangle
                ctx.strokeStyle = color;
                ctx.lineWidth = 4;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                
                // Draw label background
                const label = face.status === 'recognized' 
                    ? `${face.name} (${face.distance.toFixed(2)})` 
                    : 'Unknown';
                
                ctx.font = 'bold 16px Arial';
                const textWidth = ctx.measureText(label).width;
                
                ctx.fillStyle = color;
                ctx.fillRect(x1, y1 - 30, textWidth + 20, 30);
                
                // Draw label text
                ctx.fillStyle = '#000';
                ctx.fillText(label, x1 + 10, y1 - 8);
            });
        }
        
        // Download image without detection boxes
        function downloadOriginalImage(imageData, filename) {
            const link = document.createElement('a');
            link.href = imageData;
            link.download = filename || 'image_original.jpg';
            link.click();
        }
        
        // Download image with detection boxes
        function downloadImageWithBoxes(canvasId, filename) {
            const canvas = document.getElementById(canvasId);
            const link = document.createElement('a');
            link.href = canvas.toDataURL('image/jpeg');
            link.download = filename || 'image_with_detection.jpg';
            link.click();
        }
        
        function showTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
            
            if (tabName === 'review') loadPendingClusters();
            if (tabName === 'users') loadUsers();
        }
        
        function toggleRecognizeMode(mode) {
            if (mode === 'camera') {
                document.getElementById('cameraMode').style.display = 'block';
                document.getElementById('uploadMode').style.display = 'none';
                stopCamera();
            } else {
                document.getElementById('cameraMode').style.display = 'none';
                document.getElementById('uploadMode').style.display = 'block';
                stopCamera();
            }
            document.getElementById('recognizeStatus').style.display = 'none';
        }
        
        // Multiple file upload handling
        function handleFileUpload(event) {
            const files = Array.from(event.target.files);
            
            files.forEach(file => {
                if (!file.type.match('image.*')) return;
                
                const reader = new FileReader();
                reader.onload = function(e) {
                    uploadedImages.push({
                        data: e.target.result,
                        name: file.name,
                        id: Date.now() + Math.random()
                    });
                    displayImageGrid();
                };
                reader.readAsDataURL(file);
            });
        }
        
        function displayImageGrid() {
            const grid = document.getElementById('imageGrid');
            grid.style.display = 'grid';
            document.getElementById('uploadControls').style.display = 'flex';
            
            grid.innerHTML = uploadedImages.map((img, idx) => `
                <div class="image-preview">
                    <img src="${img.data}" alt="${img.name}">
                    <div class="remove-btn" onclick="removeImage(${idx})">×</div>
                </div>
            `).join('');
        }
        
        function removeImage(index) {
            uploadedImages.splice(index, 1);
            if (uploadedImages.length === 0) {
                document.getElementById('imageGrid').style.display = 'none';
                document.getElementById('uploadControls').style.display = 'none';
                document.getElementById('uploadResults').style.display = 'none';
            } else {
                displayImageGrid();
            }
        }
        
        function clearAllUploads() {
            uploadedImages = [];
            processedResults = [];
            document.getElementById('imageGrid').style.display = 'none';
            document.getElementById('uploadControls').style.display = 'none';
            document.getElementById('uploadResults').style.display = 'none';
            document.getElementById('fileInput').value = '';
            showStatus('recognizeStatus', 'All uploads cleared', 'info');
        }
        
        // Drag & drop support
        const uploadArea = document.getElementById('uploadArea');
        
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
                        uploadedImages.push({
                            data: e.target.result,
                            name: file.name,
                            id: Date.now() + Math.random()
                        });
                        displayImageGrid();
                    };
                    reader.readAsDataURL(file);
                }
            });
        });
        
        // Auto process all images
        async function processAllImages() {
            if (uploadedImages.length === 0) {
                alert('Please upload images first!');
                return;
            }
            
            showStatus('recognizeStatus', `🔍 Processing ${uploadedImages.length} image(s)...`, 'info');
            processedResults = [];
            
            for (let i = 0; i < uploadedImages.length; i++) {
                const img = uploadedImages[i];
                showStatus('recognizeStatus', `🔍 Processing image ${i + 1}/${uploadedImages.length}...`, 'info');
                
                try {
                    // Save image to server
                    const saveResponse = await fetch('/api/save_uploaded_image', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            image: img.data,
                            filename: img.name
                        })
                    });
                    
                    const saveData = await saveResponse.json();
                    
                    // Recognize faces
                    const response = await fetch('/api/recognize', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({image: img.data})
                    });
                    
                    const data = await response.json();
                    
                    if (data.success && data.faces && data.faces.length > 0) {
                        processedResults.push({
                            imageName: img.name,
                            faces: data.faces,
                            imageData: img.data,
                            savedPath: saveData.saved_path
                        });
                    } else {
                        processedResults.push({
                            imageName: img.name,
                            faces: [],
                            imageData: img.data,
                            savedPath: saveData.saved_path
                        });
                    }
                } catch (err) {
                    console.error('Error processing image:', err);
                    processedResults.push({
                        imageName: img.name,
                        faces: [],
                        error: err.message,
                        imageData: img.data
                    });
                }
            }
            
            displayAllResults();
            showStatus('recognizeStatus', `✅ Processed ${uploadedImages.length} image(s)! Saved to upload-foto folder`, 'success');
        }
        
        function displayAllResults() {
            const resultsDiv = document.getElementById('uploadResults');
            resultsDiv.style.display = 'block';
            
            let totalFaces = 0;
            let recognizedFaces = 0;
            
            processedResults.forEach(result => {
                totalFaces += result.faces.length;
                recognizedFaces += result.faces.filter(f => f.status === 'recognized').length;
            });
            
            resultsDiv.innerHTML = `
                <div class="section-title">
                    Detection Results
                    <span class="badge" style="float: right;">${totalFaces} faces found</span>
                </div>
                
                <div style="background: #fafafa; padding: 15px; border-radius: 6px; margin-bottom: 20px; border: 1px solid #e0e0e0;">
                    <strong>Summary:</strong> Found ${totalFaces} face(s) in ${processedResults.length} image(s)<br>
                    <strong>Recognized:</strong> ${recognizedFaces} | <strong>Unknown:</strong> ${totalFaces - recognizedFaces}<br>
                    <strong>Saved to:</strong> upload-foto folder on server
                </div>
                
                ${processedResults.map((result, idx) => {
                    const canvasId = `canvas_${idx}`;
                    const originalFilename = result.imageName.replace(/\.[^/.]+$/, '_original.jpg');
                    const detectionFilename = result.imageName.replace(/\.[^/.]+$/, '_detection.jpg');
                    
                    // Create canvas and draw
                    setTimeout(() => {
                        const canvas = document.getElementById(canvasId);
                        if (canvas && result.faces.length > 0) {
                            const img = new Image();
                            img.onload = function() {
                                canvas.width = img.width;
                                canvas.height = img.height;
                                const ctx = canvas.getContext('2d');
                                ctx.drawImage(img, 0, 0);
                                drawDetectionBoxes(canvas, result.faces);
                            };
                            img.src = result.imageData;
                        }
                    }, 100);
                    
                    return `
                    <div style="margin-bottom: 30px; padding-bottom: 30px; border-bottom: 2px solid #f0f0f0;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                            <strong>${result.imageName}</strong>
                            <span class="badge">${result.faces.length} face(s)</span>
                        </div>
                        
                        ${result.savedPath ? `<div style="font-size: 12px; color: #666; margin-bottom: 10px;">💾 Saved: ${result.savedPath}</div>` : ''}
                        
                        ${result.error ? 
                            `<div class="status warning">Error: ${result.error}</div>` :
                            result.faces.length === 0 ?
                            `<div class="status warning">No faces detected</div>
                             <div class="controls">
                                <button onclick="downloadOriginalImage('${result.imageData}', '${originalFilename}')">
                                    📥 Download Original
                                </button>
                            </div>` :
                            `<div class="image-with-detection">
                                <canvas id="${canvasId}" style="width: 100%; border: 1px solid #e0e0e0; border-radius: 6px; background: #000;"></canvas>
                            </div>
                            
                            <div class="controls">
                                <button class="btn-primary" onclick="downloadImageWithBoxes('${canvasId}', '${detectionFilename}')">
                                    📥 Download with Detection
                                </button>
                                <button onclick="downloadOriginalImage('${result.imageData}', '${originalFilename}')">
                                    📥 Download Original
                                </button>
                            </div>
                            
                            ${result.faces.map((face, faceIdx) => `
                                <div class="face-result ${face.status === 'recognized' ? 'recognized' : 'unknown'}">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                        <div>
                                            <div style="font-size: 16px; font-weight: 600;">
                                                ${face.status === 'recognized' ? '✅ ' + face.name : '⚠️ Unknown'}
                                            </div>
                                            ${face.status === 'recognized' ? 
                                                `<div style="font-size: 13px; color: #666;">Distance: ${face.distance.toFixed(3)}</div>` : 
                                                '<div style="font-size: 13px; color: #666;">Not in database</div>'
                                            }
                                        </div>
                                        <span class="badge">Face ${faceIdx + 1}</span>
                                    </div>
                                    <div style="font-size: 12px; color: #666;">
                                        Confidence: ${(face.confidence * 100).toFixed(1)}%
                                    </div>
                                </div>
                            `).join('')}`
                        }
                    </div>
                `}).join('')}
            `;
        }
        
        // Camera functions
        async function startCamera() {
            try {
                const video = document.getElementById('video');
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
                
                recognizeInterval = setInterval(recognizeFace, 1500);
                showStatus('recognizeStatus', '✅ Camera started!', 'success');
            } catch (err) {
                console.error('Camera error:', err);
                showStatus('recognizeStatus', '❌ Error: ' + err.message, 'warning');
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
            if (!video.videoWidth) return;
            
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
        
        // Enrollment functions
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
                showStatus('enrollStatus', '❌ Error: ' + err.message, 'warning');
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
                // Save to server
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
                    // Save to localStorage
                    saveEnrollmentToLocalStorage(userId, userName, enrollmentData.embeddings);
                    
                    showStatus('enrollStatus', 
                        `✅ User ${userName} registered with ${data.clusters} clusters! Saved to localStorage`, 'success');
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
        
        // Folder enrollment
        async function enrollFromFolder() {
            const userId = document.getElementById('folderUserId').value;
            const userName = document.getElementById('folderUserName').value;
            const folderPath = document.getElementById('folderPath').value;
            
            if (!userId || !userName) {
                showStatus('folderEnrollStatus', 'Please fill User ID and Name!', 'warning');
                return;
            }
            
            showStatus('folderEnrollStatus', '🔄 Processing images from folder...', 'info');
            
            try {
                const response = await fetch('/api/enroll_from_folder', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        user_id: userId,
                        user_name: userName,
                        folder_path: folderPath
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    showStatus('folderEnrollStatus', 
                        `✅ Success! User ${userName} enrolled with ${data.clusters} clusters from ${data.images_processed}/${data.total_images} images`, 
                        'success');
                } else {
                    showStatus('folderEnrollStatus', `❌ Error: ${data.error}`, 'warning');
                }
            } catch (err) {
                showStatus('folderEnrollStatus', 'Error: ' + err.message, 'warning');
            }
        }
        
        // Review and users functions
        async function loadPendingClusters() {
            try {
                const response = await fetch('/api/pending_clusters');
                const data = await response.json();
                
                const pendingList = document.getElementById('pendingList');
                
                if (data.pending.length === 0) {
                    pendingList.innerHTML = '<p style="text-align:center; color:#666; padding: 40px;">No pending clusters</p>';
                    return;
                }
                
                pendingList.innerHTML = data.pending.map(item => `
                    <div class="pending-item">
                        <strong>Cluster ID:</strong> ${item.cluster_id}<br>
                        <strong>Detected:</strong> ${new Date(item.detected_at).toLocaleString()}<br>
                        <img src="${item.image_url}" alt="Face" style="width:100%; border-radius:6px; margin:10px 0; border: 1px solid #e0e0e0;">
                        
                        <div class="form-group">
                            <label>Assign to User:</label>
                            <select id="user_${item.cluster_id}">
                                <option value="">-- Select User --</option>
                                ${data.users.map(u => `<option value="${u.id}">${u.name} (${u.id})</option>`).join('')}
                                <option value="NEW_USER">➕ New User</option>
                            </select>
                        </div>
                        
                        <div id="newUserForm_${item.cluster_id}" style="display:none;">
                            <input type="text" id="newUserId_${item.cluster_id}" placeholder="User ID" style="margin-bottom:10px;">
                            <input type="text" id="newUserName_${item.cluster_id}" placeholder="Name">
                        </div>
                        
                        <button onclick="verifyCluster('${item.cluster_id}')" class="btn-primary" style="width:100%; margin-top:15px;">
                            Verify
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
                
                // Get localStorage enrollments
                const localEnrollments = getEnrollmentsFromLocalStorage();
                
                if (data.users.length === 0 && localEnrollments.length === 0) {
                    userList.innerHTML = '<p style="text-align:center; color:#666; padding: 40px;">No users registered</p>';
                    return;
                }
                
                let html = '';
                
                // Server users
                if (data.users.length > 0) {
                    html += '<div class="section-title">Server Database</div>';
                    html += data.users.map(user => `
                        <div class="user-item">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <strong style="font-size: 16px;">${user.name}</strong><br>
                                    <small style="color:#666;">ID: ${user.id}</small>
                                </div>
                                <span class="badge">${user.clusters} clusters</span>
                            </div>
                        </div>
                    `).join('');
                }
                
                // LocalStorage enrollments
                if (localEnrollments.length > 0) {
                    html += '<div class="section-title" style="margin-top: 30px;">LocalStorage Enrollments</div>';
                    html += localEnrollments.map(enroll => `
                        <div class="user-item">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <strong style="font-size: 16px;">${enroll.user_name}</strong><br>
                                    <small style="color:#666;">ID: ${enroll.user_id}</small><br>
                                    <small style="color:#999;">Enrolled: ${new Date(enroll.enrolled_at).toLocaleString()}</small>
                                </div>
                                <span class="badge">${enroll.embeddings.length} embeddings</span>
                            </div>
                        </div>
                    `).join('');
                    
                    html += `
                        <button onclick="clearLocalStorageEnrollments()" style="width:100%; margin-top:20px; background:#dc3545; color:white; border-color:#dc3545;">
                            🗑️ Clear LocalStorage
                        </button>
                    `;
                }
                
                userList.innerHTML = html;
                
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
        
        // Show localStorage info on page load
        window.addEventListener('load', () => {
            const enrollments = getEnrollmentsFromLocalStorage();
            console.log(`📦 LocalStorage: ${enrollments.length} enrollment(s) found`);
        });
    </script>
</body>
</html>
'''

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

@app.route('/api/enroll_from_folder', methods=['POST'])
def api_enroll_from_folder():
    try:
        data = request.json
        user_id = data['user_id']
        user_name = data['user_name']
        folder_path = data.get('folder_path', 'faces')
        
        result = face_system.enroll_from_folder(user_id, user_name, folder_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/save_uploaded_image', methods=['POST'])
def api_save_uploaded_image():
    try:
        data = request.json
        image_data = data['image']
        filename = data.get('filename', 'uploaded_image.jpg')
        
        saved_path = face_system.save_uploaded_image(image_data, filename)
        return jsonify({'success': True, 'saved_path': saved_path})
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
    try:
        from OpenSSL import crypto
    except ImportError:
        return False
    
    if os.path.exists('cert.pem') and os.path.exists('key.pem'):
        print("✅ SSL Certificate sudah ada")
        return True
    
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
    cert.gmtime_adj_notAfter(365*24*60*60)
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, 'sha256')
    
    with open("cert.pem", "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    
    with open("key.pem", "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
    
    print("✅ SSL Certificate created: cert.pem & key.pem")
    return True

if __name__ == '__main__':
    import socket
    
    # Create folders
    os.makedirs('faces', exist_ok=True)
    os.makedirs('upload-foto', exist_ok=True)
    print("📁 Folders ready: faces, upload-foto")
    
    # Generate SSL certificate
    has_ssl = generate_self_signed_cert()
    
    if not has_ssl:
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
    print("🚀 FACE RECOGNITION - FULL ENHANCED")
    print("="*60)
    print(f"🌐 Akses dari laptop: https://localhost:5000")
    print(f"📱 Akses dari iPhone: https://{local_ip}:5000")
    print("="*60)
    print("✨ FEATURES:")
    print("   • LocalStorage enrollment data")
    print("   • Upload images saved to upload-foto/")
    print("   • Detection boxes on canvas")
    print("   • Download with/without detection boxes")
    print("   • Multiple image upload & auto-detection")
    print("="*60)
    print("💡 PENTING untuk iPhone/iOS:")
    print("   1. WAJIB pakai HTTPS (bukan HTTP)")
    print("   2. Browser akan warning 'Not Secure'")
    print("   3. Klik 'Advanced' → 'Proceed to {local_ip}'")
    print("   4. Izinkan akses kamera saat diminta")
    print("="*60)
    print("📁 Upload images → saved to upload-foto/")
    print("📁 Enroll from folder → use faces/")
    print("="*60)
    print("📱 Pastikan laptop dan HP di WiFi yang SAMA!")
    print("="*60 + "\n")
    
    # Create SSL context
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain('cert.pem', 'key.pem')
    
    # Run Flask dengan HTTPS
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, ssl_context=context)