"""
Sistem Pengenalan Wajah Advanced dengan Human-in-the-Loop & Biometric Enrollment
Instalasi library yang diperlukan:
pip install torch torchvision facenet-pytorch opencv-python pillow numpy scikit-learn
"""

import os
import pickle
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.cluster import DBSCAN
from datetime import datetime
import json

class AdvancedFaceRecognitionSystem:
    def __init__(self, database_path='face_database_advanced.pkl', threshold=0.6):
        """
        Sistem pengenalan wajah dengan Human-in-the-Loop
        
        Args:
            database_path: Path database wajah
            threshold: Threshold kesamaan wajah
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Menggunakan device: {self.device}")
        
        # Inisialisasi MTCNN dan FaceNet
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
        
        self.database_path = database_path
        self.threshold = threshold
        
        # Database struktur baru dengan cluster support
        self.face_database = {
            'users': {},  # {user_id: {'name': str, 'cluster_ids': [ids]}}
            'clusters': {},  # {cluster_id: {'embeddings': [], 'mean_embedding': array, 'verified': bool, 'user_id': str/None}}
            'pending_clusters': {},  # Cluster yang menunggu verifikasi
            'training_log': []  # Log untuk training ulang model
        }
        
        self.next_cluster_id = 1
        self.load_database()
    
    def extract_embedding(self, face_img):
        """Ekstraksi embedding dari gambar wajah"""
        with torch.no_grad():
            face_img = face_img.to(self.device)
            embedding = self.resnet(face_img.unsqueeze(0))
        return embedding.cpu().numpy().flatten()
    
    def biometric_enrollment(self, user_id, user_name):
        """
        Pendaftaran biometrik dengan instruksi gerakan wajah
        Mengambil multiple foto dari berbagai sudut dan ekspresi
        
        Args:
            user_id: ID unik user
            user_name: Nama user
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Tidak dapat membuka webcam")
            return False
        
        instructions = [
            ("Lihat lurus ke kamera", 15),
            ("Putar kepala ke KIRI", 10),
            ("Putar kepala ke KANAN", 10),
            ("Putar kepala ke ATAS", 10),
            ("Putar kepala ke BAWAH", 10),
            ("Kedipkan mata 3x", 10),
            ("Senyum", 10),
            ("Ekspresi normal", 10)
        ]
        
        collected_embeddings = []
        instruction_idx = 0
        frame_count = 0
        frames_needed = instructions[instruction_idx][1]
        
        print(f"\n🎥 PENDAFTARAN BIOMETRIK untuk {user_name}")
        print(f"💡 Ikuti instruksi di layar. Total: {len(instructions)} gerakan")
        print("="*60)
        
        while instruction_idx < len(instructions):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Instruksi saat ini
            current_instruction = instructions[instruction_idx][0]
            progress = f"{frame_count}/{frames_needed}"
            
            # Deteksi wajah
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            boxes, probs = self.mtcnn.detect(img_pil)
            
            if boxes is not None and len(boxes) > 0:
                box = boxes[0]
                x1, y1, x2, y2 = [int(coord) for coord in box]
                
                # Crop dan ekstrak embedding setiap 3 frame
                if frame_count % 3 == 0:
                    try:
                        x1, y1 = max(0, x1), max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)
                        
                        face_img = img_pil.crop((x1, y1, x2, y2))
                        face_img = face_img.resize((160, 160))
                        
                        face_array = np.array(face_img).astype(np.float32)
                        face_array = (face_array - 127.5) / 128.0
                        face_tensor = torch.from_numpy(face_array).permute(2, 0, 1)
                        
                        embedding = self.extract_embedding(face_tensor)
                        collected_embeddings.append(embedding)
                        
                        frame_count += 1
                        
                        # Gambar box hijau jika berhasil
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    except Exception as e:
                        print(f"⚠️  Error capturing: {e}")
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            else:
                # Tidak ada wajah terdeteksi
                cv2.putText(frame, "TIDAK ADA WAJAH TERDETEKSI", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Tampilkan instruksi
            cv2.putText(frame, f"Instruksi: {current_instruction}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Progress: {progress}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Step {instruction_idx+1}/{len(instructions)}", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            cv2.imshow('Biometric Enrollment', frame)
            
            # Pindah ke instruksi berikutnya
            if frame_count >= frames_needed:
                instruction_idx += 1
                frame_count = 0
                if instruction_idx < len(instructions):
                    frames_needed = instructions[instruction_idx][1]
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n❌ Pendaftaran dibatalkan")
                cap.release()
                cv2.destroyAllWindows()
                return False
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Berhasil mengumpulkan {len(collected_embeddings)} embeddings")
        
        # Simpan ke database dengan multiple cluster IDs
        cluster_ids = []
        for embedding in collected_embeddings:
            cluster_id = self.next_cluster_id
            self.next_cluster_id += 1
            
            self.face_database['clusters'][cluster_id] = {
                'embeddings': [embedding],
                'mean_embedding': embedding,
                'verified': True,
                'user_id': user_id,
                'created_at': datetime.now().isoformat()
            }
            cluster_ids.append(cluster_id)
        
        # Daftarkan user
        self.face_database['users'][user_id] = {
            'name': user_name,
            'cluster_ids': cluster_ids,
            'created_at': datetime.now().isoformat()
        }
        
        print(f"✅ User {user_name} terdaftar dengan {len(cluster_ids)} cluster IDs")
        self.save_database()
        return True
    
    def detect_and_cluster_faces(self, image_path):
        """
        Deteksi wajah dan buat cluster untuk wajah yang belum dikenali
        
        Returns:
            List of dict dengan info wajah dan cluster assignment
        """
        try:
            img = Image.open(image_path).convert('RGB')
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            boxes, probs = self.mtcnn.detect(img)
            
            if boxes is None:
                print("❌ Tidak ada wajah terdeteksi")
                return []
            
            results = []
            
            for i, box in enumerate(boxes):
                try:
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2 = min(img.width, x2)
                    y2 = min(img.height, y2)
                    
                    face_img = img.crop((x1, y1, x2, y2))
                    face_img = face_img.resize((160, 160))
                    
                    face_array = np.array(face_img).astype(np.float32)
                    face_array = (face_array - 127.5) / 128.0
                    face_tensor = torch.from_numpy(face_array).permute(2, 0, 1)
                    
                    embedding = self.extract_embedding(face_tensor)
                    
                    # Cari match di database
                    best_match = self._find_best_match(embedding)
                    
                    if best_match is None:
                        # Buat cluster baru untuk pending verification
                        cluster_id = f"pending_{self.next_cluster_id}"
                        self.next_cluster_id += 1
                        
                        self.face_database['pending_clusters'][cluster_id] = {
                            'embeddings': [embedding],
                            'mean_embedding': embedding,
                            'image_path': image_path,
                            'box': box.tolist(),
                            'detected_at': datetime.now().isoformat()
                        }
                        
                        results.append({
                            'cluster_id': cluster_id,
                            'user_id': None,
                            'name': 'Unknown',
                            'distance': None,
                            'box': box,
                            'confidence': probs[i],
                            'status': 'pending_verification'
                        })
                        
                        # Gambar box kuning untuk pending
                        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(img_cv, f"Pending: {cluster_id}", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        # Wajah dikenali
                        results.append({
                            'cluster_id': best_match['cluster_id'],
                            'user_id': best_match['user_id'],
                            'name': best_match['name'],
                            'distance': best_match['distance'],
                            'box': box,
                            'confidence': probs[i],
                            'status': 'recognized'
                        })
                        
                        # Gambar box hijau untuk recognized
                        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{best_match['name']} ({best_match['distance']:.2f})"
                        cv2.putText(img_cv, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                except Exception as e:
                    print(f"⚠️  Error processing face {i}: {e}")
                    continue
            
            # Tampilkan hasil
            cv2.imshow('Face Detection & Clustering', img_cv)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            self.save_database()
            return results
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return []
    
    def _find_best_match(self, embedding):
        """Cari match terbaik di database verified clusters"""
        best_match = None
        best_distance = float('inf')
        
        for cluster_id, cluster_data in self.face_database['clusters'].items():
            if not cluster_data.get('verified', False):
                continue
            
            mean_emb = cluster_data['mean_embedding']
            distance = np.linalg.norm(embedding - mean_emb)
            
            if distance < best_distance and distance < self.threshold:
                best_distance = distance
                user_id = cluster_data.get('user_id')
                user_name = self.face_database['users'].get(user_id, {}).get('name', 'Unknown')
                
                best_match = {
                    'cluster_id': cluster_id,
                    'user_id': user_id,
                    'name': user_name,
                    'distance': distance
                }
        
        return best_match
    
    def review_pending_clusters(self):
        """
        Review cluster yang pending dan minta user untuk verifikasi
        Human-in-the-Loop process
        """
        if not self.face_database['pending_clusters']:
            print("✅ Tidak ada cluster pending untuk direview")
            return
        
        print(f"\n📋 Ada {len(self.face_database['pending_clusters'])} cluster pending")
        print("="*60)
        
        for cluster_id, cluster_data in list(self.face_database['pending_clusters'].items()):
            print(f"\n🔍 Cluster ID: {cluster_id}")
            print(f"   Detected at: {cluster_data['detected_at']}")
            
            # Tampilkan gambar wajah
            try:
                img = cv2.imread(cluster_data['image_path'])
                box = cluster_data['box']
                x1, y1, x2, y2 = [int(coord) for coord in box]
                
                face_crop = img[y1:y2, x1:x2]
                cv2.imshow(f'Cluster {cluster_id}', face_crop)
                cv2.waitKey(1)
            except:
                print("   ⚠️  Tidak dapat menampilkan gambar")
            
            # Tampilkan daftar user yang ada
            print("\n   👥 User terdaftar:")
            users_list = list(self.face_database['users'].items())
            for idx, (uid, udata) in enumerate(users_list, 1):
                print(f"      {idx}. {udata['name']} (ID: {uid})")
            
            print(f"      {len(users_list)+1}. User baru")
            print(f"      0. Skip cluster ini")
            
            # Input user
            try:
                choice = int(input(f"\n   Pilih user untuk cluster {cluster_id} (0-{len(users_list)+1}): "))
                
                if choice == 0:
                    print("   ⏭️  Cluster di-skip")
                    continue
                
                elif choice == len(users_list) + 1:
                    # User baru
                    new_user_id = input("   Masukkan User ID baru: ")
                    new_user_name = input("   Masukkan nama: ")
                    
                    # Buat user baru
                    if new_user_id not in self.face_database['users']:
                        self.face_database['users'][new_user_id] = {
                            'name': new_user_name,
                            'cluster_ids': [],
                            'created_at': datetime.now().isoformat()
                        }
                    
                    user_id = new_user_id
                    user_name = new_user_name
                
                elif 1 <= choice <= len(users_list):
                    # User existing
                    user_id, user_data = users_list[choice-1]
                    user_name = user_data['name']
                
                else:
                    print("   ❌ Pilihan tidak valid")
                    continue
                
                # Pindahkan dari pending ke clusters verified
                new_cluster_id = self.next_cluster_id
                self.next_cluster_id += 1
                
                self.face_database['clusters'][new_cluster_id] = {
                    'embeddings': cluster_data['embeddings'],
                    'mean_embedding': cluster_data['mean_embedding'],
                    'verified': True,
                    'user_id': user_id,
                    'verified_at': datetime.now().isoformat()
                }
                
                # Update user's cluster list
                self.face_database['users'][user_id]['cluster_ids'].append(new_cluster_id)
                
                # Hapus dari pending
                del self.face_database['pending_clusters'][cluster_id]
                
                # Log untuk training
                self.face_database['training_log'].append({
                    'timestamp': datetime.now().isoformat(),
                    'action': 'verified',
                    'cluster_id': new_cluster_id,
                    'user_id': user_id,
                    'user_name': user_name
                })
                
                print(f"   ✅ Cluster {cluster_id} → User {user_name} (Cluster ID: {new_cluster_id})")
                
            except ValueError:
                print("   ❌ Input tidak valid")
                continue
            except KeyboardInterrupt:
                print("\n\n⏸️  Review dihentikan")
                break
        
        cv2.destroyAllWindows()
        self.save_database()
        print(f"\n💾 Database updated dengan verifikasi user")
    
    def retrain_with_feedback(self):
        """
        Update embeddings berdasarkan feedback user
        Merge embeddings untuk user yang sama
        """
        print("\n🔄 Updating face embeddings berdasarkan feedback...")
        
        updated_count = 0
        for user_id, user_data in self.face_database['users'].items():
            cluster_ids = user_data['cluster_ids']
            
            if len(cluster_ids) > 1:
                # Kumpulkan semua embeddings untuk user ini
                all_embeddings = []
                for cid in cluster_ids:
                    cluster = self.face_database['clusters'].get(cid)
                    if cluster:
                        all_embeddings.extend(cluster['embeddings'])
                
                # Hitung mean embedding baru
                if all_embeddings:
                    mean_embedding = np.mean(all_embeddings, axis=0)
                    
                    # Update mean untuk semua cluster user ini
                    for cid in cluster_ids:
                        if cid in self.face_database['clusters']:
                            self.face_database['clusters'][cid]['mean_embedding'] = mean_embedding
                    
                    updated_count += 1
        
        print(f"✅ Updated {updated_count} users dengan embeddings baru")
        self.save_database()
    
    def recognize_webcam_hitl(self):
        """
        Webcam recognition dengan Human-in-the-Loop
        Tekan 's' untuk save pending face untuk review nanti
        Tekan 'q' untuk keluar
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Tidak dapat membuka webcam")
            return
        
        print("\n📹 Webcam HITL Mode")
        print("   's' - Save pending face untuk review")
        print("   'q' - Keluar")
        
        pending_faces = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            boxes, probs = self.mtcnn.detect(img_pil)
            
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
                        
                        if best_match is None:
                            label = "Unknown - Press 's' to save"
                            color = (0, 165, 255)  # Orange
                            pending_faces.append({
                                'embedding': embedding,
                                'box': box,
                                'frame': frame.copy()
                            })
                        else:
                            label = f"{best_match['name']} ({best_match['distance']:.2f})"
                            color = (0, 255, 0)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    except Exception as e:
                        continue
            
            cv2.putText(frame, f"Pending: {len(pending_faces)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Webcam HITL Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and pending_faces:
                # Save pending face
                pending_data = pending_faces[-1]
                cluster_id = f"pending_{self.next_cluster_id}"
                self.next_cluster_id += 1
                
                # Save frame
                temp_path = f"temp_{cluster_id}.jpg"
                cv2.imwrite(temp_path, pending_data['frame'])
                
                self.face_database['pending_clusters'][cluster_id] = {
                    'embeddings': [pending_data['embedding']],
                    'mean_embedding': pending_data['embedding'],
                    'image_path': temp_path,
                    'box': pending_data['box'].tolist(),
                    'detected_at': datetime.now().isoformat()
                }
                
                print(f"💾 Saved {cluster_id} untuk review")
                pending_faces = []
        
        cap.release()
        cv2.destroyAllWindows()
        self.save_database()
    
    def save_database(self):
        """Simpan database"""
        with open(self.database_path, 'wb') as f:
            pickle.dump(self.face_database, f)
    
    def load_database(self):
        """Load database"""
        if os.path.exists(self.database_path):
            with open(self.database_path, 'rb') as f:
                self.face_database = pickle.load(f)
            
            # Update next_cluster_id
            all_ids = list(self.face_database['clusters'].keys())
            if all_ids:
                self.next_cluster_id = max([int(str(x).replace('pending_', '')) for x in all_ids if str(x).replace('pending_', '').isdigit()]) + 1
            
            print(f"📂 Database loaded:")
            print(f"   Users: {len(self.face_database['users'])}")
            print(f"   Clusters: {len(self.face_database['clusters'])}")
            print(f"   Pending: {len(self.face_database['pending_clusters'])}")
        else:
            print("📂 Database baru dibuat")
    
    def show_user_clusters(self, user_id):
        """Tampilkan semua cluster untuk user tertentu"""
        if user_id not in self.face_database['users']:
            print(f"❌ User {user_id} tidak ditemukan")
            return
        
        user_data = self.face_database['users'][user_id]
        print(f"\n👤 User: {user_data['name']} (ID: {user_id})")
        print(f"📊 Total clusters: {len(user_data['cluster_ids'])}")
        print(f"   Cluster IDs: {user_data['cluster_ids']}")


# ===== MENU UTAMA =====
if __name__ == "__main__":
    system = AdvancedFaceRecognitionSystem(threshold=0.6)
    
    while True:
        print("\n" + "="*60)
        print("🔍 ADVANCED FACE RECOGNITION SYSTEM (HITL)")
        print("="*60)
        print("1. 📝 Biometric Enrollment (Daftar user baru)")
        print("2. 🔎 Detect & Cluster Faces (Deteksi dari gambar)")
        print("3. ✅ Review Pending Clusters (Human-in-the-Loop)")
        print("4. 🔄 Retrain dengan Feedback User")
        print("5. 📹 Webcam Recognition (HITL Mode)")
        print("6. 👥 Lihat User & Clusters")
        print("7. 📊 Lihat Detail User")
        print("8. 💾 Save & Exit")
        print("="*60)
        
        choice = input("Pilih menu (1-8): ")
        
        if choice == '1':
            user_id = input("Masukkan User ID (unik): ")
            user_name = input("Masukkan Nama: ")
            system.biometric_enrollment(user_id, user_name)
        
        elif choice == '2':
            img_path = input("Masukkan path gambar: ")
            results = system.detect_and_cluster_faces(img_path)
            print(f"\n📊 Hasil: {len(results)} wajah terdeteksi")
            for r in results:
                print(f"   - {r['status']}: {r.get('name', 'Pending')} (Cluster: {r['cluster_id']})")
        
        elif choice == '3':
            system.review_pending_clusters()
        
        elif choice == '4':
            system.retrain_with_feedback()
        
        elif choice == '5':
            system.recognize_webcam_hitl()
        
        elif choice == '6':
            print(f"\n👥 Total Users: {len(system.face_database['users'])}")
            for uid, udata in system.face_database['users'].items():
                print(f"   - {udata['name']} (ID: {uid}): {len(udata['cluster_ids'])} clusters")
        
        elif choice == '7':
            user_id = input("Masukkan User ID: ")
            system.show_user_clusters(user_id)
        
        elif choice == '8':
            system.save_database()
            print("👋 Terima kasih!")
            break
        
        else:
            print("❌ Pilihan tidak valid")