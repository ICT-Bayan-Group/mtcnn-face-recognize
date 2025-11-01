"""
Sistem Pengenalan Wajah menggunakan MTCNN dan FaceNet
Instalasi library yang diperlukan:
pip install torch torchvision facenet-pytorch opencv-python pillow numpy
"""

import os
import pickle
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceRecognitionSystem:
    def __init__(self, database_path='face_database.pkl', threshold=0.6):
        """
        Inisialisasi sistem pengenalan wajah
        
        Args:
            database_path: Path untuk menyimpan database wajah
            threshold: Threshold untuk menentukan kesamaan wajah (semakin kecil semakin strict)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Menggunakan device: {self.device}")
        
        # Inisialisasi MTCNN untuk deteksi wajah
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=0, 
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709, 
            post_process=True,
            device=self.device
        )
        
        # Inisialisasi FaceNet untuk ekstraksi embedding
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        self.database_path = database_path
        self.threshold = threshold
        self.face_database = {}
        
        # Load database jika ada
        self.load_database()
    
    def extract_embedding(self, face_img):
        """
        Ekstraksi embedding dari gambar wajah
        
        Args:
            face_img: Tensor gambar wajah yang sudah di-align
            
        Returns:
            embedding: Vector embedding 512-dimensi
        """
        with torch.no_grad():
            face_img = face_img.to(self.device)
            embedding = self.resnet(face_img.unsqueeze(0))
        return embedding.cpu().numpy().flatten()
    
    def register_face(self, image_path, name):
        """
        Daftarkan wajah baru ke database
        
        Args:
            image_path: Path gambar wajah
            name: Nama/ID untuk wajah tersebut
            
        Returns:
            success: Boolean apakah berhasil mendaftar
        """
        try:
            # Baca gambar
            img = Image.open(image_path).convert('RGB')
            
            # Deteksi wajah
            face_tensor = self.mtcnn(img)
            
            if face_tensor is None:
                print(f"❌ Tidak ada wajah terdeteksi pada {image_path}")
                return False
            
            # Ekstrak embedding
            embedding = self.extract_embedding(face_tensor)
            
            # Simpan ke database
            self.face_database[name] = {
                'embedding': embedding,
                'image_path': image_path
            }
            
            print(f"✅ Wajah '{name}' berhasil didaftarkan")
            return True
            
        except Exception as e:
            print(f"❌ Error saat mendaftar {name}: {str(e)}")
            return False
    
    def register_faces_from_folder(self, folder_path):
        """
        Daftarkan semua wajah dari folder
        Struktur folder: folder_path/nama_orang/foto.jpg
        
        Args:
            folder_path: Path folder berisi subfolder untuk setiap orang
        """
        if not os.path.exists(folder_path):
            print(f"❌ Folder {folder_path} tidak ditemukan")
            return
        
        registered_count = 0
        
        # Loop setiap subfolder (nama orang)
        for person_name in os.listdir(folder_path):
            person_folder = os.path.join(folder_path, person_name)
            
            if not os.path.isdir(person_folder):
                continue
            
            # Loop setiap gambar dalam subfolder
            for img_file in os.listdir(person_folder):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_folder, img_file)
                    
                    # Gunakan nama folder sebagai ID
                    if self.register_face(img_path, person_name):
                        registered_count += 1
                        break  # Hanya ambil 1 foto per orang
        
        print(f"\n📊 Total {registered_count} wajah berhasil didaftarkan")
        self.save_database()
    
    def recognize_face(self, image_path, draw_boxes=True, save_result=True):
        """
        Kenali wajah dari gambar
        
        Args:
            image_path: Path gambar untuk dianalisis
            draw_boxes: Apakah menggambar kotak di sekitar wajah
            save_result: Apakah menyimpan hasil
            
        Returns:
            results: List hasil deteksi
        """
        try:
            # Baca gambar
            img = Image.open(image_path).convert('RGB')
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Deteksi wajah dengan bounding box
            boxes, probs = self.mtcnn.detect(img)
            
            if boxes is None:
                print("❌ Tidak ada wajah terdeteksi")
                return []
            
            results = []
            
            # Proses setiap wajah yang terdeteksi
            for i, box in enumerate(boxes):
                # Crop wajah menggunakan MTCNN langsung
                face_tensor = self.mtcnn(img, save_path=None)
                
                if face_tensor is None:
                    continue
                
                # Ekstrak embedding
                embedding = self.extract_embedding(face_tensor)
                
                # Cari wajah yang cocok di database
                best_match = None
                best_distance = float('inf')
                
                for name, data in self.face_database.items():
                    # Hitung jarak euclidean
                    distance = np.linalg.norm(embedding - data['embedding'])
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match = name
                
                # Tentukan apakah cocok berdasarkan threshold
                if best_distance < self.threshold:
                    label = f"{best_match} ({best_distance:.2f})"
                    color = (0, 255, 0)  # Hijau untuk dikenali
                else:
                    label = f"Unknown ({best_distance:.2f})"
                    color = (0, 0, 255)  # Merah untuk tidak dikenali
                
                results.append({
                    'name': best_match if best_distance < self.threshold else 'Unknown',
                    'distance': best_distance,
                    'box': box,
                    'confidence': probs[i]
                })
                
                # Gambar bounding box
                if draw_boxes:
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img_cv, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Simpan hasil
            if save_result:
                output_path = image_path.replace('.', '_result.')
                cv2.imwrite(output_path, img_cv)
                print(f"💾 Hasil disimpan ke: {output_path}")
            
            # Tampilkan hasil
            cv2.imshow('Face Recognition', img_cv)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            return results
            
        except Exception as e:
            print(f"❌ Error saat mengenali wajah: {str(e)}")
            return []
    
    def recognize_webcam(self):
        """
        Kenali wajah dari webcam secara real-time
        Tekan 'q' untuk keluar
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Tidak dapat membuka webcam")
            return
        
        print("📹 Webcam aktif. Tekan 'q' untuk keluar.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Konversi ke PIL Image
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Deteksi wajah
            boxes, probs = self.mtcnn.detect(img_pil)
            
            if boxes is not None:
                for i, box in enumerate(boxes):
                    try:
                        # Crop wajah manual dari bounding box
                        x1, y1, x2, y2 = [int(coord) for coord in box]
                        
                        # Pastikan koordinat dalam batas
                        x1, y1 = max(0, x1), max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)
                        
                        # Crop wajah dari frame
                        face_img = img_pil.crop((x1, y1, x2, y2))
                        
                        # Resize ke ukuran yang dibutuhkan MTCNN (160x160)
                        face_img = face_img.resize((160, 160))
                        
                        # Konversi ke tensor
                        face_array = np.array(face_img).astype(np.float32)
                        face_array = (face_array - 127.5) / 128.0  # Normalisasi
                        face_tensor = torch.from_numpy(face_array).permute(2, 0, 1)
                        
                        # Ekstrak embedding
                        embedding = self.extract_embedding(face_tensor)
                        
                        # Cari match
                        best_match = None
                        best_distance = float('inf')
                        
                        for name, data in self.face_database.items():
                            distance = np.linalg.norm(embedding - data['embedding'])
                            if distance < best_distance:
                                best_distance = distance
                                best_match = name
                        
                        # Gambar hasil
                        if best_distance < self.threshold:
                            label = f"{best_match} ({best_distance:.2f})"
                            color = (0, 255, 0)
                        else:
                            label = f"Unknown ({best_distance:.2f})"
                            color = (0, 0, 255)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    except Exception as e:
                        print(f"Error processing face: {e}")
                        continue
            
            cv2.imshow('Webcam Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def save_database(self):
        """Simpan database wajah ke file"""
        with open(self.database_path, 'wb') as f:
            pickle.dump(self.face_database, f)
        print(f"💾 Database disimpan ke {self.database_path}")
    
    def load_database(self):
        """Load database wajah dari file"""
        if os.path.exists(self.database_path):
            with open(self.database_path, 'rb') as f:
                self.face_database = pickle.load(f)
            print(f"📂 Database dimuat: {len(self.face_database)} wajah")
        else:
            print("📂 Database baru akan dibuat")
    
    def list_registered_faces(self):
        """Tampilkan daftar wajah yang terdaftar"""
        print("\n👥 Wajah Terdaftar:")
        if not self.face_database:
            print("   Belum ada wajah terdaftar")
        else:
            for i, name in enumerate(self.face_database.keys(), 1):
                print(f"   {i}. {name}")


# ===== CONTOH PENGGUNAAN =====
if __name__ == "__main__":
    # Inisialisasi sistem
    system = FaceRecognitionSystem(threshold=0.6)
    
    # Menu interaktif
    while True:
        print("\n" + "="*50)
        print("🔍 SISTEM PENGENALAN WAJAH")
        print("="*50)
        print("1. Daftar wajah dari folder")
        print("2. Daftar wajah dari file tunggal")
        print("3. Kenali wajah dari gambar")
        print("4. Kenali wajah dari webcam")
        print("5. Lihat daftar wajah terdaftar")
        print("6. Keluar")
        print("="*50)
        
        choice = input("Pilih menu (1-6): ")
        
        if choice == '1':
            folder = input("Masukkan path folder (contoh: ./faces): ")
            system.register_faces_from_folder(folder)
        
        elif choice == '2':
            img_path = input("Masukkan path gambar: ")
            name = input("Masukkan nama/ID: ")
            system.register_face(img_path, name)
            system.save_database()
        
        elif choice == '3':
            img_path = input("Masukkan path gambar untuk dikenali: ")
            results = system.recognize_face(img_path)
            print("\n📊 Hasil Deteksi:")
            for i, result in enumerate(results, 1):
                print(f"   Wajah {i}: {result['name']} "
                      f"(jarak: {result['distance']:.3f}, "
                      f"confidence: {result['confidence']:.3f})")
        
        elif choice == '4':
            system.recognize_webcam()
        
        elif choice == '5':
            system.list_registered_faces()
        
        elif choice == '6':
            print("👋 Terima kasih!")
            break
        
        else:
            print("❌ Pilihan tidak valid")