"""
bulk_upload.py
==============
Script untuk upload foto secara massal dari sebuah folder ke server
Soccer Clinic Face Recognition (app.py).

Cara pakai
----------
1. Pastikan server app.py sudah berjalan (http://localhost:5000).
2. Jalankan script ini:

   python bulk_upload.py

   Atau dengan argumen langsung tanpa prompt interaktif:

   python bulk_upload.py \\
       --folder     /path/to/photos \\
       --server     http://localhost:5000 \\
       --password   Bayansoccer123! \\
       --event      "Latihan Rutin Mei 2025" \\
       --location   "Lapangan Utama" \\
       --photographer "Budi" \\
       --workers    3

Parameter opsional
------------------
--workers   Jumlah thread paralel (default: 3).
            Jangan terlalu tinggi agar server tidak kewalahan.
--recursive Sertakan sub-folder secara rekursif (flag, tidak perlu nilai).
--dry-run   Tampilkan daftar file tanpa benar-benar mengupload.
"""

import os
import sys
import glob
import base64
import json
import argparse
import getpass
import time
import threading
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    sys.exit(
        "❌  Library 'requests' belum terpasang.\n"
        "    Jalankan:  pip install requests"
    )

# ─────────────────────────────────────────────────────────────
# Ekstensi gambar yang didukung
# ─────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"
}

# ─────────────────────────────────────────────────────────────
# Warna terminal (opsional, fallback ke teks biasa jika tidak didukung)
# ─────────────────────────────────────────────────────────────
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    GREEN  = Fore.GREEN
    RED    = Fore.RED
    YELLOW = Fore.YELLOW
    CYAN   = Fore.CYAN
    BOLD   = Style.BRIGHT
    RESET  = Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = CYAN = BOLD = RESET = ""


# ─────────────────────────────────────────────────────────────
# Fungsi utilitas
# ─────────────────────────────────────────────────────────────

def collect_images(folder: str, recursive: bool) -> list[Path]:
    """Kumpulkan semua file gambar dari folder (dan sub-folder jika recursive)."""
    folder_path = Path(folder)
    if not folder_path.exists():
        sys.exit(f"❌  Folder tidak ditemukan: {folder}")
    if not folder_path.is_dir():
        sys.exit(f"❌  Path bukan folder: {folder}")

    if recursive:
        files = [
            p for p in folder_path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    else:
        files = [
            p for p in folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

    # Urutkan agar urutan upload konsisten
    files.sort(key=lambda p: p.name.lower())
    return files


def image_to_base64(filepath: Path) -> str:
    """Baca file gambar dan encode ke base64 data URL."""
    ext = filepath.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".bmp": "image/bmp",
        ".webp": "image/webp", ".tiff": "image/tiff", ".tif": "image/tiff",
    }
    mime = mime_map.get(ext, "image/jpeg")

    with open(filepath, "rb") as f:
        raw = f.read()

    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def check_server(base_url: str) -> bool:
    """Cek apakah server sedang berjalan melalui /api/health."""
    try:
        resp = requests.get(f"{base_url}/api/health", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            print(f"{GREEN}✅  Server OK — "
                  f"{data.get('total_photos', '?')} foto, "
                  f"{data.get('total_faces', '?')} wajah di database")
            return True
    except requests.exceptions.ConnectionError:
        pass
    return False


def upload_one(
    filepath: Path,
    base_url: str,
    metadata: dict,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> dict:
    """
    Upload satu foto ke /api/photographer/upload.
    Kembalikan dict hasil: {'file', 'success', 'faces', 'error'}.
    """
    result = {"file": filepath.name, "success": False, "faces": 0, "error": ""}

    for attempt in range(1, retries + 1):
        try:
            image_data = image_to_base64(filepath)
            payload = {
                "image":    image_data,
                "filename": filepath.name,
                "metadata": metadata,
            }

            resp = requests.post(
                f"{base_url}/api/photographer/upload",
                json=payload,
                timeout=120,   # foto besar bisa butuh waktu
            )

            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    result["success"] = True
                    result["faces"]   = data.get("faces_detected", 0)
                else:
                    result["error"] = data.get("error", "Unknown error")
            else:
                result["error"] = f"HTTP {resp.status_code}"

            break   # berhasil atau gagal tanpa exception — tidak perlu retry

        except requests.exceptions.Timeout:
            result["error"] = "Timeout"
            if attempt < retries:
                time.sleep(retry_delay)
        except requests.exceptions.ConnectionError as e:
            result["error"] = f"Connection error: {e}"
            if attempt < retries:
                time.sleep(retry_delay)
        except Exception as e:
            result["error"] = str(e)
            break

    return result


# ─────────────────────────────────────────────────────────────
# Progress tracker thread-safe
# ─────────────────────────────────────────────────────────────

class ProgressTracker:
    def __init__(self, total: int):
        self.total    = total
        self.done     = 0
        self.success  = 0
        self.failed   = 0
        self.faces    = 0
        self._lock    = threading.Lock()

    def update(self, result: dict):
        with self._lock:
            self.done += 1
            if result["success"]:
                self.success += 1
                self.faces   += result["faces"]
            else:
                self.failed += 1

    def print_result(self, result: dict):
        """Cetak baris status untuk satu file (thread-safe via GIL pada print)."""
        with self._lock:
            pct = self.done / self.total * 100
            if result["success"]:
                status = f"{GREEN}✅"
                detail = f"{result['faces']} wajah terdeteksi"
            else:
                status = f"{RED}❌"
                detail = f"Error: {result['error']}"

            print(
                f"  {status} [{self.done:>4}/{self.total}] "
                f"({pct:5.1f}%)  {result['file']:<45}  {detail}{RESET}"
            )

    def summary(self):
        print()
        print("=" * 65)
        print(f"{BOLD}📊  RINGKASAN BULK UPLOAD{RESET}")
        print("=" * 65)
        print(f"  Total foto diproses : {self.done}")
        print(f"  {GREEN}Berhasil           : {self.success}{RESET}")
        print(f"  {RED}Gagal              : {self.failed}{RESET}")
        print(f"  Wajah terdeteksi   : {self.faces}")
        print("=" * 65)


# ─────────────────────────────────────────────────────────────
# Fungsi interaktif (jika tidak ada argumen CLI)
# ─────────────────────────────────────────────────────────────

def prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val    = input(f"  {label}{suffix}: ").strip()
    return val if val else default


def interactive_mode() -> argparse.Namespace:
    print(f"\n{BOLD}{CYAN}⚽  Soccer Clinic — Bulk Upload (Mode Interaktif){RESET}")
    print("─" * 55)

    folder       = prompt("Path folder foto")
    server       = prompt("URL server", "http://localhost:5000")
    password_in  = getpass.getpass("  Password fotografer: ")
    event        = prompt("Nama event", "Soccer Training")
    location     = prompt("Lokasi", "Lapangan Utama")
    photographer = prompt("Nama fotografer", "Anonymous")
    workers_str  = prompt("Jumlah upload paralel (1-10)", "3")
    recursive_in = prompt("Sertakan sub-folder? (y/n)", "n")

    ns = argparse.Namespace(
        folder       = folder,
        server       = server.rstrip("/"),
        password     = password_in,
        event        = event,
        location     = location,
        photographer = photographer,
        workers      = max(1, min(10, int(workers_str or "3"))),
        recursive    = recursive_in.lower() == "y",
        dry_run      = False,
        retries      = 3,
    )
    return ns


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bulk upload foto ke Soccer Clinic Face Recognition server"
    )
    parser.add_argument("--folder",       required=False, help="Path folder foto")
    parser.add_argument("--server",       default="http://localhost:5000", help="URL server")
    parser.add_argument("--password",     default="", help="Password fotografer")
    parser.add_argument("--event",        default="Soccer Training", help="Nama event")
    parser.add_argument("--location",     default="Lapangan Utama", help="Lokasi")
    parser.add_argument("--photographer", default="Anonymous", help="Nama fotografer")
    parser.add_argument("--workers",      type=int, default=3, help="Jumlah thread paralel")
    parser.add_argument("--recursive",    action="store_true", help="Cari foto di sub-folder")
    parser.add_argument("--dry-run",      action="store_true", help="Tampilkan daftar file saja, jangan upload")
    parser.add_argument("--retries",      type=int, default=3, help="Jumlah retry jika gagal")

    args = parser.parse_args()

    # Jika folder tidak diberikan via CLI → mode interaktif
    if not args.folder:
        args = interactive_mode()
    else:
        args.server = args.server.rstrip("/")
        if not args.password:
            args.password = getpass.getpass("  Password fotografer: ")

    print(f"\n{BOLD}⚽  Soccer Clinic — Bulk Upload{RESET}")
    print("=" * 65)

    # ── 1. Cek koneksi server ──────────────────────────────────
    print(f"\n🔌  Mengecek koneksi ke {args.server} ...")
    if not check_server(args.server):
        sys.exit(
            f"\n{RED}❌  Tidak dapat terhubung ke server {args.server}\n"
            f"    Pastikan app.py sudah berjalan.{RESET}"
        )

    # ── 2. Verifikasi password (cek ke endpoint health sekaligus) ──
    #    Server tidak punya endpoint verifikasi password tersendiri;
    #    password hanya divalidasi di sisi browser. Kita tetap tampilkan
    #    peringatan jika password tidak sesuai konstanta default.
    DEFAULT_PW = "Bayansoccer123!"
    if args.password != DEFAULT_PW:
        print(
            f"{YELLOW}⚠️   Password yang dimasukkan berbeda dari default. "
            f"Pastikan sudah benar.{RESET}"
        )

    # ── 3. Kumpulkan file gambar ────────────────────────────────
    print(f"\n🔍  Mencari foto di: {args.folder}")
    if args.recursive:
        print("     (termasuk sub-folder)")
    images = collect_images(args.folder, args.recursive)

    if not images:
        sys.exit(
            f"{RED}❌  Tidak ada file gambar yang ditemukan di folder tersebut.{RESET}"
        )

    print(f"\n{GREEN}📷  {len(images)} foto ditemukan:{RESET}")
    for i, img in enumerate(images, 1):
        size_kb = img.stat().st_size / 1024
        print(f"     {i:>4}. {img.name:<50} ({size_kb:>8.1f} KB)")

    if args.dry_run:
        print(f"\n{YELLOW}🔎  Dry-run mode — tidak ada yang diupload.{RESET}")
        return

    # ── 4. Konfirmasi ──────────────────────────────────────────
    print()
    print(f"  Event       : {args.event}")
    print(f"  Lokasi      : {args.location}")
    print(f"  Fotografer  : {args.photographer}")
    print(f"  Paralel     : {args.workers} thread")
    print(f"  Retry       : {args.retries}x jika gagal")
    confirm = input(f"\n{BOLD}🚀  Upload {len(images)} foto sekarang? (y/n): {RESET}").strip().lower()
    if confirm != "y":
        print("Upload dibatalkan.")
        return

    # ── 5. Metadata ────────────────────────────────────────────
    metadata = {
        "event_name":   args.event,
        "location":     args.location,
        "photographer": args.photographer,
        "date":         datetime.now().isoformat(),
    }

    # ── 6. Upload paralel ──────────────────────────────────────
    print(f"\n{BOLD}📤  Memulai upload...{RESET}\n")
    tracker   = ProgressTracker(total=len(images))
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                upload_one, img, args.server, metadata, args.retries
            ): img
            for img in images
        }

        for future in as_completed(futures):
            result = future.result()
            tracker.update(result)
            tracker.print_result(result)

    elapsed = time.time() - start_time

    # ── 7. Ringkasan ───────────────────────────────────────────
    tracker.summary()
    print(f"  Waktu total        : {elapsed:.1f} detik")
    avg = elapsed / len(images) if images else 0
    print(f"  Rata-rata per foto : {avg:.1f} detik")

    if tracker.failed:
        print(
            f"\n{YELLOW}⚠️   {tracker.failed} foto gagal diupload. "
            f"Periksa koneksi atau log server.{RESET}"
        )
    else:
        print(f"\n{GREEN}🎉  Semua foto berhasil diupload!{RESET}")


if __name__ == "__main__":
    main()