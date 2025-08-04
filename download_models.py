#!/usr/bin/env python3
"""
COVID-19 X-Ray Detection - Model Downloader
Tải xuống các model AI từ Google Drive
"""

import os
import requests
import zipfile
from tqdm import tqdm
import sys

def download_file_from_google_drive(file_id, destination):
    """Download file từ Google Drive"""
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    save_response_content(response, destination)

def get_confirm_token(response):
    """Lấy token xác nhận từ Google Drive"""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    """Lưu nội dung response với progress bar"""
    CHUNK_SIZE = 32768
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, "wb") as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination) as pbar:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def main():
    print("🩺 COVID-19 X-Ray Detection - Model Downloader")
    print("=" * 50)
    
    # Tạo thư mục models nếu chưa có
    if not os.path.exists('models'):
        os.makedirs('models')
        print("✅ Đã tạo thư mục models/")
    
    # Danh sách models cần download
    models = {
        'densenet201_covid_model.h5': '1-ABC123DEF456',  # Thay bằng ID thực
        'optimized_covid_model.h5': '1-XYZ789GHI012',    # Thay bằng ID thực
        'simple_covid_model.h5': '1-DEF345JKL678'        # Thay bằng ID thực
    }
    
    print("\n📥 Bắt đầu download models...")
    
    for model_name, file_id in models.items():
        model_path = os.path.join('models', model_name)
        
        if os.path.exists(model_path):
            print(f"⚠️ {model_name} đã tồn tại, bỏ qua...")
            continue
        
        print(f"\n📥 Đang download {model_name}...")
        try:
            download_file_from_google_drive(file_id, model_path)
            print(f"✅ Đã download thành công: {model_name}")
        except Exception as e:
            print(f"❌ Lỗi download {model_name}: {str(e)}")
    
    print("\n🎉 Hoàn thành download models!")
    print("\n📋 Hướng dẫn tiếp theo:")
    print("1. Chạy: python app.py")
    print("2. Mở browser: http://localhost:5000/multi")
    print("3. Upload ảnh X-ray để test")

if __name__ == "__main__":
    main() 