import os
import pandas as pd
import numpy as np
import ssl

# Bypass SSL certificate verification issues on local environments
ssl._create_default_https_context = ssl._create_unverified_context

def check_and_download_datasets():
    """
    Check if Boston Housing, Auto-MPG, and California Housing datasets exist in demo/data/.
    If they are missing, download, clean, rename to Vietnamese, and cache them as clean CSVs.
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Boston Housing
    boston_path = os.path.join(data_dir, "boston.csv")
    if not os.path.exists(boston_path):
        try:
            print("[DOWNLOAD] Downloading Boston Housing dataset...")
            url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
            df = pd.read_csv(url)
            
            # Keep all 14 columns as requested for completeness
            df = df.dropna()
            
            # Rename to Vietnamese
            df.columns = [
                'Tỷ Lệ Tội Phạm', 'Đất Ở Quy Hoạch', 'Tỷ Lệ Đất Công Nghiệp', 
                'Giáp Sông Charles', 'Khí Thải Độc Hại', 'Số Phòng Trung Bình', 
                'Thâm Niên Nhà Ở', 'Khoảng Cách Việc Làm', 'Khả Năng Tiếp Cận Cao Tốc', 
                'Thuế Bất Động Sản', 'Tỷ Lệ Học Sinh/Thầy Cô', 'Chỉ Số Sắc Tộc', 
                'Dân Cư Lao Động', 'Giá Nhà'
            ]
            
            df.to_csv(boston_path, index=False)
            print(f"[SUCCESS] Boston Housing saved at {boston_path} ({df.shape[0]} samples)")
        except Exception as e:
            print(f"[ERROR] Failed to fetch Boston Housing: {e}")
            
    # 2. California Housing
    california_path = os.path.join(data_dir, "california.csv")
    if not os.path.exists(california_path):
        try:
            print("[DOWNLOAD] Downloading California Housing dataset...")
            url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
            df = pd.read_csv(url)
            
            # Map categorical column to numeric categories to prevent tensor errors
            if 'ocean_proximity' in df.columns:
                df['ocean_proximity'] = df['ocean_proximity'].astype('category').cat.codes
                
            # Keep all 10 columns as requested for completeness
            df = df.dropna()
            
            # Rename to Vietnamese
            df.columns = [
                'Kinh Độ', 'Vĩ Độ', 'Tuổi Thọ Nhà', 'Tổng Số Phòng', 
                'Tổng Số Phòng Ngủ', 'Dân Số', 'Số Hộ Gia Đình', 
                'Thu Nhập Trung Vị', 'Giá Nhà', 'Khoảng Cách Đại Dương'
            ]
            
            # Downsample california for extremely fast training on web UI
            if df.shape[0] > 1000:
                df = df.sample(n=1000, random_state=42)
                
            df.to_csv(california_path, index=False)
            print(f"[SUCCESS] California Housing saved at {california_path} ({df.shape[0]} samples)")
        except Exception as e:
            print(f"[ERROR] Failed to fetch California Housing: {e}")
            
    # 3. Cleveland Heart Disease (Medical/Clinical)
    heart_path = os.path.join(data_dir, "y_te_benh_tim.csv")
    if not os.path.exists(heart_path):
        try:
            print("[DOWNLOAD] Downloading Cleveland Heart Disease dataset...")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
            columns = [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
            ]
            df = pd.read_csv(url, names=columns, na_values="?")
            
            # Keep all 14 original columns as requested for completeness
            df = df.dropna()
            
            # Rename all 14 columns to Vietnamese
            df.columns = [
                'Tuổi Tác', 'Giới Tính', 'Loại Đau Ngực', 'Huyết Áp Lúc Nghỉ', 
                'Cholesterol', 'Đường Huyết Lúc Đói', 'Điện Tâm Đồ Lúc Nghỉ', 
                'Nhịp Tim Tối Đa', 'Đau Ngực Khi Gắng Sức', 'Suy Giảm ST Gắng Sức', 
                'Độ Dốc Đoạn ST', 'Số Mạch Máu Nhuộm', 'Kiểu Thal', 'Nguy Cơ Bệnh Tim'
            ]
            
            df.to_csv(heart_path, index=False)
            print(f"[SUCCESS] Cleveland Heart Disease saved at {heart_path} ({df.shape[0]} samples)")
        except Exception as e:
            print(f"[ERROR] Failed to fetch Cleveland Heart Disease: {e}")

if __name__ == '__main__':
    check_and_download_datasets()
