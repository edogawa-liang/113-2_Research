import zipfile
import argparse
import os

# python tools/unzip.py --zip_path datasets.zip

def parse_args():
    parser = argparse.ArgumentParser(description="Unzip file to specified folder.")
    parser.add_argument("--zip_path", type=str, required=True, help="Path to the .zip file")
    parser.add_argument("--extract_dir", type=str, default=None, help="Extract directory (default: same as zip file)")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    zip_path = args.zip_path

    # 預設解壓縮路徑：與 zip 同資料夾下，資料夾名稱為 zip 檔名去掉副檔名
    if args.extract_dir is None:
        base_name = os.path.splitext(os.path.basename(zip_path))[0]  # 去掉副檔名
        extract_dir = os.path.join(os.path.dirname(zip_path), base_name)
    else:
        extract_dir = args.extract_dir

    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"已解壓縮到 {extract_dir}")
