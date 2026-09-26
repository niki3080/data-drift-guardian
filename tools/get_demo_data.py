import os
import re
import gzip
import shutil
import zipfile
import urllib.request
import urllib.parse
import http.cookiejar


def load_env(path='.env'):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value


def extract_file_id(url):
    """Извлекает FILE_ID из ссылки Google Drive."""
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    match = re.search(r'id=([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    raise ValueError("Не удалось извлечь ID файла из ссылки Google Drive")


def download_from_gdrive(url, destination):
    """Скачивание с Google Drive с обработкой подтверждения для больших файлов."""
    file_id = extract_file_id(url)

    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

    base_url = f"https://drive.google.com/uc?id={file_id}"
    response = opener.open(base_url)

    token = None
    for cookie in cj:
        if cookie.name.startswith('download_warning'):
            token = cookie.value

    if token:
        confirm_url = f"{base_url}&confirm={token}"
        response = opener.open(confirm_url)

    with open(destination, 'wb') as f:
        f.write(response.read())

    return destination


def is_gzip_file(path):
    with open(path, 'rb') as f:
        return f.read(2) == b'\x1f\x8b'


def extract_archive(archive_path, output_path):
    """Распаковывает архив, если он сжат, иначе просто оставляет файл как есть."""
    if zipfile.is_zipfile(archive_path):
        print("Обнаружен ZIP-архив, распаковываю...")
        with zipfile.ZipFile(archive_path, 'r') as zf:
            names = zf.namelist()
            zf.extract(names[0], path=os.path.dirname(output_path))
            extracted_path = os.path.join(os.path.dirname(output_path), names[0])
            os.replace(extracted_path, output_path)
        print(f"Файл распакован: {output_path}")

    elif is_gzip_file(archive_path):
        print("Обнаружен GZIP-архив, распаковываю...")
        with gzip.open(archive_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(archive_path)
        print(f"Файл распакован: {output_path}")

    else:
        print("Файл не сжат, переименовываю...")
        if archive_path != output_path:
            os.replace(archive_path, output_path)


if __name__ == "__main__":
    load_env()
    url = os.environ.get('DATASET_URL')

    if url is None:
        raise ValueError("DATASET_URL не найден в .env файле")

    os.makedirs('data', exist_ok=True)

    temp_path = 'data/_temp_download'
    final_path = 'data/reference.csv'

    # Если ссылка на Google Drive - используем специальную логику
    if 'drive.google.com' in url:
        download_from_gdrive(url, temp_path)
    else:
        urllib.request.urlretrieve(url, temp_path)

    extract_archive(temp_path, final_path)
