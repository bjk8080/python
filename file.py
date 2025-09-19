import json
import csv
import logging
from pathlib import Path

# -----------------------------
# 1️⃣ 사용자 정의 예외
# -----------------------------
class FileHandlerError(Exception):
    """기본 파일 처리 예외"""
    pass

class FileNotExistError(FileHandlerError):
    """파일이 존재하지 않을 때 발생"""
    pass

class FilePermissionError(FileHandlerError):
    """파일 접근 권한이 없을 때 발생"""
    pass

class FileFormatError(FileHandlerError):
    """파일 형식이 잘못되었을 때 발생"""
    pass

# -----------------------------
# 2️⃣ 파일 처리기 클래스
# -----------------------------
# 로깅 설정
logging.basicConfig(
    filename='file_handler.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FileHandler:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotExistError(f"{self.filepath} 파일이 존재하지 않습니다.")
    
    # --- 텍스트 ---
    def read_text(self):
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except PermissionError:
            logging.error(f"{self.filepath} 파일 권한 오류")
            raise FilePermissionError(f"{self.filepath} 접근 권한이 없습니다.")
        except Exception as e:
            logging.error(f"텍스트 파일 읽기 오류: {e}")
            raise FileHandlerError(str(e))
    
    def write_text(self, content):
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        except PermissionError:
            logging.error(f"{self.filepath} 파일 권한 오류")
            raise FilePermissionError(f"{self.filepath} 접근 권한이 없습니다.")
        except Exception as e:
            logging.error(f"텍스트 파일 쓰기 오류: {e}")
            raise FileHandlerError(str(e))
    
    # --- JSON ---
    def read_json(self):
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.error(f"{self.filepath} JSON 형식 오류")
            raise FileFormatError(f"{self.filepath} JSON 형식이 잘못되었습니다.")
        except PermissionError:
            logging.error(f"{self.filepath} 파일 권한 오류")
            raise FilePermissionError(f"{self.filepath} 접근 권한이 없습니다.")
        except Exception as e:
            logging.error(f"JSON 파일 읽기 오류: {e}")
            raise FileHandlerError(str(e))
    
    def write_json(self, data):
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except PermissionError:
            logging.error(f"{self.filepath} 파일 권한 오류")
            raise FilePermissionError(f"{self.filepath} 접근 권한이 없습니다.")
        except Exception as e:
            logging.error(f"JSON 파일 쓰기 오류: {e}")
            raise FileHandlerError(str(e))
    
    # --- CSV ---
    def read_csv(self):
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                return list(reader)
        except PermissionError:
            logging.error(f"{self.filepath} 파일 권한 오류")
            raise FilePermissionError(f"{self.filepath} 접근 권한이 없습니다.")
        except Exception as e:
            logging.error(f"CSV 파일 읽기 오류: {e}")
            raise FileHandlerError(str(e))
    
    def write_csv(self, data):
        try:
            with open(self.filepath, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data)
        except PermissionError:
            logging.error(f"{self.filepath} 파일 권한 오류")
            raise FilePermissionError(f"{self.filepath} 접근 권한이 없습니다.")
        except Exception as e:
            logging.error(f"CSV 파일 쓰기 오류: {e}")
            raise FileHandlerError(str(e))
    
    # --- 바이너리 ---
    def read_binary(self):
        try:
            with open(self.filepath, 'rb') as f:
                return f.read()
        except PermissionError:
            logging.error(f"{self.filepath} 파일 권한 오류")
            raise FilePermissionError(f"{self.filepath} 접근 권한이 없습니다.")
        except Exception as e:
            logging.error(f"바이너리 파일 읽기 오류: {e}")
            raise FileHandlerError(str(e))
    
    def write_binary(self, data):
        try:
            with open(self.filepath, 'wb') as f:
                f.write(data)
        except PermissionError:
            logging.error(f"{self.filepath} 파일 권한 오류")
            raise FilePermissionError(f"{self.filepath} 접근 권한이 없습니다.")
        except Exception as e:
            logging.error(f"바이너리 파일 쓰기 오류: {e}")
            raise FileHandlerError(str(e))
