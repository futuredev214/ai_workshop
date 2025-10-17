# utils.py
from datetime import datetime

def log(message: str):
    """
    타임스탬프와 함께 로그 출력

    Args:
        message (str): 출력할 메시지
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")