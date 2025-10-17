from stt_engine import SpeechToTextEngine
from ai_nlu_engine import UniversalNluEngine
from utils import log

class VoiceController:
    def __init__(self, model_path: str, max_port: int = 8):
        """
        음성 제어 컨트롤러 초기화
        @param model_path: Vosk 모델 경로
        @param max_port:   최대 포트 번호 (기본 8)
        """
        # STT 엔진
        self.stt_engine = SpeechToTextEngine(model_path)
        # NLU 엔진 (what–how–action)
        self.nlu_engine = UniversalNluEngine(max_port=max_port)

    def start_command_recognition(self):
        """
        음성 명령 인식 시작
        @return dict: what–how–action 결과 JSON
        """
        log("=" * 50)
        log("🎙️  음성 제어 시스템 시작")
        log("=" * 50)

        while True:
            text = self.stt_engine.listen_and_transcribe()
            if not text:
                log("⚠️  인식 실패. 다시 말씀해주세요.\n")
                continue

            log(f"\n📝 인식된 텍스트: '{text}'")
            command = self.nlu_engine.parse_text(text)

            if "error" in command:
                log(f"❌ {command['error']}")
                log("💡 예시: '디오 1번 꺼줘', '시스템 재부팅해', '지난 10분 로그 가져와'\n")
                continue

            log("\n" + "=" * 50)
            log("✅ 명령어 생성 완료!")
            log("=" * 50)
            return command
