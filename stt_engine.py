import json
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from utils import log

class SpeechToTextEngine:
    def __init__(self, model_path: str):
        """
        음성 인식 엔진 초기화

        Args:
            model_path (str): Vosk 모델이 저장된 폴더 경로
        """
        # Vosk 모델 로드
        self.model = Model(model_path)

        # 16kHz 샘플레이트로 인식기 생성
        # 16000 = 음성 인식에 적합한 표준 주파수
        self.recognizer = KaldiRecognizer(self.model, 16000)

        # 오디오 데이터를 임시 저장할 큐 (queue = 대기열)
        self.audio_queue = queue.Queue()

    def _audio_callback(self, indata, frames, time, status):
        """
        마이크에서 오디오가 들어올 때마다 자동으로 호출되는 함수

        Args:
            indata: 마이크로 들어온 오디오 데이터 (numpy array)
            frames: 오디오 프레임 개수
            time: 타임스탬프
            status: 오디오 스트림 상태
        """
        if status:
            log(f"[오디오 에러] {status}")

        # 들어온 오디오 데이터를 큐에 저장
        self.audio_queue.put(bytes(indata))

    def listen_and_transcribe(self):
        """
        마이크로 음성을 듣고 텍스트로 변환

        Returns:
            str: 인식된 텍스트 (실패 시 빈 문자열)
        """
        log("\n🎤 듣고 있습니다... 말씀하세요!")

        try:
            # 마이크 스트림 시작
            # samplerate=16000: 초당 16000번 샘플링 (음성 인식 표준)
            # blocksize=8000: 한 번에 처리할 샘플 개수
            # channels=1: 모노 (스테레오는 2)
            with sd.RawInputStream(
                    samplerate=16000,
                    blocksize=8000,
                    dtype='int16',
                    channels=1,
                    callback=self._audio_callback
            ):
                while True:
                    # 큐에서 오디오 데이터 가져오기
                    data = self.audio_queue.get()

                    # Vosk 인식기에 데이터 전달
                    if self.recognizer.AcceptWaveform(data):
                        # 문장이 완성되었을 때
                        result = json.loads(self.recognizer.Result())
                        text = result.get('text', '')

                        if text:
                            log(f"✅ 인식됨: {text}")
                            return text
                    else:
                        # 중간 인식 결과 (partial result)
                        partial = json.loads(self.recognizer.PartialResult())
                        partial_text = partial.get('partial', '')

                        if partial_text:
                            log(f"🔄 인식 중: {partial_text}", end='\r')

        except KeyboardInterrupt:
            log("\n⚠️ 사용자가 중단했습니다.")
            return ""
        except Exception as e:
            log(f"\n❌ 에러 발생: {e}")
            return ""