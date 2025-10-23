import json
import re

from controller import VoiceController
from ai_nlu_engine import UniversalNluEngine  # 텍스트 모드 직사용

def text_mode():
    """
    텍스트 입력 모드 (마이크 없이 테스트)
    """
    nlu = UniversalNluEngine()

    # 테스트용 코드
    test_cases = [
        "경보국 볼륨 30으로 방류 방송해줘",
        "경보국 시험 방송 시작",
        "수위국 데이터 가져와",
        "경보국 장비 점검해줘",
        "경바 볼륨 제일 작게 시험 방송",  # 오타 + 의미 변환
    ]

    # 실전 코드
    for text in test_cases:
        nlu.parse_text(text)
        input("계속하려면 Enter... (다음 테스트)")

    # print("\n" + "=" * 60)
    # print("⌨️  텍스트 입력 모드 (what–how–action)")
    # print("=" * 60)
    # print("예시:")
    # examples = [
    #     "DO 1~4 끄고, DI 21, 22번 상태 확인해",
    #     "현재 계측값 알려줘"          # "수위는 ~~m, 우량은 ~~m, 배터리 전압은 ~V 입니다."
    #     "경고등 빨간색 켜",
    #     "경고등 노란색 켜",
    #     "경고등 전체 다 깜빡 거려"     # -> 중지 명령을 할 때 까지 on, off 반복
    #     "경고등 전체 꺼"              # == "경고등 다 꺼"
    #     "지난 10분 동안의 로그 알려줘", # == "지난 10분 동안의 로그 읽어줘",
    # ]
    # for e in examples: print(" -", e)
    #
    # while True:
    #     text = input("\n📝 명령어 입력 (q 종료): ").strip()
    #     if text.lower() in ['q', 'quit', '종료', '끝']:
    #         print("\n👋 텍스트 모드 종료")
    #         break
    #     if not text:
    #         print("⚠️  입력 없음")
    #         continue
    #
    #     cmd = nlu.parse_text(text)
    #     print("\n" + "-" * 60)
    #     if "error" in cmd:
    #         print("❌", cmd["error"])
    #     else:
    #         print("✅ what–how–action 결과:")
    #         result = json.dumps(cmd, indent=4, ensure_ascii=False)
    #         result = re.sub(r'\[\s*([\d,\s]+)\s*\]', lambda m: '[' + re.sub(r'\s+', ' ', m.group(1).strip()) + ']', result)
    #         print(result)
    #     print("-" * 60)


def voice_mode():
    """
    음성 입력 모드 (마이크 사용)
    """
    MODEL_PATH = "model"
    try:
        print("\n" + "=" * 60)
        print("🎤 음성 입력 모드")
        print("=" * 60)

        controller = VoiceController(MODEL_PATH, max_port=8)
        command = controller.start_command_recognition()

        print("\n🎯 최종 명령어:")
        print(json.dumps(command, indent=2, ensure_ascii=False))

    except FileNotFoundError:
        print("\n❌ 'model' 폴더 없음. Vosk 한국어 모델을 넣어주세요.")
        print("   https://alphacephei.com/vosk/models")
    except Exception as e:
        print(f"\n❌ 예외: {e}")


def main():
    print("=" * 60)
    print("🤖 대화형 RTU NLU 데모 (what–how–action)")
    print("=" * 60)
    print("  1. 🎤 음성 입력 모드")
    print("  2. ⌨️  텍스트 입력 모드")
    print("  q. 종료\n")

    while True:
        text_mode()

    # while True:
    #     choice = input("선택 (1/2/q): ").strip()
    #     if choice == '1':
    #         voice_mode()
    #         break
    #     elif choice == '2':
    #         text_mode()
    #         break
    #     elif choice.lower() in ['q', 'quit', '종료']:
    #         print("\n👋 종료")
    #         break
    #     else:
    #         print("⚠️  1, 2, 또는 q 입력")

if __name__ == "__main__":
    main()
