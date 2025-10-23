from flask import Flask, request, jsonify
from ai_nlu_engine import UniversalNluEngine
import json

app = Flask(__name__)

# NLU 엔진 초기화 (한 번만)
print("🤖 AI 엔진 로딩 중...")
nlu_engine = UniversalNluEngine()
print("✅ AI 엔진 준비 완료!")


@app.route('/process', methods=['POST'])
def process_voice_command():
    """
    Android에서 받은 음성 텍스트 처리

    요청: {"text": "디지털 출력 1번 켜"}
    응답: {"response": "디지털 출력 1번을 켰습니다", "command": {...}}
    """
    try:
        # 1. 텍스트 수신
        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({
                "error": "텍스트가 없습니다",
                "response": "명령을 인식하지 못했습니다"
            }), 400

        print(f"\n{'=' * 60}")
        print(f"📱 Android로부터 수신: {text}")
        print(f"{'=' * 60}")

        # 2. NLU: 텍스트 → JSON 명령
        command = nlu_engine.parse_text(text)

        if "error" in command:
            print(f"❌ 에러: {command['error']}")
            return jsonify({
                "response": command['error'],
                "command": command
            })

        print(f"✅ 명령 생성:")
        print(json.dumps(command, indent=2, ensure_ascii=False))

        # 3. NLG: JSON → 자연스러운 응답
        response_text = generate_response(command)

        print(f"📣 응답: {response_text}")
        print(f"{'=' * 60}\n")

        # 4. 하드웨어 제어 (시리얼 통신 등)
        # execute_hardware_command(command)  # 나중에 구현

        return jsonify({
            "response": response_text,
            "command": command,
            "status": "success"
        })

    except Exception as e:
        print(f"❌ 예외 발생: {e}")
        import traceback
        traceback.print_exc()

        return jsonify({
            "error": str(e),
            "response": "명령을 처리할 수 없습니다"
        }), 500


def generate_response(command: dict) -> str:
    """
    NLG: JSON 명령을 자연스러운 한국어로 변환

    @param command: NLU가 생성한 JSON 명령
    @return: 자연스러운 응답 문장
    """
    try:
        cmd_type = command.get('type', '')
        what = command.get('what', '')
        action = command.get('action', {})

        # 장치 이름 매핑
        device_names = {
            'DO': '디지털 출력',
            'DI': '디지털 입력',
            'AO': '아날로그 출력',
            'AI': '아날로그 입력',
            'ALERT': '경보',
            'COM': '통신',
            'WATERLEVEL': '수위',
            'RAINFALL': '우량',
            'BATTERY_VOLTAGE': '배터리 전압'
        }

        device = device_names.get(what, what)

        # 제어 명령
        if cmd_type == 'control':
            action_type = action.get('action', '')
            target = command.get('target')

            if action_type == 'on':
                if isinstance(target, list):
                    return f"{device} {', '.join(map(str, target))}번을 켰습니다"
                else:
                    return f"{device} {target}번을 켰습니다"

            elif action_type == 'off':
                if isinstance(target, list):
                    return f"{device} {', '.join(map(str, target))}번을 껐습니다"
                else:
                    return f"{device} {target}번을 껐습니다"

        # 방송 명령
        elif cmd_type == 'broadcast':
            return "경보 방송을 시작합니다"

        # 읽기 명령
        elif cmd_type == 'read':
            return f"{device} 값을 조회하고 있습니다"

        # 로그 명령
        elif cmd_type == 'log':
            return "로그를 조회하고 있습니다"

        return "명령을 수행했습니다"

    except Exception as e:
        print(f"⚠️ NLG 에러: {e}")
        return "명령을 수행했습니다"


@app.route('/health', methods=['GET'])
def health_check():
    """헬스 체크 엔드포인트"""
    return jsonify({"status": "ok", "message": "라즈베리파이 서버 정상 작동 중"})


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🍓 라즈베리파이 음성 제어 서버")
    print("=" * 60)
    print("📡 서버 시작: http://0.0.0.0:5000")
    print("💡 Android 앱에서 연결하세요")
    print("=" * 60 + "\n")

    # 모든 네트워크 인터페이스에서 접속 허용
    app.run(host='0.0.0.0', port=5000, debug=True)