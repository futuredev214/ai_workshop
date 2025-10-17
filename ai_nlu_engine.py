import os
from utils import log
import pandas as pd
from keybert import KeyBERT

# 오프라인 허용 (모델이 로컬에 있을 때)
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ["TRANSFORMERS_OFFLINE"] = os.getenv("TRANSFORMERS_OFFLINE", "0")

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re

class UniversalNluEngine:
    """범용 제어 시스템 NLU 엔진"""
    def __init__(self):
        log("🤖 AI 모델 로딩 중...")
        model_dir = r"D:\models\xlmR_xnli"

        # 옵션 1: 영어 학습 모델
        # self.classifier = pipeline(
        #     "zero-shot-classification",
        #     model="facebook/bart-large-mnli",
        #     what=-1
        # )

        # 옵션 2: 한국어 특화 모델 (KoBERT 기반)
        self.classifier = pipeline(
            "zero-shot-classification",
            model=AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True),
            tokenizer=AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=False),
        )

        self.keyword_extractor = KeyBERT('distiluse-base-multilingual-cased-v2')

        log("✅ AI 모델 로딩 완료!")

        # 구분 타입
        self.what_TYPES = {
            # 기본 I/O
            "do": "DO", "디지털 출력": "DO", "digital output": "DO",
            "di": "DI", "디지털 입력": "DI", "digital input": "DI",
            "ao": "AO", "아날로그 출력": "AO", "analog output": "AO",
            "ai": "AI", "아날로그 입력": "AI", "analog input": "AI",

            # 긴급 시, 경보 후 로그 남기기
            "경보": "ALERT", "ALERT": "ALERT",
            "로그": "LOG", "log": "LOG",

            # 통신 상태
            "uart": "COM", "com": "COM", "시리얼": "COM", "serial": "COM",

            # 센서
            "수위": "WATERLEVEL", "waterlevel": "WATERLEVEL",
            "우량": "RAINFALL", "rainfall": "RAINFALL",
            "배터리 전압": "BATTERY_VOLTAGE", "batteryvoltage": "BATTERY_VOLTAGE",
        }

        # 지원하는 명령 타입
        self.COMMAND_TYPES = [
            "control", "broadcast", "log", "read", "write"
        ]

        self.ACTION_ON = ["켜", "켜줘", "on", "start", "활성", "활성화", "작동"]
        self.ACTION_OFF = ["꺼", "꺼줘", "끄", "off", "stop", "비활성", "비활성화", "정지"]
        self.ACTION_READ = ["읽", "read", "조회", "확인", "가져와"]
        self.ACTION_WRITE = ["쓰", "write", "설정", "세팅", "변경", "바꿔"]
        self.ACTION_QUERY = ["상태", "status", "어때", "어떻게", "query"]

        # 잡음 키워드
        self.NOISE = ["음","어","으","아","이제","좀","약간","그","저","뭐","뭐시기",
                     "그거","저거","이거","있잖아","있잖아요","요","잠깐","빨리"]

    def _extract_keywords(self, text: str, top_n: int = 5):
        """문장에서 주요 키워드 추출 (명사, 행위어 등)"""
        text = self._preprocess(text)
        keywords = self.keyword_extractor.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words=None,
            top_n=top_n
        )
        # [('경보국', 0.74), ('포트', 0.68)...] → ['경보국', '포트', ...]
        return [kw for kw, score in keywords]

    # 0. 잡음 처리
    def _preprocess(self, text: str) -> str:
        """잡음 제거 전처리"""
        t = text.strip().lower()
        for n in self.NOISE:
            t = re.sub(r'\b' + re.escape(n) + r'\b', ' ', t)
            t = t.replace(' ' + n + ' ', ' ').replace(' ' + n, ' ').replace(n + ' ', ' ')
        return re.sub(r'\s+', ' ', t).strip()

    # 1. (AI) 명령 Type 추론
    def _classify_command_type(self, text: str):
        """명령 타입 분류 (수정!)"""

        # 1. control (장비 제어: 켜기/끄기/작동)
        if any(kw in text for kw in ["켜", "꺼", "작동", "정지", "on", "off", "start", "stop", "activate", "deactivate"]):
            return "control"

        # 2. broadcast (방송 또는 알림 송출)
        if any(kw in text for kw in ["방송", "안내", "송출", "출력", "재생", "broadcast", "announce", "play"]):
            return "broadcast"

        # 3. log (기록 또는 로그 조회)
        if any(kw in text for kw in ["로그", "기록", "이력", "history", "log"]):
            return "log"

        # 4. read (값 조회 / 센서 데이터 읽기)
        if any(kw in text for kw in ["읽어", "확인", "조회", "측정", "read", "measure", "get", "status"]):
            return "read"

        # 5. write (설정 변경 / 데이터 쓰기)
        if any(kw in text for kw in ["설정", "변경", "입력", "저장", "쓰기", "write", "set", "update"]):
            return "write"

        # 기타: 부정확한 발음은 AI 분류기 사용
        z = self.classifier(
            text, candidate_labels=self.COMMAND_TYPES, multi_label=False
        )

        return z["labels"][0]

    # 2. (AI) 무엇을 수행할지 추론
    def _extract_what(self, text: str):
        """ What 추출 """
        text_lower = text.lower()

        # 명확하게 DI(디아이) 라고 입력 받은 경우 즉시 return
        for keyword, what_code in self.what_TYPES.items(): # "디아이", "DI"
            if keyword in text_lower:
                return what_code # "DI"

        # 음성 인식 등 명확하게 DI 라고 입력 받지 못한 경우 NLU model을 동작시킴
        # 후보중 의미적 유사도에 가까운 값을 반환함
        z = self.classifier(text, candidate_labels=list(set(self.what_TYPES.values())),
                            multi_label=True)

        df = pd.DataFrame({
            '후보': z['labels'],  # 레이블 목록
            '정확도': z['scores']  # 해당 점수
        })

        # 점수 기준 내림차순 정렬 (이미 정렬되어 있지만 명시적으로)
        df = df.sort_values('정확도', ascending=False).reset_index(drop=True)
        print(df) # 보기 좋게 출력

        return z["labels"][0]

    # 3. Target (write 타입) 추론
    def _extract_value_for_write(self, text: str, numbers: list):
        """write 명령용 값 추출"""
        target_match = re.search(r'(\d+)\s*번', text)
        target = int(target_match.group(1)) if target_match else numbers[0]

        percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
        if percent_match:
            value = float(percent_match.group(1)) / 100.0
            return target, value

        value_match = re.search(r'(?:값|설정|세팅).*?(\d+(?:\.\d+)?)', text)
        if value_match:
            value = float(value_match.group(1))
            if value > 1:
                value = value / 100.0
            return target, value

        return target, None

    # O1. 다중 명령 처리
    def _split_commands(self, text: str):
        """여러 명령 분리"""
        parts = re.split(r'((?:끄|켜|설정하|활성화하|비활성화하|읽)고)', text)

        commands = []
        for i in range(0, len(parts) - 1, 2):
            connector = parts[i + 1]
            action_word = connector.replace("고", "")
            cmd_text = (parts[i].strip() + " " + action_word).strip()
            cmd_text = cmd_text.strip(',').strip()

            if cmd_text:
                commands.append(cmd_text)
                log(f"  [분리] '{cmd_text}'")

        if len(parts) % 2 == 1:
            last_cmd = parts[-1].strip().strip(',').strip()
            if last_cmd:
                commands.append(last_cmd)
                log(f"  [분리] '{last_cmd}'")

        return commands

    # 3. Target 추출
    def _extract_target(self, text: str):
        """대상 추출"""
        numbers = [int(m) for m in re.findall(r'\d+', text)]

        if not numbers:
            if any(kw in text for kw in ["모두", "전체", "다", "all"]):
                return {"type": "target", "values": list(range(1, 9))}
            return None

        if any(kw in text for kw in ["포트", "target"]):
            target_type = "target"
        elif any(kw in text for kw in ["채널", "channel", "ch"]):
            target_type = "channel"
        else:
            target_type = "target"

        range_patterns = [
            r'(\d+)\s*(?:번)?\s*(?:에서)\s*(\d+)\s*(?:번)?',
            r'(\d+)\s*(?:번)?\s*(?:부터)\s*(\d+)\s*(?:번)?(?:\s*까지)?',
            r'(\d+)\s*[~\-]\s*(\d+)',
        ]

        for pattern in range_patterns:
            match = re.search(pattern, text)
            if match:
                start, end = int(match.group(1)), int(match.group(2))

                if start > end:
                    start, end = end, start

                range_nums = set(range(start, end + 1))

                if "그리고" in text or "하고" in text:
                    all_nums = range_nums.union(set(numbers))
                    values = sorted(list(all_nums))
                else:
                    values = list(range(start, end + 1))

                log(f"  [범위 인식] {start}~{end} → {values}")
                return {"type": target_type, "values": values}

        log(f"  [개별 인식] {numbers}")
        return {"type": target_type, "values": numbers}

    # 4. Action 추출
    def _extract_action(self, text: str, cmd_type: str):
        """동작 추출"""
        if cmd_type == "control":
            if any(kw in text for kw in ["켜", "on", "start", "활성"]):
                return {"action": "on"}
            elif any(kw in text for kw in ["꺼", "끄", "off", "stop", "비활성"]):
                return {"action": "off"}

        elif cmd_type == "query":
            return {"action": "query"}

        elif cmd_type == "read":
            return {"action": "read"}

        elif cmd_type == "write":
            numbers = [int(m) for m in re.findall(r'\d+', text)]
            if numbers:
                target, value = self._extract_value_for_write(text, numbers)
                return {"action": "write", "target": target}
            return None

        elif cmd_type == "status":
            return {"action": "status"}

        elif cmd_type == "log":
            return {"action": "log"}

    def _parse_single_command_with_what(self, text: str, forced_what: str = None):
        """단일 명령 분석"""
        cmd_type = self._classify_command_type(text)
        log(f"  타입: {cmd_type}")

        if forced_what:
            what = forced_what
        else:
            what = self._extract_what(text)
            log(f"  장치: {what}")

        if not what:
            return {"error": "장치를 인식하지 못했습니다."}

        action = self._extract_action(text, cmd_type)

        if cmd_type == "write" and action and "target" in action:
            command = {
                "type": cmd_type,
                "what": what,
                "target": action["target"],
                "action": action["action"],
            }
            log(f"  포트: {action['target']}")
            log(f"  값: {action['value']}")
            return command


        target = self._extract_target(text)

        if not target:
            print("타켓을 찾을 수 없습니다. 건너 뜁니다.")
            # return {"error": "대상을 찾을 수 없습니다."}

        else:
            log(f"  {target['type']}: {target['values']}")

        if not action:
            print("동작을 인식할 수 없습니다.")
            # return {"error": "동작을 인식하지 못했습니다."}

        command = {"type": cmd_type, "what": what}

        if len(target['values']) == 1:
            command[target['type']] = target['values'][0]
        else: # 다중 명령 인식
            command[target['type']] = target['values']
            log(f"  동작: {action}")

            command.update(action)

        return command

    def _parse_single_command(self, text: str):
        """단일 명령 분석"""

        keywords = self._extract_keywords(text)
        log(f"  [키워드] {keywords}")

        return self._parse_single_command_with_what(keywords, forced_what=None)


    def parse_text(self, text: str):
        """텍스트 분석 (다중 명령 지원)"""
        if not text or not text.strip():
            return {"error": "입력된 텍스트가 없습니다."}

        log(f"[감지한 텍스트] {text}")
        log("=" * 60)

        if any(kw in text for kw in ["끄고", "켜고", "하고", "다음에"]):
            log("[다중 명령 감지]")

            common_what = self._extract_what(text)
            log(f"  공통 장치: {common_what}")

            commands_text = self._split_commands(text)
            log(f"  분리된 명령: {len(commands_text)}개")

            results = []
            last_what = common_what

            for i, cmd_text in enumerate(commands_text, 1):
                log(f"[명령 {i}] {cmd_text}")

                what = self._extract_what(cmd_text)

                if not what:
                    what = last_what
                    log(f"  장치: {what} (상속됨)")
                else:
                    last_what = what
                    log(f"  장치: {what}")

                if not what:
                    log(f"  ⚠️ 장치를 찾을 수 없어 스킵합니다.")
                    continue

                result = self._parse_single_command_with_what(cmd_text, what)

                if "error" not in result:
                    results.append(result)

            log("=" * 60)

            if len(results) == 0:
                return {"error": "유효한 명령을 찾을 수 없습니다."}
            elif len(results) == 1:
                return results[0]
            else:
                return {"명령어": results, "명령 개수": len(results)}

        result = self._parse_single_command(text)
        log("=" * 60)
        return result