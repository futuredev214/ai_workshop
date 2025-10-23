import os
from utils import log
import pandas as pd
from keybert import KeyBERT

# 오프라인 허용 (모델이 로컬에 있을 때)
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ["TRANSFORMERS_OFFLINE"] = os.getenv("TRANSFORMERS_OFFLINE", "0")

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
import torch

class UniversalNluEngine:
    """범용 제어 시스템 NLU 엔진"""
    def __init__(self):
        # 지원하는 명령 타입
        self.COMMAND_TYPES = [
            "control", "broadcast", "log", "read", "write"
        ]

        # 레이블 → 가설문 매핑
        self.HYP_DETAILED = {
            "ALERT": {
                "description": "사용자 또는 관리자에게 경고, 알림, 비상 상황을 알리는 방송이나 메시지 전송을 지시합니다.",
                "examples": [
                    "침수 위험 경보를 발령하라.",
                    "관리자에게 즉시 문자 알림을 보내.",
                    "비상 방송 시스템을 작동시켜라.",
                    "사이렌을 울려."
                ],
                "action_verbs": ["알리다", "경보하다", "방송하다", "발령하다", "전송하다", "보내다", "울리다"],
                "target_objects": ["경보", "알림", "비상 방송", "사이렌", "안내 방송", "관리자", "사용자"]
            },

            "LOG": {
                "description": "현재 상태, 특정 이벤트 발생, 센서 데이터 값 등을 시스템 로그 파일이나 데이터베이스에 기록(저장)하라는 지시입니다.",
                "examples": [
                    "현재 수위를 로그에 기록해.",
                    "모든 센서 값을 데이터베이스에 저장해.",
                    "펌프 작동 이벤트 로그를 남겨라."
                ],
                "action_verbs": ["기록하다", "저장하다", "로그를 남기다", "쓰다"],
                "target_objects": ["로그", "데이터", "이벤트", "현재 상태", "값", "파일", "DB"]
            },

            "DO": {
                "description": "디지털 출력(DO) 포트를 제어하여 특정 장치(예: 릴레이, 펌프, 밸브)를 켜거나(ON) 끄는(OFF) 동작을 지시합니다.",
                "examples": [
                    "1번 펌프를 켜라.",
                    "밸브 3번을 닫아.",
                    "DO 2번 포트 OFF 시켜.",
                    "경광등을 작동시켜라."
                ],
                "action_verbs": ["켜다", "끄다", "작동시키다", "정지시키다", "열다", "닫다", "ON", "OFF"],
                "target_objects": ["펌프", "밸브", "릴레이", "모터", "팬", "경광등", "DO 포트"]
            },
            "DI": {
                "description": "디지털 입력(DI) 포트의 현재 상태(ON/OFF, High/Low)를 확인하거나 읽어오라는 지시입니다.",
                "examples": [
                    "1번 스위치 상태 확인해.",
                    "DI 3번 입력 값이 뭐야?",
                    "문 열림 센서가 감지됐는지 알려줘.",
                    "비상 정지 버튼이 눌렸어?"
                ],
                "action_verbs": ["확인하다", "읽다", "상태를 보다", "감지하다", "입력값"],
                "target_objects": ["스위치", "센서", "버튼", "문 열림", "DI 포트", "입력 상태"]
            },
            "AO": {
                "description": "아날로그 출력(AO) 포트를 통해 특정 전압이나 전류(예: 0-10V, 4-20mA) 값을 설정하거나 제어하라는 지시입니다.",
                "examples": [
                    "AO 1번 포트에 5V를 출력해.",
                    "밸브 개방도를 50%로 설정해.",
                    "아날로그 출력으로 모터 속도를 80%로 조절해."
                ],
                "action_verbs": ["출력하다", "설정하다", "제어하다", "조절하다", "맞추다", "보내다"],
                "target_objects": ["전압", "전류", "밸브 개방도", "모터 속도", "AO 포트", "출력값", "퍼센트"]
            },
            "AI": {
                "description": "아날로그 입력(AI) 포트에 연결된 센서의 현재 값(전압, 전류, 또는 변환된 물리량)을 읽어오라는 지시입니다.",
                "examples": [
                    "AI 2번 포트 값 읽어와.",
                    "현재 온도 센서 값 몇 도야?",
                    "압력 센서가 측정한 값이 뭐야?",
                    "아날로그 입력 1번 채널 전압 확인해."
                ],
                "action_verbs": ["읽다", "확인하다", "측정하다", "값을 가져오다", "알려주다"],
                "target_objects": ["온도 센서", "압력 센서", "전압값", "전류값", "AI 포트", "센서 값", "측정값"]
            },

            "COM": {
                "description": "시리얼 포트(RS-232, 485)나 특정 통신 채널(TCP/IP)을 통해 데이터를 전송하거나 수신하라는 지시입니다.",
                "examples": [
                    "COM1 포트로 'START' 문자열을 보내.",
                    "RS485 통신을 통해 외부 장치 값을 읽어와.",
                    "시리얼 통신 연결을 확인해."
                ],
                "action_verbs": ["전송하다", "보내다", "수신하다", "읽다", "쓰다", "연결하다", "확인하다"],
                "target_objects": ["COM 포트", "시리얼", "RS485", "RS232", "데이터", "문자열", "외부 장치"]
            },

            # "WATERLEVEL": {
            #     "description": "수위 센서의 현재 값(수위 높이)을 묻거나, 특정 수위 값과 관련된 동작(예: 펌프 제어)을 지시합니다.",
            #     "examples": [
            #         "현재 저수지 수위 몇 미터야?",
            #         "수위가 3m를 넘으면 1번 펌프를 켜.",
            #         "수위 센서 값 실시간으로 알려줘."
            #     ],
            #     "action_verbs": ["읽다", "확인하다", "측정하다", "알려주다", "...이면 ...해라"],
            #     "target_objects": ["수위", "수위값", "수위 센서", "수위계", "저수지", "수조"]
            # },
            # "RAINFALL": {
            #     "description": "우량계(강수량 센서)가 측정한 값(누적 강우, 시간당 강우 등)을 묻는 지시입니다.",
            #     "examples": [
            #         "오늘 누적 강수량이 얼마야?",
            #         "시간당 강우량 알려줘.",
            #         "우량 센서 값 좀 읽어와."
            #     ],
            #     "action_verbs": ["읽다", "확인하다", "측정하다", "알려주다"],
            #     "target_objects": ["강수량", "강우량", "누적 강수량", "시간당 강우량", "우량계", "강수 센서"]
            # },
            # "BATTERY_VOLTAGE": {
            #     "description": "시스템의 주 전원 또는 배터리의 현재 전압 값이나 전원 상태(잔량)를 묻는 지시입니다.",
            #     "examples": [
            #         "배터리 전압 몇 볼트 남았어?",
            #         "현재 전원 상태 어때?",
            #         "배터리 잔량 확인해 줘.",
            #         "UPS 전압 체크해 봐."
            #     ],
            #     "action_verbs": ["확인하다", "읽다", "알려주다", "체크하다", "측정하다"],
            #     "target_objects": ["배터리", "전압", "배터리 전압", "전원 상태", "잔량", "UPS"]
            # }
        }

        # 구분 타입
        self.what_TYPES = {
            # 긴급 시, 경보 후 로그 남기기
            "ALERT": [
                "경보", "경고", "알람", "비상", "긴급", "alert", "alarm",
                "사이렌", "경고음", "비상신호", "경보방송", "경보등", "경보발생", "alert signal"
            ],
            "LOG": [
                "로그", "기록", "저장", "log", "history", "데이터기록", "이력남기기", "log save", "log record"
            ],

            # 기본 I/O
            "DO": ["do", "디오", "디지털 출력", "digital output"],
            "DI": ["di", "디아이", "디지털 입력", "digital input"],
            "AO": ["ao", "에이오", "아날로그 출력", "analog output"],
            "AI": ["ai", "에이아이", "아날로그 입력", "analog input"],

            # 통신 상태
            # "COM": ["uart", "com", "시리얼", "serial", "통신포트", "포트상태", "통신연결", "tx", "rx"],

            # 센서
            # "WATERLEVEL": [
            #     "수위", "waterlevel", "수위센서", "level", "water level",
            #     "수위값", "수위측정", "저수위", "고수위", "water sensor"
            # ],
            # "RAINFALL": [
            #     "우량", "rainfall", "rain", "비", "강수", "강우", "rain sensor",
            #     "우량센서", "강수량", "rain gauge", "rain data"
            # ],
            # "BATTERY_VOLTAGE": [
            #     "배터리 전압", "batteryvoltage", "배터리", "전압", "전원전압",
            #     "배터리 상태", "battery voltage", "battery level", "전원상태",
            #     "battery sensor", "전압값"
            # ]
        }

        self.ACTION_ON = ["켜", "켜줘", "on", "start", "활성", "활성화", "작동"]
        self.ACTION_OFF = ["꺼", "꺼줘", "끄", "off", "stop", "비활성", "비활성화", "정지"]
        self.ACTION_READ = ["읽", "read", "조회", "확인", "가져와"]
        self.ACTION_WRITE = ["쓰", "write", "설정", "세팅", "변경", "바꿔"]
        self.ACTION_QUERY = ["상태", "status", "어때", "어떻게", "query"]

        # 잡음 키워드
        self.NOISE = ["야", "음", "어", "으", "아", "이제", "좀", "약간", "그", "저", "뭐", "뭐시기",
                      "그거", "저거", "이거", "있잖아", "있잖아요", "요", "잠깐", "빨리"]


        log("🤖 AI 모델 로딩 중...")
        model_dir = r"D:\models\xlmR_xnli"

        # 옵션 2: 한국어 특화 모델 (KoBERT 기반)
        self.classifier = pipeline(
            "zero-shot-classification",
            model=AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True),
            tokenizer=AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True),
        )

        self.keyword_extractor = KeyBERT('distiluse-base-multilingual-cased-v2')

        # HYP_DETAILED 검증
        for label in self.what_TYPES.keys():
            if label not in self.HYP_DETAILED:
                log(f"⚠️ 경고: {label}에 대한 HYP_DETAILED 없음")

        log("✅ HYP_DETAILED 검증 완료")

        log("✅ AI 모델 로딩 완료!")

    # 0. 잡음 처리
    def _preprocess(self, text: str) -> str:
        """잡음 제거 전처리"""
        t = text.strip().lower()
        for n in self.NOISE:
            t = re.sub(r'\b' + re.escape(n) + r'\b', ' ', t)
            t = t.replace(' ' + n + ' ', ' ').replace(' ' + n, ' ').replace(n + ' ', ' ')
        return re.sub(r'\s+', ' ', t).strip()

    def _extract_keywords(self, text: str, top_n: int = 5):
        """문장에서 주요 키워드 추출 (명사, 행위어 등)"""
        text = self._preprocess(text)
        log(f"   잡음 제거 후 : {text}")
        keywords = self.keyword_extractor.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 1), # 연속된 단어 n 부터 m개 까지 하나의 키워드 후보로 본다.
            stop_words=self.NOISE,        # 불필요한 불용어 제거
            top_n=top_n                   # 몇 개의 키워드를 추출할 지 결정
        )
        print(f"키워드 중요도 분석 : {[(kw, score) for kw, score in keywords]}")
        # [('경보국', 0.74), ('포트', 0.68)...] → ['경보국', '포트', ...]
        return [kw for kw, score in keywords]

    def _build_hypothesis(self, label: str) -> str:
        """
        레이블에 맞는 상세한 가설 문장 생성

        Args:
            label: "DO", "ALERT" 같은 장치 코드

        Returns:
            str: "이 문장은 디지털 출력(DO) 포트를 제어하여..."
        """
        if label not in self.HYP_DETAILED:
            return f"이 문장은 {label}에 관한 지시다."

        detail = self.HYP_DETAILED[label]

        # description 활용
        desc = detail["description"]

        # examples 일부 추가 (선택 사항)
        examples = detail["examples"][:2]  # 상위 2개만
        example_text = " 예: " + ", ".join(examples) if examples else ""

        return f"{desc}{example_text}"


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
        """
        What 추출 (NLI 모델 직접 사용)

        Args:
            text: 사용자 입력 텍스트

        Returns:
            str: 장치 코드
        """
        text_lower = text.lower()

        # 1단계: 키워드 체크
        for what_code, keywords in self.what_TYPES.items():
            if any(kw in text_lower for kw in keywords):
                log(f"  [키워드 매칭] {what_code}")
                return what_code

        # 2단계: AI 분류 (직접 추론)
        log("  [AI 분류 시작]")

        # Tokenizer와 Model 직접 사용
        tokenizer = self.classifier.tokenizer
        model = self.classifier.model

        scores = []

        for what_code in self.what_TYPES.keys():
            # Hypothesis 생성
            if what_code in self.HYP_DETAILED:
                hypothesis = self.HYP_DETAILED[what_code]["description"]
            else:
                hypothesis = f"{what_code} 장치를 제어하는 명령이다."

            # Tokenization (premise, hypothesis 쌍)
            inputs = tokenizer(
                text,  # Premise (입력 텍스트)
                hypothesis,  # Hypothesis (가설 문장)
                return_tensors="pt",  # PyTorch 텐서로 반환
                truncation=True,  # 긴 문장 자르기
                max_length=512  # 최대 길이
            )

            # 모델 추론
            with torch.no_grad():  # 기울기 계산 안 함 (속도 향상)
                outputs = model(**inputs)
                logits = outputs.logits[0]  # [contradiction, neutral, entailment]

            # Entailment 점수 추출 (마지막 인덱스)
            entailment_score = torch.softmax(logits, dim=0)[-1].item()

            scores.append((what_code, entailment_score))
            log(f"    {what_code}: {entailment_score:.4f}")

        # 3단계: 최고 점수 선택
        scores.sort(key=lambda x: x[1], reverse=True)
        best_label, best_score = scores[0]

        log(f"  [AI 판정] {best_label} (확신도: {best_score:.2%})")

        return best_label

    # def _extract_what(self, text: str):
    #     """What 추출 (개선된 버전)"""
    #     text_lower = text.lower()
    #
    #     # 명확하게 키워드가 있는 경우 즉시 return
    #     for what_code, keywords in self.what_TYPES.items():
    #         for keyword in keywords:
    #             if keyword in text_lower:
    #                 return what_code
    #
    #     # 음성 인식 등 명확하게 입력 받지 못한 경우 NLU model 동작
    #     # 후보 중 의미적 유사도에 가까운 값을 반환
    #
    #     log(f"[NLU-INPUT] {text}")
    #     log(f"[NLU-LABELS] {self.what_TYPES.keys()}")
    #     candidate_labels = list(self.what_TYPES.keys())  # ["DO", "DI", "ALERT", ...]
    #     log(f"[NLU-RAW] {candidate_labels}")
    #
    #     z = self.classifier(
    #         text,
    #         candidate_labels=candidate_labels,
    #         hypothesis_template=[],
    #         multi_label=True
    #     )
    #
    #     df = pd.DataFrame({
    #         '후보': z['labels'],
    #         '정확도': z['scores']
    #     })
    #
    #     # 점수 기준 내림차순 정렬
    #     df = df.sort_values('정확도', ascending=False).reset_index(drop=True)
    #     print(df)
    #
    #     return z["labels"][0]

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
            print("동작을 인식할 수 없습니다. 건너 뜁니다.")
            # return {"error": "동작을 인식하지 못했습니다."}

        command = {"type": cmd_type, "what": what}

        if target and len(target['values']) == 1:
            command[target['type']] = target['values'][0]
            command.update(action)

        return command

    def _parse_single_command(self, text: str):
        """단일 명령 분석"""

        keywords = self._extract_keywords(text, 5)
        log(f"  [키워드] : {keywords}")

        return self._parse_single_command_with_what(keywords, forced_what=None)


    def parse_text(self, text: str):
        """텍스트 분석 (다중 명령 지원)"""
        if not text or not text.strip():
            return {"error": "입력된 텍스트가 없습니다."}

        log(f"[감지한 텍스트] {text}")
        log("=" * 60)

        # if any(kw in text for kw in ["끄고", "켜고", "하고", "다음에"]):
        #     log("[다중 명령 감지]")
        #
        #     common_what = self._extract_what(text)
        #     log(f"  공통 장치: {common_what}")
        #
        #     commands_text = self._split_commands(text)
        #     log(f"  분리된 명령: {len(commands_text)}개")
        #
        #     results = []
        #     last_what = common_what

            # log("=" * 60)
            #
            # if len(results) == 0:
            #     return {"error": "유효한 명령을 찾을 수 없습니다."}
            # elif len(results) == 1:
            #     return results[0]
            # else:
            #     return {"명령어": results, "명령 개수": len(results)}
            #
            # for i, cmd_text in enumerate(commands_text, 1):
            #     log(f"[명령 {i}] {cmd_text}")
            #
            #     what = self._extract_what(cmd_text)
            #
            #     if not what:
            #         what = last_what
            #         log(f"  장치: {what} (상속됨)")
            #     else:
            #         last_what = what
            #         log(f"  장치: {what}")
            #
            #     if not what:
            #         log(f"  ⚠️ 장치를 찾을 수 없어 스킵합니다.")
            #         continue
            #
            #     result = self._parse_single_command_with_what(cmd_text, what)
            #
            #     if "error" not in result:
            #         results.append(result)

        what = self._extract_what(text)
        log(f"  장치: {what}")

        if not what:
            log(f"  ⚠️ 장치를 찾을 수 없어 스킵합니다.")

        result = self._parse_single_command(text)

        log("=" * 60)
        return result