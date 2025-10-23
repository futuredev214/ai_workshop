import os
from utils import log
import pandas as pd
from keybert import KeyBERT
import json

# 오프라인 허용 (모델이 로컬에 있을 때)
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ["TRANSFORMERS_OFFLINE"] = os.getenv("TRANSFORMERS_OFFLINE", "0")

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
import torch

"""
    범용 제어 시스템 NLU 엔진

    역할:
        - 음성 인식 텍스트를 구조화된 JSON 명령으로 변환
        - Zero-Shot Classification (AI 기반 의미 분류)
        - 패턴 매칭 (정규식 기반 빠른 추출)
"""
class UniversalNluEngine:
    MAX_LENGTH = 128
    """
        NLU 엔진 초기화

        수행 작업:
            1. Command Hypotheses 정의 (AI 분류용 가설 문장)
            2. XLM-RoBERTa XNLI 모델 로드 (다국어 NLI)
            3. KeyBERT 키워드 추출기 초기화

        설정:
            - 모델: XLM-RoBERTa Large + XNLI
            - 키워드 추출: KeyBERT (다국어 지원)
            - 잡음 필터: 한국어 불용어 리스트
    """
    def __init__(self):
        self.SCENARIO_TO_FILE = {
            "시험": "A.wav",
            "방류": "B.wav",
        }

        # Command 레벨 가설 상세
        # Command 레벨 가설 상세
        self.COMMAND_HYPOTHESES = {
            # 1) 경보 방송
            "alert.broadcast": {
                "description": "경보국 방송 수행. 시나리오(시험/방류)와 볼륨을 슬롯으로 추출.",
                "examples": [
                    "경보국 볼륨 1로 시험 방송 시작",
                    "경보국 볼륨 제일 작게 시험 방송 시작",
                    "경보국 볼륨 30으로 방류 방송해줘",
                    "경보국 시험 방송 시작",
                    "방류 안내 방송 송출",
                ],
                "keywords": ["경보", "방송", "안내", "재생", "송출", "시험", "테스트", "방류", "경보국"],
                "slots": {
                    "scenario": ["시험", "방류", "테스트"],
                    "volume": list(range(0, 101)),
                    "action": ["시작", "정지", "중단", "켜기", "끄기"],
                },
                "slot_patterns": {
                    "scenario": r"(시험|방류|테스트)",
                    "volume": r"(?:볼륨|volume)\s*(\d{1,3})",
                    "action": r"(시작|정지|중단|켜|꺼)",
                },
                "defaults": {"station": "경보국", "volume": 10, "action": "시작"}
            },

            # 2) 수위국 데이터 호출
            "data.fetch.level": {
                "description": "수위국의 수위, 우량, 배터리 전압 데이터를 조회해 응답.",
                "examples": [
                    "부천 수위국 데이터 호출해줘",
                    "수위국 데이터 가져와",
                    "부천 수위, 우량, 배터리 전압 조회",
                    "수위국 값 불러와"
                ],
                "keywords": ["수위국", "데이터", "호출", "가져와", "조회", "불러와", "수위", "우량", "배터리"],
                "slots": {
                    "station": ["수위국", "우량국"],
                    "data_type": ["수위", "우량", "배터리전압", "전체"],
                },
                "slot_patterns": {
                    "station": r"(수위국|우량국)",
                    "data_type": r"(수위|우량|배터리\s*전압)",
                },
                "defaults": {"station": "수위국", "data_type": "전체"}
            },

            # 3) 장비 점검
            "device.inspect": {
                "description": "지정 국의 장비·센서 상태 점검 후 이상 여부 리포트.",
                "examples": [
                    "울산 경보국 장비 점검해줘",
                    "부천 경보국 점검",
                    "경보국 장비 상태 체크",
                    "장비 진단 실행"
                ],
                "keywords": ["점검", "진단", "체크", "검사", "장비", "상태", "경보국", "수위국"],
                "slots": {
                    "station": ["경보국", "수위국"],
                },
                "slot_patterns": {
                    "station": r"(경보국|수위국)",
                },
                "defaults": {"station": "경보국"}
            },
        }

        # 잡음 키워드
        self.NOISE = ["음", "어", "으", "아", "이제", "좀", "약간", "그", "저", "뭐", "뭐시기",
                      "그거", "저거", "이거", "있잖아", "있잖아요", "요", "잠깐", "빨리"]


        log("🤖 AI 모델 로딩 중...")
        model_dir = r"D:\models\xlmR_xnli"

        # 옵션 2: 한국어 특화 모델 (KoBERT 기반)
        # Zero-Shot Classification 파이프라인 생성
        self.classifier = pipeline(
            "zero-shot-classification",               # 작업 타입: 제로샷 분류
            model=AutoModelForSequenceClassification.from_pretrained(
                model_dir,                                 # 모델 경로
                local_files_only=True                      # 로컬 파일만 사용 (인터넷 차단)
            ),
            tokenizer=AutoTokenizer.from_pretrained(
                model_dir,                                 # 토크나이저 경로
                local_files_only=True,                     # 로컬 파일만 사용
                use_fast=True                              # Rust 기반 Fast Tokenizer (2~3배 빠름)
            ),
        )

        # KeyBERT 키워드 추출기 초기화
        # - 모델: DistilUSE (다국어 문장 임베딩)
        # - 용도: 중요 키워드 자동 추출
        self.keyword_extractor = KeyBERT('distiluse-base-multilingual-cased-v2')

        log("✅ AI 모델 로딩 완료!")

    """
        상태 체크 타입 분류
    
        Returns:
            "communication" | "power" | "all"
    """
    def _classify_check_type(self, text: str) -> str:
        hypotheses = {
            "communication": "이 명령은 통신 연결 상태나 네트워크 상태 확인입니다.",
            "power": "이 명령은 전원 상태나 배터리 전압 확인입니다.",
            "all": "이 명령은 전체 시스템 상태를 종합적으로 확인합니다."
        }

        return self._classify_with_hypotheses(text, hypotheses)

    """
        데이터 타입 분류
    
        Returns:
            "waterlevel" | "rainfall" | "all"
    """
    def _classify_data_type(self, text: str) -> str:
        hypotheses = {
            "waterlevel": "이 명령은 수위 센서 데이터나 수위계 측정값과 관련됩니다.",
            "rainfall": "이 명령은 강수량, 우량 데이터와 관련됩니다.",
            "all": "이 명령은 모든 종류의 센서 데이터를 포함합니다."
        }

        return self._classify_with_hypotheses(text, hypotheses)

    """
        공통 분류 헬퍼 함수
    
        Args:
            text: 입력 텍스트
            hypotheses: {label: hypothesis} 딕셔너리
    
        Returns:
            최고 점수 레이블
    """
    def _classify_with_hypotheses(self, text: str, hypotheses: dict) -> str:
        tokenizer = self.classifier.tokenizer
        model = self.classifier.model

        scores = []

        for label, hypothesis in hypotheses.items():
            inputs = tokenizer(
                text,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0]
                probs = torch.softmax(logits, dim=0)
                entailment_prob = probs[-1].item()

            scores.append((label, entailment_prob))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    def _classify_intent(self, text: str):
        """
        Intent 분류: 사용자 입력을 COMMAND_HYPOTHESES의 Intent로 분류

        Args:
            text: 사용자 입력 텍스트

        Returns:
            str: 분류된 Intent (예: "alert.broadcast.start")
        """
        # 1단계: 키워드 기반 빠른 필터링
        candidate_intents = []

        for intent, config in self.COMMAND_HYPOTHESES.items():
            # 키워드 매칭 점수 계산
            keyword_match_count = sum(
                1 for keyword in config["keywords"]
                if keyword in text.lower()
            )

            if keyword_match_count > 0:
                candidate_intents.append(intent)

        # 키워드 매칭된 Intent가 없으면 전체 Intent 사용
        if not candidate_intents:
            candidate_intents = list(self.COMMAND_HYPOTHESES.keys())

        log(f"  [Intent 후보] {candidate_intents}")

        # 2단계: NLU 모델로 정확한 Intent 분류
        intent_labels = [
            self.COMMAND_HYPOTHESES[intent]["description"]
            for intent in candidate_intents
        ]

        result = self.classifier(
            text,
            candidate_labels=intent_labels,
            multi_label=False
        )

        # 결과를 DataFrame으로 보기 좋게 출력
        df = pd.DataFrame({
            'Intent': candidate_intents,
            '설명': intent_labels,
            '정확도': result['scores']
        })
        df = df.sort_values('정확도', ascending=False).reset_index(drop=True)
        print(df)

        # 가장 높은 확률의 Intent 반환
        best_intent_idx = intent_labels.index(result['labels'][0])
        selected_intent = candidate_intents[best_intent_idx]

        log(f"  [선택된 Intent] {selected_intent}")
        return selected_intent

    def _extract_slot_with_regex(self, text: str, slot_name: str, pattern: str):
        """
        정규식으로 슬롯 추출 (1차 시도)

        Args:
            text: 사용자 입력 텍스트
            slot_name: 슬롯 이름 (예: "scenario", "volume")
            pattern: 정규식 패턴

        Returns:
            추출된 값 또는 None
        """
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1) if match.groups() else match.group(0)
            log(f"    [정규식 성공] {slot_name} = {value}")
            return value
        return None

    def _extract_slot_with_nlu(self, text: str, slot_name: str, candidates: list):
        """
        NLU 모델로 슬롯 추출 (2차 시도)

        Args:
            text: 사용자 입력 텍스트
            slot_name: 슬롯 이름
            candidates: 가능한 후보 값 리스트

        Returns:
            추출된 값 또는 None
        """
        # 특수 처리: volume 같은 숫자 슬롯
        if slot_name == "volume":
            # 텍스트에서 숫자 추출
            numbers = re.findall(r'\d+', text)
            if numbers:
                value = int(numbers[0])
                # 범위 체크 (0~100)
                if 0 <= value <= 100:
                    log(f"    [NLU-숫자] {slot_name} = {value}")
                    return value

            # "제일 작게", "최소", "최대" 같은 표현 처리
            if any(kw in text for kw in ["제일 작게", "최소", "작게"]):
                log(f"    [NLU-의미] {slot_name} = 1 (최소)")
                return 1
            if any(kw in text for kw in ["제일 크게", "최대", "크게"]):
                log(f"    [NLU-의미] {slot_name} = 100 (최대)")
                return 100

            return None

        # 일반 슬롯: NLU 분류
        if not candidates or len(candidates) == 0:
            return None

        # 후보가 1개면 바로 반환
        if len(candidates) == 1:
            return candidates[0]

        # 문자열 후보만 필터링 (NLU 입력용)
        str_candidates = [str(c) for c in candidates if isinstance(c, str)]

        if not str_candidates:
            return None

        try:
            result = self.classifier(
                text,
                candidate_labels=str_candidates,
                multi_label=False
            )

            value = result['labels'][0]
            score = result['scores'][0]

            # 신뢰도가 낮으면 무시 (threshold: 0.3)
            if score < 0.3:
                log(f"    [NLU-실패] {slot_name} 신뢰도 낮음 ({score:.2f})")
                return None

            log(f"    [NLU 성공] {slot_name} = {value} (신뢰도: {score:.2f})")
            return value

        except Exception as e:
            log(f"    [NLU-오류] {slot_name}: {e}")
            return None

    def _extract_slots(self, text: str, intent: str):
        """
        Intent에 정의된 모든 슬롯 추출

        Args:
            text: 사용자 입력 텍스트
            intent: 분류된 Intent

        Returns:
            dict: 추출된 슬롯들 {slot_name: value}
        """
        config = self.COMMAND_HYPOTHESES[intent]
        slots = {}

        log(f"  [슬롯 추출 시작] Intent: {intent}")

        # 각 슬롯별로 추출 시도
        for slot_name, candidates in config.get("slots", {}).items():
            log(f"  [{slot_name}] 추출 시도...")

            # 1차: 정규식 시도
            pattern = config.get("slot_patterns", {}).get(slot_name)
            if pattern:
                value = self._extract_slot_with_regex(text, slot_name, pattern)
                if value is not None:
                    slots[slot_name] = value
                    continue

            # 2차: NLU 시도
            value = self._extract_slot_with_nlu(text, slot_name, candidates)
            if value is not None:
                slots[slot_name] = value

        # 기본값 적용
        defaults = config.get("defaults", {})
        for key, default_value in defaults.items():
            if key not in slots:
                slots[key] = default_value
                log(f"    [기본값 적용] {key} = {default_value}")

        log(f"  [최종 슬롯] {slots}")
        return slots

    # 0. 잡음 처리
    def _preprocess(self, text: str) -> str:
        """잡음 제거 전처리"""
        t = text.strip().lower()
        for n in self.NOISE:
            t = re.sub(r'\b' + re.escape(n) + r'\b', ' ', t)
            t = t.replace(' ' + n + ' ', ' ').replace(' ' + n, ' ').replace(n + ' ', ' ')
        return re.sub(r'\s+', ' ', t).strip()

    def _extract_keywords(self, text: str, top_n: int = 5):
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

    """
        명령 타입 분류 (순수 AI)

        Args:
            text: 사용자 입력 (예: "서버에서 상수도 데이터 호출해줘")

        Returns:
            (command_type, confidence_score)
            예: ("data.fetch", 0.92)
    """
    def _classify_command(self, text: str) -> tuple:
        log("=" * 60)
        log(f"[1단계: Command 분류] 입력: {text}")
        log("=" * 60)

        # Tokenizer와 Model
        tokenizer = self.classifier.tokenizer
        model = self.classifier.model

        scores = []

        for cmd_type, hypothesis_data in self.COMMAND_HYPOTHESES.items():
            # Hypothesis 생성 (description + examples)
            hypothesis = hypothesis_data["description"]

            # 예시 추가
            examples = hypothesis_data["examples"]
            if examples:
                hypothesis += f" 예시: {' '.join(examples)}"

            # NLI 추론
            inputs = tokenizer(
                text,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0]

                # Softmax로 확률 변환
                probs = torch.softmax(logits, dim=0)
                entailment_prob = probs[-1].item()  # XNLI: 마지막 인덱스가 entailment

            scores.append((cmd_type, entailment_prob))

            log(f"  {cmd_type:20s} → {entailment_prob:.4f}")

        # 최고 점수 선택
        scores.sort(key=lambda x: x[1], reverse=True)
        best_cmd, best_score = scores[0]

        log(f"  ✅ 추론 : {best_cmd} (확신도: {best_score:.2%})")
        log("=" * 60)

        return best_cmd, best_score

    """
        명령 대상 범위 분류 (로컬/원격)

        Args:
            text: "서버에서 상수도 데이터 호출"

        Returns:
            "local" or "remote"
    """
    def _classify_target_scope(self, text: str) -> str:
        log("=" * 60)
        log(f"[2단계: Target Scope 분류]")
        log("=" * 60)

        # Hypothesis 정의
        target_hypotheses = {
            "local": "이 명령은 현재 기기에서 직접 실행하는 로컬 작업입니다.",
            "remote": "이 명령은 원격 서버나 다른 장소의 시스템에 요청하는 작업입니다."
        }

        tokenizer = self.classifier.tokenizer
        model = self.classifier.model

        scores = []

        for scope, hypothesis in target_hypotheses.items():
            inputs = tokenizer(
                text,                                  # premise: 사용자 원문 텍스트
                hypothesis,                            # hypothesis: 레이블별 가설 문장
                return_tensors="pt",                   # PyTorch 텐서로 반환
                truncation=True,                       # 최대 길이를 초과하면 잘라낸다
                max_length=self.MAX_LENGTH             # 문장쌍 총 길이 상한(모델 한계에 맞춤)
            )

            with torch.no_grad():                      # 추론 모드(gradient 비계산)로 메모리/속도 절약
                outputs = model(**inputs)              # NLI 모델 전향 패스(forward) 호출
                logits = outputs.logits[0]             # 배치 첫 항목의 로짓 벡터 취득
                probs = torch.softmax(logits, dim=0)   # 로짓을 소프트맥스로 확률로 변환
                entailment_prob = probs[-1].item()     # 마지막 인덱스를 함의(entailment)로 가정해 확률 추출

            scores.append((scope, entailment_prob))
            log(f"  {scope:10s} → {entailment_prob:.4f}")

        scores.sort(key=lambda x: x[1], reverse=True)
        best_scope = scores[0][0]

        log(f"  현장/서버 : {best_scope}\n")

        return best_scope

    """
        장소명 추출 (하이브리드)
    
        Args:
            text: "서버에서 소양강댐 데이터 호출"
    
        Returns:
            "소양강댐" or None
    """
    def _extract_location(self, text: str) -> str:
        log("[장소 추출]")

        # ========================================
        # Step 1: 패턴 기반 추출 (빠른 처리)
        # ========================================

        # 패턴 1: "서버에서 [장소] ..."
        pattern1 = r'서버에서\s+(\S+?)\s+(?:데이터|경보|장비|상태)'
        match = re.search(pattern1, text)

        if match:
            candidate = match.group(1)
            log(f"  패턴 후보: {candidate}")

            # 2. 시설 키워드: 댐, 교, 국 등 접미사 검색
            if any(kw in candidate for kw in ["댐", "교", "국"]):
                log(f"  ✅ 시설명 확정: {candidate}")
                return candidate

            # 애매한 경우 AI 검증으로 넘김
            else:
                log(f"  ⚠️ 애매함 → AI 검증 필요")
                return self._verify_location_with_ai(text, candidate)

        # 3. AI 검증: 애매한 경우 NLI 모델로 확인
        facility_patterns = [
            r'(\S+댐 \S+교 \S+국)',
            r'(\S+댐 \S+교)',
            r'(\S+댐 \S+국)',
            r'(\S+교 \S+국)',
            r'(\S+댐)',
            r'(\S+교)',
            r'(\S+국)',
        ]

        for pattern in facility_patterns:
            match = re.search(pattern, text)
            if match:
                location = match.group(1)
                log(f"  ✅ 시설 패턴: {location}")
                return location

        # 4. AI 분류: 등록된 주요 장소 중 선택 (10개 이하 권장)
        common_locations = [
            "수위우량국",
            "수위국",
            "우량국",
            "경보국",
            "통신실"
        ]

        return self._classify_location_ai(text, common_locations)

    """
        후보 장소명 AI 검증

        Args:
            text: 전체 문장
            candidate: 추출된 후보 (예: "상수도")

        Returns:
            검증된 장소명 or None
    """
    def _verify_location_with_ai(self, text: str, candidate: str) -> str:
        log(f"  [AI 검증] 후보: {candidate}")

        hypothesis = f"이 문장에서 '{candidate}'는 특정 장소나 시설을 가리킵니다."

        tokenizer = self.classifier.tokenizer
        model = self.classifier.model

        inputs = tokenizer(
            text,                                   # premise: 사용자 원문 텍스트
            hypothesis,                             # hypothesis: 레이블별 가설 문장
            return_tensors="pt",                    # PyTorch 텐서로 반환
            truncation=True,                        # 최대 길이를 초과하면 잘라낸다
            max_length=self.MAX_LENGTH              # 문장쌍 총 길이 상한(모델 한계에 맞춤)
        )

        with torch.no_grad():                       # 추론 모드(gradient 비계산)로 메모리/속도 절약
            outputs = model(**inputs)               # NLI 모델 전향 패스(forward) 호출
            logits = outputs.logits[0]              # 배치 첫 항목의 로짓 벡터 취득
            probs = torch.softmax(logits, dim=0)    # 로짓을 소프트맥스로 확률로 변환
            entailment_prob = probs[-1].item()      # 마지막 인덱스를 함의(entailment)로 가정해 확률 추출

        log(f"    확률: {entailment_prob:.4f}")

        # 임계값 설정
        if entailment_prob > 0.7:
            log(f"  ✅ 검증 성공: {candidate}")
            return candidate
        else:
            log(f"  ❌ 검증 실패")
            return None

    """
        등록된 장소 중 AI 분류

        Args:
            text: 입력 문장
            candidates: 후보 장소 목록 (작은 리스트만!)

        Returns:
            가장 적합한 장소 or None
    """
    def _classify_location_ai(self, text: str, candidates: list) -> str:
        log(f"  [AI 분류] 후보: {candidates}")

        tokenizer = self.classifier.tokenizer
        model = self.classifier.model

        # "장소 없음" 케이스 추가
        candidates_with_none = candidates + ["없음"]

        hypotheses = {
            loc: f"이 문장은 {loc}와 관련된 명령입니다."
            for loc in candidates
        }
        hypotheses["없음"] = "이 문장에는 특정 장소가 언급되지 않았습니다."

        scores = []

        for loc, hypothesis in hypotheses.items():
            # hypotheses 딕셔너리에서 (레이블코드/키, 가설문) 쌍을 한 개씩 가져온다. 예: ("ALERT", "이 문장은 경보...")
            inputs = tokenizer(
                text,                         # premise: 사용자 원문 텍스트
                hypothesis,                   # hypothesis: 레이블별 가설 문장
                return_tensors="pt",          # PyTorch 텐서로 반환
                truncation=True,              # 최대 길이를 초과하면 잘라낸다
                max_length=self.MAX_LENGTH    # 문장쌍 총 길이 상한(모델 한계에 맞춤)
            )

            with torch.no_grad():                      # 추론 모드(gradient 비계산)로 메모리/속도 절약
                outputs = model(**inputs)              # NLI 모델 전향 패스(forward) 호출
                logits = outputs.logits[0]             # 배치 첫 항목의 로짓 벡터 취득
                probs = torch.softmax(logits, dim=0)   # 로짓을 소프트맥스로 확률로 변환
                entailment_prob = probs[-1].item()     # 마지막 인덱스를 함의(entailment)로 가정해 확률 추출

            scores.append((loc, entailment_prob))
            log(f"    {loc:15s} → {entailment_prob:.4f}")

        scores.sort(key=lambda x: x[1], reverse=True)
        best_location = scores[0][0]

        if best_location == "없음":
            log("  ✅ 추론 결과: 장소 없음")
            return None
        else:
            log(f"  ✅ 추론 결과: {best_location}")
            return best_location

    """
                텍스트 분석 - Intent & Slot 기반 파싱

                Args:
                    text: 사용자 입력 텍스트

                Returns:
                    dict: 파싱 결과 {intent, slots, ...}
            """

    def parse_text(self, text: str):

        if not text or not text.strip():
            return {"error": "입력된 텍스트가 없습니다."}

        log(f"[감지한 텍스트] {text}")
        log("=" * 60)

        # 전처리
        text = self._preprocess(text)

        # 키워드 추출 (디버깅용)
        keywords = self._extract_keywords(text)
        log(f"  [키워드] {keywords}")

        # 1단계: Intent 분류
        try:
            intent = self._classify_intent(text)
        except Exception as e:
            log(f"  [Intent 분류 실패] {e}")
            return {"error": "명령을 인식하지 못했습니다."}

        # 2단계: Slot 추출
        try:
            slots = self._extract_slots(text, intent)
        except Exception as e:
            log(f"  [Slot 추출 실패] {e}")
            return {"error": "파라미터를 추출하지 못했습니다."}

        # 3단계: 최종 결과 구성
        result = {
            "intent": intent,
            "slots": slots,
        }

        # 특수 처리: 방송 명령의 경우 파일 매핑
        if intent == "alert.broadcast.start" and "scenario" in slots:
            scenario = slots["scenario"]
            if scenario in self.SCENARIO_TO_FILE:
                result["file"] = self.SCENARIO_TO_FILE[scenario]
                log(f"  [파일 매핑] {scenario} → {result['file']}")

        log("=" * 60)
        return result

