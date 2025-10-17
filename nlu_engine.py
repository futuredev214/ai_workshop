# import re
# import json
#
# class NluEngine:
#     # 장치 이름을 표준 코드로 매핑
#     DEVICE_MAP = {
#         "디지털 출력": "DO",
#         "디지털출력": "DO",
#         "디지털 아웃풋": "DO",
#         "디지털아웃풋": "DO",
#         "디지털 아웃": "DO",
#         "디오": "DO",
#         "디아웃": "DO",
#         "do": "DO",
#         "DO": "DO",
#         "d o": "DO",
#
#         # 디지털 입력 (Digital Input) - 나중을 위해
#         "디지털 입력": "DI",
#         "디지털입력": "DI",
#         "디지털 인풋": "DI",
#         "디인": "DI",
#         "디아이": "DI",
#         "di": "DI",
#         "DI": "DI",
#
#         # 아날로그 출력 (Analog Output)
#         "아날로그 출력": "AO",
#         "아날로그출력": "AO",
#         "아날로그 아웃풋": "AO",
#         "아웃": "AO",
#         "에이오": "AO",
#         "ao": "AO",
#         "AO": "AO",
#
#         # 아날로그 입력 (Analog Input)
#         "아날로그 입력": "AI",
#         "아날로그입력": "AI",
#         "아날로그 인풋": "AI",
#         "아인": "AI",
#         "에이아이": "AI",
#         "ai": "AI",
#         "AI": "AI"
#     }
#
#     # ON 상태를 의미하는 키워드들
#     STATE_ON_KEYWORDS = [
#         # 기본
#         "켜", "켜줘", "켜줄래", "켜라", "켜요", "켜주세요",
#         "켤래", "켜줄", "켜주",
#
#         # 영어
#         "on", "ON", "온",
#
#         # 활성화
#         "활성화", "활성", "활성화해줘", "활성화시켜",
#
#         # 숫자
#         "1로", "1", "일로", "하나로",
#
#         # 구어체
#         "틀어", "틀어줘", "틀어줄래", "돌려", "돌려줘",
#         "작동", "작동시켜", "작동해줘",
#         "올려", "올려줘",
#
#         # 기타
#         "시작", "시작해줘", "실행", "실행해줘"
#     ]
#
#     # OFF 상태를 의미하는 키워드들
#     STATE_OFF_KEYWORDS = [
#         # 기본
#         "꺼", "꺼줘", "꺼줄래", "꺼라", "꺼요", "꺼주세요",
#         "껄래", "꺼줄", "꺼주",
#
#         # 영어
#         "off", "OFF", "오프",
#
#         # 비활성화
#         "비활성화", "비활성", "비활성화해줘", "비활성화시켜",
#
#         # 숫자
#         "0으로", "0", "영으로", "제로로",
#
#         # 구어체
#         "끄", "끄줘", "끄줄래",
#         "멈춰", "멈춰줘", "정지", "정지해줘",
#         "내려", "내려줘",
#         "닫아", "닫아줘", "차단", "차단해줘",
#
#         # 기타
#         "중지", "중지해줘", "종료", "종료해줘"
#     ]
#
#     # 제거할 잡음 단어들 (filler words)
#     NOISE_WORDS = [
#         # 추임새
#         "음", "어", "으음", "어어", "음음",
#
#         # 지시어
#         "그", "저", "이", "요", "거기",
#         "그거", "저거", "이거", "요거",
#         "그것", "저것", "이것",
#         "그놈", "저놈",
#
#         # 불확실성 표현
#         "뭐", "뭔가", "뭐시기", "뭐드라", "뭐였더라", "뭐라고",
#         "어디", "어디더라", "어디였지",
#
#         # 시간/정도 부사
#         "이제", "좀", "막", "약간", "조금", "살짝", "잠깐", "빨리",
#
#         # 확인 표현
#         "있잖아", "있잖아요", "알지", "그치", "맞지",
#
#         # 접속사/감탄사
#         "그래", "그래서", "그런데", "근데", "아", "에", "으",
#         "어그", "아그", "어이구",
#
#         # 기타
#         "한번", "한", "좀", "뭐냐", "해줘"
#     ]
#
#     def __init__(self):
#         """NLU 엔진 초기화"""
#         pass
#
#     def _preprocess_text(self, text: str) -> str:
#         """
#         텍스트 전처리 - 잡음 제거
#
#         Args:
#             text (str): 원본 텍스트
#
#         Returns:
#             str: 전처리된 텍스트
#         """
#         processed = text.lower()
#
#         # 단어 경계 고려한 제거
#         for noise in self.NOISE_WORDS:
#             processed = re.sub(r'\b' + re.escape(noise) + r'\b', '', processed)
#
#         # 공백과 함께 있는 잡음 제거
#         # for noise in self.NOISE_WORDS:
#         #     processed = processed.replace(' ' + noise + ' ', ' ')
#         #     processed = processed.replace(' ' + noise, ' ')
#         #     processed = processed.replace(noise + ' ', ' ')
#
#         # 연속 공백 정리
#         processed = re.sub(r'\s+', ' ', processed).strip()
#
#         return processed
#
#     def _extract_port_range(self, text: str):
#         """
#         텍스트에서 포트 번호 또는 범위 추출
#
#         Args:
#             text (str): 전처리된 텍스트
#
#         Returns:
#             list: 포트 번호 리스트
#             None: 포트를 찾지 못한 경우
#         """
#         ports = []
#
#         # 🆕 "전체" 키워드 먼저 체크 (최우선)
#         all_keywords = ["모두", "전부", "전체", "다", "all", "올"]
#         has_all_keyword = any(keyword in text for keyword in all_keywords)
#
#         # 🆕 패턴 1: "N번까지 (모두/다/전체)" - 1부터 N까지
#         # "4번까지 다", "5까지 모두", "3번까지 전체"
#         until_all_patterns = [
#             r'(\d+)\s*(?:번)?\s*까지\s*(?:다|모두|전부|전체)',  # "4번까지 다"
#             r'(\d+)\s*(?:번)?\s*(?:까지|이하)\s*(?:다|모두|전부|전체)?',  # "4까지", "4이하"
#         ]
#
#         if has_all_keyword:
#             for pattern in until_all_patterns:
#                 match = re.search(pattern, text)
#                 if match:
#                     end = int(match.group(1))
#                     start = 1  # 암묵적 시작점
#
#                     # 범위가 유효한지 체크
#                     if end > self.max_port:
#                         print(f"⚠️ 경고: {end}번은 최대 포트({self.max_port})를 초과합니다. {self.max_port}로 제한합니다.")
#                         end = self.max_port
#
#                     ports = list(range(start, end + 1))
#                     print(f"[암묵적 범위 감지] 1부터 {end}까지 → {ports}")
#                     return ports
#
#         # 🆕 패턴 2: "N번부터 (모두/다/전체)" - N부터 max_port까지
#         # "4번부터 다", "5부터 모두", "3번부터 전체"
#         from_all_patterns = [
#             r'(\d+)\s*(?:번)?\s*부터\s*(?:다|모두|전부|전체)',  # "4번부터 다"
#             r'(\d+)\s*(?:번)?\s*(?:부터|이상)\s*(?:다|모두|전부|전체)?',  # "4부터", "4이상"
#         ]
#
#         if has_all_keyword:
#             for pattern in from_all_patterns:
#                 match = re.search(pattern, text)
#                 if match:
#                     start = int(match.group(1))
#                     end = self.max_port  # 암묵적 끝점
#
#                     # 범위가 유효한지 체크
#                     if start > self.max_port:
#                         return {"error": f"{start}번은 최대 포트({self.max_port})를 초과합니다."}
#
#                     ports = list(range(start, end + 1))
#                     print(f"[암묵적 범위 감지] {start}부터 {end}까지 → {ports}")
#                     return ports
#
#         # 기존 범위 패턴들
#         range_patterns = [
#             # 1. "부터 ~ 까지" 패턴
#             r'(\d+)\s*(?:번)?\s*부터\s*(\d+)\s*(?:번)?\s*까지',  # "1부터 4까지"
#             r'(\d+)\s*(?:번)?\s*부터\s*(\d+)\s*(?:번)?',  # "1부터 4"
#
#             # 2. "에서 ~ 까지" 패턴
#             r'(\d+)\s*(?:번)?\s*에서\s*(\d+)\s*(?:번)?\s*까지',  # "1에서 4까지"
#             r'(\d+)\s*(?:번)?\s*에서\s*(\d+)\s*(?:번)?',  # "1에서 4"
#
#             # 3. "사이" 패턴
#             r'(\d+)\s*(?:번)?\s*(?:부터|에서)?\s*(\d+)\s*(?:번)?\s*사이',
#
#             # 4. 기호 패턴
#             r'(\d+)\s*~\s*(\d+)',  # "1~4"
#             r'(\d+)\s*-\s*(\d+)',  # "1-4"
#             r'(\d+)\s*·\s*(\d+)',  # "1·4"
#             r'(\d+)\s*\.\.\s*(\d+)',  # "1..4"
#
#             # 5. "통해" 패턴
#             r'(\d+)\s*(?:번)?\s*(?:를)?\s*통해\s*(\d+)\s*(?:번)?',
#         ]
#
#         # 범위 패턴 매칭 시도
#         for pattern in range_patterns:
#             match = re.search(pattern, text)
#             if match:
#                 start = int(match.group(1))
#                 end = int(match.group(2))
#
#                 # 시작이 끝보다 크면 스왑
#                 if start > end:
#                     start, end = end, start
#
#                 # 범위가 너무 크면 경고
#                 if end - start > 100:
#                     print(f"⚠️ 경고: 범위가 너무 큽니다 ({start}~{end}). 최대 100개로 제한합니다.")
#                     end = start + 100
#
#                 ports = list(range(start, end + 1))
#                 print(f"[범위 감지] {start}부터 {end}까지 → {ports}")
#                 return ports
#
#         # 개별 포트 패턴들
#         individual_patterns = [
#             r'(\d+)\s*,',  # "1, 2, 3"
#             r'(\d+)\s*(?:번)?\s*(?:하고|랑|이랑)',  # "1하고 2하고 3"
#             r'(\d+)\s*(?:번)?\s*(?:와|과)',  # "1과 2와 3"
#         ]
#
#         # 개별 포트 패턴 체크
#         for pattern in individual_patterns:
#             if re.search(pattern, text):
#                 matches = re.findall(r'\d+', text)
#                 if len(matches) >= 2:
#                     ports = [int(m) for m in matches]
#                     ports = sorted(list(set(ports)))
#                     print(f"[개별 포트 감지] {ports}")
#                     return ports
#
#         # "모두", "전체", "all" 단독 사용 (숫자 없음)
#         if has_all_keyword:
#             ports = list(range(1, self.max_port + 1))
#             print(f"[전체 포트 감지] 1~{self.max_port} → {ports}")
#             return ports
#
#         # 단일 포트 패턴
#         single_match = re.search(r'(\d+)\s*(?:번)?', text)
#         if single_match:
#             port = int(single_match.group(1))
#             print(f"[단일 포트 감지] {port}")
#             return [port]
#
#         return None
#
#     def parse_text(self, text: str):
#         """
#         텍스트 분석해서 명령어 생성
#
#         Args:
#             text (str): 사용자 입력
#
#         Returns:
#             dict: 명령어 JSON 또는 에러
#         """
#         if not text or not text.strip():
#             return {"error": "입력된 텍스트가 없습니다."}
#
#         print(f"[원본] {text}")
#
#         # 전처리
#         processed_text = self._preprocess_text(text)
#         print(f"[전처리 후] {processed_text}")
#
#         if not processed_text:
#             return {"error": "유효한 명령어가 없습니다."}
#
#         # 1️⃣ 장치 찾기
#         device = None
#         for keyword, device_code in self.DEVICE_MAP.items():
#             if keyword.lower() in processed_text:
#                 device = device_code
#                 break
#
#         if not device:
#             return {"error": "장치를 인식하지 못했습니다. (예: DO, DI, AO, AI)"}
#
#         # 2️⃣ 포트 번호 또는 범위 찾기
#         ports = self._extract_port_range(processed_text)
#
#         if not ports:
#             return {"error": "포트 번호를 찾을 수 없습니다. (예: 1번, 1부터 4까지)"}
#
#         # 3️⃣ 상태 찾기 (ON=1 or OFF=0)
#         value = None
#
#         # OFF 먼저 확인
#         for keyword in self.STATE_OFF_KEYWORDS:
#             if keyword in processed_text:
#                 value = 0
#                 break
#
#         # ON 확인
#         if value is None:
#             for keyword in self.STATE_ON_KEYWORDS:
#                 if keyword in processed_text:
#                     value = 1
#                     break
#
#         if value is None:
#             return {"error": "상태를 인식하지 못했습니다. (예: 켜줘, 꺼줘, on, off)"}
#
#         # 4️⃣ 최종 명령어 생성
#         # 단일 포트인 경우
#         if len(ports) == 1:
#             command = {
#                 "device": device,
#                 "port": ports[0],
#                 "value": value
#             }
#         else:
#             # 여러 포트인 경우 (배열로 반환)
#             command = {
#                 "device": device,
#                 "ports": ports,  # port가 아닌 ports (복수형)
#                 "value": value
#             }
#
#         return command