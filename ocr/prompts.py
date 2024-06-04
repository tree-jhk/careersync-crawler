PreprocessPrompt = """### Instruction:
아래에 제공된 OCR 텍스트(list 형식으로 제공됨)를 정제하고 구조화해 주세요. 오타를 수정하고 불필요한 문자를 제거한 후, 섹션을 구분하고 포맷을 유지해 주세요.

**요구사항**:
1. **텍스트 정제**:
   - 오타를 수정하세요 (예: "스스파트" -> "스마트", "스대트팜" -> "스마트팜", "피래농업" -> "미래농업", "Nol." -> "No.1").
   - 불필요한 문자를 제거하세요 (예: 불필요한 공백이나 특수 문자).

2. **구조화**:
   - 각 섹션을 명확히 구분하고 라벨링하세요 (예: "회사 소개", "채용 공고", "채용 절차").
   - 목록 형식을 유지하세요 (예: 채용 절차 아래의 리스트 형식 유지).

### OCR text:
회사 이름: 대동
["SMARTAGRICULTURALIMACHINE SMARTMOBILITY SMARTFARM", "daedong", "애그테크&플랫폼 기업 주)대동 각 부문 신입(경력 채용", "대동과 함께 미래농업 세상을 열어갈 우수한 인재를 기다립니다.", "대동은 지난 70년이 넘는 시간 동안 대한민국의 농업 발전을 이끌어 왔습니다. 수많은 최초의 역사를 쓰며 국내 농기계 Nol. 회사가 되었지만, "우리 농업의 미래와 새로운 역사를 위해 미래농업으로 도전합니다. 스스파트(농기계), 스마트 모빌리티, 스대트팜으로 대동과 함께 피래농업을 이끌어 갈 우수 인재를 포집합니다."]

### Response:
{{"Co_name":"대동","OCR_result":"SMART AGRICULTURAL MACHINE, SMART MOBILITY, SMART FARM\ndaedong\n애그테크&플랫폼 기업 (주)대동 각 부문 신입(경력) 채용\n대동과 함께 미래농업 세상을 열어갈 우수한 인재를 기다립니다.\n\n대동은 지난 70년이 넘는 시간 동안 대한민국의 농업 발전을 이끌어 왔습니다. 수많은 최초의 역사를 쓰며 국내 농기계 No.1 회사가 되었지만, 우리 농업의 미래와 새로운 역사를 위해 미래농업으로 도전합니다. 스마트 농기계, 스마트 모빌리티, 스마트팜으로 대동과 함께 미래농업을 이끌어 갈 우수 인재를 모집합니다."}}

### OCR text:
회사 이름: 티이엠씨(주)
["면접", "인-적성검사 (면접대상자 선정)", "채용건강검진", "서류심사", "채용진행"]

### Response:
{{"Co_name":"티이엠씨(주)","OCR_result":"채용 절차\n- 서류심사\n- 인적성검사 (면접대상자 선정)\n- 면접\n- 채용 건강검진\n- 채용 진행"}}

### OCR text:
회사 이름:{Co_name}
{ocr_result}

### Response:
"""