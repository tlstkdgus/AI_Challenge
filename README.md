# SSAFY 15기 AI 챌린지 - 재활용품 VQA

## 과제 개요

재활용품 이미지를 보고 관련 질문에 4지선다(a/b/c/d)로 답하는 **Visual Question Answering(VQA)** 태스크

- **입력**: 재활용품 이미지 + 질문 + 4개 선지
- **출력**: a, b, c, d 중 하나
- **평가**: Accuracy

---

## 데이터셋

| 분류 | 파일 | 수량 |
|------|------|------|
| 학습 | `train.csv` + `train/` | 5,073장 |
| 검증 | `dev.csv` + `dev/` | 4,413장 (정답 5개 앵커) |
| 테스트 | `test.csv` + `test/` | 5,074장 |

- `dev.csv`는 `answer1~answer5` 5개 정답 컬럼 → majority vote로 최종 정답 결정

---

## 환경

- GPU: NVIDIA RTX 5060 Ti (16GB VRAM)
- OS: Windows 11
- CUDA: 12.8
- Python: 3.12
- 가상환경: `baseline/` (venv)

### 패키지 설치

```bash
# PyTorch nightly (RTX 5060 Ti Blackwell 지원)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 주요 패키지
pip install "transformers>=5.0.0" "accelerate>=0.34.2" "peft>=0.13.2" "bitsandbytes>=0.43.3" datasets pillow pandas
```

---

## 파일 구조

```
AI_Challenge/
├── 2026-ssafy-ai-15-2/
│   ├── train.csv
│   ├── dev.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   ├── train/                          # 학습 이미지 (gitignore)
│   ├── dev/                            # 검증 이미지 (gitignore)
│   ├── test/                           # 테스트 이미지 (gitignore)
│   ├── (260324)_baseline_colab.ipynb              # 베이스라인 (Colab)
│   ├── (260325)_baseline_desktop5060ti.ipynb      # 베이스라인 (데스크탑)
│   ├── (260325)_baseline_desktop5060ti_최적화코드.ipynb  # 최적화 실험
│   └── upgraded_qwen3vl.ipynb                     # 업그레이드 버전
├── baseline/                           # 가상환경 (gitignore)
├── .gitignore
├── CLAUDE.md
└── README.md
```

---

## 모델 실험 기록

### Baseline (베이스라인)

| 항목 | 설정 |
|------|------|
| 모델 | `Qwen/Qwen2.5-VL-3B-Instruct` |
| 학습 데이터 | 200개 (전체 중 샘플링) |
| Epoch | 1 |
| LoRA | r=8, alpha=16 |
| LR schedule | linear |
| 학습 시간 | ~2분 |
| 평가 | val loss만 측정 |

### Upgraded (업그레이드)

| 항목 | 설정 |
|------|------|
| 모델 | `Qwen/Qwen3-VL-4B-Instruct` |
| 학습 데이터 | 2,000개 (오버피팅 방지) |
| Epoch | 3 |
| LoRA | r=16, alpha=32 |
| LR schedule | cosine + warmup |
| Attention | sdpa |
| 평가 | dev accuracy (200 샘플 majority vote) |
| Best checkpoint | dev accuracy 기준 자동 저장 |

### Exp4 - 최적화 (팀원 4, 데스크탑 RTX 5060 Ti)

| 항목 | 설정 |
|------|------|
| 모델 | `Qwen/Qwen3-VL-8B-Instruct` |
| 학습 데이터 | ~4,565개 (전체 90%, 층화 분리) |
| Epoch | 1 |
| LoRA | r=16, alpha=32, dropout=0 (Unsloth 최적화) |
| LR | 6e-5, cosine + warmup 10% |
| Label smoothing | 0.05 |
| 전처리 | 패딩 (검정 여백, 비율 유지) |
| 오버샘플링 | 없음 (데이터 균형 양호) |
| 추론 | TTA 3회 다수결 |

### 주요 개선 전략

- **층화 추출**: answer × 질문유형(재질/개수/색상/종류) 복합 key로 train/valid 분리
- **labels 마스킹**: assistant 응답 토큰만 loss 반영 (이미지·질문 토큰 제외)
- **TTA**: 추론 시 선지 순서를 3회 셔플 후 다수결로 positional bias 제거
- **lora_dropout=0**: Unsloth fast patching 활성화로 속도 19배 향상
- **선지 셔플 제거**: 학습/추론 일관성 유지

---

## 실행 방법

### 학습 + 추론

```
upgraded_qwen3vl.ipynb 순서대로 실행
```

### 제출 파일

추론 완료 후 `submission.csv` 생성됨

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `Qwen3VLForConditionalGeneration` import 오류 | transformers < 5.0 | `pip install "transformers>=5.0.0"` |
| VRAM 부족 (8B 모델) | 16GB 한계 | 4B 모델로 변경 |
| 학습 속도 18s/batch | VRAM 97% 포화 | 모델 크기 축소 |
| unsloth 설치 후 torch 다운그레이드 | unsloth가 torch<2.11 요구 | unsloth 제거 후 nightly 재설치 |
| 학습 속도 53s/batch (8B Unsloth) | lora_dropout=0.05로 fast patching 비활성화 | `lora_dropout=0` 으로 변경 → 2.75s/batch |
| loss 15~20 수렴 안 됨 | labels에 이미지·질문 토큰 포함 | DataCollator에서 assistant 응답 전까지 -100 마스킹 |
| `torch.cuda.amp.GradScaler` FutureWarning | deprecated API | `torch.amp.GradScaler('cuda')` 로 변경 |

## 실험 기록

> 점수/메모 열은 직접 채워주세요.

| 파일명 | 날짜 | 주요 변경 | 점수 | 메모 |
|--------|------|-----------|------|------|
| submission_Qwen3_8B_origin.csv | 2026-04-02 | 베이스라인 | - | |
| submission_TTA_v1.csv | 2026-04-03 | Exp4: 층화분리+labels마스킹+TTA | - | Qwen3-VL-8B, lora_dropout=0 |

## 최근 커밋

| 날짜 | 내용 | 작성자 |
|------|------|--------|
| 2026-04-03 | Exp4 최적화: 층화분리, labels마스킹, TTA, lora_dropout=0 | sanghyeon shin |
| 2026-04-02 | update readme | tlstkdgus |
| 2026-04-02 | Merge branch 'sanghyeon' | tlstkdgus |
| 2026-04-02 | readme update 수정 | tlstkdgus |
| 2026-04-02 | Update scripts/update_readme.py | sanghyeon shin |
| 2026-04-02 | origin code result | tlstkdgus |
| 2026-04-02 | 자동 readme 업데이트 적용 | tlstkdgus |
| 2026-04-02 | unsloth_compiled_cache add to gitignore | tlstkdgus |
| 2026-04-02 | baseline 추가 금지 | tlstkdgus |
| 2026-04-02 | 기본 셋팅 | tlstkdgus |
