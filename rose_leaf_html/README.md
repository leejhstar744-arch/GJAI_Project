# 장미 잎 이상 탐지: HTML 버전

`rose_leaf_colab_keyfile.ipynb`(package/leafdx)와 같은 계산을 **브라우저 안에서** 합니다. 서버도 Python도 필요 없습니다.
인터넷이 되는 곳에서 `index.html`을 열면 됩니다. 사진은 브라우저 밖으로 나가지 않습니다 (VLM 검증을 켰을 때 고른 잎 사진만 Claude로 보냄).

## 폴더 구성
```
rose_leaf_html/
├── index.html        화면
├── app.js            화면 동작 (모델 불러오기, 결과 표시, VLM 검증, zip)
├── leafdx.js         leafdx를 JavaScript로 옮긴 것 (1단계 잎 탐색, 2단계 상태 분류, VLM 사진)
├── models/           모델 4개 (ONNX, 약 94MB)
├── samples/          샘플 사진 1장
└── tools/pack_models.py   .pt → 브라우저용 모델로 바꾸는 스크립트 (모델을 새로 학습했을 때)
```
라이브러리(onnxruntime-web 1.31.1, OpenCV.js 5.0.0, JSZip 3.10.2)는 인터넷(cdn.jsdelivr.net)에서 받습니다.

## 여는 방법

### 1. 웹에 올려서 링크로 쓰기 (권장)
이 저장소에서는 GitHub Pages로 엽니다. 저장소 **Settings → Pages → Build and deployment**에서
Source를 `Deploy from a branch`, 브랜치를 `Minirose_Disease_show`, 폴더를 `/ (root)`로 두고 Save하면 1~2분 뒤 아래 주소가 열립니다.

**https://leejhstar744-arch.github.io/GJAI_Project/rose_leaf_html/**

폴더를 그대로 다른 정적 호스팅에 올려도 링크만으로 누구나 씁니다.
- **GitHub Pages**: 저장소에 폴더째 올린 뒤 Settings → Pages에서 브랜치를 고르면 `https://<계정>.github.io/<저장소>/rose_leaf_html/` 주소가 생깁니다. 모델 파일은 15MB 이하로 나눠 두어 GitHub 파일 크기 제한(100MB)에 걸리지 않습니다.
- Netlify, Cloudflare Pages, 학교·회사 웹 서버도 같습니다 (폴더째 올리기).
- 처음 열 때 모델을 약 52MB(잎 탐색 + 기본 분류 모델) 받고, 이후에는 브라우저 캐시를 씁니다.

### 2. 내 PC에서 바로 열기
`index.html`을 더블클릭해 열 수도 있습니다. 이때는 브라우저 보안 규칙 때문에 모델을 자동으로 읽지 못하므로,
화면의 **models 폴더 고르기**로 이 폴더 안의 `models` 폴더를 한 번 골라 주세요. (샘플 사진 버튼은 숨겨집니다.)
또는 이 폴더에서 `python -m http.server 8000`을 실행하고 `http://localhost:8000`을 열면 1번과 똑같이 동작합니다.

## 화면에서 할 수 있는 것
| 노트북 단계 | HTML 화면 |
|---|---|
| 3. 사진 올리기 | 끌어다 놓기·여러 장, **샘플 사진으로 실행** |
| 4. 실행 (`CLS_MODEL`) | 분류 모델 3개 중 선택, 계산 장치(자동 / GPU(WebGPU) / CPU(WASM)) |
| 5. 결과 보기 | 결과 이미지, 상태별 개수, 이상·의심 잎 확대 사진과 클래스별 확률, 모든 잎 표 |
| 6. VLM 검증 | Claude API 키(직접 입력 또는 키 파일)로 잎마다 확대 + 주변 사진 2장 검증, verdict 표, `_vlm.jpg` |
| 7. 결과 내려받기 | `results.zip` (사진별 `.json`, `_vis.jpg`, `summary.csv`, `vlm_review.csv`, `_vlm.json/jpg`) |

## 결과가 Python 버전과 같은가
CPU(WASM)로 계산하면 **잎 단위로 같습니다.** 샘플 사진과 변형 2장(세로로 자른 사진, 1280×960으로 줄인 사진), 분류 모델 3개로 비교했습니다.
- 잎 수, 잎 ID, 박스, 윤곽선(polygon), 상태: 모두 같음 (308 / 290 / 225개)
- 클래스 확률: 소수 넷째 자리까지 같음 (YOLO26m만 최대 0.0008 차이)
- 잎 면적: 수천 개 중 1개에서 몇 픽셀 차이 (마스크 경계 반올림)
- sharpness는 최대 1.1 차이 (표시용 값, 판정에 안 쓰임)

이렇게 맞춘 방법: 모델은 `.pt`의 fp16 가중치를 손실 없이 옮겼고(ONNX 출력이 원래 ONNX와 비트 단위로 같음), 이미지 축소는 OpenCV와 같은 함수,
잎 캔버스 만들기는 Pillow의 Lanczos/Bilinear 리샘플링을 그대로 옮겨 Python과 바이트 단위로 같게 했습니다.

**GPU(WebGPU)** 는 훨씬 빠르지만 그래픽 카드마다 계산 순서가 달라 확률이 조금 다를 수 있습니다. 결과를 노트북과 똑같이 맞춰야 하면 CPU(WASM)를 고르세요. (WebGPU 경로는 소프트웨어 GPU에서 오류 없이 도는 것까지만 확인했고, 실제 그래픽 카드에서의 결과 비교는 하지 못했습니다.) 결과 화면에 어떤 장치로 계산했는지 표시됩니다.
CPU는 사진 1장(2560×1920)에 1~2분 걸립니다.

## VLM(Claude) 검증
- Claude API 키가 필요합니다. 키는 이 페이지의 메모리에만 있고 저장하지 않습니다. 새로 고치면 사라집니다. 브라우저에서 Anthropic API로 바로 요청합니다.
- 잎 하나에 요청 한 번, 사용량만큼 요금이 나갑니다 (검증할 잎 수 확인 → 실행).
- 공개 주소에 올려도 키는 각자 자기 브라우저에서 입력하므로 서버에 키를 둘 일이 없습니다.
- Qwen(로컬 GPU 모델)은 브라우저에서 돌리기에 너무 커서 빠졌습니다. 필요하면 Colab 노트북이나 `web/` 서버 버전을 쓰세요.
- 건강 잎 표본(FRESH 무작위 5장)은 Python과 다른 난수를 써서 뽑히는 잎이 다를 수 있습니다. 이상·의심 잎 목록은 같습니다.

## 모델을 다시 학습했을 때
```bash
pip install ultralytics==8.4.198 onnx onnxslim
yolo export model=leaf_seg_v8r2.pt format=onnx imgsz=640 dynamic=True simplify=True opset=17
yolo export model=leaf_cls_yolov8m.pt format=onnx imgsz=224 dynamic=True simplify=True opset=17
python tools/pack_models.py models leaf_seg_v8r2.onnx:leaf_seg_v8r2.pt leaf_cls_yolov8m.onnx:leaf_cls_yolov8m.pt
```
새 분류 모델을 추가하면 `app.js` 위쪽 `CLS` 목록에 한 줄 넣습니다.

## 알아 둘 점
- 크롬, 엣지, 웨일 최신 버전을 권장합니다. 사파리·파이어폭스는 WebGPU가 없거나 제한적이라 CPU로 계산합니다.
- 메모리를 1~2GB 씁니다. 휴대폰에서는 느리거나 멈출 수 있습니다.
- 결과는 새로 고치면 사라집니다. zip으로 받아 두세요.
