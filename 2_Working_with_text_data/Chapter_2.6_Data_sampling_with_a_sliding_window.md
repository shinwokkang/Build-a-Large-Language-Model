# 📖 [심층 해설] 2.6: 슬라이딩 윈도우를 활용한 데이터 샘플링 (Data sampling with a sliding window) 완벽 정복

지금까지 우리는 텍스트를 기계가 읽을 수 있는 '숫자(토큰 ID)'로 바꾸는 완벽한 토크나이저(BPE)를 만들었습니다. 이제 남은 것은 이 숫자들을 딥러닝 모델(LLM)의 입에 직접 떠먹여 주기 위한 **데이터 급식판(Data Loader)**을 짜는 일입니다.

2.6절에서는 거대 언어 모델이 도대체 어떤 방식으로 글을 학습하는지 그 본질을 꿰뚫어 보고, 실무 딥러닝의 표준인 파이토치(PyTorch)를 이용해 훈련 데이터를 가장 효율적으로 떠먹여 주는 **슬라이딩 윈도우(Sliding Window)** 기법을 코드로 완벽하게 구현해 봅니다.

---

## 1. 🧠 LLM은 어떻게 똑똑해질까? '다음 단어 맞추기'의 마법 (Figure 2.12)

우리가 쓰는 ChatGPT가 그토록 엄청난 지식을 자랑하지만, 녀석이 공장에서 학습(Pretraining)받을 때 했던 시험 문제는 딱 하나뿐입니다. 바로 **"다음에 올 단어 맞추기(Next-word prediction)"** 입니다.

그림 2.12를 자세히 보면 LLM의 학습법이 얼마나 단순무식하고 확실한지 알 수 있습니다.
* 모델에게 `"LLMs learn to predict"` 라는 문장(Input)을 보여줍니다.
* 그리고 모델에게 **"자, 이 뒤에 올 단어 1개가 뭘까?"** 라고 질문합니다.
* 정답(Target)은 `"one"` 입니다. 
* 모델이 틀리면 매를 맞고(가중치 업데이트), 맞추면 칭찬을 받으며 다음 문제로 넘어갑니다. 

이 단순한 '스무고개' 놀이를 인터넷에 있는 수조 개의 단어에 대해 죽을 때까지 반복하면, 모델은 단순한 문법뿐만 아니라 수학, 코딩, 철학적 지식까지 스스로 깨우치게 됩니다.

---

## 2. 💻 코드로 보는 입력(x)과 정답(y) 쌍 만들기

가장 먼저 2.5절에서 배운 `tiktoken` BPE를 이용해 소설 전체를 숫자로 바꿉니다.
```python
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 소설 전체를 토큰 ID(숫자)로 변환
enc_text = tokenizer.encode(raw_text)
print(len(enc_text)) 
# 출력: 5145 (총 5,145개의 토큰으로 이루어짐)

# 설명의 편의를 위해 흥미로운 문장이 나오는 50번째 토큰부터 시작합니다.
enc_sample = enc_text[50:] 
```

### 📝 코드 1: 파이썬 리스트 슬라이싱으로 문제(x)와 정답(y) 만들기
모델이 한 번에 볼 수 있는 글자의 최대 길이(Context Size)를 `4`로 정해봅시다.
```python
context_size = 4  # 모델이 한 번에 읽을 수 있는 토큰의 개수 (창문의 크기)

# 문제 데이터(x): 0번째부터 3번째 토큰까지 (총 4개)
x = enc_sample[:context_size] 

# 정답 데이터(y): 1번째부터 4번째 토큰까지 (총 4개)
y = enc_sample[1:context_size+1] 

print(f"x: {x}")
print(f"y:      {y}")
```
* **결과**:
  - `x: [290, 4920, 2241, 287]` (and established himself in)
  - `y:      [4920, 2241, 287, 257]` (established himself in a)
* **핵심 원리**: 정답 데이터 `y`는 문제 데이터 `x`에서 **정확히 오른쪽으로 1칸씩(Shifted by 1)** 이동한 형태입니다. 
  - `x`가 "나는 밥을 먹고" 라면, 
  - `y`는 "밥을 먹고 왔다" 가 되는 식입니다.

### 📝 코드 2: 모델 머릿속에서 일어나는 일 (for 루프 시뮬레이션)
입력값 4개가 모델에 들어가면, 모델은 순차적으로 4개의 문제를 동시에 풉니다.
```python
for i in range(1, context_size+1):
    context = enc_sample[:i]    # 모델이 읽은 문맥
    desired = enc_sample[i]     # 맞춰야 할 다음 단어(정답)
    
    # 숫자(ID)를 다시 사람이 읽을 수 있는 텍스트로 디코딩하여 출력
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
```
* **출력 결과**:
  ```text
   and ---->  established
   and established ---->  himself
   and established himself ---->  in
   and established himself in ---->  a
  ```
* 왼쪽(---->)이 모델에게 주어지는 힌트이고, 오른쪽이 모델이 맞춰야 할 정답입니다. 이처럼 단어 1칸을 밀어서 무한한 퀴즈(Input-Target Pair)를 만들어내는 것이 LLM 훈련의 알파이자 오메가입니다.

---

## 3. 🖼️ 파이토치(PyTorch)의 꽃: 텐서(Tensor)와 데이터 로더 (Figure 2.13)

<img width="648" height="314" alt="Figure 2 13" src="https://github.com/user-attachments/assets/72cc1238-9af4-4a40-a2a7-4d2ead7eb1b7" />


위에서는 1차원 배열(리스트)로 퀴즈를 하나 만들었습니다. 하지만 실전에서 수십억 개의 퀴즈를 CPU로 하나하나 풀게 하면 학습에만 100년이 걸립니다. 우리는 엔비디아(NVIDIA) GPU의 막강한 병렬 처리 능력을 써야 합니다.

* **텐서 (Tensor)**: GPU가 연산할 수 있도록 고안된 고성능 다차원 배열(행렬)입니다.
* **배치 (Batch)**: GPU가 한 번에 병렬로 풀 수 있는 문제 뭉치(묶음)를 말합니다.

그림 2.13을 보면 1차원 리스트였던 입력 데이터들이 차곡차곡 쌓여서 거대한 2차원 블록(Matrix) 형태의 텐서인 **대문자 X (문제지 모음)**와 **대문자 Y (정답지 모음)**로 변환되는 것을 볼 수 있습니다.

이 변환을 자동으로, 그리고 메모리 과부하 없이 효율적으로 해주는 도구가 바로 파이토치의 `Dataset`과 `DataLoader` 클래스입니다.

---

## 4. 💻 실전 코드 분석: `GPTDatasetV1` 클래스 완벽 해부 (Listing 2.5)

메모리(RAM)가 32GB인데 텍스트 데이터가 500GB라면 어떻게 될까요? 한 번에 데이터를 읽으려 하면 컴퓨터가 터집니다. 그래서 필요한 만큼만 데이터를 썰어서(Chunk) 내보내 주는 자판기(`Dataset`)를 만들어야 합니다.

```python
import torch
from torch.utils.data import Dataset, DataLoader

# 파이토치의 내장 클래스인 Dataset을 상속(Inheritance)받아 나만의 데이터셋을 만듭니다.
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 1. 원본 텍스트 전체를 일단 숫자로 바꿉니다.
        token_ids = tokenizer.encode(txt)    

        # 2. 슬라이딩 윈도우 (Sliding Window) 알고리즘
        for i in range(0, len(token_ids) - max_length, stride):
            # 문제지: i부터 max_length 개수만큼 자름
            input_chunk = token_ids[i:i + max_length]
            
            # 정답지: 문제지보다 오른쪽으로 딱 1칸 더 간 곳부터 max_length 개수만큼 자름
            target_chunk = token_ids[i + 1: i + max_length + 1]
            
            # 파이썬 리스트를 GPU가 좋아하는 torch.tensor 형태로 변환하여 보관
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # 3. 데이터셋의 총 길이(문제의 총 개수)를 반환하는 필수 매직 메서드
    def __len__(self):    
        return len(self.input_ids)

    # 4. 데이터 로더가 "idx번째 문제 좀 줘!" 라고 요청할 때 데이터를 뱉어내는 필수 매직 메서드
    def __getitem__(self, idx):         
        return self.input_ids[idx], self.target_ids[idx]
```
* **슬라이딩 윈도우 알고리즘의 핵심**: `for i in range(0, 마지막, stride)`
  - 창문(`max_length`) 크기만큼 데이터를 보고 나면, 그 창문을 오른쪽으로 몇 칸(`stride`) 이동시킬지를 결정하여 데이터를 썰어냅니다.

---

## 5. 💻 실전 코드 분석: `create_dataloader_v1` 완벽 해부 (Listing 2.6)

`Dataset`이 문제를 만들어두는 창고라면, `DataLoader`는 창고에서 문제들을 묶음(Batch) 단위로 예쁘게 포장해서 GPU에게 배달해 주는 트럭입니다.

```python
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    # 1. 토크나이저 준비
    tokenizer = tiktoken.get_encoding("gpt2")                         
    
    # 2. 방금 만든 자판기(Dataset) 객체 생성
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)   
    
    # 3. 배달 트럭(DataLoader) 객체 생성
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,   # 한 번에 GPU로 실어보낼 문제의 개수
        shuffle=shuffle,         # 문제를 랜덤하게 섞을지 여부 (학습 시에는 True가 정석)
        drop_last=drop_last,     # 마지막에 꼬투리로 남는 데이터 쪼가리를 버릴지 여부
        num_workers=num_workers  # 데이터 포장에 동원할 CPU 코어의 개수 (0은 메인 스레드만 씀)
    )

    return dataloader
```
* **💡 실무 꿀팁: `drop_last=True`를 하는 치명적인 이유**
  - 총 문제가 101개인데 `batch_size`가 10이라면, 10개씩 10번 포장하고 마지막에 1개짜리 묶음(Batch)이 남습니다. 
  - 딥러닝 모델은 항상 10개짜리를 풀다가 갑자기 1개짜리 문제를 풀면 텐서 크기가 안 맞거나 오차(Loss)가 비정상적으로 치솟는(Loss Spike) 현상이 발생합니다. 훈련 안정성을 위해 꼬투리는 과감히 버리는 것이 딥러닝 실무의 철칙입니다.

---

## 6. 🖼️ 창문 크기(max_length)와 보폭(stride)의 마법 (Figure 2.14)

<img width="642" height="458" alt="Figure 2 14" src="https://github.com/user-attachments/assets/da21b71b-0126-42f0-a807-40dfdf8f5e7b" />


방금 만든 데이터 로더를 이용해 `stride`의 의미를 시각적으로 살펴보겠습니다. (그림 2.14)

### 📝 코드 테스트 1: Stride = 1 (1칸씩 이동)
```python
dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)

print(next(data_iter)) # 1번째 묶음
# 입력: [[40, 367, 2885, 1464]], 정답: [[367, 2885, 1464, 1807]]

print(next(data_iter)) # 2번째 묶음
# 입력: [[367, 2885, 1464, 1807]], 정답: [[2885, 1464, 1807, 3619]]
```
* **해설**: 창문을 1칸만 움직였으므로, 1번째 묶음과 2번째 묶음의 데이터가 `[367, 2885, 1464]` 만큼 **심하게 겹칩니다(Overlap).**

### 📝 코드 테스트 2: Stride = 4 (창문 크기와 동일하게 이동)
실무에서 배치 사이즈를 8 이상으로 늘릴 때 코드를 보겠습니다.
```python
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
```
* 이 경우 창문 크기(`max_length`)가 4인데, 오른쪽으로 이동하는 보폭(`stride`)도 4입니다.
* **효과**: 창문이 겹치지 않고 딱딱 끊어져서 다음 데이터로 넘어갑니다. (Overlap 없음)
* **💡 실무적 이유**: 데이터가 너무 겹치면 모델이 똑같은 문구를 너무 여러 번 보게 되어 **과적합(Overfitting, 암기해 버리는 현상)**이 발생합니다. 모델이 문장을 통째로 외우는 것을 막기 위해 실무에서는 보통 `stride`를 `max_length`와 동일하게 맞추어 훈련 데이터를 구축합니다.

---

## 7. 🌐 [2027년 최신 실무 트렌드] 무한히 늘어나는 컨텍스트 윈도우

우리가 방금 작성한 코드에서는 설명의 편의상 `max_length`(문맥 길이)를 4나 256 정도로 아주 작게 잡았습니다.

* **[최신 트렌드 1] 무한의 컨텍스트**: 2027년 최신 실무 환경에서 GPT-4o나 Google Gemini 1.5 Pro 모델들의 `max_length`는 무려 **100만~200만(1M~2M) 토큰**에 달합니다. 모델이 책 수십 권 분량의 텍스트를 "한 번에(하나의 입력으로)" 읽고 기억한다는 뜻입니다.
* **[최신 트렌드 2] GPU 메모리의 장벽**: `max_length`가 256에서 100만으로 커지면 텐서의 크기도 기하급수적으로 폭발합니다. VRAM(비디오 메모리) 80GB짜리 최상급 GPU인 H100을 수천 대 이어 붙여도 `batch_size`를 겨우 1이나 2로 설정해야 할 만큼 메모리 압박이 심합니다. 
* **[최신 해결책]**: 현대 딥러닝 엔지니어들은 이 문제를 해결하기 위해 파이토치의 `DataLoader` 레벨에서 메모리를 쪼개는 기술(Tensor Parallelism)이나, 메모리 입출력 속도를 극한으로 끌어올리는 **Flash Attention(플래시 어텐션)** 등의 최첨단 최적화 알고리즘을 도입하여 초거대 슬라이딩 윈도우 훈련을 성공시켜 내고 있습니다.

---

# 📝 이해도 점검 (Verification Q&A)

### ❓ 질문 1
거대 언어 모델(LLM)을 학습시킬 때, 훈련 데이터를 `x(입력)` 텐서와 `y(정답)` 텐서로 분리하여 구성합니다. 이때 두 텐서의 데이터는 정확히 어떤 관계(규칙)를 가지며, 모델에게 어떤 방식의 퀴즈를 내기 위해 이렇게 구성하는 것인가요?

### ❓ 질문 2
파이토치(PyTorch)의 `DataLoader`를 생성할 때 `drop_last=True` 매개변수를 사용하는 것이 실무적으로 왜 중요한가요? (배치 사이즈와 관련된 딥러닝 훈련의 안정성 관점에서 설명해 보세요.)

### ❓ 질문 3
슬라이딩 윈도우 방식으로 데이터를 자를 때, `max_length`를 256으로 설정했다면 `stride`(보폭)도 동일하게 256으로 맞추는 것이 실무적으로 권장됩니다. 만약 `stride`를 1로 설정하여 배치 간에 데이터가 극심하게 겹치도록(Overlap) 만들면 학습 시 어떤 부작용이 발생하나요?

---
---

### 💡 모범 답안 (정확한 이해를 위한 체크)

**답안 1**
정답 텐서 `y`는 입력 텐서 `x`에서 정확히 오른쪽으로 1칸 시프트(Shifted by 1)된 구조를 가집니다. 이는 모델에게 이전까지의 모든 문맥(x)을 보여주고, 그 문장 바로 다음에 등장할 단 1개의 새로운 단어(y)를 예측하게 만드는 **"다음 단어 예측(Next-word prediction)"** 퀴즈를 무한히 생성하기 위함입니다. 이 단순한 방식이 LLM 사전 훈련(Pretraining)의 핵심 원리입니다.

**답안 2**
데이터셋의 총개수가 배치 사이즈(Batch Size)로 나누어 떨어지지 않으면, 마지막 훈련 스텝에서 지정된 배치 사이즈보다 훨씬 작은 꼬투리 배치가 GPU로 들어오게 됩니다. 모델은 항상 일정한 크기의 배치에 맞춰서 오차(Loss)의 평균을 내며 가중치를 업데이트하는데, 갑자기 크기가 작은 배치가 들어오면 업데이트 계산 비율이 어긋나 오차 값이 튀어 오르는(Loss Spike) 등 학습 과정이 매우 불안정해질 수 있습니다. 이를 방지하기 위해 꼬투리를 깔끔히 버리는(`drop_last=True`) 것이 안전합니다.

**답안 3**
`stride`를 1로 설정하면, 첫 번째 배치의 내용과 두 번째 배치의 내용이 단 한 글자를 제외하고 100% 동일하게 구성됩니다. 이렇게 데이터가 극심하게 겹치게 되면 모델이 동일한 문구를 훈련 과정에서 불필요하게 많이 반복해서 보게 됩니다. 이로 인해 모델이 언어의 범용적 규칙을 깨우치는 대신 해당 특정 문장 자체를 맹목적으로 암기해 버리는 **과적합(Overfitting)** 문제가 심각하게 발생하며, 학습 연산량도 비효율적으로 폭증하게 됩니다.
