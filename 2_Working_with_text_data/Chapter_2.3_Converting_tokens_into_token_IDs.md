# 📖 [심층 해설] 2.3: 토큰을 토큰 ID로 변환하기 (Converting tokens into token IDs) 및 파이썬 코드 완벽 분석

앞선 2.2절에서 우리는 텍스트 문장을 '토큰(Token)'이라는 의미 있는 글자 조각들로 완벽하게 쪼개는 데 성공했습니다. 하지만 컴퓨터, 특히 신경망(Neural Network)은 `"apple"`이나 `"hello"` 같은 문자열(String) 데이터를 전혀 이해하지 못합니다. 오직 '숫자(Integer, Float)'만 계산할 수 있죠.

따라서 2.3절의 핵심 목표는 **문자열로 이루어진 토큰들을 고유한 '정수 번호(Token ID)'로 번역해 주는 작업**입니다. 이 과정은 추후에 정수 번호를 임베딩 벡터(실수)로 변환하기 위한 필수적인 중간 다리 역할을 합니다.

---

## 1. 🖼️ 단어 사전(Vocabulary) 구축 원리 (Figure 2.6 해설)

<img width="662" height="471" alt="Figure 2 6" src="https://github.com/user-attachments/assets/a8e052f5-04f6-4120-9a36-7be2d92e01bd" />


가장 먼저 해야 할 일은 우리가 가진 모든 텍스트 데이터를 분석하여 **"이 세상에 어떤 단어들이 존재하는가?"**를 규정하는 **단어 사전(Vocabulary)**을 만드는 것입니다.

그림 2.6을 보면 단어 사전을 만드는 과정이 직관적으로 나와 있습니다.
1. 훈련 데이터 전체를 토큰으로 쪼갭니다.
2. 중복되는 단어를 모두 제거하고 **고유한(Unique) 토큰**들만 남깁니다. (예: "the"가 100번 등장해도 사전에 등재할 때는 딱 1번만 들어갑니다.)
3. 이 고유한 토큰들을 알파벳 순서대로 정렬(Sorting)합니다.
4. 순서대로 0번부터 시작하는 **고유한 정수 번호(Token ID)**를 부여합니다. (`brown` -> 0, `dog` -> 1, `fox` -> 2...)

### 📝 코드 분석: 고유 단어 추출과 사전 크기 확인
```python
# 1. set() 함수로 중복을 제거하고, sorted() 함수로 알파벳 순 정렬을 수행합니다.
all_words = sorted(set(preprocessed))

# 2. len() 함수를 통해 총 고유 단어의 개수(사전의 크기)를 구합니다.
vocab_size = len(all_words)
print(vocab_size)
```
* **코드 해설**: 파이썬의 `set()` 자료형은 수학의 '집합'과 같아서 중복된 원소를 허용하지 않습니다. 2.2절에서 만든 4,690개의 단편 소설 토큰 조각들을 `set()`에 넣고 다시 `sorted()`로 빼내면, 알파벳 순서대로 정렬된 고유 단어 리스트 `all_words`가 만들어집니다.
* **출력 결과**: `1130`. 즉, 에디스 워튼의 단편 소설 안에는 총 1,130개의 고유한 단어와 기호들이 사용되었음을 알 수 있습니다.

### 📝 Listing 2.2: 단어 사전(Vocabulary) 딕셔너리 만들기
```python
# 파이썬의 딕셔너리 컴프리헨션(Dictionary Comprehension)을 사용한 우아한 코드
vocab = {token:integer for integer, token in enumerate(all_words)}

# 사전이 잘 만들어졌는지 앞의 50개만 출력해서 확인해 보기
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break
```
* **코드 해설**: 
  - `enumerate(all_words)`는 리스트의 원소를 하나씩 꺼낼 때, `0, 1, 2...` 같은 인덱스(integer)를 함께 묶어주는 파이썬 내장 함수입니다.
  - `{token: integer}` 형태로 작성하여, 문자열(단어)을 열쇠(Key)로, 정수(ID)를 값(Value)으로 가지는 파이썬 딕셔너리(Dictionary) 구조를 단 한 줄 만에 만들어 냅니다.
* **결과**: `('!', 0)`, `('"', 1)`, `("'", 2)` ... `('Hermia', 50)` 처럼, 기호부터 대문자 A, B, C 순으로 고유 번호표가 예쁘게 매겨진 것을 볼 수 있습니다.

---

## 2. 🖼️ 인코딩(Encoding)과 디코딩(Decoding)의 개념 (Figure 2.7 & 2.8)

<img width="649" height="468" alt="Figure 2 7" src="https://github.com/user-attachments/assets/d5a69107-de72-420c-a4de-9afd7ab34ffd" />

<img width="652" height="400" alt="Figure 2 8" src="https://github.com/user-attachments/assets/4244bc3b-a074-4d8d-90e1-58dfadddbc9c" />


단어 사전을 완성했다면 이제 통역을 할 차례입니다.
* **인코딩 (Encode)**: 사람이 읽는 **문자열 텍스트** ➡️ 기계가 읽는 **숫자(Token IDs)** 로 변환하는 과정입니다. 모델에게 데이터를 먹일 때 사용합니다. (Figure 2.7)
* **디코딩 (Decode)**: 기계가 뱉어낸 **숫자(Token IDs)** ➡️ 사람이 읽는 **문자열 텍스트** 로 복원하는 과정입니다. 모델의 답변을 화면에 띄워줄 때 사용합니다. (Figure 2.8)

이를 프로그래밍으로 완벽하게 제어하기 위해, 책에서는 `SimpleTokenizerV1` 이라는 파이썬 클래스(Class)를 직접 구현합니다.

### 📝 Listing 2.3: SimpleTokenizerV1 클래스 완벽 해부

```python
import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        # 1. 문자를 숫자로 바꾸는 사전 (Encoding용)
        self.str_to_int = vocab            
        
        # 2. 숫자를 문자로 바꾸는 역방향 사전 (Decoding용)
        self.int_to_str = {i:s for s,i in vocab.items()}        

    def encode(self, text):
        # 3. 2.2절에서 만든 궁극의 정규식으로 문장을 토큰으로 조각냄
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        
        # 4. 쪼개진 단어들을 하나씩 꺼내어 str_to_int 사전을 통해 정수 ID로 변환
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        # 5. 역방향 사전(int_to_str)을 통해 정수 ID를 다시 문자열로 바꾼 뒤, 공백(" ")을 넣어 하나로 합침
        text = " ".join([self.int_to_str[i] for i in ids]) 

        # 6. 보기 좋게 만들기: 구두점 앞의 쓸데없는 공백 제거 (예: "Hello ," -> "Hello,")
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)    
        return text
```
* **코드 상세 해설**:
  - `__init__`: 초기화 메서드입니다. 클래스를 생성할 때 방금 만든 1,130개짜리 `vocab` 사전을 입력받습니다. 그리고 인코딩을 위한 원본 사전(`str_to_int`)과, 딕셔너리의 Key와 Value를 뒤집어 디코딩을 위한 역방향 사전(`int_to_str`) 두 개를 메모리에 저장해 둡니다.
  - `encode`: 입력받은 문장을 정규식으로 잘게 쪼갠 뒤, 리스트 컴프리헨션을 이용해 사전에 있는 정수 번호들로 치환하여 `[1, 56, 2, ...]` 같은 숫자 배열(`ids`)을 반환합니다.
  - `decode`: 숫자를 문자로 다시 바꿉니다. 이때 `" ".join()` 함수를 쓰면 모든 단어 사이에 강제로 공백이 들어가기 때문에, 쉼표나 마침표 앞에도 어색한 공백이 생기게 됩니다. 이를 해결하기 위해 파이썬 정규식 치환 함수인 `re.sub`를 사용하여 구두점 앞의 공백(`\s+`)을 지워주고 자연스러운 인간의 문장으로 복원합니다.

---

## 3. 토크나이저 실전 테스트와 치명적인 문제점 (KeyError)

이제 우리가 만든 이 훌륭한 `SimpleTokenizerV1` 객체를 직접 사용해 봅시다.

### 📝 정상 작동 테스트 (소설 본문 내의 문장)
```python
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
# 출력: [1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108, 754, 793, 7]

print(tokenizer.decode(ids))
# 출력: " It's the last he painted, you know," Mrs. Gisburn said with pardonable pride.
```
* 소설 "The Verdict" 안에 등장했던 문장을 그대로 넣었기 때문에, 인코딩과 디코딩이 소름 돋게 완벽하게 작동합니다.

### 📝 에러 발생 테스트 (소설에 없던 새로운 문장)
```python
text = "Hello, do you like tea?"
print(tokenizer.encode(text))
```
* **출력 결과**: 🚨 **`KeyError: 'Hello'`**

### 💣 KeyError 분석 및 2027년 최신 실무 관점 (OOV 문제 해결)
이 치명적인 에러가 발생한 이유는, `"Hello"`라는 단어가 에디스 워튼의 단편 소설 안에서는 단 한 번도 사용되지 않았기 때문입니다. 즉, 우리의 1,130개짜리 단어 사전(`vocab`) 안에 `"Hello"`라는 항목 자체가 없으므로 파이썬이 딕셔너리에서 열쇠(Key)를 찾지 못해 에러를 뿜어낸 것입니다.

기계 학습 분야에서는 이를 **OOV (Out-Of-Vocabulary, 미등록 단어)** 문제라고 부릅니다.

* **과거의 땜질식 해결책**: 
  - 과거에는 사전에 없는 단어가 나오면 `[UNK]` (Unknown)이라는 퉁명스러운 특수 기호 하나로 뭉뚱그려 치환해 버렸습니다. 이 방식은 문맥의 핵심 의미가 완전히 날아가버린다는 심각한 단점이 있었습니다.
* **[2027년 최신 실무] BPE (Byte Pair Encoding)의 대중화**: 
  - 우리가 사용하는 최신 ChatGPT, Llama 3/4 등은 훈련 데이터셋을 수조 개(Trillion) 단위로 거대하게 늘려 사전 자체의 크기를 수십만 개(예: 128K) 규모로 키웁니다.
  - 하지만 아무리 사전을 키워도 신조어나 외계어 오타는 계속 생겨납니다. 최신 모델들은 이를 해결하기 위해 단어를 통째로 등록하지 않고, **알파벳 문자 단위나 바이트(Byte) 단위의 서브워드(Subword)**로 쪼개어 사전을 구축하는 방식을 사용합니다. 
  - 만약 최신 모델에 `"Hello"`라는 미등록 단어가 들어온다면 에러를 내거나 무시하는 대신, `"Hel" + "lo"` 혹은 `"H" + "e" + "l" + "l" + "o"` 처럼 사전에 이미 등록되어 있는 더 작은 조각들로 쪼개어 완벽하게 해석해 냅니다. 이 BPE(바이트 쌍 인코딩) 기법은 바로 2.5절에서 등장할 핵심 기술입니다.

---

# 📝 이해도 점검 (Verification Q&A)

### ❓ 질문 1
단어 사전을 구축할 때, 파이썬의 `set()` 함수를 거친 뒤 다시 `sorted()` 함수로 알파벳 정렬을 수행했습니다. 이처럼 중복 단어를 제거하고 하나의 고유 번호만을 부여해야 하는 이유는 무엇인가요?

### ❓ 질문 2
`SimpleTokenizerV1` 클래스 내의 `decode()` 메서드에서, 정수 배열을 문자열로 바꾼 직후에 `re.sub(r'\s+([,.?!"()\'])', r'\1', text)` 라는 코드를 굳이 추가하여 구두점 앞의 공백을 제거한 이유는 무엇인가요?

### ❓ 질문 3
작성한 토크나이저에 `"Hello"`라는 텍스트를 `encode()` 하려 하자 `KeyError`가 발생했습니다. 실무(현대의 LLM)에서는 이러한 OOV(Out-Of-Vocabulary) 문제를 해결하기 위해, 에러를 내거나 통째로 무시하는 대신 어떤 혁신적인 접근법(개념)을 사용하여 단어를 처리하나요?

---
---

### 💡 모범 답안 (정확한 이해를 위한 체크)

**답안 1**
학습 데이터 안에는 "the"나 "a" 같은 단어들이 수만 번씩 등장합니다. 만약 중복을 허용하여 "the"라는 똑같은 단어에 1번, 50번, 100번 등 서로 다른 ID가 부여된다면, 딥러닝 모델은 이 번호들을 제각각 다른 의미의 단어로 착각하여 학습 효율이 극도로 떨어집니다. 따라서 똑같은 형태와 의미를 지닌 토큰은 무조건 단 하나의 고유 정수(ID)로 1:1 매핑되어야만 모델이 일관성 있게 언어 규칙을 학습할 수 있습니다.

**답안 2**
숫자 ID 배열을 문장으로 합칠 때 `" ".join()` 방식을 사용하면, 원래는 띄어쓰기가 없어야 할 구두점 앞에도 무조건 공백이 추가됩니다 (예: `"world ,"`). 이는 인간의 언어 규칙에 맞지 않는 부자연스러운 문장이므로, 디코딩 단계에서 정규식을 이용해 구두점 앞의 불필요한 공백을 삭제하여 사람이 읽기에 자연스럽고 올바른 문장 구조(`"world,"`)로 복원해 주기 위함입니다.

**답안 3**
실무에서는 단어를 띄어쓰기 기준으로 통째로 사전에 등록하는 대신, **서브워드(Subword) 토큰화 기법(주로 BPE, Byte Pair Encoding)**을 사용합니다. 이를 통해 사전에 없는 낯선 단어나 신조어, 오타가 입력으로 들어오더라도 에러를 발생시키지 않고, `"H"`, `"el"`, `"lo"` 등 모델 사전에 이미 등록되어 있는 더 작은 문자 혹은 바이트 조각들로 유연하게 분해하여 처리합니다.
