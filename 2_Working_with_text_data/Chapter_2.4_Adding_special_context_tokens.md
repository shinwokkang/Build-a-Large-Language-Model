# 📖 [심층 해설] 2.4: 특수 문맥 토큰 추가하기 (Adding special context tokens) 및 파이썬 코드 완벽 분석

바로 앞선 2.3절에서 우리는 아주 치명적인 에러(`KeyError: 'Hello'`)를 마주했습니다. 훈련 데이터(사전)에 없는 낯선 단어가 들어오면 파이썬 프로그램 자체가 뻗어버리는 **OOV(Out-Of-Vocabulary, 미등록 단어)** 문제였습니다.

이번 2.4절에서는 이 문제를 임시로 땜질하는 방법과 함께, 모델에게 **"문맥이 바뀌었다"**는 중요한 힌트를 주기 위해 일반적인 단어가 아닌 **'특수 토큰(Special Tokens)'**을 사전에 추가하고 토크나이저를 업그레이드(V2)하는 과정을 실습해 봅니다.

---

## 1. 🖼️ 특수 토큰의 도입 (Figure 2.9 해설)

<img width="653" height="421" alt="Figure 2 9" src="https://github.com/user-attachments/assets/4952e25e-b4cb-4fe4-a8da-bb4ea50d2425" />


우리가 평소에 쓰는 언어에는 단어만 있는 것이 아닙니다. 기계에게 '상황'을 알려주는 특별한 기호가 필요합니다. 그림 2.9는 기존 단어 사전(`brown`, `dog`, `fox`...)의 맨 마지막에 파란색으로 두 개의 새로운 특수 기호를 추가하는 모습을 보여줍니다.

1. **`<|unk|>` 토큰 (Unknown)**: 
   - 사전에 없는 낯선 단어(신조어, 오타, 학습하지 않은 단어)가 들어왔을 때 에러를 뿜고 멈추는 대신, **"이건 내가 모르는 미지의 단어다"**라는 의미로 퉁치고 넘어가게 해주는 방어막 역할을 합니다.
2. **`<|endoftext|>` 토큰 (End of Text)**: 
   - 텍스트 문서 하나가 완전히 끝났음을 알리는 표지판입니다.

---

## 2. 🖼️ 독립적인 텍스트 병합하기 (Figure 2.10 해설)

<img width="639" height="393" alt="Figure 2 10" src="https://github.com/user-attachments/assets/99b347e3-f033-4325-83f4-b6b606a0df91" />


LLM을 훈련시킬 때는 책 한 권만 읽히는 것이 아니라, 인터넷에 있는 수천만 개의 위키백과 문서와 뉴스 기사들을 **일렬로 쭉 이어 붙여서(Concatenate)** 거대한 하나의 문자열로 만들어 모델에게 먹여줍니다.

* **문제점**: [스포츠 뉴스 기사] 바로 뒤에 [판타지 소설]이 띄어쓰기 하나만 두고 바로 이어지면, 모델은 "스포츠 선수가 갑자기 마법을 쓴다"고 문맥을 크게 오해하며 헷갈리게 됩니다.
* **해결책 (Figure 2.10)**: 서로 전혀 상관없는 문서 A와 문서 B를 이어 붙일 때, 그 사이에 **`<|endoftext|>`**라는 거대한 벽(토큰)을 세워둡니다. 
  - 모델이 이 토큰을 읽는 순간, **"아! 여기까지가 앞이야기 끝이고, 지금부터는 전혀 상관없는 새로운 문맥이 시작되는구나!"** 하고 뇌 속의 문맥 기억(Context)을 깔끔하게 리셋하게 됩니다.

---

## 3. 💻 파이썬 코드 분석: 사전 업데이트 및 토크나이저 V2

이제 파이썬 코드를 통해 우리의 단어 사전을 업데이트하고, 업그레이드된 `SimpleTokenizerV2`를 만들어 보겠습니다.

### 📝 코드 1: 단어 사전에 특수 토큰 추가하기
```python
# 1. 2.2절에서 만든 원본 토큰 리스트를 고유 단어 집합으로 만듭니다.
all_tokens = sorted(list(set(preprocessed)))

# 2. 파이썬 리스트의 extend() 함수를 사용해 두 개의 특수 토큰을 꼬리에 추가합니다.
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

# 3. 다시 딕셔너리 컴프리헨션을 돌려 정수 ID를 매깁니다.
vocab = {token:integer for integer,token in enumerate(all_tokens)}

print(len(vocab.items())) 
# 출력: 1132 (기존 1130개 + 특수 토큰 2개 추가됨)
```
* 사전에 새로운 단어를 추가하려면 단순히 리스트 끝에 밀어 넣고 번호표(ID)를 새로 부여하면 됩니다.

### 📝 코드 2: 잘 들어갔는지 뒤에서 5개만 확인하기
```python
# 리스트 슬라이싱 [-5:]를 사용하여 뒤에서 5개 항목만 가져와 출력합니다.
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)
```
* **출력 결과**:
  ```python
  ('younger', 1127)
  ('your', 1128)
  ('yourself', 1129)
  ('<|endoftext|>', 1130)  # 1130번으로 성공적으로 등록됨
  ('<|unk|>', 1131)        # 1131번으로 성공적으로 등록됨
  ```

### 📝 Listing 2.4: SimpleTokenizerV2 완벽 해부
이제 모르는 단어를 만나면 `<|unk|>`로 바꿔치기하는 방어 로직을 추가한 V2 버전을 만듭니다.

```python
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        
        # ★ V2의 핵심 로직: 삼항 연산자(if-else)를 포함한 리스트 컴프리헨션 ★
        preprocessed = [item if item in self.str_to_int            
                        else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
```
* **V2의 핵심 코드 완벽 해설 (`encode` 메서드 내부)**:
  - `item if item in self.str_to_int else "<|unk|>"` : 이 구문이 바로 `KeyError` 에러를 막아주는 생명줄입니다.
  - "방금 쪼갠 텍스트 조각(`item`)이 만약 우리가 가진 사전(`self.str_to_int`) 안에 존재한다면 원래 단어 그대로 놔두고, **만약 사전에 없는 단어라면 무조건 `<|unk|>` 토큰으로 강제 둔갑시켜라!**" 라는 뜻입니다.

---

## 4. 🚀 실전 테스트: 낯선 단어와 문서 구분자 처리

### 📝 코드 3: 독립적인 두 문장 이어 붙이기
```python
text1 = "Hello, do you like tea?" # 소설에 없는 Hello 등장
text2 = "In the sunlit terraces of the palace." # 소설에 없는 palace 등장

# 두 문장 사이에 <|endoftext|> 토큰을 끼워 넣어서 하나로 합칩니다.
text = " <|endoftext|> ".join((text1, text2))
print(text)
# 출력: Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace.
```

### 📝 코드 4: 인코딩 및 디코딩(Sanity Check) 테스트
```python
tokenizer = SimpleTokenizerV2(vocab)
ids = tokenizer.encode(text)
print(ids)
# 출력: [1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7]
# 1131번(<|unk|>)이 두 번 등장했고, 1130번(<|endoftext|>)이 한 번 등장했습니다!

print(tokenizer.decode(ids))
# 출력: <|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>.
```
* **결과 분석**: V1 에서는 에러를 뿜으며 프로그램이 멈췄지만, V2 에서는 사전에 없던 `"Hello"`와 `"palace"`가 얌전하게 `<|unk|>` 토큰으로 변환되어 안전하게 통과했습니다. 또한 두 문장 사이에 `<|endoftext|>` 토큰도 무사히 자리 잡았습니다.

---

## 5. 💡 [2027년 최신 실무 관점] 기타 특수 토큰과 GPT의 '충격적인 진실'

자연어 처리 학계에는 `<|unk|>`나 `<|endoftext|>` 외에도 유명한 특수 토큰들이 있습니다. (특히 구글의 BERT 모델 등에서 자주 쓰입니다.)
* `[BOS]` (Beginning Of Sequence): 문장의 시작을 알림.
* `[EOS]` (End Of Sequence): 문장의 끝을 알림. (GPT의 `<|endoftext|>`와 같은 역할)
* `[PAD]` (Padding): 딥러닝 훈련 시 GPU 연산 효율을 높이기 위해 문장 길이를 통일해야 하는데, 짧은 문장의 남는 빈칸을 채워 넣는 깍두기(스펀지) 역할.

**🚨 그런데 충격적인 진실 하나!**
사실 최신 GPT 모델, Llama 3, 4 모델 등 현대의 모든 최고 성능 LLM들은 **`<|unk|>` 토큰을 단 한 번도 사용하지 않습니다.**

우리가 방금 `SimpleTokenizerV2`에서 구현한 `<|unk|>` 변환 방식은 실무에서는 **최악의 방식**입니다. 왜냐하면 `"Hello"`나 `"palace"` 같은 핵심 명사들이 그저 `<|unk|>`라는 멍청한 껍데기로 치환되면서, **단어가 원래 가지고 있던 소중한 의미와 철자 정보가 100% 날아가 버리기 때문입니다.** 모델 입장에서는 갑자기 눈과 귀가 가려지는 것과 같습니다.

그렇다면 현대의 최신 모델들은 모르는 단어를 어떻게 처리할까요? 에러도 안 내고, `<|unk|>`로 뭉뚱그리지도 않는 마법 같은 기술이 바로 **BPE (Byte Pair Encoding, 바이트 쌍 인코딩)** 기반의 서브워드(Subword) 토큰화입니다. 모르는 단어가 나오면 글자나 바이트 단위로 더 잘게 찢어서라도 기어코 사전에 있는 조각들로 맞춰냅니다.

이 위대한 BPE 기술이 바로 다음 2.5절에서 펼쳐질 내용입니다!

---

# 📝 이해도 점검 (Verification Q&A)

### ❓ 질문 1
LLM을 훈련시킬 때 서로 전혀 다른 출처의 책이나 기사 수천 개를 한 줄로 이어 붙여서 학습시킵니다. 이때 서로 다른 문서 사이에 `<|endoftext|>` 와 같은 특수 토큰을 삽입하지 않으면 모델의 학습 과정에 어떤 악영향이 발생하나요?

### ❓ 질문 2
파이썬 코드 `preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]` 가 이전 버전(V1)에서 발생했던 `KeyError`를 어떻게 막아주는지 논리적인 흐름을 설명해 보세요.

### ❓ 질문 3
과거에는 OOV(미등록 단어) 문제가 발생하면 이처럼 `<|unk|>` 토큰으로 치환했지만, 2027년 최신 거대 언어 모델(GPT, Llama 등)의 실무에서는 이 방식을 **절대 사용하지 않습니다.** 그 이유는 무엇이며, 대신 어떤 방향성(개념)을 가진 기술을 사용하나요?

---
---

### 💡 모범 답안 (정확한 이해를 위한 체크)

**답안 1**
서로 다른 출처의 문서 사이에 구분선이 없으면 모델은 두 문서가 문맥적으로 이어지는 하나의 이야기라고 심각하게 착각하게 됩니다. 예를 들어 경제 기사 바로 뒤에 마법 소설이 이어지면 "주식 가격이 오르자 주인공이 마법을 썼다"는 식으로 문맥 간섭이 일어납니다. 문서와 문서 사이에 특수 토큰을 넣어주어야 모델이 "여기서부터는 새로운 맥락이구나" 하고 기억을 초기화하여 각 문서의 올바른 문맥 구조를 정확히 학습할 수 있습니다.

**답안 2**
이 코드는 파이썬의 리스트 컴프리헨션과 조건문(if-else)을 결합한 것입니다. 텍스트를 쪼갠 조각(`item`)이 모델의 단어 사전(`self.str_to_int`)의 열쇠(Key)로 등록되어 있는지 먼저 검사(`if item in ...`)합니다. 등록되어 있다면 본래 단어를 그대로 유지하고, 만약 사전에 없는 낯선 단어라면 프로그램이 에러를 내기 전에 즉시 `" <|unk|> "` 라는 문자열로 덮어쓰기(`else "<|unk|>"`) 해버립니다. 결과적으로 사전에서 항상 찾을 수 있는 단어들만 남게 되므로 `KeyError`가 원천 차단됩니다.

**답안 3**
`<|unk|>` 토큰으로 치환해 버리면, 비록 에러는 막을 수 있지만 새로 입력된 단어(예: 신조어나 고유명사)가 원래 가지고 있던 구체적인 의미, 철자, 형태소 정보가 **완벽하게 소실(삭제)**되기 때문입니다. 현대 모델들은 이런 정보의 손실을 막기 위해 단어를 통째로 등록하지 않고 더 작은 의미 조각이나 철자, 바이트(Byte) 단위로 잘게 쪼개어 인식하는 **BPE (Byte Pair Encoding, 서브워드 토큰화)** 기술을 사용합니다. 이를 통해 세상의 어떤 낯선 단어가 들어와도 이미 아는 조각들로 분해하여 의미 손실 없이 완벽하게 처리합니다.
