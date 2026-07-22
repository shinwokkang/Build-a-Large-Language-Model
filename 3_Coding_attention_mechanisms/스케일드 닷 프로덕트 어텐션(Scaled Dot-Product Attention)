    ## 1. 3.4 Implementing self-attention with trainable weights 완벽 해부
    
    ### (1) 3.4절 도입부: 왜 학습 가능한 가중치가 필요한가?
    
    - **교재의 문장 연결:** 저자는 3.4절 문단에서 이전 3.3절의 '단순 버전'과 '상용 버전'의 차이를 이렇게 연결합니다.
        
        > *"Our next step will be to implement the self-attention mechanism used in the original transformer architecture, the GPT models, and most other popular LLMs. This self-attention mechanism is also called scaled dot-product attention... The most notable difference is the introduction of weight matrices that are updated during model training."*
        > 
        > 
        > (우리의 다음 단계는 원래의 트랜스포머 아키텍처, GPT 모델, 그리고 대부분의 다른 대중적인 LLM에서 사용되는 셀프 어텐션 메커니즘을 구현하는 것입니다. 이 셀프 어텐션 메커니즘은 '스케일드 닷 프로덕트 어텐션'이라고도 불립니다... 가장 주목할 만한 차이점은 모델 학습 과정 동안 업데이트되는 '가중치 행렬(Weight Matrices)'의 도입입니다.)
        > 
    - **개념 설명:**
        
        3.3절에서 배운 단순 셀프 어텐션은 문장 속 단어 벡터들을 아무런 변형 없이 그대로 내적(곱하기)했습니다. 이는 고정된 사전식 매칭과 다를 바가 없어서, 인공지능이 데이터를 아무리 많이 읽어도 문맥을 파악하는 실력이 늘지 않습니다.
        
        인공지능이 "이 문장에서는 이 단어가 저 단어보다 중요하구나"라는 것을 데이터 학습을 통해 스스로 깨닫게 만들려면, 단어 벡터를 입력받았을 때 그 벡터의 성질을 유연하게 바꾸어 줄 수 있는 학습 가능한 필터(가중치 행렬, Weight Matrices)가 필요합니다. 이 가중치 행렬들을 도입한 실제 아키텍처의 이름이 바로 스케일드 닷 프로덕트 어텐션(Scaled Dot-Product Attention)이며, 이것이 모든 현대 트랜스포머 LLM의 코어 엔진입니다.
        
    
    - **[교재 그림 해설] Figure 3.13 분석:**

        
    
    - **책의 시각 자료 설명:** Figure 3.13은 3장에서 구현하는 어텐션 발달 단계를 로드맵으로 보여줍니다.
    - `1) Simplified self-attention` 단계를 지나 점선 박스로 하이라이트된 `2) Self-attention (with trainable weights)` 단계를 가리키고 있습니다. 즉, 단순 연산 메커니즘 위에 학습 알고리즘이 개입할 수 있는 가중치 레이어를 얹는 구조적 전환점임을 시각적으로 묘사하고 있습니다.
    
    ---
    
    ### (2) 3.4.1 쿼리(Query), 키(Key), 값(Value) 벡터의 연산 원리
    
    - **교재의 문장 연결:** 저자는 데이터 학습을 위해 단어가 3가지의 새로운 신분으로 변신해야 함을 문장으로 기술합니다.
        
        > *"We will implement the self-attention mechanism step by step by introducing the three trainable weight matrices $W_q$, $W_k$, and $W_v$. These three matrices are used to project the embedded input tokens, $x^{(i)}$, into query, key, and value vectors, respectively..."*
        > 
        > 
        > (우리는 세 개의 학습 가능한 가중치 행렬 $W_q, W_k, W_v$를 도입하여 셀프 어텐션 메커니즘을 단계별로 구현할 것입니다. 이 세 행렬은 임베드된 입력 토큰 $x^{(i)}$를 각각 쿼리(Query), 키(Key), 값(Value) 벡터로 투영(Project)하는 데 사용됩니다...)
        > 
    
    - **[교재 그림 해설] Figure 3.14 분석:**
        
        !image.png
        
    
    - **책의 시각 자료 설명:** **Figure 3.14**는 하나의 단어 벡터가 3개의 경로로 갈라지는 모습을 보여줍니다. 기준 단어인 두 번째 토큰 $x^{(2)}$("journey")가 각각 $W_q$, $W_k$, $W_v$라는 세 가지 다른 모양의 행렬과 곱해져서, 결과물로 $q^{(2)}$(Query), $k^{(2)}$(Key), $v^{(2)}$(Value)라는 전혀 다른 3개의 벡터로 분화되는 매커니즘을 시각화했습니다.
    
    - **개념 설명 (왜 3개나 필요한가?):**
        
        데이터베이스나 정보 검색 시스템의 개념에서 이름을 빌려온 것입니다. 
        
        1. **쿼리 (Query, $q$):** "질문자"입니다. 현재 문장에서 내가 집중해서 쳐다보고 있는 기준 단어입니다. (예: "내가 지금 'journey'라는 단어인데...")
        2. **키 (Key, $k$):** "명함/주소록"입니다. 문장 속의 모든 단어가 나를 찾아오라고 내미는 고유한 인덱스 속성입니다. 쿼리는 이 키들과 내적을 해서 유사도를 구합니다. (예: "...너네 다른 단어들 명함 좀 나랑 맞춰보자.")
        3. **값 (Value, $v$):** "실제 알맹이 정보"입니다. 쿼리와 키가 서로 궁합이 잘 맞는다고 판정되면, 최종적으로 그 단어가 가진 진짜 의미(값)를 가중치만큼 가져와서 섞게 됩니다.
    
    ---
    
    ### (3) 가상 데이터 정의 및 파이토치 초기화 코드 분석
    
    교재에서는 2차원에서 배운 토큰 임베딩의 연장선으로, 6개의 단어("Your journey starts with one step")가 각각 3차원(`d_in=3`) 벡터로 이루어진 가상의 텐서 `inputs`를 설정하고 코드를 작성합니다.
    
    #### 코드 흐름 분석 (입력 및 가중치 행렬 초기화):
    
    ```python
    x_2 = inputs[1]       # 문장의 두 번째 단어인 "journey" 벡터 추출
    d_in = inputs.shape[1] # 입력 임베딩 차원 크기 = 3
    d_out = 2             # 출력 임베딩 차원 크기 = 2 (이해를 위해 입력과 다르게 설정)
    
    torch.manual_seed(123)
    # 학습 가능한 파라미터로 세 가지 가중치 행렬 정의
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    ```
    
    - **코드 작동 상세 설명:**
        
        `inputs[1]`을 통해 쿼리로 삼을 기준 단어 벡터 `x_2`를 가져옵니다. 저자는 실무적인 LLM(GPT 등)은 보통 입력 차원(`d_in`)과 출력 차원(`d_out`)이 똑같지만, 컴퓨터가 행렬 곱셈을 수행할 때 차원이 어떻게 바뀌는지 시각적으로 쉽게 추적할 수 있도록 일부러 `d_in=3`, `d_out=2`로 다르게 설정했다고 친절히 덧붙입니다.
        
        `torch.nn.Parameter`를 통해 가중치 행렬들을 감싸줌으로써, 이 행렬들이 인공지능이 학습해야 할 대상임을 파이토치 엔진에 선언합니다. (단, 출력 청결성을 위해 코드 수준에서는 `requires_grad=False`로 임시 동결했습니다.)
        
    
    #### 코드 흐름 분석 (Q, K, V 벡터 생성):
    
    ```python
    query_2 = x_2 @ W_query
    keys = inputs @ W_key
    values = inputs @ W_value
    ```
    
    - **코드 작동 상세 설명:**
        
        기준 단어 `x_2`에 가중치 행렬 `W_query`를 행렬곱(`@`)하여 차원이 3에서 2로 변환된 쿼리 벡터 `query_2`를 얻습니다.
        
        그리고 나 자신이 아닌 문장 전체 단어들의 명함(`keys`)과 알맹이 정보(`values`)를 한꺼번에 구하기 위해, 전체 행렬 `inputs`에 각각 `W_key`와 `W_value`를 행렬곱합니다. 그 결과 `keys`와 `values`는 6개의 단어가 각각 2차원 벡터를 가진 `[6, 2]` 모양의 텐서가 됩니다.
        
    
    ---
    
    ### (4) 어텐션 점수(Attention Scores) 및 스케일링(Scaling) 메커니즘
    
    - **교재의 문장 연결:** 저자는 쿼리와 키의 결합, 그리고 '스케일드(Scaled)'라는 수식어가 붙은 핵심 이유를 기술합니다.
        
        > *"The second step is to compute the attention scores... We compute the attention weights by scaling the attention scores and using the softmax function. However, now we scale the attention scores by dividing them by the square root of the embedding dimension of the keys..."*
        > 
        > 
        > (두 번째 단계는 어텐션 점수를 계산하는 것입니다... 우리는 어텐션 점수를 스케일링하고 소프트맥스 함수를 사용하여 어텐션 가중치를 계산합니다. 그러나 이제 우리는 어텐션 점수를 키의 임베딩 차원의 제곱근으로 나누어 스케일링합니다...)
        > 
    
    - **[교재 그림 해설] Figure 3.15 & 3.16 분석:**
        
        !image.png
        
    
    !image.png
    
    - **Figure 3.15:** 쿼리 벡터 $q^{(2)}$가 전체 단어들의 명함 벡터인 $k^{(1)}, k^{(2)}, \dots, k^{(6)}$ 전치 행렬과 행렬곱(`@`)되어 날것의 어텐션 점수 행렬 $\omega$를 낳는 과정을 보여줍니다.
    - **Figure 3.16:** 이 점수 벡터 $\omega$를 단순히 소프트맥스에 넣는 것이 아니라, 키 차원의 제곱근인 $\sqrt{d_k}$로 나누어 크기를 깎아준 뒤 소프트맥스(Softmax)를 먹여 최종 확률 분포 가중치 $\alpha$를 도출하는 단계를 묘사합니다.
    
    #### 코드 흐름 분석 (점수 계산 및 스케일링):
    
    ```python
    attn_scores_2 = query_2 @ keys.T
    d_k = keys.shape[-1]  # 키의 차원 크기 = 2
    attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
    ```
    
    - **코드 해설 (왜 제곱근으로 나누는가?):**
        
        저자는 'THE RATIONALE BEHIND SCALED-DOT PRODUCT ATTENTION' 박스를 통해 이 스케일링 공식을 매우 전문적으로 해설합니다.
        
        실제 상용 LLM은 단어 임베딩 차원이 1,000이 넘고, 심하면 수만 차원에 달합니다. 차원이 이렇게 거대해지면 두 벡터를 내적했을 때 결과 숫자(점수)가 수백, 수천 단위로 엄청나게 커질 수 있습니다.
        
        문제는 이 큰 숫자들을 그대로 소프트맥스 함수에 집어넣으면, 지수 함수 특성상 가장 큰 값 하나가 99.999%를 독점하고 나머지 단어들은 0%에 수렴하는 **'계단 함수(Step Function)'처럼 돌변**합니다. 이렇게 되면 미분을 했을 때 그래디언트(기울기)가 거의 0이 되어버리는 역전파 학습 정체 현상(Gradients Nearing Zero)이 일어나 인공지능 학습이 멈춰버립니다. 따라서 차원의 크기가 커지더라도 점수가 폭발하지 않도록 차원의 제곱근인 `d_k**0.5`($\sqrt{d_k}$)로 나누어 점수들을 예쁘게 스케일링(축소)해 주는 것입니다.
        
    
    ---
    
    ### (5) 최종 단계: 컨텍스트 벡터(Context Vector)의 가중합 연산
    
    - **교재의 문장 연결:** 스케일링된 확률을 이용해 알맹이 정보를 결합하는 마지막 연산입니다.
        
        > *"Now, the final step is to compute the context vectors... we now compute the context vector as a weighted sum over the value vectors. Here, the attention weights serve as a weighting factor that weighs the respective importance of each value vector."*
        > 
        > 
        > (이제 마지막 단계는 컨텍스트 벡터를 계산하는 것입니다... 우리는 이제 컨텍스트 벡터를 값(Value) 벡터들에 대한 가중합으로 계산합니다. 여기서 어텐션 가중치는 각 값 벡터의 상대적 중요도를 계량하는 가중 요인 역할을 합니다.)
        > 
    - **[교재 그림 해설] Figure 3.17 분석:**
        
        !image.png
        
    - **책의 시각 자료 설명:** **Figure 3.17**은 완성된 어텐션 가중치 확률 행렬 $\alpha_{2}$와 원본 정보의 알맹이 변신체인 값 벡터들 $v^{(1)} \sim v^{(6)}$를 행렬곱하여 최종적으로 문맥 정보가 완벽하게 주입된 출력 컨텍스트 벡터 $z^{(2)}$가 탄생하는 종착지를 보여줍니다.
    
    #### 코드 흐름 분석 (컨텍스트 벡터 생성):
    
    ```
    context_vec_2 = attn_weights_2 @ values
    ```
    
    - **코드 작동 상세 설명:**
        
        방금 소프트맥스로 정돈한 확률적 가중치 `attn_weights_2`와 모든 단어의 알맹이 데이터인 `values` 행렬을 곱합니다. 이 과정을 통해 쿼리 단어 "journey"는 문장 내 모든 단어들의 가치 정보 중 자신과 연관된 만큼의 양을 골고루 흡수하여, 문맥적으로 극도로 풍부해진 최종 2차원 벡터 `context_vec_2`로 새로 태어나게 됩니다.
        
    
    ---
    
    ### (6) 3.4.2 파이토치 클래스 모듈화 (`SelfAttention_v1` & `v2`)
    
    저자는 임의로 한 단어씩 루프를 도는 단계를 지나, 실전 LLM 아키텍처에 모듈로 직접 끼워 넣을 수 있는 정돈된 두 가지 파이토치 클래스 코드를 제시합니다.
    
    #### [Listing 3.1] 기본 텐서 파라미터 기반 `SelfAttention_v1` 흐름 분석:
    
    ```python
    class SelfAttention_v1(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()
            # 3개의 가중치 행렬을 파이토치 파라미터 텐서로 선언
            self.W_query = nn.Parameter(torch.rand(d_in, d_out))
            self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
            self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    
        def forward(self, x):
            keys = x @ self.W_key
            queries = x @ self.W_query
            values = x @ self.W_value
    
            # 문장 전체 단어 쌍의 연관 점수를 일시에 행렬곱 연산
            attn_scores = queries @ keys.T
            attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
            context_vec = attn_weights @ values
            return context_vec
    ```
    
    - **코드 흐름 및 아키텍처 매핑 (Figure 3.18 연결):**
        
        !image.png
        
    
    이 클래스의 `forward` 함수는 우리가 앞에서 한 단어(`x_2`) 단위로 쪼개어 수동으로 코딩했던 모든 스텝을 **행렬 대수 연산 하나로 완전히 통합**했습니다.
    
    **Figure 3.18**은 이 클래스의 대규모 병렬 연산 메커니즘을 시각화한 그림으로, 입력 행렬 $X$가 세 갈래의 학습용 가중치 행렬 그리드를 통과해 거대한 $Q, K, V$ 행렬이 되고, $Q$와 $K^T$의 거대한 행렬곱을 통해 전체 단어의 쌍을 그리드로 기록하는 '어텐션 가중치 매트릭스'를 일시에 형성한 뒤, $V$와 결합하여 출력 행렬 $Z$를 쏟아내는 트랜스포머의 전체 매트릭스 파이프라인을 보여줍니다.
    
    #### [Listing 3.2] 상용 표준 리니어 레이어 기반 `SelfAttention_v2` 흐름 분석:
    
    ```python
    class SelfAttention_v2(nn.Module):
        def __init__(self, d_in, d_out, qkv_bias=False):
            super().__init__()
            # 파이토치 빌트인 nn.Linear 레이어로 교체하여 대치
            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
        def forward(self, x):
            keys = self.W_key(x)
            queries = self.W_query(x)
            values = self.W_value(x)
    
            attn_scores = queries @ keys.T
            attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
            context_vec = attn_weights @ values
            return context_vec
    ```
    
    - **코드 작동 상세 설명 및 실무적 이유:**
        
        저자는 `v1`에서 텐서를 직접 생성하던 방식 대신 파이토치 내장 레이어인 `nn.Linear`를 사용하는 `v2` 코드를 제시하며, 이것이 상용 LLM 코딩의 글로벌 표준이라고 강조합니다.
        
        그 이유는 `nn.Linear` 레이어는 바이어스(Bias) 유닛을 끄고(`bias=False`) 사용하면 수학적으로 단순 행렬곱(`@`)과 완벽히 동일하게 작동하면서도, 내부적으로 Xavier 초기화 또는 Kaiming 초기화 같은 고도로 최적화된 가중치 초기화 스키마(Optimized Weight Initialization Scheme)를 자동으로 적용해 주기 때문에 딥러닝 모델의 초기 학습 안정성이 수십 배 향상되기 때문입니다.
        
    
    ---
    
    ## 3. 2026~2027년 최신 트렌드 반영: 실무에서의 맥락 확장과 아키텍처적 응용
    
    교재 3.4절은 트랜스포머의 심장인 '가중치 기반 셀프 어텐션'의 정석을 다루고 있습니다. 2026~2027년 현재 빅데이터 및 대규모 언어 모델 서빙/학습 실무 환경에서는 이 원초적인 계산식이 하드웨어 가속기 및 인프라 엔지니어링과 결합하여 극도로 변형 및 고도화되어 쓰이고 있습니다.
    
    ### (1) 실무적 사용: $Q, K, V$ 통합 투영(Fused QKV Projection) 커널 최적화
    
    교재 3.4절의 코드(Listing 3.2)에서는 `self.W_query`, `self.W_key`, `self.W_value`라는 3개의 별도 리니어 레이어를 선언하여 단어 행렬을 3번 따로 곱했습니다.
    
    - **2027년 실무 최적화 트렌드:** 실제 엔터프라이즈급 LLM(예: Llama 3 계열 수정본, Mistral 최신 아키텍처)의 소스코드를 보면 성능 극대화를 위해 레이어를 3개로 나누지 않습니다.
        
        출력 차원을 3배 키운 단 하나의 거대한 융합 레이어 `self.W_qkv = nn.Linear(d_in, d_out * 3)`를 선언하여 GPU 메모리 인터페이스 단에서 단 한 번의 행렬곱으로 $Q, K, V$ 행렬을 동시에 쪼개어 짜냅니다. 이를 **Fused QKV Projection** 기법이라고 부르며, GPU의 메모리 대역폭 정체를 줄여 추론 및 학습 속도를 약 15%~20% 향상시키는 필수 실무 인프라 기술입니다.
        
    
    ### (2) 최신 아키텍처 트렌드: RoPE(Rotary Position Embedding)와의 융합
    
    3.4절에서 계산된 `queries`와 `keys` 벡터는 서로 내적(`@`)되기 직전에 최신 LLM 아키텍처 실무에서 매우 중요한 변형을 거칩니다.
    
    - **실무 엔지니어링 기술:** 트랜스포머는 단어의 순서를 모르는 단점이 있어서 위치 정보를 주입해야 합니다. 2026~2027년 현재 실무 표준으로 자리 잡은 **로터리 위치 임베딩(RoPE)** 기술은 3.4절의 `forward` 함수 내에서 `queries`와 `keys` 행렬을 구한 직후, 내적을 수행하기 바로 전 단계에서 이 벡터들을 고차원 복소수 공간에서 회전(Rotation)시키는 커스텀 커널을 실행합니다. 이를 통해 수백만 토큰의 초장문 컨텍스트 속에서도 단어 간의 상대적 거리 감각을 오차 없이 완벽하게 유지합니다.
    
    ### (3) 실무 응용 예시: 금융 이상 거래 탐지(Fraud Detection)를 위한 실시간 가중치 스코어링 시스템
    
    3.4절의 학습 가능한 셀프 어텐션 매커니즘은 자연어 처리를 넘어 대규모 빅데이터 로그 분석 인프라에서 핵심 보안 탐지 엔진으로 널리 사용됩니다.
    
    ```mermaid
    graph TD
        A["신용카드 실시간 결제 이력 시퀀스 데이터"] --> B["시간/금액/가맹점 임베딩 행렬 X"]
        B --> C{"학습 가능한 QKV 가중치 행렬 투영"}
        C -->|질문자 벡터| D["Query 행렬"]
        C -->|주소록 벡터| E["Key 행렬"]
        C -->|알맹이 벡터| F["Value 행렬"]
        
        %% @ 기호와 .T 연산을 화살표 라벨 안으로 이동하고 큰따옴표로 안전하게 처리했습니다.
        D & E -->|"Scaled Dot-Product (Q @ K.T)"| G["어텐션 점수 매트릭스"]
        G -->|"Softmax 가중합 정규화"| H["어텐션 가중치"]
        H & F -->|"가중합 연산 (Softmax @ V)"| I["컨텍스트 임베딩 결과물 Z"]
        I --> J["초고속 AI 판별기: 이상 패턴 실시간 차단 에이전트"]
    ```
    
    - **구체적 실무 예시:**
        
        글로벌 핀테크 기업의 실시간 이상 금융 거래 탐지(FDS) 인프라를 설계한다고 가정해 봅시다. 한 사용자가 최근 1시간 동안 결제한 수십 건의 카드 결제 이력(시퀀스 데이터)이 입력됩니다.
        
        엔지니어들은 3.4절의 코드를 확장하여 결제 시퀀스를 입력 행렬 $X$로 변환합니다. 인공지능은 학습 가능한 가중치 행렬 `W_query`와 `W_key`를 통해 "방금 발생한 해외 결제 건(Query)"이 "몇 분 전 국내 가맹점 결제 건들(Key)"과 물리적으로 양립할 수 없는 속성을 지녔는지를 내적 연산 그리드로 실시간 탐색합니다.
        
        스케일링 및 소프트맥스를 거쳐 도출된 문맥적 변동 위험도 벡터(`context_vec`)를 금융 AI 판별 에이전트에 주입함으로써, 0.01초 만에 이상 금융 사기 거래를 정밀 포착하여 자동 승인 거절을 내리는 핵심 빅데이터 보안 인프라의 핵심 엔진으로 강력하게 구동되고 있습니다.
        
    
    ---
    
    ## 4. 요약 및 차기 학습 가이드
    
    3.4절의 핵심 엔지니어링 개념을 명확히 요약해 드리겠습니다.
    
    - 인공지능이 데이터 문맥을 학습하게 만들기 위해 학습 가능한 3가지 매개변수 행렬 $W_q, W_k, W_v$를 도입합니다. (**Figure 3.14**)
    - 입력 단어는 질문자(Query), 명함(Key), 알맹이(Value)로 분화되어 상호 매칭 연산에 돌입합니다. (**Figure 3.15**)
    - 고차원 환경에서 그래디언트 소실로 인한 학습 정체 현상을 막기 위해 키 차원의 제곱근($\sqrt{d_k}$)으로 점수를 깎아주는 **스케일드 닷 프로덕트 어텐션**을 완성합니다. (**Figure 3.16**)
    - 실무에서는 초기화 스키마 최적화가 보장된 파이토치의 `nn.Linear` 레이어를 사용하여 모듈을 컴팩트하게 구현합니다. (**Listing 3.2**)
    
    이 가중치 기반 셀프 어텐션의 행렬곱 파이프라인과 파이토치 모듈 구조를 머릿속에 완벽히 정돈하신 상태에서 다음 단계인 3.5절 "Hiding future words with causal attention"로 전진하시면, 문장을 생성할 때 미래 단어를 물리적으로 지워버리는 상용 GPT LLM의 마스킹(Masking) 트릭을 막힘없이 완벽하게 이해하실 수 있습니다.
