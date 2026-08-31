# Critical Period as Structural-Plasticity Termination — Implementation Spec

**목적**: 생물학적 critical period(CP)를 ANN에 구현한다. 핵심 아이디어는 "새로운 가소성 규칙을 발명"하는 게 아니라, **기존 dynamic sparse training(구조 가소성)에 발달적·비가역적 종료(성숙-latch)를 추가**하는 것이다.

이 문서는 구현자(Claude Code)를 위한 사양서다. 코드 컨벤션이 아니라 *설계 의도와 반드시 지켜야 할 제약*에 초점을 둔다. 애매하면 "정직한 경계" 섹션과 "실패 판정" 섹션을 먼저 볼 것.

> **REV 2 (실험 반영) — 바뀐 것 요약**
> 1. **성숙 신호 교체**: raw churn(§3.4 negative result: 성숙해도 안 떨어지고 ~0.065 평형)와 loss-slope(§3.5: 작동하나 생물학적으로 부적합 — 순환·synaptic 오염) 모두 폐기. → **silent-synapse 소진율**(§3.1, Huang 2015 직결)로 확정.
> 2. **엔진**: SET → **RigL**(§3.6) — 전환율 신호 품질 + activity-dependent 주장.
> 3. **프레이밍 재정렬**: 이 메커니즘은 memory-protection이 아니라 **representation-locking**(§5.2). 척추를 H3(forgetting)에서 **H4/H6(robustness/human-likeness)**로 이동. forgetting을 못 막는 건 설계상 정상이며 실패 판정 아님.
> 4. §5.1 위상 발산 sanity는 **통과함**(§5.2).

---

## 0. 한 줄 요약

표준 딥러닝은 가소성 축이 하나다 — weight 크기(synaptic plasticity). 생물학의 CP는 두 번째 축(structural plasticity: 연결의 생성·소멸)을 발달 특정 시점에 **비가역적으로 닫고**, synaptic 축은 열어둔다. DST(SET/RigL)는 두 번째 축을 켰지만 **끄지 않았다**(CP가 없다). 이 프로젝트는 그 축을 성숙 시점에 latch해 CP를 만들고, 그것이 (1) stability-plasticity와 (2) human-like 표상을 동시에 만드는지 본다.

---

## 1. 개념 모델 (구현 전 반드시 이해할 것)

두 개의 독립적 가소성 축:

- **Synaptic plasticity** = 이미 존재하는 연결의 weight를 gradient로 바꿈. → 표준 backprop. **항상 켜져 있음.**
- **Structural plasticity** = 어떤 연결이 존재하는가(위상/topology)를 바꿈. 연결 prune + 새 연결 regrow. → SET/RigL이 하는 것. **CP가 닫히면 꺼짐.**

CP의 생애주기:

1. **Juvenile (CP open)**: 두 축 다 활성. 위상이 활발히 rewire됨 (spine turnover).
2. **성숙 (closing)**: 표상이 수렴하며 위상이 안정됨. 성숙 지표가 임계에 도달.
3. **Closure (CP closed, 비가역)**: 구조 축 latch. 더 이상 prune/regrow 없음. **살아남은 연결의 weight는 계속 학습됨.**

> 생물학 대응: 여는 신호(GABA 회로 성숙)와 닫는 신호(PNN/Lynx1 구조적 브레이크)는 **다른 것**이다. 구현에서 "여는 것"은 그냥 juvenile 단계 시작(별도 모델링 불필요)이고, 모델링해야 할 것은 **닫는 브레이크**뿐이다.

> 백내장 대응: closure 이후엔 입력(눈)이 정상이어도 위상을 못 바꾸므로 회복이 안 된다 — synaptic만으로는 못 메꾸는 구조적 상한.

---

## 2. 핵심 메커니즘 (최소 로직)

SET(또는 RigL)를 **juvenile 엔진**으로 그대로 쓰고, 레이어별 상태 하나와 latch 조건만 추가한다.

레이어 `l`마다: `frozen[l]: bool = False`

매 rewiring 주기(SET가 원래 prune-regrow 하는 시점)에 레이어별로:

```
for l in layers:
    if not frozen[l]:
        prune_and_regrow(l)              # SET/RigL 그대로 (juvenile 구조 가소성)
        if maturity(l) >= threshold:     # 성숙 브레이크
            frozen[l] = True             # 위상 영구 latch (비가역)
    # frozen이든 아니든: weight는 backward에서 항상 정상 업데이트
```

이게 전부다. `frozen[l]=True`가 되면:
- 그 레이어는 prune도 regrow도 **하지 않는다** (위상 고정).
- backward는 정상 — **남은 연결의 weight는 계속 gradient로 업데이트된다.**
- **다시 False로 돌아가지 않는다** (irreversible; 이게 "진짜 자물쇠").

파라미터는 이상적으로 **`threshold` 하나**.

---

## 3. `maturity(l)` — silent-synapse 소진 신호 (개정판)

> **개정 이력 (실험으로 확정)**: 초기 스펙은 raw churn(mask turnover)이 성숙하면 감소할 거라 가정했으나, **이 가정은 틀렸다**(§3.4 negative result 참고). 대안으로 시도한 loss-slope 트리거도 작동은 하나 생물학적으로 부적합하다(§3.5). 확정 신호는 **silent-synapse 소진율**이다(§3.1). 이 신호는 Huang et al. 2015에 직결되고, endogenous·structural하며 계산이 싸다.

### 3.1 확정 신호: silent-synapse → load-bearing 전환율

**생물학 근거 (Huang, Schlüter et al., *PNAS* 112, 2015)**: 눈뜰 때 ~50%였던 silent synapse(AMPA 수용체가 아직 없는 미성숙 연결) 비율이 성체가 되며 ~5%로 소진되는 것이 CP 종료 시점을 결정한다. PSD-95 제거로 이 성숙을 멈추면 juvenile 가소성이 평생 지속된다. 즉 **"미성숙 연결 풀의 소진"이 곧 closure 신호**다.

**ANN 대응**:
- **silent synapse = grown-but-not-yet-load-bearing 연결**: regrow됐지만 아직 magnitude가 작아 core에 편입되지 않은 연결.
- **성숙 신호 = 전환율(conversion rate)**: 최근 주기에 grow된 연결 중, 살아남아 magnitude가 "load-bearing core" 수준(예: 레이어 magnitude 분포 상위 분위, 또는 prune threshold의 k배)에 **새로 도달한** 연결의 비율.
- **closure = 전환율이 0**: 새로 발굴할 productive 연결이 고갈됨 = silent pool 소진. 이때 latch.

측정은 mask + magnitude 장부만 보면 되므로 loss만큼 싸다. 구현: 주기마다 "이번에 grow된 edge들의 이후 magnitude 궤적"을 추적해, load-bearing 임계를 처음 넘는 비율을 전환율로 집계.

### 3.2 왜 이 신호는 실제로 0으로 가는가 (raw churn과의 결정적 차이)

gross churn = **productive turnover(쓸모 있는 연결 발굴) + unproductive flickering(prune threshold 근처 marginal edge가 계속 들락날락)**. magnitude prune-regrow는 후자가 **non-zero 평형에 영구히 남는다**(실측 ~0.065, §3.4). raw churn이 실패한 건 이 두 성분이 섞였기 때문이다.

silent-synapse 전환율은 **productive 성분만** 잰다. flickering edge는 grow돼도 곧 죽어 load-bearing에 도달하지 못하므로 전환율에 안 잡힌다. 따라서 gross churn이 평형에 붙어 있어도 **전환율은 0으로 떨어진다** — pool이 소진되면 새로 core에 편입되는 연결이 사라지기 때문. 이게 성숙의 진짜 readout이다.

### 3.3 latch 타이밍 — "바닥"이 아니라 "꺾임"

전환율이 완전히 0에 도달한 뒤 잠그면 문제가 생긴다: 그 시점에 얼리는 위상 ≈ 자연히 도달했을 위상이라 **weight 동역학이 안 갈린다**(§5). 따라서 전환율이 **peak 대비 유의하게 꺾이기 시작하는 시점**(pool이 아직 남아 탐색 여지가 있을 때)에 latch해 미래 자유도를 실제로 자른다. 구현: 전환율의 절대 임계가 아니라 감소율/변곡점, 또는 "peak 전환율 대비 X% 지점"으로 트리거.

### 3.4 ⚠️ Negative result (기록 — 스펙에서 폐기된 가정)

**raw churn(mask symmetric difference)은 성숙해도 감소하지 않는다.** 40 rewire cycle, Task A 94%+ saturate 후 30+ cycle을 봐도 churn은 단조 상승 후 **non-zero plateau(~0.065–0.07)에 영구히 머문다** — peak도 kink도 없음. 이유: magnitude prune-regrow는 threshold 근처 marginal edge를 영구히 flicker시킨다(§3.2). 이것이 실제 SET/RigL이 외부 decay 스케줄을 강제하는 이유이기도 하다. **결론: §3.1의 productive 성분(전환율)만 신호로 쓸 것. gross churn은 성숙 신호로 부적합.**

### 3.5 ⚠️ loss-slope 트리거를 쓰지 않는 이유 (검토했으나 폐기)

raw churn 실패 후 loss-slope로 prune 강도를 modulate하면 churn이 인위적으로 kink하게 만들 수 있고, 실험상 §5.1/§6 시그니처도 깨끗하게 나온다. **그러나 생물학 방향에서 부적합하며 채택하지 않는다**:
- **순환**: loss-slope로 만든 스케줄의 결과(churn 하락)를 다시 latch 트리거로 감지 = 자기가 만든 스케줄을 자기가 감지. churn이 독립적 성숙 readout으로서의 의미를 잃는다.
- **축 오염**: loss는 synaptic(weight) 신호다. 이 프로젝트의 척추는 structural 축과 synaptic 축의 분리인데, loss로 closure를 트리거하면 두 축을 다시 묶는다 — "emergent CP는 weight 동역학의 readout일 뿐"이라던 비판에 스스로 들어간다.
- **closure 시점이 외부화**: 순수 emergent가 아니라 부분적으로 외부 타이밍이 된다.

silent-synapse 전환율은 이 세 문제를 전부 피한다(structural 신호, endogenous, 순환 없음). loss-slope는 "작동하는 우회로"였을 뿐 옳은 신호가 아니다.

### 3.6 엔진 권장: SET → RigL

silent-synapse 전환율은 **RigL(gradient-based regrow)에서 훨씬 깨끗**하다. SET의 random regrow는 grow된 연결이 productive인지가 노이즈투성이라 전환율 신호가 지저분하다. RigL은 high-gradient 위치에 grow하므로 초기 편입이 뚜렷하고 pool 소진도 선명하게 잡힌다. 덤으로 grow가 gradient(활동)에 의해 결정되므로 **activity-dependent라는 CP의 핵심 생물학적 성질**(Hensch 계열)을 비로소 주장할 수 있다 — random regrow로는 못 하는 주장. 단순함을 조금 내주고 신호 품질 + 생물학적 정당성을 크게 얻는 트레이드라 여기서는 RigL을 채택한다.

---

## 4. 위계(hierarchy)는 손으로 주지 말고 창발시킨다

깊이별로 threshold를 다르게 hard-code하지 **말 것**. 대신 **coarse-to-fine 입력 curriculum**과 결합한다:

- 얕은 레이어는 저해상도·저대비 입력에서 먼저 수렴 → churn 먼저 꺾임 → **먼저 latch**.
- 깊은 레이어는 고해상도 디테일을 기다리느라 늦게 수렴 → **늦게 latch**.
- 결과: 생물학의 위계적 CP 순서(V1 먼저, 고차 영역 나중)가 **설계 가정이 아니라 curriculum의 결과로 창발**한다. threshold는 전 레이어 공통 단일값 하나만 쓴다.

curriculum 파이프라인은 Lu et al. 2026(Developmental Visual Diet: acuity/contrast sensitivity/색각 발달 궤적)을 재사용 대상으로 우선 검토. 1차 프로토타입은 Achille식 단순 blur(저해상도 down/up-sample)로 시작해도 됨.

---

## 5. ⚠️ 가장 중요 — "weight 동역학이 정말 달라지는가"를 먼저 증명

이 설계의 최대 리스크: **단일 정적 task에서는 latch가 weight 동역학을 안 바꿀 수 있다.** latch가 "이미 일어난 수렴을 사후 승인"에 그치면 latch 있는 망과 없는(SET 계속) 망의 trajectory가 거의 같아진다.

weight 동역학이 **확정적으로** 갈리려면 두 조건이 필요하다:

- **조건 A — distribution shift가 있어야 한다 (continual learning).** task A에서 latch한 뒤 task B가 오면, latch 없는 망은 B용 새 연결을 recruit(위상 rewire)하지만 latch된 망은 못 하고 기존 위상 안에서 weight만 재배치한다. **여기서 처음으로 두 trajectory가 측정 가능하게 갈린다.** 정적 단일 task에선 안 갈린다 → 세팅이 continual learning인 것은 필수다.
- **조건 B — 수렴 *전에* latch (섹션 3.2).** 위상이 아직 유동적일 때 잠가야 잘라낸 자유도가 실재한다.

### 5.1 최소 sanity 실험 (다른 것보다 먼저 돌릴 것)

주장으로 두지 말고 측정량으로 증명한다. **latch 망 vs SET-계속 망**을 task A→B로 학습시키고:

1. **위상 발산 시점** (1차 sanity check): task B 투입 후 두 망의 연결 마스크가 갈라지는 지점. **안 갈리면 실패** → threshold를 더 이른 시점으로 당김(조건 B). 갈리면 통과.
2. **weight-change 부분공간의 rank/방향**: latch 망은 새 task 학습이 기존 위상이 span하는 부분공간에 갇히고, SET 망은 새 방향으로 샌다. 부분공간 겹침(subspace overlap) 또는 코사인 유사도로 정량화.
3. **최종 solution basin**: 같은 task B에서 두 망이 도달하는 해가 다른 basin인지.

**1번이 안 나오면 2·3번도 없다.** 그러니 1번을 최우선으로 확인하고, 실패 시 조건 B로 threshold를 당기는 루프를 먼저 확립할 것.

### 5.2 ✅ 확정된 결과 (실험 완료) + 프레이밍 재정렬

digits 미니 실험(Task A→B, latch @ epoch 29, switch @ epoch 80)에서 §5.1 측정 1이 **깨끗하게 통과**했다:
- **위상 발산**: latch 시점(epoch 29)에서 mask divergence 1차 계단, task switch(epoch 80)에서 훨씬 큰 2차 계단 — §6 시그니처 그대로.
- **churn resurgence**: CONTINUE 망은 switch에서 churn 0.001→0.028(>20배)로 급증, LATCH는 구조적으로 불가(epoch 29부터 frozen) — "shift가 측정 가능한 divergence를 만든다"의 가장 깨끗한 형태.
- **구조 용량 차이**: switch 후 CONTINUE 연결의 ~20–23%가 freeze 시점엔 존재하지 않던 것(layer weight mass의 5–6% 담당). LATCH는 이 용량을 원천 차단당함.

**그러나 결정적 재해석 — freeze는 forgetting을 막지 못한다.** 두 망 다 Task A가 **0%로 붕괴**했다. 위상만 잠그고 공유 연결의 weight는 Task B가 덮어쓰기 때문(스펙 §9의 weight-stays-open 제약대로). 이건 실패가 아니라 **이 메커니즘이 무엇을 잠그는지에 대한 명료화**다:

> **이 메커니즘은 memory-protection이 아니라 representation-locking이다.** 잠그는 대상은 "어느 task의 weight가 살아남느냐"(=forgetting 방어)가 아니라 "**어떤 종류의 feature 위상이 형성되느냐**(shape vs texture, causal vs shortcut)"다.

따라서 이 메커니즘의 자연스러운 payoff는 **H3(forgetting)이 아니라 H4(robustness) + H6(human-likeness)**다. 실험이 이 둘을 깨끗이 분리해줬다: freeze는 **구조 가용성**을 바꾸지만(20–23% 용량 차이) **weight-overwrite 동역학**은 안 바꾼다(forgetting 동일). → **논문 척추를 H3 중심에서 H4/H6 중심으로 옮긴다**(§7 반영).

### 5.3 다음 실험 (재정렬된 척추의 1차 검증)

forgetting이 아니라 **representation 보존**을 잰다. Task B 학습 후:
- LATCH 망이 shape bias / spurious-correlation robustness를 **보존**하는가?
- CONTINUE 망은 위상을 자유롭게 rewire해 texture/shortcut 채널을 새로 recruit하므로 이 값들이 **깎일** 것으로 예측.
- 지금 확인된 20배 churn resurgence가 이미 "CONTINUE는 shift 때 새 구조를 판다"를 보였으니, 다음 질문은 **"그 새 구조가 texture/shortcut이고 LATCH는 그것을 막아 robust한가"**다. 이게 §7.1 인과 사슬의 직접 실증.

---

## 6. 확정적 차이 시그니처 (pre-register / 성공의 정의)

baseline이 **원리적으로 못 만드는** 세 곡선. 이걸 성공 기준으로 미리 못박는다.

1. **위상 turnover staircase**: 레이어별 구조 turnover가 높다가 성숙 시점에 0으로 떨어져 **되살아나지 않고**, 깊이순으로 계단식. — dense는 항상 0(구조 축 없음), SET는 항상 높음(안 꺼짐), emergent DNN-CP는 latch가 없음.
2. **weight-update 부분공간 붕괴**: closure 이후 weight 변화 중 *새 구조 방향*(닫힌/silent 연결 방향) 성분이 0, 기존 위상 재가중만 남음 → 도달 가능한 weight-change manifold의 rank가 레이어별로 closure에서 하락. baseline은 내내 full-rank.
3. **post-closure 학습의 old-basis 감금**: 닫힌 뒤 새 task 학습이 전부 기존 부분공간 방향으로 분해됨(새 방향 recruit = 0). baseline은 새 방향 자유 recruit → 여기서 ±CP가 갈림.

이 셋이 안 나오면 메커니즘 실패이고, baseline이 이 셋을 내면 novelty가 죽는다.

---

## 7. 실험 매트릭스 (2×2)

조건: **(± CP latch) × (± coarse-to-fine curriculum)**

| 조건 | 예측 (representation 보존 중심) |
|---|---|
| − CP, − curriculum | texture bias, robustness 낮음 (표준 baseline) |
| − CP, + curriculum | shape bias·robustness ↑ (Lu 2026 입증) — 그러나 이후 continual learning에서 위상이 자유 rewire되어 human-like 표상이 texture/shortcut 쪽으로 다시 깎일 것 (핵심 미검증 지점) |
| + CP, − curriculum | 구조 봉쇄는 있으나 압축 표상 형성 유인이 약해 "무엇을 잠갔는지" 모호 — 효과 약할 것 |
| + CP, + curriculum | curriculum이 만든 human-like 위상을 CP latch가 continual learning 동안 **보존** — 핵심 가설(H6). CONTINUE 대비 shape bias/spurious robustness가 안 깎이는 것이 성공 신호 |

> **주의**: 위 표는 **forgetting(Task A 유지)이 아니라 representation 보존**을 예측한다. Task A 정확도는 ±CP 모두 붕괴할 수 있고(§5.2), 그건 이 메커니즘의 실패가 아니다. 갈리는 건 "Task B를 배운 뒤에도 human-like/robust한 feature 위상을 유지하는가"다.

평가 축 (**§5.2 재정렬 반영 — human-like/robustness가 1차, forgetting은 부차**):
- **P1 human-likeness triad** (핵심 novelty): (a) **shape bias** (Geirhos 2019 프로토콜; 목표 감각 0.90=인간 4–6세, Lu 2026), (b) **global/gist**: coarse readout에서 전역 구조 우선 추출 정도, (c) **spurious-correlation robustness**: shortcut(배경·색·텍스처)에 안 넘어가는 정도. 각 조건 × 각 task 전환 시점마다 측정해 **궤적 보존** 여부 비교. ← 이 메커니즘이 실제로 서빙하는 축(representation-locking).
- **P2 robustness**: closure 이후 noise/OOD 성능 저하를 ±CP 간 비교.
- **P3 quality-gap / irreversibility**: closure 후 이상적 입력으로 재학습해도 CP-open 수준으로 회복 안 됨을 확인(넓이·구조성·robustness 셋 다).
- **P4 stability-plasticity (부차)**: forgetting, backward/forward transfer. **주의: latch 단독으로는 forgetting을 막지 못함이 실측 확인됨**(§5.2). 이 축에서 이득을 보려면 §9의 graded weight consolidation이 추가로 필요하며, 그건 별도 확장이다 — 기본 메커니즘의 성공 기준에서 제외한다.

### 7.1 human-like triad의 인과 사슬 (왜 하나의 엔진에서 셋이 다 나오는가)

"얕은 층이 입력이 아직 저해상도(global·저SF)일 때 위상을 확정한다"는 단일 원인에서:

- **shape bias**: 텍스처(고SF) 도착 전에 얕은 위상이 shape용 저SF 구조로 latch → 이후 텍스처는 전용 채널 recruit 불가, 기존 shape basis 위 재가중만 → shape가 substrate로 남음.
- **global/gist**: 확정된 저층 basis가 global 저SF → 빠른 feedforward 표상이 전역 구조 지배. *(주의: Navon식 top-down global precedence 완전 재현은 recurrence 필요. feedforward net에선 "확정된 저층 basis가 global하다"까지만 정직하게 주장할 것.)*
- **spurious robustness**: shortcut learning을 "여분 구조 용량의 기회주의적 recruit"로 재정의 → spurious feature는 대개 전용 pathway 신설을 요구 → closure가 구조 용량을 고갈시켜 이미 확정된 causal(shape/global) basis에 의존하게 강제 → shortcut 봉쇄.

---

## 8. 구현 순서 (권장)

1. ✅ **완료**: SET juvenile 엔진 + `frozen` latch + §5.1 위상 발산 sanity. 통과함(§5.2).
2. **신호 교체**: raw churn/loss-slope → **silent-synapse 전환율**(§3.1). 이 전환율이 gross churn과 달리 **정말 0으로 가는지** 먼저 확인(§3.2 예측 검증). 가면 그걸 latch 신호로 확정.
3. **엔진 교체**: SET random regrow → **RigL gradient regrow**(§3.6). 전환율 신호 품질 + activity-dependent 주장 확보.
4. **척추 재정렬 실험**(§5.3): Task B 학습 후 LATCH vs CONTINUE의 shape bias / spurious robustness 보존 비교. ← 다음 핵심 실험.
5. 통과 후 2×2 매트릭스 + 평가 P1–P4(재정렬 순서)로 확장.
6. curriculum은 1차 blur → 유의미하면 Lu 2026 파이프라인으로 정교화.

---

## 9. 정직한 경계 (related work에서 선제 방어할 것)

- **구조 가소성 자체는 선행 있음**: SET(Mocanu 2018), RigL(Evci 2020), DEEP R(Bellec 2018), neurogenesis-CL(Parisi 2018), Grow-Prune-Freeze(2606.25170류), Fauth 2014(potential-synapse 풀 고갈 이론). novelty는 "구조를 바꾼다"가 **아니다**.
- **novelty의 정확한 위치**: 구조 가소성에 **① 성숙 신호로 위상만 비가역 latch + ② coarse-to-fine과 결합해 위계 창발 + ③ human-like triad(shape/gist/spurious) 및 섹션 6 시그니처로 검증**. 이 특정 조합의 선행 부재를 확인해 명시할 것.
- **주장 정정**: "AI엔 구조 가소성이 없다"(X, DST가 반례) → **"AI엔 구조 가소성을 발달적으로 *종료*시키는 CP가 없다"**(O).
- **hard-freeze 함정**: 위상을 너무 세게 닫고 weight까지 얼리면 순수 layer-freezing이 되어 plasticity를 잃는다. **살아남은 연결의 weight는 반드시 열어둘 것** — 그 잔존 synaptic 가소성이 stability-plasticity의 plasticity 절반이자 "학습은 되되 다르게(Ranson 2012, Sato-Stryker 2008)"의 실체다.
- **gist 과대주장 금지**: feedforward net에서 top-down global precedence를 완전 주장하지 말 것(섹션 7.1 주의).
- **신호↔생물학 매핑 (novelty 방어에 유리)**: 성숙 신호를 silent-synapse 전환율로 잡으면 latch 트리거가 Huang et al. 2015(silent synapse 소진이 CP 종료를 결정)에 **직접 대응**한다 — "임의로 고른 지표"가 아니라 생물학적으로 동기화된 신호. DST 선행(SET/RigL)은 pool 소진을 성숙 신호로 쓴 적이 없으므로 여기서도 차별화된다.
- **forgetting을 꼭 쫓을 경우 — graded consolidation (선택 확장, EWC 방어 필요)**: 기본 메커니즘은 위상만 잠그고 weight를 열어두므로 forgetting을 못 막는다(§5.2, 설계상). 굳이 H3를 살리려면 closure가 **살아남은 연결 weight도 부분 consolidate**(LR을 0이 아니라 낮게)하게 확장한다. 이건 더 충실한 PNN 대응이기도 하다 — **PNN은 새 연결을 막을 뿐 아니라 기존 시냅스를 물리적으로 안정화**하는데, 현재 스펙은 앞 절반(신규 차단)만 구현했다. 단, 이 확장은 EWC 인접 영역이므로 novelty를 (i) 발달적 타이밍, (ii) layer-wise 구조 결합, (iii) binary 아닌 graded 소비로 방어해야 하고, Task B가 consolidated 연결을 재사용 못 하도록 pathway overlap을 줄여야 한다(안 그러면 그냥 weight-freezing).

---

## 10. 실패 판정 (언제 이 접근을 접거나 수정하나)

- ~~섹션 5.1 측정 1(위상 발산)~~ → **통과함**(§5.2). 이 판정은 해소됨.
- silent-synapse 전환율이 gross churn처럼 **0으로 안 가고 평형에 남음**(§3.2 예측 실패) → 신호를 §3.3 effective-dimensionality 대안으로 교체. (loss-slope로는 돌아가지 말 것 — §3.5.)
- 섹션 6 시그니처가 안 나옴 → 메커니즘이 baseline과 구분 안 됨.
- baseline(SET/RigL 계속)이 섹션 6 시그니처를 냄 → novelty 소멸.
- **+ CP 조건이 human-like triad(P1)에서 − CP(CONTINUE) 대비 유의한 보존 이득 없음 → 핵심 가설(H6) 기각.** ← 이게 이제 1차 성공/실패 기준(forgetting 아님).
- **주의**: "Task A forgetting을 못 막음"은 **실패 판정이 아니다**(§5.2). 이 메커니즘은 representation-locking이지 memory-protection이 아니므로 forgetting은 애초 성공 기준에서 제외.

---

## 참고 (핵심만)

- 구조 가소성/DST: Mocanu et al., *Nat. Commun.* 2018 (SET); Evci et al., ICML 2020 (RigL); Bellec et al., ICLR 2018 (DEEP R).
- 구조 가소성 × CL: Parisi et al., 2018 (neurogenesis); Fauth et al., *PLOS ONE* 2014 (potential-synapse 풀 고갈).
- 생물 CP: Hensch, *Nat. Rev. Neurosci.* 2005; Morishita et al., *Science* 2010 (Lynx1); Huang et al., *PNAS* 2015 (silent synapse 성숙); Ranson et al., *PNAS* 2012 & Sato-Stryker, *J. Neurosci.* 2008 (성체 가소성은 메커니즘부터 다름).
- curriculum/human-like: Lu et al., *Nat. Mach. Intell.* 2026 (Developmental Visual Diet); Geirhos et al., ICLR 2019 (shape bias).
- DNN-CP 선행: Achille et al., ICLR 2019 (Critical Learning Periods).

(전체 서지는 프로젝트 proposal.md / notes.md 참조.)