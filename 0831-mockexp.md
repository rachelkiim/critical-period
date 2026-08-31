# Critical Period as Structural-Plasticity Termination — Implementation Spec

**목적**: 생물학적 critical period(CP)를 ANN에 구현한다. 핵심 아이디어는 "새로운 가소성 규칙을 발명"하는 게 아니라, **기존 dynamic sparse training(구조 가소성)에 발달적·비가역적 종료(성숙-latch)를 추가**하는 것이다.

이 문서는 구현자(Claude Code)를 위한 사양서다. 코드 컨벤션이 아니라 *설계 의도와 반드시 지켜야 할 제약*에 초점을 둔다. 애매하면 "정직한 경계" 섹션과 "실패 판정" 섹션을 먼저 볼 것.

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

## 3. `maturity(l)` — 유일한 실질적 설계 선택

### 3.1 1차 후보: churn rate (새 지표 안 만듦, 최우선)

SET는 매 주기 magnitude 낮은 연결을 prune하고 새 연결을 regrow한다. 위상이 수렴하면 "regrow했다가 다음 주기에 곧바로 다시 prune되는" 연결 비율이 올라간다(churn이 헛돎). 이 **churn이 낮아지는 것 = 위상 수렴 = 성숙**.

- 정의(예): 최근 주기에 regrow된 연결 중 다음 주기에 prune된 비율, 또는 연속 주기 간 연결 마스크의 symmetric difference 비율.
- 장점: SET가 이미 만들어내는 양만 읽으면 됨. 계산 추가 없음. 가장 단순.

### 3.2 ⚠️ 반드시 지킬 것 — threshold는 "바닥"이 아니라 "꺾임"에 건다

churn이 **완전히 바닥 친 뒤** 잠그면 문제가 생긴다: 그 시점에 얼리는 연결은 "어차피 죽어 있던" 것들이라, latch한 위상 ≈ SET가 자연히 도달했을 위상. → **weight 동역학이 안 갈린다** (섹션 5 참고). 이건 이 설계의 핵심 실패 모드다.

따라서 latch는 churn이 **바닥 치기 전, 꺾이기 시작하는 시점**(위상이 아직 유동적일 때)에 걸어야 "지금 이 위상으로 확정, 추가 탐색 금지"가 되어 미래 자유도를 실제로 자른다. 구현: churn의 절대 임계가 아니라 churn 감소율/변곡점, 또는 "peak churn 대비 X% 지점"으로 트리거.

### 3.3 대안 후보 (나중에 비교용, 지금은 구현 안 해도 됨)

- 그 레이어 표상의 압축도/유효 차원(effective dimensionality)이 평평해지는 시점. human-likeness 지표와 연동되지만 계산 추가.
- RigL식 gradient-based regrow를 쓰면 grow 신호가 gradient라 "activity-dependent"라는 생물학적 주장이 살짝 강해짐. churn 정의는 동일하게 유지. (단순함 우선이면 SET random regrow가 제일 깔끔.)

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

| 조건 | 예측 |
|---|---|
| − CP, − curriculum | texture bias, forgetting 큼, robustness 낮음 (표준 baseline) |
| − CP, + curriculum | shape bias·robustness ↑ (Lu 2026이 입증) — 그러나 이후 continual learning에서 human-like 표상이 다시 깎일 것(미검증) |
| + CP, − curriculum | 구조 봉쇄는 있으나 압축 표상 형성 유인이 약해 "무엇을 잠갔는지" 모호 — 효과 약할 것 |
| + CP, + curriculum | curriculum이 만든 human-like 표상을 CP latch가 continual learning 동안 보존 — 핵심 가설 |

평가 축:
- **P1 stability-plasticity**: forgetting, backward/forward transfer (task-incremental).
- **P2 robustness**: closure 이후 noise/OOD 성능 저하를 ±CP 간 비교.
- **P3 human-likeness triad** (핵심 novelty): (a) **shape bias** (Geirhos 2019 프로토콜; 목표 감각 0.90=인간 4–6세, Lu 2026), (b) **global/gist**: coarse readout에서 전역 구조 우선 추출 정도, (c) **spurious-correlation robustness**: shortcut(배경·색·텍스처)에 안 넘어가는 정도. 각 조건 × 각 task 전환 시점마다 측정해 궤적 비교.
- **P4 quality-gap / irreversibility**: closure 후 이상적 입력으로 재학습해도 CP-open 수준으로 회복 안 됨을 확인(넓이·구조성·robustness 셋 다).

### 7.1 human-like triad의 인과 사슬 (왜 하나의 엔진에서 셋이 다 나오는가)

"얕은 층이 입력이 아직 저해상도(global·저SF)일 때 위상을 확정한다"는 단일 원인에서:

- **shape bias**: 텍스처(고SF) 도착 전에 얕은 위상이 shape용 저SF 구조로 latch → 이후 텍스처는 전용 채널 recruit 불가, 기존 shape basis 위 재가중만 → shape가 substrate로 남음.
- **global/gist**: 확정된 저층 basis가 global 저SF → 빠른 feedforward 표상이 전역 구조 지배. *(주의: Navon식 top-down global precedence 완전 재현은 recurrence 필요. feedforward net에선 "확정된 저층 basis가 global하다"까지만 정직하게 주장할 것.)*
- **spurious robustness**: shortcut learning을 "여분 구조 용량의 기회주의적 recruit"로 재정의 → spurious feature는 대개 전용 pathway 신설을 요구 → closure가 구조 용량을 고갈시켜 이미 확정된 causal(shape/global) basis에 의존하게 강제 → shortcut 봉쇄.

---

## 8. 구현 순서 (권장)

1. **SET(또는 RigL) juvenile 엔진**을 sparse 레이어로 세팅. prune-regrow 정상 작동 + 주기별 churn 측정 확보.
2. 레이어별 `frozen` 상태 + churn 기반 latch(섹션 3.2의 "꺾임" 트리거) 추가.
3. **섹션 5.1 최소 sanity 실험 먼저.** 위상 발산(측정 1) 확인. 안 나오면 threshold 당김.
4. 통과 후 2×2 매트릭스 + 평가 P1–P4로 확장.
5. curriculum은 1차 blur → 유의미하면 Lu 2026 파이프라인으로 정교화.

---

## 9. 정직한 경계 (related work에서 선제 방어할 것)

- **구조 가소성 자체는 선행 있음**: SET(Mocanu 2018), RigL(Evci 2020), DEEP R(Bellec 2018), neurogenesis-CL(Parisi 2018), Grow-Prune-Freeze(2606.25170류), Fauth 2014(potential-synapse 풀 고갈 이론). novelty는 "구조를 바꾼다"가 **아니다**.
- **novelty의 정확한 위치**: 구조 가소성에 **① 성숙 신호로 위상만 비가역 latch + ② coarse-to-fine과 결합해 위계 창발 + ③ human-like triad(shape/gist/spurious) 및 섹션 6 시그니처로 검증**. 이 특정 조합의 선행 부재를 확인해 명시할 것.
- **주장 정정**: "AI엔 구조 가소성이 없다"(X, DST가 반례) → **"AI엔 구조 가소성을 발달적으로 *종료*시키는 CP가 없다"**(O).
- **hard-freeze 함정**: 위상을 너무 세게 닫고 weight까지 얼리면 순수 layer-freezing이 되어 plasticity를 잃는다. **살아남은 연결의 weight는 반드시 열어둘 것** — 그 잔존 synaptic 가소성이 stability-plasticity의 plasticity 절반이자 "학습은 되되 다르게(Ranson 2012, Sato-Stryker 2008)"의 실체다.
- **gist 과대주장 금지**: feedforward net에서 top-down global precedence를 완전 주장하지 말 것(섹션 7.1 주의).

---

## 10. 실패 판정 (언제 이 접근을 접거나 수정하나)

- 섹션 5.1 측정 1(위상 발산)이 threshold를 아무리 당겨도 안 나옴 → latch가 자유도를 못 자름. 접근 재검토.
- 섹션 6 시그니처가 안 나옴 → 메커니즘이 baseline과 구분 안 됨.
- baseline(SET 계속)이 섹션 6 시그니처를 냄 → novelty 소멸.
- + CP 조건이 human-like triad(P3)에서 − CP 대비 유의한 보존 이득 없음 → 핵심 가설 기각.

---

## 참고 (핵심만)

- 구조 가소성/DST: Mocanu et al., *Nat. Commun.* 2018 (SET); Evci et al., ICML 2020 (RigL); Bellec et al., ICLR 2018 (DEEP R).
- 구조 가소성 × CL: Parisi et al., 2018 (neurogenesis); Fauth et al., *PLOS ONE* 2014 (potential-synapse 풀 고갈).
- 생물 CP: Hensch, *Nat. Rev. Neurosci.* 2005; Morishita et al., *Science* 2010 (Lynx1); Huang et al., *PNAS* 2015 (silent synapse 성숙); Ranson et al., *PNAS* 2012 & Sato-Stryker, *J. Neurosci.* 2008 (성체 가소성은 메커니즘부터 다름).
- curriculum/human-like: Lu et al., *Nat. Mach. Intell.* 2026 (Developmental Visual Diet); Geirhos et al., ICLR 2019 (shape bias).
- DNN-CP 선행: Achille et al., ICLR 2019 (Critical Learning Periods).

(전체 서지는 프로젝트 proposal.md / notes.md 참조.)
