Developmental Training Framework — 구현 스펙 (Cl

0. 목표

세 가지 developmental mechanism을 하나의 training framework에서 각각 독립적으로 on/off 가능하게 통합하고, 3-mechanism factorial ablation(2³ = 8 조건)을 자동 실행한다.

Random-noise warm-up (기존 논문 그대로)
Gradual sensory maturation (기존 논문 그대로,)
LR suppression / stabilization 

1. 참고 구현 (clone해서 그대로 재사용)
Random-noise warm-up: https://github.com/cogilab/Random2
Gradual sensory maturation: https://github.com/cogilab/Gradual-sensory-maturation

2. 확정된 설계
2.1 Dataset / Architecture — ResNet-18 + CIFAR-10

2.2 전체 타임라인
RANDOM WARM-UP (real-data clock과 분리된 독립 pretraining)
        │
        ▼
┌──────────────────────────────────────────────────┐
│                REAL DATA TRAIN (100%)             │
├───────────────┬────────────────┬─────────────────┤
│   0–5%         │   5–10%        │   10–100%       │
│ degraded input │ degraded input │ normal input    │
│ normal LR      │ LR × α         │ normal LR       │
└───────────────┴────────────────┴─────────────────┘
Random warm-up epoch은 real-data epoch 카운트에 포함하지 않음.
세 mechanism은 각각 독립 boolean — 꺼도 나머지 파이프라인은 그대로 동작해야 함 (예: warm-up 끄면 그냥 real-data train만 실행, gradual maturation 끄면 처음부터 normal input, LR suppression 끄면 5–10% 구간도 그냥 normal LR).

2.3 Mechanism별 구현 방법

① Random-noise warm-up — Random2 random_train/random_input/random_label 그대로 사용.

독립 optimizer: SGD(lr=0.1, momentum=0.9, weight_decay=1e-4) (Random2 benchmark 기본값)
num_noise=50000/epoch, epochs_noise config로 노출 (기본 5)
real-data training의 optimizer/scheduler와는 별개의 인스턴스로 취급 (완전히 분리된 pretraining phase)

② Gradual sensory maturation — Gradual repo get_transforms(dataset="cifar10c", blur=7, color=0)를 degraded, get_transforms(dataset="cifar10c", blur=0, color=1)을 normal input transform으로 그대로 사용.

Binary transition: real-data epoch의 0~10%까지 degraded, 10% 이후 normal.
원 논문은 50% 지점에서 전환했지만 protocol(transform 함수 자체)은 그대로 두고 전환 위치만 우리 developmental window(0–10%)로 이동.

③ LR suppression (신규) — 기존 optimizer/scheduler 객체는 건드리지 않음. 매 epoch scheduler가 정한 LR을 그대로 두되, 5–10% 구간에서만 실제로 optimizer.param_groups[i]['lr']에 적용되는 값을 scheduled_lr × α로 스케일. scheduler.step()은 정상적으로 매 epoch 계속 호출 (내부 스케줄 상태 자체는 안 건드림 — 그 구간이 지나면 원래 스케줄로 자동 복귀).

α는 hyperparameter — grid search 대상 (기본 후보 [0.1, 0.3, 0.5, 0.7, 0.9])


3. Config (CLI/config로 노출)
use_warmup: bool
use_gradual_maturation: bool
use_lr_suppression: bool
total_epochs: int          # 기본 100, 확인 필요 (3-1)
alpha: float | list[float] # LR suppression multiplier, grid search 대상, 확인 필요 (3-2)
epochs_noise: int = 5
num_noise: int = 50000
batch_size: int = 128
seed: int = 42

