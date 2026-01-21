# Project Texo

## 1. 프로젝트 개요 (Project Overview)

**Project Texo**은 실제 사용자의 LLM 상호작용 데이터를 구조화하여 실시간 모니터링과 디버깅을 가능하게 하는 시스템입니다. 

단순한 로그 저장을 넘어, 멀티턴(Multi-turn) 대화의 흐름을 시각화하고, 네트워크 재시도(Retry)와 같은 엣지 케이스까지 정밀하게 추적하여 데이터의 일관성과 비용 효율성을 동시에 확보했습니다.

본 시스템은 사용자가 실제로 LLM과 어떻게 상호작용하는지를 있는 그대로 반영하면서, 엔지니어링 관점에서의 정밀한 분석이 가능하도록 설계되었습니다.

### 1.1 목표 (Objective)
*   **완전한 관측성 확보 (Full Observability)**: 실시간 대화 흐름뿐만 아니라 재시도(Retry) 내역까지 포함한 모든 상호작용을 시각화합니다.
*   **데이터 정합성 보장 (Idempotency)**: 네트워크 오류로 인한 재시도 시 불필요한 중복 트레이스(Trace) 생성을 방지하고 데이터 정확성을 유지합니다.
*   **비용 및 효율 최적화 (Cost Optimization)**: 전체 히스토리는 보존하되, 평가는 필요한 턴(Turn)에만 수행하여 비용을 절감합니다.
*   **OpenAI Compatible Server**: Custom OpenAI Compatible Server를 구축하여 API key 등 시스템 관리 효율성을 높입니다.

### 1.2 문제 정의 (Problem Statement)
LLM 서비스 운영 시, 단순한 입출력 로그만으로는 복잡한 멀티턴 대화의 맥락이나 간헐적인 네트워크 오류로 인한 재시도 상황을 파악하기 어렵습니다.
*   **디버깅의 어려움**: 전체 대화 문맥(Context)이 소실되거나, 특정 턴에서의 오류 원인을 파악하기 위해 재현(Re-run)하는 것이 불가능했습니다.
*   **데이터 중복 및 비용**: 재시도 로직이 돌 때마다 중복된 트레이스가 생성되어 통계가 왜곡되거나, 불필요한 전체 히스토리 평가로 인해 토큰 비용이 낭비되는 문제가 있었습니다.

---

## 2. 해결 방안 및 핵심 기술 (Solution & Key Features)

### 2.1 관측성 강화 (Observability)
사용자와 LLM 간의 복잡한 상호작용을 투명하게 파악할 수 있는 체계를 구축했습니다.
*   **Live Monitoring**: 진행 중인 멀티턴 대화를 실시간으로 모니터링하여 이탈 지점이나 오류 발생 구간을 즉시 파악합니다.
*   **Session Replay**: 전체 대화 흐름을 처음부터 끝까지 다시보기(Replay) 할 수 있어, 사용자 경험(UX)을 정밀하게 분석할 수 있습니다.
*   **Retry Visibility**: 한 번의 응답을 위해 내부적으로 몇 번의 재시도가 있었는지(Retry Attempts)를 명확하게 시각화하여, 모델의 불안정성이나 네트워크 이슈를 식별합니다.

### 2.2 멱등성 보장 (Idempotency)
네트워크 불안정으로 인한 재시도 상황에서도 데이터의 유일성을 보장합니다.
*   **Deterministic IDs**: 요청(Request)별로 고정된 ID를 부여하여, 동일한 요청이 재시도되더라도 중복된 트레이스로 기록되지 않도록 방지합니다.
*   **Accurate Metrics**: 재시도로 인해 발생할 수 있는 중복 카운팅을 제거하여, 정확한 사용량 및 성공률 지표를 유지합니다.

### 2.3 완벽한 디버깅 환경 (Debugging)
개발자가 언제든 과거의 상황을 완벽하게 재현하고 분석할 수 있는 환경을 제공합니다.
*   **Full Context Availability**: 오류가 발생한 시점의 모든 문맥 정보(Prompt, Parameter, Context)를 그대로 보존합니다.
*   **Exact Inputs Re-run**: 과거의 특정 턴(Turn)을 당시와 완전히 동일한 입력값으로 다시 실행(Re-run)해볼 수 있어, 문제의 원인을 정확하게 격리하고수정할 수 있습니다.
*   **Trace History**: 단순 결과뿐만 아니라 성공하기까지의 모든 시도(Attempt) 내역을 추적하여 간헐적 오류까지 잡아냅니다.

### 2.4 커스텀 OpenAI 호환 서버 (Custom OpenAI Compatible Server)
Cursor와 IDE에서 원하는 LLM 모델을 코딩 어시스턴트로 활용하기 위해 OpenAI 호환 엔드포인트를 구축하고, API key로 팀 단위의 체계적인 관리를 지원합니다.
*   **OpenAI 호환성**: vLLM 등의 추론 엔진을 활용하여 쉽게 호환 엔드포인트를 구축할 수 있습니다.
*   **팀 단위 트래킹**: 개인 용도를 넘어 기업 도입 시 필수적인 개인/팀/부서별 트래킹 시스템을 통해 사용 현황을 투명하게 관리합니다. 
*   **FastAPI 기반 분석**: FastAPI로 구현된 커스텀 엔드포인트를 통해 팀 내 사용 패턴을 분석하고, 개발 생산성을 높이는 인사이트를 도출하여 시스템을 최적화합니다.

---

## 3. System Flow
    ┌─────────────┐
    │   Frontend  │
    │  (User UI)  │
    └──────┬──────┘
           │ Sends: messages, session_id
           ▼
    ┌─────────────────────────────────────────────────────────┐
    │                      Backend                            │
    ├─────────────────────────────────────────────────────────┤
    │  1. Generate trace_id = f"{session_id}_turn_{turn_#}"   │
    │  2. Create/retrieve trace (reuses ID for retries)       │
    │  3. Create span for this attempt                        │
    │  4. Execute LLM call (streaming)                        │
    │  5. Update span with output                             │
    │  6. Update trace with final output                      │
    └──────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                 ┌─────────────────┐
                 │    Langfuse     │
                 │  (Observability)│
                 ├─────────────────┤
                 │ • Session view  │
                 │ • Trace view    │
                 │ • Span details  │
                 │ • Scores        │
                 │ • Datasets      │
                 └─────────────────┘

---

## 4. 기술 스택 (Tech Stack)

*   **Observability**: Langfuse
*   **LLM Integration**: OpenAI or OpenAI compatible server
*   **Backend/Server**: FastAPI, Gunicorn
*   **Infrastructure**: Docker
*   **Language**: Python
