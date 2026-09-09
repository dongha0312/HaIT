# 차봇(carbot) 8주 구현 계획

> 이 계획은 **사용자가 직접 구현**한다. Claude는 카드를 주고, 막히면 힌트, 그래도 막히면 함수 골격, 끝나면 리뷰한다. 코드는 요청할 때만 쓴다. 체크박스는 진행 표시용.

**목표:** 차 사진이나 이름 → 유튜브 리뷰어 의견(타임스탬프 링크) + 월 유지비(공공데이터 계산)를 한 화면에 보여주는 로컬 RAG. 파인튜닝·비전·평가 리포트 포함.

**구조:** 정형 데이터는 Postgres, 벡터는 Qdrant, 둘은 모델 id로 연결. LLM은 Qwen(MLX). 핵심 로직(청킹, RRF, 지표, 확신도, 유지비)은 직접 구현. LangChain은 5주차 비교 챕터에서만.

**스택:** Python 3.12, uv, Docker Compose(Postgres, Qdrant), FlagEmbedding(bge-m3), bge-reranker-v2-m3, SigLIP, mlx-lm, yt-dlp, faster-whisper(예외용), FastAPI, Streamlit, pytest.

**설계 문서:** `docs/superpowers/specs/2026-09-09-carbot-design.md` (이 계획은 그 문서를 근거로 한다. 둘 다 읽는다.)

## 전역 제약 (설계 문서에서 그대로)

- 유료 AI API 사용 금지. 임베딩·리랭커·LLM·OCR·비전 전부 로컬.
- 5주차 전 LangChain 계열 금지.
- 숫자는 LLM이 계산하지 않는다. `cost`가 계산하고 LLM은 설명만.
- LLM 없이도 검색·계산·식별은 동작해야 한다.
- 자막 원문은 로컬에만. 결과에는 짧은 발췌와 링크만.
- 사진은 저장하지 않는다.
- 링크 오차 중앙값 목표 8초 이내. 겹침 기준 Recall@5가 주 지표.
- 수동 평가셋 50개는 최종 보고 전까지 튜닝에 쓰지 않는다.
- 깃: GitHub Flow. 카드 하나 = 브랜치 하나(`week1/setup` 식), squash PR, 주차 태그 `v0.N-weekN`.
- 모든 모듈은 CLI로 단독 실행되고, 실행 가능한 체크 하나를 남긴다.

## 카드 형식

각 카드는 다음 칸을 가진다. **완성 판정**을 통과하면 PR을 올린다.

- 무엇 / 왜
- 파일
- 인터페이스: 다른 카드가 의존하는 함수·타입의 정확한 이름과 시그니처
- 완성 판정: 실행 명령과 기대 출력
- 테스트 케이스: 직접 쓸 테스트의 입력 → 기대값 (코드는 본인이 쓴다)
- 힌트 / 함정

---

## 파일 구조 (전체)

```
carbot/
  pyproject.toml           uv 관리. src 레이아웃
  Makefile                 setup / up / down / ingest / index / eval
  docker-compose.yml       postgres, qdrant (8주차에 api, ui 추가)
  .env.example             DB URL, QDRANT_URL, LLM_BASE_URL, FUEL_PRICE 기본값
  channels.yaml            유튜브 채널 5개
  models.yaml              모델 사전 (3주차)
  src/carbot/
    types.py               공용 데이터클래스 (Segment, VideoMeta, Chunk, Embedding, Hit)
    config.py              .env·경로·상수 로딩
    cli.py                 typer 엔트리. 서브커맨드가 각 모듈을 부른다
    ingest/youtube.py      채널 → 영상 목록 → 자막 JSON
    ingest/fuel.py         에너지공단 API (3주차)
    ingest/recall.py       리콜 API (3주차)
    ingest/insurance.py    보험개발원 페이지 (3주차)
    ingest/images.py       AI Hub 이미지 정리 (7주차)
    index/chunk.py         세그먼트 → 청크
    index/embed.py         bge-m3 dense+sparse (1주차), SigLIP (7주차)
    index/store.py         Qdrant 컬렉션 생성·적재·검색 래퍼
    index/build.py         색인 파이프라인
    retrieve/search.py     dense / sparse / hybrid(RRF) / rerank
    retrieve/link.py       문장 단위 링크 (2주차)
    retrieve/models.py     질문에서 모델명 추출 (3주차)
    cost/calc.py           유지비 계산 (3주차)
    db/schema.sql          Postgres 스키마 (3주차)
    db/conn.py             psycopg 연결 (3주차)
    generate/prompt.py     프롬프트 조립 (3주차)
    generate/llm.py        mlx_lm.server 클라이언트 (3주차)
    generate/cite.py       [n] → 링크 후처리 (3주차)
    generate/verify.py     NLI 환각 검증 (4주차)
    eval/synth.py          합성 질문 생성 (2주차)
    eval/retrieval.py      Recall@k, MRR, 링크 오차 (2주차)
    eval/answers.py        환각률, 인용 정확도, 거절률 (4주차)
    eval/report.py         reports/날짜.md 생성 (2주차)
    eval/bakeoff.py        Qwen 크기 비교 (2주차)
    finetune/synth_raft.py RAFT 데이터 합성 (5주차)
    finetune/train.sh      mlx_lm.lora 실행 (5주차)
    vision/index.py        AI Hub 이미지 색인 (7주차)
    vision/identify.py     식별·확신도·판정 (7주차)
    vision/head.py         분류 헤드 학습 (7주차)
    framework/lc_pipeline.py  LangChain + pgvector (6주차)
    api/app.py             FastAPI (8주차)
    ui/app.py              Streamlit (8주차)
  tests/                   모듈당 파일 하나
  data/raw/videos/         {video_id}.json (git 제외)
  data/eval/               평가셋 jsonl (git 포함)
  reports/                 평가 리포트 (git 포함)
```

---

# 1주차: 수집 → 청킹 → 색인 → 검색 CLI

**주차 완성 판정:** `uv run carbot search "그랜저 승차감"` 이 영상 제목, mm:ss, 발췌, 링크가 붙은 상위 5개를 찍는다.
**태그:** `v0.1-week1`

## 카드 1-1: 환경 세팅

**브랜치:** `week1/setup`

**무엇 / 왜**
uv 프로젝트, Docker Compose(Postgres + Qdrant), Makefile, 공용 타입 파일을 만든다. 이후 모든 카드가 이 위에서 돈다. Postgres는 3주차부터 쓰지만 compose에 지금 넣어 두면 나중에 손댈 일이 없다.

**파일**
- 생성: `pyproject.toml`, `docker-compose.yml`, `Makefile`, `.env.example`, `src/carbot/__init__.py`, `src/carbot/types.py`, `src/carbot/config.py`, `src/carbot/cli.py`, `tests/__init__.py`
- 수정: `.gitignore` (이미 있음, 필요 시)

**인터페이스** (`src/carbot/types.py`, 이후 모든 카드가 import)
```python
from dataclasses import dataclass, field

@dataclass
class Segment:            # 자막 한 줄
    start: float          # 초
    end: float            # 초
    text: str
    words: list[tuple[float, str]] = field(default_factory=list)  # (시작 초, 단어). 자동 자막 json3에서만 채움

@dataclass
class VideoMeta:
    video_id: str
    title: str
    channel: str          # channels.yaml의 name
    handle: str           # "@mocar_official"
    published: str        # "YYYYMMDD"
    duration: int         # 초

@dataclass
class Chunk:
    video_id: str
    idx: int              # 영상 내 순번 0부터
    start: float
    end: float
    text: str

@dataclass
class Embedding:
    dense: list[float]            # bge-m3 1024차원
    sparse: dict[int, float]      # 토큰 id → 가중치

@dataclass
class Hit:
    chunk_id: str         # f"{video_id}:{idx}"
    score: float
    payload: dict         # video_id, idx, start, end, text, title, channel, handle
```

`src/carbot/config.py`는 `.env`를 읽어 다음 상수를 노출한다: `DATA_DIR`(기본 `data`), `QDRANT_URL`(기본 `http://localhost:6333`), `DATABASE_URL`(기본 `postgresql://carbot:carbot@localhost:5432/carbot`), `LLM_BASE_URL`(기본 `http://localhost:8080/v1`), `FUEL_PRICE`(기본값은 본인이 정해 `.env.example`에 적는다).

`src/carbot/cli.py`는 typer 앱 하나. 1주차 서브커맨드: `ingest`, `index`, `search`. 카드마다 추가한다.

**완성 판정**
```bash
make up          # docker compose up -d
curl -s localhost:6333/healthz && echo
docker compose exec postgres pg_isready -U carbot
uv run python -c "import carbot.types, mlx_lm, FlagEmbedding, qdrant_client; print('ok')"
uv run pytest -q   # 테스트 0개여도 에러 없이 끝나야 함
```
기대: `healthz ok`, `accepting connections`, `ok`, pytest 정상 종료.

**힌트**
- Python 3.12로 고정한다. `uv python install 3.12` 후 `uv init --package carbot --python 3.12`. 3.13은 torch·FlagEmbedding 휠이 늦게 나와 막힐 수 있다.
- 의존성: `uv add yt-dlp FlagEmbedding torch qdrant-client pyyaml typer python-dotenv mlx-lm` / `uv add --dev pytest ruff`.
- compose: `qdrant/qdrant` 이미지에 6333 포트와 볼륨, `postgres:16`에 `POSTGRES_USER/PASSWORD/DB=carbot`과 볼륨.
- Makefile 타깃 이름은 설계 문서와 같게: `setup`, `up`, `down`, `ingest`, `index`, `eval`.

**함정**
- FlagEmbedding 설치가 torch 버전을 끌어올 수 있다. 실패하면 torch를 먼저 깔고 FlagEmbedding을 `--no-deps`로 깐 뒤 부족한 것만 추가한다.
- `uv init --package`는 `src/carbot/` 레이아웃을 만든다. 테스트에서 `import carbot`이 되려면 `uv run pytest`로 실행해야 한다(uv가 패키지를 editable로 넣는다).

---

## 카드 1-2: 채널 → 영상 목록

**브랜치:** `week1/ingest` (카드 1-3과 같은 브랜치)

**무엇 / 왜**
`channels.yaml`의 채널에서 최근 영상 N개의 메타를 가져온다. 자막 수집(1-3)의 입력이다.

**파일**
- 생성: `channels.yaml`, `src/carbot/ingest/__init__.py`, `src/carbot/ingest/youtube.py`, `tests/test_youtube.py`

**`channels.yaml`**
```yaml
channels:
  - handle: "@mocar_official"
    name: "김한용의 MOCAR"
  - handle: "@Motline"
    name: "Motline"
  - handle: "@StationB"
    name: "강병휘의 Station.B"
  - handle: "@Woopa87"
    name: "우파푸른하늘Woopa TV"
  - handle: "@bpd"
    name: "비피디 BPD"
```

**인터페이스**
```python
def load_channels(path: Path = Path("channels.yaml")) -> list[dict]        # [{"handle","name"}]
def list_videos(handle: str, name: str, limit: int) -> list[VideoMeta]     # 최신순
```

**완성 판정**
```bash
uv run carbot ingest list --handle @mocar_official --limit 5
```
기대: 5줄, 각 줄에 `video_id | published | duration | title`.

**테스트 케이스** (`tests/test_youtube.py`, 네트워크 없이)
- yt-dlp가 돌려주는 flat entry dict 하나(직접 만든 샘플: `id`, `title`, `upload_date`, `duration`)를 `VideoMeta`로 바꾸는 함수 `entry_to_meta(entry, handle, name)`를 분리해서 테스트한다. `upload_date`가 없으면 `""`, `duration`이 None이면 0.

**힌트**
- yt-dlp는 파이썬 API가 있다. `yt_dlp.YoutubeDL({"extract_flat": True, "playlist_end": limit, "quiet": True}).extract_info(f"https://www.youtube.com/{handle}/videos", download=False)` → `["entries"]`.
- flat 모드에선 `upload_date`가 비어 있을 수 있다. 비어 있으면 1-3에서 자막 받을 때 채운다(그때는 flat이 아니라 전체 메타가 온다).

**함정**
- 쇼츠가 섞인다. `duration < 90`은 건너뛴다. 리뷰 영상은 대부분 5분 이상이다.

---

## 카드 1-3: 자막 수집 → JSON 저장

**브랜치:** `week1/ingest`

**무엇 / 왜**
영상마다 한국어 자동 자막을 json3 형식으로 받아 `Segment` 목록으로 바꾸고 `data/raw/videos/{video_id}.json`에 저장한다. 이미 있으면 건너뛴다. 이 파일이 색인의 유일한 입력이다.

**파일**
- 수정: `src/carbot/ingest/youtube.py`, `src/carbot/cli.py`
- 생성: `tests/fixtures/json3_sample.json`, `tests/test_youtube.py`에 케이스 추가

**인터페이스**
```python
def fetch_captions(video_id: str, lang: str = "ko") -> tuple[VideoMeta, list[Segment]] | None
    # 자막이 없으면 None. VideoMeta는 전체 메타에서 채운다(제목·게시일·길이 확정).
def parse_json3(data: dict) -> list[Segment]         # json3 dict → Segment 목록. 순수 함수
def save_video(meta: VideoMeta, segments: list[Segment], out_dir: Path) -> Path
def load_video(path: Path) -> tuple[VideoMeta, list[Segment]]
def ingest_channel(handle: str, name: str, limit: int, out_dir: Path) -> dict   # {"fetched": n, "skipped": n, "no_captions": n}
```

**저장 형식** (`data/raw/videos/{video_id}.json`)
```json
{
  "meta": {"video_id": "...", "title": "...", "channel": "...", "handle": "@...", "published": "20250812", "duration": 812},
  "segments": [
    {"start": 0.0, "end": 2.5, "text": "안녕하세요 김한용입니다", "words": [[0.0, "안녕하세요"], [0.8, "김한용입니다"]]}
  ]
}
```

**완성 판정**
```bash
uv run carbot ingest run --limit 2        # 채널 5개 × 2개 = 10개
ls data/raw/videos | wc -l                # 10
uv run carbot ingest run --limit 2        # 다시 실행
```
기대: 두 번째 실행 출력에 `skipped: 10`, 새로 받은 것 0.

**테스트 케이스** (`parse_json3`, 픽스처는 아래 구조로 직접 만든다)
- 이벤트 2개, 각각 segs에 단어 2개 → Segment 2개, `words` 길이 2, `start`는 `tStartMs/1000`, `end`는 `(tStartMs + dDurationMs)/1000`.
- `segs`가 없는 이벤트 → 무시.
- segs 텍스트가 `"\n"`뿐인 이벤트 → 무시.
- `aAppend: 1`이 붙은 이벤트 → 무시.
- 단어 앞뒤 공백은 합칠 때 하나로.

json3 구조 (자동 자막):
```json
{"events": [
  {"tStartMs": 0, "dDurationMs": 2500, "segs": [{"utf8": "안녕하세요", "tOffsetMs": 0}, {"utf8": " 김한용입니다", "tOffsetMs": 800}]},
  {"tStartMs": 2500, "dDurationMs": 10, "segs": [{"utf8": "\n"}]},
  {"tStartMs": 2500, "dDurationMs": 3000, "aAppend": 1, "segs": [{"utf8": "오늘은"}]}
]}
```

**힌트**
- yt-dlp 옵션: `{"skip_download": True, "writeautomaticsub": True, "subtitleslangs": [lang], "subtitlesformat": "json3", "outtmpl": str(tmp_dir / "%(id)s"), "quiet": True}`. 받은 파일은 `{id}.ko.json3`. 수동 자막이 있으면 `writesubtitles: True`도 켜서 그쪽을 우선한다(지금 5개 채널은 전부 자동뿐이지만 코드는 둘 다 받게).
- 전체 메타(`extract_info(url, download=False)`)에서 `title`, `upload_date`, `duration`, `channel`을 가져와 VideoMeta를 확정한다.
- 저장은 `dataclasses.asdict`로 dict 만들고 `json.dump(ensure_ascii=False)`.

**함정**
- 자동 자막은 요청이 몰리면 429가 난다. 영상 사이에 `time.sleep(1~2)`를 둔다.
- `words`의 시각은 `tStartMs + tOffsetMs`. 오프셋을 빼먹으면 모든 단어가 줄 시작 시각이 된다. 2주차 문장 링크가 이 값에 의존한다.
- 1주차에는 자막 없는 영상은 `no_captions`로 세고 건너뛴다. Whisper는 4주차.

---

## 카드 1-4: 청킹

**브랜치:** `week1/chunk`

**무엇 / 왜**
세그먼트를 60초 목표, 15초 오버랩, 문장 경계 정렬로 묶는다. 청크 길이는 2주차에 30·60·120초로 비교하니 인자로 받는다. 이 함수는 순수 함수라 테스트하기 가장 좋은 곳이다.

**파일**
- 생성: `src/carbot/index/__init__.py`, `src/carbot/index/chunk.py`, `tests/test_chunk.py`

**인터페이스**
```python
def chunk_segments(video_id: str, segments: list[Segment],
                   target_sec: float = 60.0, overlap_sec: float = 15.0) -> list[Chunk]
def is_sentence_end(text: str) -> bool    # "다." "요." "죠." "까?" "!" 등으로 끝나면 True
```

**규칙**
1. 세그먼트를 순서대로 누적한다. 누적 길이(마지막 end - 첫 start)가 `target_sec`를 넘기 직전 세그먼트에서 청크를 끊되, 뒤로 최대 3개 세그먼트 안에 문장 끝이 있으면 거기서 끊는다.
2. 다음 청크는 현재 청크 끝에서 `overlap_sec`만큼 앞으로 돌아간 시각 이후의 첫 세그먼트에서 시작한다.
3. 청크 텍스트는 세그먼트 텍스트를 공백으로 이어 붙인다. `idx`는 0부터.
4. 세그먼트 하나가 `target_sec`보다 길면 그 하나로 청크 하나.

**완성 판정**
```bash
uv run pytest tests/test_chunk.py -q
uv run carbot index chunk-stats            # data/raw/videos 전체에 대해
```
기대: 테스트 통과. 통계 출력에 영상 수, 청크 수, 청크 길이 평균·최소·최대(초), 텍스트 길이 평균(자).

**테스트 케이스**
- 5초짜리 세그먼트 30개(150초), 텍스트 전부 "..다."로 끝남, target 60, overlap 15 → 청크 3개 이상, 모든 청크 길이 ≤ 60 + 5(허용), 연속 청크는 시간이 겹침(뒤 청크 start < 앞 청크 end).
- 문장 끝이 없는 세그먼트 → 길이 기준으로만 끊긴다.
- 세그먼트 하나가 90초 → 청크 하나, 길이 90.
- 빈 입력 → 빈 목록.
- `is_sentence_end("좋습니다.")` True, `is_sentence_end("그래서")` False, `is_sentence_end("어때요?")` True.

**힌트**
- 한국어 자동 자막에는 마침표가 거의 없다. 종결어미로 판단한다: `다 요 죠 까 네 군 지` 뒤에 구두점이 오거나 문자열이 끝나면 문장 끝. 완벽할 필요 없다. 없으면 길이로 끊는다.

**함정**
- 오버랩을 "이전 청크의 마지막 N개 세그먼트"로 구현하면 세그먼트 길이에 따라 오버랩이 들쭉날쭉해진다. 시각 기준으로 되돌아가야 한다.

---

## 카드 1-5: 임베딩 + Qdrant 적재

**브랜치:** `week1/index`

**무엇 / 왜**
청크를 bge-m3로 dense·sparse 벡터로 만들어 Qdrant `review_chunks`에 넣는다. 임베딩 입력 텍스트 앞에 `[채널] 제목`을 붙인다(payload에는 원문만).

**파일**
- 생성: `src/carbot/index/embed.py`, `src/carbot/index/store.py`, `src/carbot/index/build.py`, `tests/test_store.py`
- 수정: `src/carbot/cli.py`

**인터페이스**
```python
# embed.py
class TextEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-m3"): ...
    def encode(self, texts: list[str], batch_size: int = 16) -> list[Embedding]
    def encode_query(self, text: str) -> Embedding

# store.py
REVIEW_COLLECTION = "review_chunks"
class VectorStore:
    def __init__(self, url: str = config.QDRANT_URL): ...
    def ensure_review_collection(self) -> None       # dense 1024 cosine + sparse 이름 "sparse". 있으면 그대로
    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[Embedding], meta: VideoMeta) -> int
    def count(self) -> int
    def search_dense(self, vec: list[float], k: int) -> list[Hit]
    def search_sparse(self, sparse: dict[int, float], k: int) -> list[Hit]
    def delete_video(self, video_id: str) -> None

# build.py
def embed_text_for(chunk: Chunk, meta: VideoMeta) -> str      # f"[{meta.channel}] {meta.title}\n{chunk.text}"
def build_index(videos_dir: Path, store: VectorStore, embedder: TextEmbedder,
                target_sec: float = 60.0, overlap_sec: float = 15.0, force: bool = False) -> dict
    # {"videos": n, "chunks": n, "skipped": n}. force=False면 이미 적재된 video_id는 건너뜀
```

**완성 판정**
```bash
uv run carbot index build
uv run carbot index count
```
기대: `build` 출력의 chunks 수 == `count` 출력. 10개 영상이면 수백 개. 두 번째 `build`는 `skipped: 10`.

**테스트 케이스** (`tests/test_store.py`, Qdrant가 떠 있어야 함. 없으면 skip 마크)
- 테스트용 컬렉션 이름을 따로 써서(예: `test_review_chunks`) 가짜 Embedding 3개 upsert → `count()` 3 → dense 검색 k=2 → Hit 2개, payload에 `video_id`·`idx`·`start` 있음 → `delete_video` 후 count 0.
- `embed_text_for`: 채널 "A", 제목 "B", 텍스트 "C" → `"[A] B\nC"`.

**힌트**
- bge-m3 sparse는 FlagEmbedding의 `BGEM3FlagModel(model, use_fp16=True).encode(texts, return_dense=True, return_sparse=True)` → `dense_vecs`, `lexical_weights`(토큰 문자열 → 가중치 dict). Qdrant sparse는 정수 인덱스가 필요하니 모델의 토크나이저로 토큰 → id를 바꿔 `dict[int, float]`로 만든다.
- Qdrant 컬렉션: `vectors_config={"dense": VectorParams(size=1024, distance=COSINE)}`, `sparse_vectors_config={"sparse": SparseVectorParams()}`. 포인트 id는 `chunk_id` 문자열을 uuid5로 바꾼다.
- "이미 적재된 영상" 판단은 payload `video_id`로 `scroll` 하거나, 별도로 `data/processed/indexed.json`에 video_id 목록을 두는 게 단순하다. 후자를 추천.
- 첫 실행 때 bge-m3(약 2.2GB)를 내려받는다. 시간이 걸린다.

**함정**
- MPS에서 fp16이 NaN을 내면 `use_fp16=False`. 느려지지만 맞다.
- 배치를 크게 하면 16GB에서 스왑이 난다. 16으로 시작.

---

## 카드 1-6: 검색 CLI (dense)

**브랜치:** `week1/index`

**무엇 / 왜**
질문 → dense 검색 → 상위 5개를 사람이 읽을 형태로 찍는다. 1주차 완성 판정이자 2주차 하이브리드의 베이스라인이다.

**파일**
- 생성: `src/carbot/retrieve/__init__.py`, `src/carbot/retrieve/search.py`, `tests/test_search.py`
- 수정: `src/carbot/cli.py`

**인터페이스**
```python
Mode = Literal["dense", "sparse", "hybrid", "rerank"]     # 1주차는 dense만 구현, 나머지는 NotImplementedError
def search(query: str, k: int = 5, mode: Mode = "dense",
           store: VectorStore | None = None, embedder: TextEmbedder | None = None) -> list[Hit]
def youtube_url(video_id: str, start: float) -> str        # https://www.youtube.com/watch?v={id}&t={int(start)}s
def fmt_time(sec: float) -> str                            # 83.0 → "01:23"
def excerpt(text: str, n: int = 80) -> str                 # 앞 n자 + "…"
```

**완성 판정**
```bash
uv run carbot search "그랜저 승차감"
```
기대 출력 형태(5줄):
```
1. [김한용의 MOCAR] 그랜저 2.5 시승기  03:20  0.71
   "승차감은 확실히 부드러워졌고…"  https://www.youtube.com/watch?v=XXXX&t=200s
```

**테스트 케이스**
- `youtube_url("abc", 83.6)` → `"https://www.youtube.com/watch?v=abc&t=83s"`.
- `fmt_time(83)` → `"01:23"`, `fmt_time(3725)` → `"1:02:05"`.
- `excerpt("가"*100, 80)` 길이 81.
- `search(..., mode="hybrid")` → `NotImplementedError`.

**힌트**
- 2주차에 `mode`별 분기가 늘어나니 `search`는 얇게 두고 모드별 함수(`_dense`, 나중에 `_sparse`, `_hybrid`, `_rerank`)로 나눈다.

---

## 1주차 마무리

- [ ] 카드 1-1 ~ 1-6 PR 머지
- [ ] `docs/progress.md` 갱신
- [ ] `git tag v0.1-week1 && git push --tags`
- [ ] 리뷰 요청: "1주차 끝났어" 한마디면 된다. 전체 흐름과 각 모듈을 본다.

---

# 2주차: 하이브리드·리랭크·문장 링크·평가셋 v1·베이크오프

**주차 완성 판정:** `uv run carbot eval retrieval` 이 `reports/YYYY-MM-DD.md`에 검색 방식 4종 × 청크 길이 3종 표를 쓴다.
**태그:** `v0.2-week2`

## 카드 2-1: sparse + RRF 하이브리드
- 파일: `retrieve/search.py`, `tests/test_search.py`
- 인터페이스: `def rrf_fuse(rankings: list[list[Hit]], k: int = 60) -> list[Hit]` (점수 = Σ 1/(k+rank), 같은 chunk_id 합산, 내림차순), `search(mode="sparse")`, `search(mode="hybrid")` (dense 50 + sparse 50 → rrf → 상위 k)
- 완성 판정: 같은 질문에 세 모드 결과가 다르게 나오고 `hybrid`에 둘의 상위가 섞여 있음.
- 테스트: 두 랭킹 [A,B,C], [B,D] → 융합 결과 1위 B. 빈 랭킹 포함해도 에러 없음.

## 카드 2-2: 리랭커 + 인접 병합
- 파일: `retrieve/rerank.py`, `retrieve/search.py`
- 인터페이스: `class Reranker: def score(self, query: str, texts: list[str]) -> list[float]` (bge-reranker-v2-m3), `def merge_adjacent(hits: list[Hit], max_per_video: int = 2) -> list[Hit]` (같은 video_id의 idx 연속이면 하나로, start=min, end=max, text 이어붙임), `search(mode="rerank")` (hybrid 20 → 리랭크 → 병합 → k)
- 완성 판정: `search "그랜저 승차감" --mode rerank`가 한 영상에서 최대 2개만 보여줌.
- 테스트: idx 3,4,5와 9 → 병합 결과 2개. 영상 3개 각 3개 히트 → 영상당 2개 이하.

## 카드 2-3: 문장 단위 링크
- 파일: `retrieve/link.py`, `tests/test_link.py`
- 인터페이스: `def best_sentence_start(query: str, chunk_payload: dict, words: list[tuple[float, str]], scorer: Reranker) -> float` (청크를 문장으로 나눠 리랭커 점수 최고 문장의 첫 단어 시각 반환. 문장 분리는 `is_sentence_end` 재사용), `search` 결과 Hit.payload에 `link_start` 추가
- 전제: 청크 payload에 words가 없으므로 `data/raw/videos/{id}.json`을 읽어 청크 구간의 words를 잘라 쓴다. 헬퍼 `def words_in(video_id, start, end) -> list[tuple[float,str]]`.
- 완성 판정: CLI 링크의 `t=`가 청크 시작이 아니라 문장 시작으로 바뀜.
- 테스트: 문장 3개짜리 청크, 두 번째 문장이 질문과 겹침 → 반환 시각이 두 번째 문장 첫 단어 시각.

## 카드 2-4: 합성 평가셋 v1 + 지표
- 파일: `eval/synth.py`, `eval/retrieval.py`, `eval/report.py`, `data/eval/synthetic_v1.jsonl`, `data/eval/manual_v1.jsonl`(20개, 직접 작성), `tests/test_metrics.py`
- 평가셋 행: `{"q": str, "video_id": str, "start": float, "end": float, "src": "synthetic"|"manual"}`
- 인터페이스: `def synth_questions(store, llm, n: int, seed: int) -> list[dict]` (청크 샘플 → 교사 모델에 "이 구간으로만 답할 수 있는 한국어 질문 하나, 모델명 포함" → n-gram 겹침 필터), `def ngram_overlap(a: str, b: str, n: int = 3) -> float`, `def recall_at_k(results, truth, k) -> float` (히트 구간과 정답 구간이 겹치면 적중), `def mrr(results, truth) -> float`, `def link_error(results, truth) -> list[float]` (link_start - truth.start 절대값, 1위 히트만), `def write_report(rows: list[dict], path: Path)`
- LLM 호출은 3주차 `generate/llm.py`가 아직 없으니 이 카드에서 `mlx_lm.server`를 띄우고 OpenAI 호환 `/v1/chat/completions`를 requests로 직접 부르는 최소 클라이언트를 `eval/llm_min.py`에 둔다. 3주차에 `generate/llm.py`로 옮긴다.
- 완성 판정: `carbot eval retrieval --modes dense,sparse,hybrid,rerank --chunk 30,60,120` → 리포트에 12행(Recall@5, MRR, 링크 오차 중앙값, ±10초 적중률, 지연 ms). 합성 100 + 수동 20.
- 테스트: `ngram_overlap("가나다라마", "가나다라마")` 1.0, 완전 다른 문자열 0.0. recall/mrr은 손으로 만든 결과 3개로 검증.
- 주의: 청크 길이 비교는 길이마다 색인을 따로 만들어야 한다. 컬렉션 이름에 길이를 붙인다(`review_chunks_30`). `ensure_review_collection(name)`으로 일반화.

## 카드 2-5: Qwen 베이크오프
- 파일: `eval/bakeoff.py`, `reports/bakeoff.md`
- 내용: mlx-community의 Qwen 1.7B·4B·8B급 4-bit를 각각 `mlx_lm.server`로 띄워 같은 30문항(근거 청크 5개 포함 프롬프트) → 응답 길이, 첫 토큰 지연, 전체 지연, 본인이 1~5점 채점.
- 완성 판정: 표 하나와 "학생 = X, 교사 = Y" 결정 한 줄이 `docs/progress.md` 결정 기록에 들어감.

## 카드 2-6: 영상 100개로 확장
- `carbot ingest run --limit 20`, `carbot index build`. 밤에 돌린다. 평가셋 재생성(시드 고정).

---

# 3주차: Postgres·유지비·모델 사전·인용 답변

**주차 완성 판정:** `uv run carbot ask "그랜저 유지비랑 리뷰 어때?"` 가 인용 링크 달린 답변과 유지비 표를 출력한다.
**태그:** `v0.3-week3`

## 카드 3-1: Postgres 스키마 + 연결
- 파일: `db/schema.sql`, `db/conn.py`, `tests/test_db.py`
- 테이블: `models(id text pk, maker text, display_name text, aliases text[])`, `fuel_economy(id serial, model_id fk, trim text, displacement_cc int, fuel text, kmpl numeric, grade int, source_name text)`, `recalls(id serial, model_id fk, title text, date date, url text)`, `insurance(model_id fk, grade int, base_price int, year int)`, `videos(video_id text pk, title, channel, handle, published, duration)`, `feedback(id serial, question text, answer text, vote int, reason text, created_at timestamptz)`
- 인터페이스: `def get_conn() -> psycopg.Connection`, `def init_schema()`; Makefile `make db-init`
- 완성 판정: `make db-init` 후 `\dt`에 6개 테이블.

## 카드 3-2: 모델 사전
- 파일: `models.yaml`, `retrieve/models.py`, `tests/test_models.py`
- 형식: `- id: hyundai_grandeur_gn7  maker: 현대  name: 그랜저  aliases: ["그랜저", "그랜져", "GN7", "신형 그랜저", "Grandeur"]`
- 인터페이스: `def load_models() -> list[dict]`, `def extract_models(question: str) -> list[str]` (별칭 긴 것부터 매칭, 대소문자 무시, 중복 제거), `def load_models_to_db()`
- 완성 판정: 50종 등록. `extract_models("그랜저랑 K8 중에 뭐가 나아")` → 두 id.
- 테스트: 별칭 대소문자, 포함 관계("그랜저 하이브리드"는 그랜저), 없으면 빈 목록.

## 카드 3-3: 연비·리콜·보험 수집
- 파일: `ingest/fuel.py`, `ingest/recall.py`, `ingest/insurance.py`, `.env.example`에 `DATA_GO_KR_KEY`
- 인터페이스: 각 모듈 `def fetch_all() -> list[dict]`(원본 JSONL 저장), `def load_to_db(rows)`(모델 사전으로 model_id 매칭, 못 맞춘 행은 `data/processed/unmatched_*.jsonl`에)
- 완성 판정: 50종 중 연비 매칭 40종 이상, 리콜 조회 됨, 보험등급 30종 이상. 미매칭 목록이 파일로 남음.
- 함정: 공공데이터포털 키는 활용신청 후 발급. 보험개발원은 페이지 구조를 먼저 눈으로 보고 파서를 짠다. 막히면 등급은 수동 CSV 30종으로 대체하고 넘어간다.

## 카드 3-4: 유지비 계산
- 파일: `cost/calc.py`, `tests/test_cost.py`
- 인터페이스: `@dataclass class CostBreakdown: model_id; annual_tax: int; annual_fuel: int; monthly_total: int; insurance_grade: int|None; base_price: int|None; recalls: list[dict]; formulas: dict[str,str]; inputs: dict`, `def estimate(model_id: str, km_per_year: int = 15000, fuel_price: int = config.FUEL_PRICE, car_age: int = 1) -> CostBreakdown`, `def car_tax(displacement_cc: int, fuel: str, car_age: int) -> int`
- 자동차세 상수는 지방세법 조문을 직접 확인해 코드 주석에 조문 번호와 함께 적는다.
- 완성 판정: `carbot cost hyundai_grandeur_gn7 --km 15000` 표 출력. 위택스 계산기와 케이스 5개 일치.
- 테스트: 1600cc 미만·이상, 전기차, 차령 3년·10년 경감, 유류비 = km/kmpl*price 반올림.

## 카드 3-5: LLM 클라이언트 + 프롬프트 + 인용 후처리 + ask CLI
- 파일: `generate/llm.py`, `generate/prompt.py`, `generate/cite.py`, `generate/answer.py`, `tests/test_cite.py`, `tests/test_prompt.py`
- 인터페이스: `class LLM: def chat(self, messages: list[dict], stream: bool = False, temperature: float = 0.2) -> str | Iterator[str]` (OpenAI 호환, base_url=config.LLM_BASE_URL), `def build_messages(question: str, hits: list[Hit], cost: CostBreakdown | None) -> list[dict]` (규칙 시스템 프롬프트 + [1]~[n] 근거 + 유지비 표), `def attach_links(answer: str, hits: list[Hit]) -> str` ([n] → 마크다운 링크, 없는 번호는 제거), `def answer(question: str, km: int, fuel_price: int) -> AnswerResult` (`@dataclass AnswerResult: text; hits; cost; models`)
- 완성 판정: `carbot ask "그랜저 유지비랑 리뷰 어때?"` → 인용 링크 달린 답변 + 유지비 표. 모델명 없는 질문은 유지비 표 없이 리뷰만.
- 테스트: `attach_links("좋다[1] 나쁘다[9]", hits 2개)` → [1]은 링크, [9]는 제거. `build_messages`에 근거 5개면 시스템 프롬프트에 "[5]"까지 등장.

---

# 4주차: 환각 검증·거절·답변 평가·리포트 v1·영상 300개

**주차 완성 판정:** 리포트에 환각률, 인용 정확도, 거절률, 링크 오차가 들어간다.
**태그:** `v0.4-week4`

## 카드 4-1: NLI 검증
- 파일: `generate/verify.py`, `tests/test_verify.py`
- 인터페이스: `@dataclass SentenceVerdict: sentence: str; cited: list[int]; supported: bool; score: float`, `class Verifier: def __init__(self, model_name: str)`, `def verify(self, answer: str, hits: list[Hit]) -> list[SentenceVerdict]` (문장 분리 → 인용 청크 텍스트를 premise로 entailment 확률 → 임계값), `def split_sentences(text: str) -> list[str]`
- NLI 후보 2~3개(다국어 xnli 계열)를 수동 채점 50문장으로 비교해 하나 고르고 `docs/progress.md`에 기록.
- `answer()`에 검증 루프 추가: 미지지 문장이 있으면 "다음 문장은 근거가 없으니 빼거나 근거 안에서 고쳐 써라"를 붙여 1회 재생성.
- 완성 판정: `carbot ask ... --verify` 출력에 문장마다 ✅/⚠.

## 카드 4-2: 거절 규칙 + 모르는 질문셋
- 파일: `generate/prompt.py` 수정, `data/eval/unknown_v1.jsonl`(30개)
- 규칙: 모델명 추출 실패 + 검색 최고 점수 임계값 미만 → 검색 없이 "리뷰 데이터에 없다" 응답. 유지비는 데이터 없으면 항목별로 "없음".
- 완성 판정: 30개 중 지어내지 않고 모른다고 한 비율(거절률) 출력.

## 카드 4-3: 답변 평가 + 리포트 v1
- 파일: `eval/answers.py`, `eval/report.py` 확장, `data/eval/manual_v1.jsonl` 50개로 확장(정답 구간 포함), `reports/`
- 지표: 환각률(미지지 문장/전체), 인용 정확도(인용이 실제 근거인 비율), 거절률, 지연(첫 토큰·전체), 자동 지표 vs 수동 채점 일치율(50개).
- 완성 판정: `carbot eval all` → 리포트 하나에 검색·링크·유지비·답변 표 전부.

## 카드 4-4: 영상 300개 + Whisper 폴백
- `ingest/whisper.py`: `def transcribe(video_id) -> list[Segment]` (faster-whisper, `word_timestamps=True`). `ingest_channel`에서 `no_captions`일 때만 호출.
- 밤에 `carbot ingest run --limit 60` → `index build`. 평가셋은 재생성하지 않고(정답 고정) 색인만 커진 뒤 지표 재측정.

---

# 5주차: RAFT 데이터 합성 + LoRA 학습

**주차 완성 판정:** 학습이 끝나고 손실 곡선 이미지가 `reports/`에 있다.
**태그:** `v0.5-week5`

## 카드 5-1: RAFT 데이터 합성
- 파일: `finetune/synth_raft.py`, `data/finetune/train.jsonl`, `valid.jsonl`
- 형식(mlx-lm chat): `{"messages": [{"role":"system",...},{"role":"user", "<근거 5개(정답 2~3 + 무관 2~3 섞음) + 질문>"},{"role":"assistant","<교사 답변, [n] 인용, 무관 근거 미인용>"}]}`
- 인터페이스: `def make_example(question: dict, store, teacher: LLM, n_distract: int = 2, seed: int) -> dict`, `def synth(n: int, out: Path, seed: int)`
- 교사 답변은 4-1 검증기를 통과한 것만 채택(미지지 문장 있으면 버림). 2~3천 개. 밤에 돌린다.
- 완성 판정: train 2000+, valid 200, 채택률과 버린 이유 집계가 `docs/progress.md`에.

## 카드 5-2: LoRA 학습
- 파일: `finetune/train.sh`, `finetune/config.yaml`
- `mlx_lm.lora --model <학생 4-bit> --train --data data/finetune --iters 500~1000 --batch-size 4 --num-layers 16 --adapter-path adapters/`
- 학습 손실·검증 손실을 로그에서 뽑아 `reports/lora_loss.png`(matplotlib).
- 완성 판정: 검증 손실이 내려가고 `mlx_lm.generate --adapter-path adapters/`로 답이 나온다.
- 함정: 16GB에서 8B 학생은 무리. 4B 4-bit로. 배치 4가 스왑 나면 2.

---

# 6주차: 파인튜닝 평가 + LangChain 비교

**주차 완성 판정:** 비교표 두 개(파인튜닝 전후, 직접 구현 vs LangChain).
**태그:** `v0.6-week6`

## 카드 6-1: 파인튜닝 전후 평가
- `mlx_lm.server --adapter-path adapters/`로 학생+어댑터 서빙. `carbot eval answers --tag base` / `--tag lora` → 같은 평가셋(수동 50 + 합성 100 + 모르는 30)으로 환각률·인용 정확도·거절률·지연 표.
- 완성 판정: 표가 리포트에. 나빠졌으면 나빠진 대로 적고 원인 가설 3줄.

## 카드 6-2: LangChain + pgvector 비교
- 파일: `framework/lc_pipeline.py`, `framework/README.md`
- 같은 청크·같은 임베딩 모델로 pgvector에 적재, LangChain retriever + 같은 프롬프트로 `answer`와 동일한 출력 형식. 같은 평가셋으로 검색·답변 지표, 코드 줄 수, 지연 비교.
- 완성 판정: 비교표 + "어디서 프레임워크가 편했고 어디서 불편했나" 5줄.

---

# 7주차: 차종 식별

**주차 완성 판정:** `carbot identify photo.jpg` 가 확정/후보/모르는 차를 내고, 식별 표가 리포트에 있다.
**태그:** `v0.7-week7`

## 카드 7-1: 이미지 색인
- 파일: `ingest/images.py`, `index/embed.py`에 `ImageEmbedder`, `vision/index.py`
- AI Hub 압축 해제 → `data/raw/car_images/{maker_model_year}/`. 모델당 100장 색인, 20장 홀드아웃 목록 `data/eval/vision_holdout.jsonl`.
- 인터페이스: `class ImageEmbedder: def encode(self, images: list[PIL.Image]) -> list[list[float]]` (SigLIP), `VectorStore.ensure_image_collection()`, `upsert_images(vectors, payloads)`, `search_image(vec, k)`
- 완성 판정: `car_images` 카운트 = 100종 × 100장.

## 카드 7-2: 식별·확신도·판정
- 파일: `vision/identify.py`, `tests/test_identify.py`
- 인터페이스: `@dataclass Identification: status: Literal["confirmed","candidates","unknown"]; model_id: str|None; confidence: float; candidates: list[tuple[str,float]]`, `def vote(hits: list[Hit]) -> list[tuple[str,float]]` (모델별 유사도 가중 합, 내림차순), `def decide(votes, min_top: float, min_margin: float) -> Identification`, `def identify(image: PIL.Image) -> Identification`
- 임계값은 홀드아웃으로 "확정 시 정확도 95% 이상" 지점을 찾아 `config`에.
- 완성 판정: `carbot eval vision` → Top-1, Top-5, 확정률, 확정 시 정확도. 실물 사진 30장 별도 표.
- 테스트: 투표 합산, margin 부족 → candidates, top 부족 → unknown.

## 카드 7-3: 분류 헤드
- 파일: `vision/head.py`
- SigLIP 벡터(색인 때 저장해 둔 npz) → 선형 분류층 학습(torch, 몇 분). `identify`에 `method="knn"|"head"` 옵션. 홀드아웃 비교표. 모르는 차 판정은 계속 knn 유사도로.
- 완성 판정: knn vs head 표.

---

# 8주차: API·UI·배포·README·영상

**주차 완성 판정:** 링크 하나(배포 1층)와 시연 영상 3분, README 완성.
**태그:** `v0.8-week8` 그리고 `v1.0`

## 카드 8-1: FastAPI
- 파일: `api/app.py`, `tests/test_api.py`
- 엔드포인트 7개(설계 12.1). `/ask`는 SSE 스트리밍, 마지막 이벤트에 JSON. `/health`는 세 의존성 각각 ok/fail.
- 완성 판정: `uv run uvicorn carbot.api.app:app` 후 curl 7개.

## 카드 8-2: Streamlit
- 파일: `ui/app.py`. 설계 12.2. `/health`의 LLM fail이면 "근거 발췌 모드" 배지.
- 완성 판정: 브라우저에서 질문·사진·피드백 전부 동작.

## 카드 8-3: 배포 1층
- compose에 api, ui 추가. Supabase Postgres + 컨테이너 내 Qdrant로 HF Spaces(Docker) 또는 Fly.io. 리랭크 끄기 옵션 `RERANK=0`.
- 완성 판정: 공개 URL에서 검색·유지비·식별 동작. LLM 없이.

## 카드 8-4: README + 영상
- README: 문제 정의, 구조도(mermaid), 비교표 3개(리포트에서 복사), 재현 5명령, 한계와 다음 단계, 채널 출처·리콜센터 출처 표기.
- 시연 영상 3분(로컬 전체 기능, 번호판 마스킹).
- `v1.0` 태그.

---

## 자르는 순서 (밀릴 때)

1. 카드 6-2 (LangChain 비교) → 2. 카드 7-3 (분류 헤드, knn만 남김) → 3. 배포 2층(LLM 터널) → 4. 영상 300개를 200개로. 파인튜닝(5-1, 5-2, 6-1)과 평가표는 끝까지.
