# HaIT 8주 구현 계획 (v2)

> **사용자가 직접 구현**한다. Claude는 카드를 주고, 막히면 힌트, 그래도 막히면 함수 골격, 끝나면 리뷰한다. 코드는 요청할 때만. 체크박스는 진행 표시용. v1(`2026-09-09-carbot-8week-plan.md`)은 자동차 판이며 대체됨.

**목표:** 폰·노트북·데스크탑 이름이나 사진 → 한·영 리뷰어 의견(타임스탬프 링크) + 공식 스펙 비교 + 실구매가를 한 화면에. 로컬 RAG. LoRA·임베딩 파인튜닝, 프레임 검색, 평가 리포트 포함.

**구조:** 정형은 Postgres, 벡터는 Qdrant(자막 청크 + 프레임), 모델 id로 연결. LLM은 Qwen(MLX). 핵심 로직 직접 구현.

**스택:** Python 3.12, uv, Docker Compose(Postgres, Qdrant), FlagEmbedding(bge-m3), bge-reranker-v2-m3, SigLIP, mlx-lm, yt-dlp, ffmpeg, faster-whisper(예외용), selectolax(HTML 파싱), OCR(PaddleOCR 또는 EasyOCR), FastAPI, Streamlit, pytest.

**설계 문서:** `docs/superpowers/specs/2026-09-10-hait-design.md`

## 전역 제약

- 유료 AI API 금지. 전부 로컬.
- LangChain 계열은 비교 챕터(8주차 여유분) 전 금지.
- 숫자는 LLM이 계산하지 않는다. `specs`·`price`가 만든다.
- LLM 없이도 검색·스펙·가격·프레임·OCR 동작.
- 자막 원문 로컬에만. 유튜브 기계번역 자막 사용 금지. 사진 저장 금지.
- 링크 오차 중앙값 8초 이내 목표. 겹침 Recall@5 주 지표. 수동셋은 튜닝에 안 씀.
- 깃: GitHub Flow, 카드 = 브랜치, squash PR, 주차 태그 `v0.N-weekN`.
- 패키지명 `hait`, 표기 `HaIT`.

## 카드 형식

무엇/왜 · 파일 · 인터페이스(다른 카드가 의존하는 이름·시그니처) · 완성 판정(명령과 기대 출력) · 테스트 케이스(입력 → 기대값, 코드는 본인이) · 힌트 · 함정

---

## 파일 구조 (전체)

```
HaIT/
  pyproject.toml, Makefile, docker-compose.yml, .env.example
  channels.yaml            채널 8개 (handle, name, lang)
  models.yaml              모델 사전 (3주차)
  src/hait/
    types.py               Segment, VideoMeta, Chunk, Embedding, Hit
    config.py              .env·경로·상수
    cli.py                 typer 엔트리
    ingest/youtube.py      채널 → 영상 목록 → 원문 자막 JSON
    ingest/whisper.py      자막 없는 영상 전사 (4주차)
    ingest/specs.py        애플·삼성 스펙 페이지 (3주차)
    ingest/subsidy.py      스마트초이스 지원금 (3주차)
    ingest/rra.py          전파인증 CSV (4주차)
    ingest/recall.py       소비자24 리콜 (4주차)
    index/chunk.py         세그먼트 → 청크
    index/embed.py         TextEmbedder(bge-m3), ImageEmbedder(SigLIP, 7주차)
    index/store.py         Qdrant 래퍼 (review_chunks, frames)
    index/build.py         자막 색인 파이프라인
    index/frames.py        장면 추출 → 프레임 색인 (7주차)
    retrieve/search.py     dense / sparse / hybrid(RRF) / rerank + 게시일 부스트
    retrieve/link.py       문장 단위 링크 (2주차)
    retrieve/models.py     모델명 추출 (3주차)
    specs/compare.py       스펙 정규화·비교 (3주차)
    price/calc.py          실구매가 (3주차)
    db/schema.sql, db/conn.py  (3주차)
    generate/{llm,prompt,cite,answer,verify}.py  (3~4주차)
    eval/{synth,retrieval,answers,report,bakeoff}.py
    finetune/{synth_raft.py,train.sh}            (5주차)
    finetune/{synth_pairs.py,train_embed.py}     (6주차)
    vision/{search.py,ocr.py}                    (7주차)
    api/app.py, ui/app.py                        (8주차)
    framework/lc_pipeline.py                     (여유 시)
  tests/
  data/raw/videos/, data/raw/frames/, data/eval/, reports/
```

---

# 1주차: 수집(한·영) → 청킹 → 색인 → 검색 CLI

**주차 완성 판정:** `uv run hait search "아이폰 17 발열"` 이 한·영 채널이 섞인 상위 5개를 제목·mm:ss·발췌·링크와 함께 찍는다.
**태그:** `v0.1-week1`

## 카드 1-1: 환경 세팅

**브랜치** `week1/setup`

**무엇 / 왜**
uv 프로젝트, Docker Compose(Postgres + Qdrant), Makefile, 공용 타입. 이후 모든 카드의 바닥.

**파일**
- 생성: `pyproject.toml`, `docker-compose.yml`, `Makefile`, `.env.example`, `src/hait/__init__.py`, `src/hait/types.py`, `src/hait/config.py`, `src/hait/cli.py`, `tests/__init__.py`

**인터페이스** (`src/hait/types.py`)
```python
from dataclasses import dataclass, field

@dataclass
class Segment:            # 자막 한 줄
    start: float          # 초
    end: float
    text: str
    words: list[tuple[float, str]] = field(default_factory=list)  # (시작 초, 단어)

@dataclass
class VideoMeta:
    video_id: str
    title: str
    channel: str          # channels.yaml의 name
    handle: str           # "@mkbhd"
    lang: str             # "ko" | "en"  채널 언어 = 자막 원문 언어
    published: str        # "YYYYMMDD"
    duration: int         # 초

@dataclass
class Chunk:
    video_id: str
    idx: int
    start: float
    end: float
    text: str

@dataclass
class Embedding:
    dense: list[float]            # 1024
    sparse: dict[int, float]

@dataclass
class Hit:
    chunk_id: str         # f"{video_id}:{idx}"
    score: float
    payload: dict         # video_id, idx, start, end, text, lang, title, channel, handle, published
```

`config.py`: `.env`에서 `DATA_DIR`(기본 `data`), `QDRANT_URL`(`http://localhost:6333`), `DATABASE_URL`(`postgresql://hait:hait@localhost:5432/hait`), `LLM_BASE_URL`(`http://localhost:8080/v1`), `DEFAULT_PLAN_KRW`(요금제 월액 기본값, 본인이 정해 `.env.example`에).
`cli.py`: typer 앱. 1주차 서브커맨드 `ingest`, `index`, `search`.

**완성 판정**
```bash
make up
curl -s localhost:6333/healthz && echo
docker compose exec postgres pg_isready -U hait
uv run python -c "import hait.types, mlx_lm, FlagEmbedding, qdrant_client; print('ok')"
uv run pytest -q
```
기대: `healthz ok`, `accepting connections`, `ok`, pytest 정상 종료(테스트 0개).

**힌트**
- `uv python install 3.12` → `uv init --package hait --python 3.12`. 3.13은 torch·FlagEmbedding 휠이 늦다.
- `uv add yt-dlp FlagEmbedding torch qdrant-client pyyaml typer python-dotenv mlx-lm selectolax` / `uv add --dev pytest ruff`.
- compose: `qdrant/qdrant` 6333 + 볼륨, `postgres:16` `POSTGRES_USER/PASSWORD/DB=hait` + 볼륨.
- Makefile 타깃: `setup up down ingest index eval`.

**함정**
- FlagEmbedding이 torch 버전을 끌어올리면 torch 먼저 깔고 FlagEmbedding `--no-deps`.
- `uv init --package`는 src 레이아웃. 테스트는 `uv run pytest`로.

---

## 카드 1-2: 채널 → 영상 목록

**브랜치** `week1/ingest` (1-3과 같은 브랜치)

**무엇 / 왜**
`channels.yaml`의 채널 8개에서 최근 영상 N개의 메타를 가져온다. 채널 언어를 메타에 붙인다.

**파일**
- 생성: `channels.yaml`, `src/hait/ingest/__init__.py`, `src/hait/ingest/youtube.py`, `tests/test_youtube.py`

**`channels.yaml`**
```yaml
channels:
  - {handle: "@ITSUB",         name: "ITSub잇섭",            lang: ko}
  - {handle: "@Underkg",       name: "underKG",              lang: ko}
  - {handle: "@zuyoni1",       name: "ZUYONI TECH",          lang: ko}
  - {handle: "@BullsLab",      name: "뻘짓연구소 BullsLab",    lang: ko}
  - {handle: "@the-edit",      name: "디에디트 THE EDIT",      lang: ko}
  - {handle: "@mkbhd",         name: "Marques Brownlee",     lang: en}
  - {handle: "@Dave2D",        name: "Dave2D",               lang: en}
  - {handle: "@LinusTechTips", name: "Linus Tech Tips",      lang: en}
```

**인터페이스**
```python
def load_channels(path: Path = Path("channels.yaml")) -> list[dict]      # [{"handle","name","lang"}]
def list_videos(handle: str, name: str, lang: str, limit: int) -> list[VideoMeta]   # 최신순, 쇼츠 제외
def entry_to_meta(entry: dict, handle: str, name: str, lang: str) -> VideoMeta
```

**완성 판정**
```bash
uv run hait ingest list --handle @mkbhd --limit 5
```
기대: 5줄, `video_id | published | duration | title`.

**테스트 케이스** (`entry_to_meta`, 네트워크 없이)
- `{"id":"x","title":"t","upload_date":"20260910","duration":600}` → VideoMeta 필드 그대로, lang 전달값.
- `upload_date` 없음 → `""`. `duration` None → 0.

**힌트**
- `yt_dlp.YoutubeDL({"extract_flat": True, "playlist_end": limit, "quiet": True}).extract_info(f"https://www.youtube.com/{handle}/videos", download=False)["entries"]`.

**함정**
- 쇼츠 제외: `duration < 90` 건너뜀. LTT는 영상이 길고 많다. limit는 채널별로 같게 두되 나중에 채널별 가중을 둘 수 있게 `channels.yaml`에 `limit` 선택 키를 허용.

---

## 카드 1-3: 원문 자막 수집 → JSON 저장

**브랜치** `week1/ingest`

**무엇 / 왜**
채널 언어의 **원문** 자막을 받는다. 우선순위: 수동(원문 언어) → 자동(원문 언어, `xx-orig` 우선) → 없으면 `no_captions`(Whisper는 4주차). 영어 채널의 "자동 ko"는 기계번역이라 받지 않는다.

**파일**
- 수정: `src/hait/ingest/youtube.py`, `src/hait/cli.py`
- 생성: `tests/fixtures/json3_sample.json`, 테스트 케이스 추가

**인터페이스**
```python
def pick_caption_track(available: dict[str, list[dict]], automatic: dict[str, list[dict]], lang: str) -> tuple[str, str] | None
    # yt-dlp info의 subtitles / automatic_captions dict → (kind "manual"|"auto", track_key) 또는 None
def fetch_captions(video_id: str, lang: str) -> tuple[VideoMeta, list[Segment]] | None
def parse_json3(data: dict) -> list[Segment]
def save_video(meta: VideoMeta, segments: list[Segment], out_dir: Path) -> Path
def load_video(path: Path) -> tuple[VideoMeta, list[Segment]]
def ingest_channel(handle: str, name: str, lang: str, limit: int, out_dir: Path) -> dict
    # {"fetched": n, "skipped": n, "no_captions": n}
```

**저장 형식** (`data/raw/videos/{video_id}.json`)
```json
{"meta": {"video_id": "...", "title": "...", "channel": "...", "handle": "@...", "lang": "en", "published": "20260910", "duration": 812},
 "caption": {"kind": "auto", "track": "en-orig"},
 "segments": [{"start": 0.0, "end": 2.5, "text": "...", "words": [[0.0, "..."], [0.8, "..."]]}]}
```

**완성 판정**
```bash
uv run hait ingest run --limit 2        # 8채널 × 2 = 16개
ls data/raw/videos | wc -l              # 16
uv run hait ingest run --limit 2        # skipped: 16
python -c "import json,glob;print({json.load(open(f))['meta']['lang'] for f in glob.glob('data/raw/videos/*.json')})"   # {'ko','en'}
```

**테스트 케이스**
- `pick_caption_track`: 수동 `ko` 있음 → ("manual","ko"). 수동 없고 자동에 `ko-orig`, `ko` 둘 다 → ("auto","ko-orig"). 자동에 `ko`만 → ("auto","ko"). lang="en"인데 자동에 `ko`만 → None(기계번역 안 받음).
- `parse_json3`: 이벤트 2개 각 단어 2개 → Segment 2개, words 길이 2, start=tStartMs/1000, end=(tStartMs+dDurationMs)/1000. `segs` 없는 이벤트 무시. 텍스트가 `"\n"`뿐인 이벤트 무시. `aAppend: 1` 이벤트 무시. 단어 사이 공백 하나로.

json3 구조:
```json
{"events": [
  {"tStartMs": 0, "dDurationMs": 2500, "segs": [{"utf8": "안녕하세요", "tOffsetMs": 0}, {"utf8": " 잇섭입니다", "tOffsetMs": 800}]},
  {"tStartMs": 2500, "dDurationMs": 10, "segs": [{"utf8": "\n"}]},
  {"tStartMs": 2500, "dDurationMs": 3000, "aAppend": 1, "segs": [{"utf8": "오늘은"}]}
]}
```

**힌트**
- 먼저 `extract_info(url, download=False)`로 메타와 `subtitles`/`automatic_captions` 키 목록을 얻고 `pick_caption_track`으로 트랙을 고른 뒤, 두 번째 호출에서 `{"skip_download": True, "writesubtitles": kind=="manual", "writeautomaticsub": kind=="auto", "subtitleslangs": [track], "subtitlesformat": "json3", "outtmpl": ...}`로 받는다.
- 단어 시각 = `tStartMs + tOffsetMs`. 2주차 문장 링크가 이 값에 의존.

**함정**
- 요청이 몰리면 429. 영상 사이 `time.sleep(1.5)`.
- 영어 자동 자막 트랙 이름이 `en-orig`가 아니라 `en`만 있는 영상도 있다. 우선순위 로직으로 둘 다 처리.

---

## 카드 1-4: 청킹

**브랜치** `week1/chunk`

**무엇 / 왜**
세그먼트를 60초 목표, 15초 오버랩, 문장 경계 정렬로 묶는다. 한·영 둘 다. 청크 길이는 2주차에 비교하니 인자.

**파일**
- 생성: `src/hait/index/__init__.py`, `src/hait/index/chunk.py`, `tests/test_chunk.py`

**인터페이스**
```python
def chunk_segments(video_id: str, segments: list[Segment], target_sec: float = 60.0, overlap_sec: float = 15.0) -> list[Chunk]
def is_sentence_end(text: str, lang: str = "ko") -> bool
    # ko: "다." "요." "죠." "까?" 등 종결어미 + 구두점 또는 문자열 끝. en: . ? ! 로 끝남
```

**규칙**
1. 세그먼트를 순서대로 누적. 누적 길이가 `target_sec`를 넘기 직전에 끊되, 뒤로 최대 3개 안에 문장 끝이 있으면 거기서.
2. 다음 청크는 현재 청크 끝에서 `overlap_sec` 앞으로 돌아간 시각 이후 첫 세그먼트부터.
3. 텍스트는 공백으로 이어 붙임. `idx` 0부터.
4. 세그먼트 하나가 `target_sec`보다 길면 그것만으로 청크 하나.

**완성 판정**
```bash
uv run pytest tests/test_chunk.py -q
uv run hait index chunk-stats     # 영상 수, 청크 수, 길이 평균·최소·최대, 텍스트 길이 평균, 언어별 청크 수
```

**테스트 케이스**
- 5초 세그먼트 30개(모두 "다."로 끝남), target 60, overlap 15 → 청크 ≥ 3, 각 길이 ≤ 65, 연속 청크는 시간이 겹침.
- 문장 끝 없는 세그먼트 → 길이로만 끊김.
- 90초 세그먼트 하나 → 청크 하나, 길이 90.
- 빈 입력 → 빈 목록.
- `is_sentence_end("좋습니다.")` True, `("그래서")` False, `("It's great.", "en")` True, `("and then", "en")` False.

**힌트**
- 한국어 자동 자막엔 마침표가 거의 없다. 종결어미(`다 요 죠 까 네 군 지`) + 구두점/문자열 끝으로 근사. 영어 자동 자막도 마침표가 없는 경우가 많다. 없으면 길이로.

**함정**
- 오버랩을 "마지막 N개 세그먼트"로 하면 들쭉날쭉. 시각 기준으로.

---

## 카드 1-5: 임베딩 + Qdrant 적재

**브랜치** `week1/index`

**무엇 / 왜**
청크를 bge-m3로 dense·sparse 벡터화해 `review_chunks`에 넣는다. 임베딩 입력 앞에 `[채널] 제목`. payload에 `lang`, `published`를 넣는다(2주차 부스트가 씀).

**파일**
- 생성: `src/hait/index/embed.py`, `src/hait/index/store.py`, `src/hait/index/build.py`, `tests/test_store.py`

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
    def ensure_review_collection(self, name: str = REVIEW_COLLECTION) -> None   # dense 1024 cosine + sparse "sparse"
    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[Embedding], meta: VideoMeta, name: str = REVIEW_COLLECTION) -> int
    def count(self, name: str = REVIEW_COLLECTION) -> int
    def search_dense(self, vec: list[float], k: int, name: str = REVIEW_COLLECTION) -> list[Hit]
    def search_sparse(self, sparse: dict[int, float], k: int, name: str = REVIEW_COLLECTION) -> list[Hit]
    def delete_video(self, video_id: str, name: str = REVIEW_COLLECTION) -> None

# build.py
def embed_text_for(chunk: Chunk, meta: VideoMeta) -> str    # f"[{meta.channel}] {meta.title}\n{chunk.text}"
def build_index(videos_dir: Path, store: VectorStore, embedder: TextEmbedder,
                target_sec: float = 60.0, overlap_sec: float = 15.0, force: bool = False,
                name: str = REVIEW_COLLECTION) -> dict     # {"videos","chunks","skipped"}
```

**완성 판정**
```bash
uv run hait index build
uv run hait index count      # build의 chunks와 같음. 다시 build → skipped: 16
```

**테스트 케이스** (Qdrant 필요, 없으면 skip)
- 테스트 컬렉션 `test_review_chunks`에 가짜 Embedding 3개 upsert → count 3 → dense 검색 k=2 → Hit 2개, payload에 video_id·idx·start·lang → delete_video 후 0.
- `embed_text_for`: 채널 "A", 제목 "B", 텍스트 "C" → `"[A] B\nC"`.

**힌트**
- `BGEM3FlagModel("BAAI/bge-m3", use_fp16=True).encode(texts, return_dense=True, return_sparse=True)` → `dense_vecs`, `lexical_weights`(토큰 문자열 → 가중치). 토크나이저로 토큰 → id 변환해 `dict[int,float]`.
- Qdrant: `vectors_config={"dense": VectorParams(1024, COSINE)}`, `sparse_vectors_config={"sparse": SparseVectorParams()}`. 포인트 id는 `chunk_id`의 uuid5.
- 적재 여부는 `data/processed/indexed_{name}.json`에 video_id 목록으로.
- 첫 실행 때 bge-m3 약 2.2GB 다운로드.

**함정**
- MPS fp16 NaN이면 `use_fp16=False`. 배치 16으로 시작.

---

## 카드 1-6: 검색 CLI (dense)

**브랜치** `week1/index`

**무엇 / 왜**
질문 → dense 검색 → 상위 5개. 한국어 질문으로 영어 청크가 나오는지 눈으로 확인하는 첫 순간이다.

**파일**
- 생성: `src/hait/retrieve/__init__.py`, `src/hait/retrieve/search.py`, `tests/test_search.py`

**인터페이스**
```python
Mode = Literal["dense", "sparse", "hybrid", "rerank"]    # 1주차 dense만. 나머지 NotImplementedError
def search(query: str, k: int = 5, mode: Mode = "dense", store=None, embedder=None) -> list[Hit]
def youtube_url(video_id: str, start: float) -> str      # https://www.youtube.com/watch?v={id}&t={int(start)}s
def fmt_time(sec: float) -> str                          # 83 → "01:23", 3725 → "1:02:05"
def excerpt(text: str, n: int = 80) -> str
```

**완성 판정**
```bash
uv run hait search "아이폰 17 발열"
```
기대 형태:
```
1. [Marques Brownlee] (en) iPhone 17 Review  04:12  0.68
   "thermals are noticeably better than…"  https://www.youtube.com/watch?v=XXXX&t=252s
2. [ITSub잇섭] (ko) 아이폰 17 일주일 써보니  02:05  0.66
   "발열은 확실히…"  https://...
```

**테스트 케이스**
- `youtube_url("abc", 83.6)` → `...watch?v=abc&t=83s`. `fmt_time(83)` "01:23", `fmt_time(3725)` "1:02:05". `excerpt("가"*100, 80)` 길이 81. `search(mode="hybrid")` → NotImplementedError.

**힌트**
- `search`는 얇게, 모드별 함수(`_dense`, 이후 `_sparse` `_hybrid` `_rerank`)로.

## 1주차 마무리
- [ ] 1-1~1-6 PR 머지 → `docs/progress.md` 갱신 → `git tag v0.1-week1 && git push --tags` → "1주차 끝났어"로 리뷰 요청.

---

# 2주차: 하이브리드·부스트·리랭크·문장 링크·평가셋 v1·베이크오프

**주차 완성 판정:** `hait eval retrieval` 이 검색 방식 4종 × 청크 길이 3종 × 부스트 유무 표를 리포트에 쓴다. 한→한, 한→영 분리.
**태그:** `v0.2-week2`

## 카드 2-1: sparse + RRF
- `def rrf_fuse(rankings: list[list[Hit]], k: int = 60) -> list[Hit]`, `search(mode="sparse"|"hybrid")` (dense 50 + sparse 50 → rrf → 30).
- 테스트: [A,B,C],[B,D] → 1위 B. 빈 랭킹 포함 OK.

## 카드 2-2: 게시일 부스트 (1차, 모델 사전 없이)
- `def recency_boost(hits: list[Hit], anchor_date: str | None, before_penalty_days: int = 180) -> list[Hit]`: anchor 이후 게시 +δ, anchor 180일 이전 −δ. 3주차에 모델 사전의 release_date를 anchor로 연결. 1차는 CLI `--after 20260901` 옵션.
- 테스트: 같은 점수 두 히트, 하나만 anchor 이후 → 그것이 1위.

## 카드 2-3: 리랭커 + 인접 병합
- `class Reranker: def score(self, query: str, texts: list[str]) -> list[float]` (bge-reranker-v2-m3), `def merge_adjacent(hits, max_per_video=2)`, `search(mode="rerank")` (hybrid 30 → 부스트 → 리랭크 → 병합 → k).
- 테스트: idx 3,4,5와 9 → 2개. 영상당 2개 이하.

## 카드 2-4: 문장 단위 링크
- `retrieve/link.py`: `def words_in(video_id, start, end) -> list[tuple[float,str]]`, `def best_sentence_start(query, chunk_payload, words, scorer) -> float`. 문장 분리는 `is_sentence_end(text, lang)`. Hit.payload에 `link_start`.
- 테스트: 문장 3개, 두 번째가 질문과 겹침 → 두 번째 첫 단어 시각.

## 카드 2-5: 합성 평가셋 v1 + 지표 + 리포트
- `eval/llm_min.py`(OpenAI 호환 최소 클라이언트, 3주차에 `generate/llm.py`로 이동), `eval/synth.py`, `eval/retrieval.py`, `eval/report.py`
- 평가셋 행: `{"q","video_id","start","end","src":"synthetic"|"manual","lang":"ko"|"en"}` (lang = 정답 청크 언어). 영어 청크에도 **한국어 질문**을 만든다.
- `synth_questions(store, llm, n, seed, lang_filter)`, `ngram_overlap(a,b,n=3)`, `recall_at_k`, `mrr`, `link_error`, `write_report`.
- 완성 판정: `hait eval retrieval --modes dense,sparse,hybrid,rerank --chunk 30,60,120 --boost on,off` → 행마다 한→한 / 한→영 분리 지표 + 링크 오차 + 지연. 합성 120(ko 80, en 40) + 수동 20.
- 청크 길이별 컬렉션 `review_chunks_30` 등.

## 카드 2-6: Qwen 베이크오프
- `eval/bakeoff.py`, `reports/bakeoff.md`. 1.7B·4B·8B급 4-bit, 같은 30문항(한·영 근거 섞임), 지연·품질. 학생·교사 결정을 `docs/progress.md`에.

## 카드 2-7: 영상 150개
- `hait ingest run --limit 20` 밤에. 평가셋 재생성(시드 고정).

---

# 3주차: Postgres·모델 사전·스펙 비교·실구매가·ask CLI

**주차 완성 판정:** `hait ask "아이폰 17 vs 갤럭시 S25 뭐 살까"` 가 인용 링크 답변 + 스펙 비교 표 + 실구매가 표를 출력한다.
**태그:** `v0.3-week3`

## 카드 3-1: Postgres 스키마 + 연결
- 테이블: `models(id pk, brand, category, name, aliases text[], release_date date, model_numbers text[])`, `specs(model_id fk, raw jsonb, display_in, resolution, refresh_hz, chip, ram_gb, storage_gb int[], battery_mah, battery_wh, weight_g, camera_main_mp, ports text[], os, prices jsonb, source_url, fetched_at)`, `subsidies(model_id fk, storage_gb, carrier, plan_name, plan_krw, subsidy_krw, extra_krw, fetched_at)`, `rra_devices(cert_no pk, applicant, model_number, product_name, cert_date, matched_model_id)`, `recalls(id, product_name, reason, date, url, matched_model_id)`, `videos(video_id pk, title, channel, handle, lang, published, duration)`, `feedback(id, question, answer, vote, reason, created_at)`
- `get_conn()`, `init_schema()`, `make db-init`.

## 카드 3-2: 모델 사전 + 추출
- `models.yaml` 60종(폰 30, 노트북 20, 데스크탑 10), `retrieve/models.py`: `load_models()`, `extract_models(question) -> list[str]`(한·영 별칭, 긴 것 우선, 대소문자 무시), `load_models_to_db()`. 2-2의 부스트 anchor를 추출 모델의 `release_date`로 연결.
- 테스트: "아이폰 17이랑 갤럭시 S25 중에" → 두 id. "iPhone 17 Pro"는 Pro id, "아이폰 17"은 기본 id(긴 별칭 우선).

## 카드 3-3: 공식 스펙 수집 + 비교
- `ingest/specs.py`: 페이지별 파서 `parse_apple(html) -> dict`, `parse_samsung(html) -> dict`, `normalize(raw, brand) -> dict`(3-1 컬럼), `fetch_specs(model_id)`, 스냅샷 `data/raw/specs/{model_id}.html`.
- `specs/compare.py`: `def compare(model_ids: list[str]) -> SpecTable`(필드별 값, 숫자 필드 차이·우세).
- 완성 판정: `hait specs compare apple_iphone_17 samsung_galaxy_s25` 표. 20종 이상 수집.
- 테스트: 저장한 HTML 스냅샷으로 파서 회귀 테스트(칩·RAM·무게 값 일치).

## 카드 3-4: 지원금 수집 + 실구매가
- `ingest/subsidy.py`: 스마트초이스 조회 폼 분석(브라우저 개발자도구로 요청 형태 확인) → `fetch_subsidies(model_query) -> list[dict]` → DB. 막히면 상위 폰 20종 수동 CSV `data/raw/subsidy_manual.csv`로 대체하고 그 사실을 기록.
- `price/calc.py`: `@dataclass PriceResult: model_id; storage_gb; carrier; plan_krw; route_subsidy: int|None; route_contract: int; route_unlocked: int|None; monthly: dict; formulas: dict; data_date: str`, `def estimate(model_id, storage_gb, carrier, plan_krw, months=24, mvno_plan_krw=None) -> PriceResult`
- 완성 판정: `hait price apple_iphone_17 --storage 256 --carrier SKT --plan 89000` 표. 스마트초이스 화면 5케이스 일치.
- 테스트: 지원금 있음/없음, 선택약정 0.75, 자급제 경로 None 처리.

## 카드 3-5: LLM 클라이언트 + 프롬프트 + 인용 + ask
- `generate/llm.py`(`LLM.chat`), `generate/prompt.py`(`build_messages(question, hits, spec_table, price_result)`: 영어 근거는 "한국어로 요약하되 원문 발췌 인용" 규칙 포함), `generate/cite.py`(`attach_links`), `generate/answer.py`(`answer(question, carrier, plan_krw) -> AnswerResult`).
- 완성 판정: 위 주차 판정. 모델명 없는 질문은 리뷰만.
- 테스트: `attach_links("좋다[1] 나쁘다[9]", hits 2개)` → [1] 링크, [9] 제거.

---

# 4주차: 검증·거절·전파인증·리콜·답변 평가·리포트 v1·영상 400개

**태그:** `v0.4-week4`

## 카드 4-1: NLI 검증
- `generate/verify.py`: `SentenceVerdict`, `Verifier.verify(answer, hits) -> list[SentenceVerdict]`, `split_sentences(text)`. 다국어 NLI 후보 2~3개를 수동 채점 50문장(영어 근거 포함)으로 비교해 선택. 미지지 문장 있으면 1회 재생성.
- 완성 판정: `hait ask ... --verify` 문장마다 ✅/⚠.

## 카드 4-2: 거절 규칙 + 모르는 질문셋 30
- 모델 추출 실패 + 최고 점수 임계값 미만 → 거절 응답. 스펙·가격 없으면 항목별 "없음". `data/eval/unknown_v1.jsonl`.

## 카드 4-3: 전파인증 + 리콜
- `ingest/rra.py`: CSV 내려받아 `rra_devices`, 모델번호·신청인으로 모델 사전 매칭(못 맞추면 NULL). `hait new-devices --brand apple` 최근 30건.
- `ingest/recall.py`: 소비자24 가전·정보통신 목록 페이지 파싱 → `recalls`, 제품명 퍼지 매칭.
- 완성 판정: 두 명령이 표를 찍고, 매칭률이 `progress.md`에.

## 카드 4-4: 답변 평가 + 리포트 v1
- `eval/answers.py`(환각률, 인용 정확도, 거절률, 지연, 수동 채점 일치율), 수동셋 50개 완성(정답 구간 포함), 스펙 20케이스·가격 10케이스 채점 스크립트. `hait eval all` → 리포트 하나.

## 카드 4-5: 영상 400개 + Whisper 폴백
- `ingest/whisper.py`: `transcribe(video_id, lang) -> list[Segment]`(faster-whisper, `word_timestamps=True`). `no_captions`일 때만.

---

# 5주차: RAFT 합성 + LoRA 학습

**태그:** `v0.5-week5`

## 카드 5-1: RAFT 데이터 합성
- `finetune/synth_raft.py`: `make_example(question, store, teacher, n_distract=2, seed) -> dict`(mlx-lm chat 형식. 근거 2~3 + 무관 2~3, 한·영 섞임, 교사 답변은 한국어·[n] 인용, 4-1 검증 통과한 것만), `synth(n, out, seed)`. train 2000+, valid 200.

## 카드 5-2: LoRA 학습
- `finetune/train.sh`: `mlx_lm.lora --model <학생 4-bit> --train --data data/finetune --iters 500~1000 --batch-size 4 --num-layers 16 --adapter-path adapters/`. 손실 곡선 `reports/lora_loss.png`.
- 완성 판정: 검증 손실 하강, `mlx_lm.generate --adapter-path` 응답.

---

# 6주차: LoRA 전후 평가 + 임베딩 파인튜닝

**태그:** `v0.6-week6`

## 카드 6-1: LoRA 전후 평가
- `mlx_lm.server --adapter-path adapters/`. `hait eval answers --tag base|lora` → 표. 나빠지면 나빠진 대로 + 원인 가설 3줄.

## 카드 6-2: 임베딩 파인튜닝
- `finetune/synth_pairs.py`: 평가셋과 **겹치지 않는 청크**에서 (한국어 질문, 청크) 쌍 3~5천 합성(영어 청크 포함). `finetune/train_embed.py`: sentence-transformers `MultipleNegativesRankingLoss`, fp16, 배치 8, 하위 레이어 동결(상위 4~6층만 학습). 결과 `models/bge-m3-hait/`.
- `hait eval retrieval --embedder models/bge-m3-hait` → 전후 표(한→한, 한→영).
- 함정: 16GB. OOM이면 상위 2층만 또는 peft LoRA.

---

# 7주차: 프레임 검색 + OCR 모델 확정

**태그:** `v0.7-week7`

## 카드 7-1: 프레임 추출 + 색인
- `index/frames.py`: `extract_frames(video_id, scene_th=0.3, min_gap=5, max_frames=120) -> list[tuple[float, Path]]`(ffmpeg `select='gt(scene,0.3)'`, 썸네일 320px, `data/raw/frames/{video_id}/{t}.jpg`), `ImageEmbedder(SigLIP).encode_images / encode_texts`, `VectorStore.ensure_frames_collection / upsert_frames / search_frames`.
- 완성 판정: 100개 영상 프레임 색인, `frames` 카운트.

## 카드 7-2: 시각 질의 검색 + 융합
- `vision/search.py`: `search_frames(query, k, translate=False) -> list[FrameHit]`(FrameHit: video_id, t, thumb_path, score, payload), `fuse_with_chunks(frame_hits, chunk_hits, window=30)`.
- 평가: 시각 질의 30개 수동 라벨 → ±10초 Recall@5, 번역 옵션 유무 비교.
- 완성 판정: `hait frames "카메라 비교 장면"` → 썸네일 경로 + 링크 5개.

## 카드 7-3: OCR 모델 확정
- `vision/ocr.py`: `ocr_lines(image) -> list[str]`(PaddleOCR/EasyOCR 샘플 비교 후 선택), `match_model(lines) -> Identification`(status confirmed|candidates|unreadable, model_id, confidence, candidates), 퍼지 매칭 + 모델번호 정규식.
- 평가: 직접 찍은 박스·설정화면 20장 → 확정 정확도·확정률.
- 완성 판정: `hait identify photo.jpg`.

---

# 8주차: API·UI·배포·README·영상

**태그:** `v0.8-week8`, `v1.0`

## 카드 8-1: FastAPI 10개 (설계 12.1). `/ask` SSE.
## 카드 8-2: Streamlit (설계 12.2). LLM fail → 근거 발췌 모드 배지. 신제품 레이더 탭.
## 카드 8-3: 배포 1층. compose에 api·ui, Supabase Postgres, 컨테이너 내 Qdrant, HF Spaces 또는 Fly.io. `RERANK=0` 옵션.
## 카드 8-4: README(문제 정의, 구조도, 비교표 4개, 재현 5명령, 한계·다음 단계, 채널·데이터 출처), 시연 영상 3분. `v1.0`.
## 카드 8-5 (여유 시): LangChain + pgvector 비교 `framework/lc_pipeline.py`, 비교표.

---

## 자르는 순서 (밀릴 때)
1. 8-5 LangChain 비교 → 2. 4-3 전파인증·리콜 → 3. 6-2 임베딩 파인튜닝을 상위 2층·1천 쌍으로 축소 → 4. 배포 2층 → 5. 영상 400 → 250. LoRA(5-1, 5-2, 6-1)와 평가표는 끝까지.
