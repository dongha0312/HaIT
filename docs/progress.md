# 차봇 진행 상황

세션 시작 시 이 파일부터 읽는다. 설계 원본: `docs/superpowers/specs/2026-09-09-carbot-design.md`

## 완료

- 2026-09-09 설계 문서 v1 작성.
- 2026-09-09 GitHub 공개 저장소 생성(dongha0312/carbot).
- 2026-09-09 유튜브 채널 5개 선정, 자동 자막 확인.

## 진행 중

- 설계 문서 사용자 검토.

## 다음

- 1주차 과제 카드: 환경 세팅 → 영상 10개 수집 → 청킹 → 색인 → 검색 CLI.

## 결정 기록

- 2026-09-09: 도메인 = 자동차(사진·리뷰·유지비). 모델 = Qwen 계열(MLX). 정형 DB = Postgres. 벡터DB = Qdrant(비교용 pgvector). 파인튜닝 = RAFT LoRA(필수) + 차종 분류 헤드(필수). 링크 오차 목표 8초.
- 2026-09-09: 유튜브 채널 = MOCAR, Motline, Station.B, Woopa TV, BPD. 전부 자동 자막만 있음. Whisper는 예외용.
- 2026-09-09: 깃 운영 = GitHub Flow(main + 주차/모듈 브랜치 + squash PR + 주차 태그). PR 템플릿 추가.
