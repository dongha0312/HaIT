# HaIT 진행 상황

세션 시작 시 이 파일부터 읽는다. 설계 원본: `docs/superpowers/specs/2026-09-10-hait-design.md`, 구현 계획: `docs/superpowers/plans/2026-09-10-hait-8week-plan.md`

## 완료

- 2026-09-09 설계 v1(자동차 판) 작성, GitHub 저장소 생성, 깃 운영 규칙·PR 템플릿.
- 2026-09-10 도메인을 폰·노트북·데스크탑으로 전환. 데이터(공식 스펙, 스마트초이스 지원금, 전파인증 CSV, 소비자24 리콜) 확인. 채널 8개(한 5, 영 3) 자막 확인. 설계 v2·계획 v2 작성. 저장소 이름 carbot → HaIT.

## 진행 중

- 1주차 카드 1-1 환경 세팅 (브랜치 `week1/setup`).

## 다음

- 카드 1-2~1-6. 전체 카드는 계획 v2 문서.

## 결정 기록

- 2026-09-09: 모델 = Qwen 계열(MLX). 정형 DB = Postgres. 벡터DB = Qdrant. 링크 오차 목표 8초. 깃 = GitHub Flow.
- 2026-09-10: 도메인 = 폰·노트북·데스크탑. 이름 = HaIT(패키지 `hait`). 채널 = 잇섭, underKG, ZUYONI, 뻘짓연구소, 디에디트, MKBHD, Dave2D, Linus Tech Tips (추가는 나중에). 영어 채널은 원문 자막만, 기계번역 자막 미사용. 파인튜닝 = RAFT LoRA + bge-m3 임베딩(차종 분류 헤드는 제외). 비전 = 프레임 검색 + OCR 모델 확정.
