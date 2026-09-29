.PHONY: dev test test-frontend test-rust build

dev:
	npm run tauri dev

test: test-frontend test-rust

test-frontend:
	npm run typecheck

test-rust:
	cd src-tauri && cargo test

build:
	npm run tauri build
