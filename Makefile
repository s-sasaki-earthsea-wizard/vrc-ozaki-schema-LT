.DEFAULT_GOAL := help

export HOST_UID := $(shell id -u)
export HOST_GID := $(shell id -g)

.PHONY: help build env-check quick bench dgemm pde analyze clean-cache

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

build: ## Build the CUDA 12 / CUDA 13.4 benchmark images
	docker compose build cuda12 cuda13

env-check: ## Print GPU / CUDA / cuBLAS / emulation info for each image
	docker compose run --rm -e BENCH_CONDITION=cuda12 cuda12 python src/common.py
	docker compose run --rm -e BENCH_CONDITION=cuda13-emu -e CUBLAS_EMULATE_DOUBLE_PRECISION=1 cuda13 python src/common.py

quick: ## Smoke test with small sizes for all conditions (-> results-quick/)
	QUICK=1 ./run_benchmarks.sh

bench: ## Full run: DGEMM + PDE for all conditions, then report (-> results/)
	./run_benchmarks.sh

dgemm: ## DGEMM only for all conditions
	SKIP_PDE=1 ./run_benchmarks.sh

pde: ## PDE only for all conditions
	SKIP_DGEMM=1 ./run_benchmarks.sh

analyze: ## Regenerate figures and report.md from results/
	docker compose run --rm -e BENCH_CONDITION=analysis cuda13 python src/analyze_results.py --results-dir results

clean-cache: ## Remove cached NumPy reference solutions
	rm -rf cache
