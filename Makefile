.DEFAULT_GOAL := help

export HOST_UID := $(shell id -u)
export HOST_GID := $(shell id -g)

.PHONY: help build env-check quick bench dgemm pde phi-sweep profile cpu-quick cpu-bench cpu-phi-sweep analyze clean-cache

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

build: ## Build the CUDA 12 / CUDA 13.4 / CPU benchmark images
	docker compose build cuda12 cuda13 cpu

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

phi-sweep: ## DGEMM over input exponent ranges phi = 0..4 (-> results-phi/)
	RESULTS_DIR=results-phi SKIP_PDE=1 DGEMM_ARGS="--sizes 2000 4000 8000 --phi 0 0.5 1 2 4" ./run_benchmarks.sh

profile: ## Nsight Systems: which DGEMM kernels run (Ozaki-II / native) (-> results/nsys/)
	./profile_kernels.sh

cpu-quick: ## Smoke test of the C++/Eigen and NumPy CPU baselines (-> results-quick/)
	QUICK=1 ./run_cpu_benchmarks.sh

cpu-bench: ## CPU baselines: C++/Eigen + NumPy, DGEMM + PDE, threads 6/14/16 (-> results/)
	./run_cpu_benchmarks.sh

cpu-phi-sweep: ## CPU DGEMM over phi = 0..4 with the given THREADS (-> results-phi/)
	RESULTS_DIR=results-phi SKIP_PDE=1 THREADS="$${THREADS:-6}" DGEMM_ARGS="--sizes 2000 4000 8000 --phi 0 0.5 1 2 4" ./run_cpu_benchmarks.sh

analyze: ## Regenerate figures and report.md from results/ and results-phi/
	docker compose run --rm -T -e BENCH_CONDITION=analysis cuda13 python src/analyze_results.py --results-dir results
	@if [ -d results-phi ]; then docker compose run --rm -T -e BENCH_CONDITION=analysis cuda13 python src/analyze_results.py --results-dir results-phi; fi

clean-cache: ## Remove cached NumPy reference solutions
	rm -rf cache
