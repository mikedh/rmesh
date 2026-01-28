.PHONY: test compare build clean test-local compare-local

# Build Docker image
build:
	docker build -t rmesh .

# Run all tests in Docker (cargo + pytest)
test: build
	docker run --rm rmesh bash -c "cargo test --package rmesh && PYTHONPATH=. uv run pytest test/"

# Run comparison against trimesh test suite in Docker
compare: build
	docker run --rm rmesh bash -c "PYTHONPATH=. uv run pytest --compare-rmesh test/trimesh/tests/"

# Local variants (for when you know your env is good)
test-local:
	cargo test --package rmesh
	uv pip install -e . && PYTHONPATH=. uv run pytest test/

compare-local:
	@if [ ! -d "test/trimesh" ]; then \
		echo "Cloning trimesh test suite..."; \
		git clone --depth 1 https://github.com/mikedh/trimesh test/trimesh; \
	fi
	uv pip install trimesh[all]
	PYTHONPATH=. uv run pytest --compare-rmesh test/trimesh/tests/

# Clean build artifacts
clean:
	cargo clean
	rm -rf target/ .pytest_cache/ __pycache__/
	rm -f python/rmesh/*.so
	find . -name "*.pyc" -delete
