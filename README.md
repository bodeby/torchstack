# 🫧 consensus [WIP]

build scaleable ensemble systems for transformer based models.

# Tools and Libraries

## Core Tooling

- Packaging : [uv](https://docs.astral.sh/uv)
- Linting / Formatting : [ruff](https://docs.astral.sh/ruff/)
- Testing: [PyTest](https://docs.pytest.org/en/8.2.x/)
- Coverage: [coverage.py](https://coverage.readthedocs.io/en/7.5.4/)
- Static Code Analysis: [CodeClimate](https://codeclimate.com/quality)

## Core dependencies

- **transformers**: async runtime
- **torch**: python redis toolkit
- **loguru**: logging rotation, retention and compression

## Running the service

```bash
# starts the service with development settings
uv run
```

```bash
# starts the service with production settings
uv build
```

## guides

- https://packaging.python.org/en/latest/tutorials/packaging-projects/

## uv build process

https://docs.astral.sh/uv/concepts/projects/#build-isolation

**uv** build will first build a source distribution, and then build a binary distribution (wheel) from that source distribution.

You can limit uv build to building a source distribution with uv build --sdist, a binary distribution with uv build --wheel, or build both distributions from source with uv build --sdist --wheel.

## Build isolation

By default, uv builds all packages in isolated virtual environments, as per PEP 517. Some packages are incompatible with build isolation, be it intentionally (e.g., due to the use of heavy build dependencies, mostly commonly PyTorch) or unintentionally (e.g., due to the use of legacy packaging setups).

To disable build isolation for a specific dependency, add it to the no-build-isolation-package list in your pyproject.toml: