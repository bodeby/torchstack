# consensus (🫧)

A silly little web browser project that I'm working on.

# Tools and Libraries

## Core Tooling

- Buildtool: [Poetry](https://docs.docker.com/compose/)
- Testing: [PyTest](https://docs.pytest.org/en/8.2.x/)
- Coverage: [coverage.py](https://coverage.readthedocs.io/en/7.5.4/)
- Static Code Analysis: [CodeClimate](https://codeclimate.com/quality)

## Core libraries

- **transformers**: async runtime
- **torch**: python redis toolkit
- **loguru**: logging rotation, retention and compression

## Example Usage

- **text-generation**: poetry run python examples/text-generation/run.py
- **text-classification**: peotry run python examples/text-classification/run.py 

```bash
# starts the service with development settings
python app/main.py
```

```bash
# starts the service with production settings
python app/main.py prod
```
