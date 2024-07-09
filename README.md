# consensus (🪼)

A silly little web browser project that I'm working on.

# Tools and Libraries

## Core Tooling

- Orchestration: [Docker Compose](https://docs.docker.com/compose/)
- Containerization: [Docker](https://docs.docker.com/manuals/)
- Static Code Analysis: [CodeClimate](https://codeclimate.com/quality)

## Core libraries

- **asyncio**: async runtime
- **redis-py**: python redis toolkit
- **pymongo**: mongodb driver for python
- **loguru**: logging rotation, retention and compression

## Running the service

```bash
# starts the service with development settigns
python app/main.py
```

```bash
# starts the service with production settigns
python app/main.py prod
```
