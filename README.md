# consensus (🫧)

A silly little web browser project that I'm working on.

# Tools and Libraries

## Core Tooling

- project managment: [Poetry]](https://docs.docker.com/compose/)
- Deep Learning: [PyTorch](https://docs.docker.com/manuals/)
- Inference: [Transformers](https://huggingface.co/docs/hub/transformers)
- Static Code Analysis: [CodeClimate](https://codeclimate.com/quality)

## Core libraries

- **asyncio**: async runtime
- **redis-py**: python redis toolkit
- **pymongo**: mongodb driver for python
- **loguru**: logging rotation, retention and compression

## Running the service

```bash
# starts the service with development settings
python app/main.py
```

```bash
# starts the service with production settings
python app/main.py prod
```
