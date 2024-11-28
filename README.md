
# 🫧 **TorchStack** [work in progress]  
**Build scalable ensemble systems for transformer-based models.**

torchstack is a library designed to simplify the creation and deployment of scalable ensemble learning systems for Hugging Face transformers. It provides tools to address challenges like tokenizer mismatch, voting strategies, and model integration, making ensemble learning accessible and efficient for natural language processing tasks.

---

## **🚀 Features**
- **High-Level API**: Simplifies ensemble learning, inspired by Keras for transformers.  
- **Tokenizer Compatibility**: Support for union vocabularies, projections (e.g., DEEPEN), and other solutions to handle tokenizer mismatches.  
- **Flexible Voting Strategies**: Includes average voting, majority voting, and extensible custom strategies.  
- **Integration with Hugging Face**: Seamlessly works with Hugging Face models and tokenizers.  
- **Production-Ready**: Tools for building, testing, and deploying your ensemble systems with ease.  

---

## **📦 Tools and Libraries**

### **Core Tooling**
- **Packaging**: [uv](https://docs.astral.sh/uv)  
- **Linting/Formatting**: [ruff](https://docs.astral.sh/ruff/)  
- **Testing**: [PyTest](https://docs.pytest.org/en/8.2.x/)  
- **Code Coverage**: [coverage.py](https://coverage.readthedocs.io/en/7.5.4/)  
- **Static Code Analysis**: [CodeClimate](https://codeclimate.com/quality)

### **Core Dependencies**
- **[Transformers](https://huggingface.co/transformers/)**: Core library for transformer-based models.  
- **[Torch](https://pytorch.org/)**: Deep learning framework for model integration and training.  
- **[Loguru](https://loguru.readthedocs.io/)**: Advanced logging with rotation, retention, and compression.

---

## **📖 Example Usage**

### **Text Generation**
```bash
poetry run python examples/text-generation/run.py
```

### **Text Classification**
```bash
poetry run python examples/text-classification/run.py
```

### **Running the Service**
- **Development Mode**:  
  ```bash
  uv run
  ```
- **Production Mode**:  
  ```bash
  uv build
  ```

---

## **🛠️ Guides**

- [Packaging Python Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/)  
- [UV Build Process](https://docs.astral.sh/uv/concepts/projects/#build-isolation)

---

## **🔧 Build Process**
The **uv** tool builds a source distribution first, followed by a binary distribution (wheel). You can customize the build process:
- Build only a source distribution:  
  ```bash
  uv build --sdist
  ```
- Build only a binary distribution:  
  ```bash
  uv build --wheel
  ```
- Build both distributions from source:  
  ```bash
  uv build --sdist --wheel
  ```

---

## **⚙️ Build Isolation**
By default, **uv** builds all packages in isolated virtual environments, following [PEP 517](https://peps.python.org/pep-0517/). However, some packages (e.g., PyTorch) may require disabling build isolation. To do so, add the dependency to the `no-build-isolation-package` list in your `pyproject.toml` file.

---

## **📝 Roadmap**
- [ ] Implement remote model integration (`ensemble.add_remote_member`).  
- [ ] Add more voting strategies and tokenization solutions.  
- [ ] Publish and manage ensembles on Hugging Face Model Repository.  
- [ ] Expand documentation with tutorials and advanced examples.

---

## **💬 Contributing**
Contributions are welcome! Feel free to open an issue or submit a pull request. See the [Contributing Guide](CONTRIBUTING.md) for more details.

---

## **📄 License**
This project is licensed under the [MIT License](LICENSE).

---

This revised README focuses on being **engaging**, **informative**, and **structured**, with clear headings, concise descriptions, and actionable examples. Let me know if you’d like further refinements or to add anything specific!