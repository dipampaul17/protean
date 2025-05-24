# Contributing to Protean

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dipampaul17/protean.git
   cd protean
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run tests**:
   ```bash
   python -m pytest tests/
   ```

## Architecture Guidelines

- Follow the modular design in `protean/` directory
- Use type hints for all function signatures
- Add docstrings for public APIs
- Follow PEP 8 style guidelines

## Testing

- Add unit tests for new functionality in `tests/unit/`
- Integration tests go in `tests/integration/`
- Run validation: `python simple_validation.py`

## Documentation

- Update `README.md` for user-facing changes
- Add technical details to `docs/ARCHITECTURE.md`
- Include examples in `examples/` directory

## Code Review Process

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request
5. Address review feedback

## Commit Messages

Use clear, descriptive commit messages:
- feat: add new pattern discovery algorithm
- fix: resolve memory leak in graph construction
- docs: update installation instructions
- test: add validation for circuit breaker patterns

## Performance Considerations

- Profile code with large datasets
- Maintain sub-second response times for validation
- Keep model sizes under 10MB for production deployment 