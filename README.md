# LLM Framework Stress Testing

A comprehensive stress testing framework to compare Ollama and vLLM deployment frameworks for production readiness.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the stress test:

```bash
python stress_test.py
```

Specify custom output filename:

```bash
python stress_test.py --output my_test_results
```

## Test Configuration

- **Concurrency Levels**: 1, 5, 10, 25, 50, 100, 200, 500
- **Requests per Level**: 100 total requests
- **Warmup**: 10 requests before each level
- **Timeout**: 30 seconds per request
- **Indonesian Test Prompts**: 5 standardized prompts

## Output Files

The script generates:
- `llm_stress_test_YYYYMMDD_HHMMSS.csv` - Performance metrics table
- `llm_stress_test_YYYYMMDD_HHMMSS.json` - Raw test data with error details

## Test Results

The framework evaluates:
- Maximum concurrent request handling capacity
- Response time under varying load conditions  
- Throughput performance (requests per second)
- Error rates and stability under stress
- Resource efficiency and scalability characteristics

## Performance Metrics

- **Concurrency Level**: Number of simultaneous requests
- **Framework**: Ollama vs vLLM
- **Average Response Time**: Mean response time in milliseconds
- **95th Percentile Response Time**: P95 response time in milliseconds  
- **Requests Per Second (RPS)**: Throughput measurement
- **Success Rate**: Percentage of successful requests
- **Error Count & Types**: Failed requests categorized by error type

## Production Recommendations

The framework automatically generates:
- Performance comparison between frameworks
- Scalability analysis and breaking point identification
- Optimal concurrency recommendations
- Suggested scaling strategies for production deployment