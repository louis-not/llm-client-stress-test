#!/usr/bin/env python3

import asyncio
import aiohttp
import json
import csv
import time
import os
import pickle
from datetime import datetime
from typing import Dict, List
import statistics
import argparse
import sys
from dotenv import load_dotenv
from datasets import load_dataset
import tiktoken
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

class VLLMTimeSliceStressTester:
    def __init__(self):
        load_dotenv()
        
        # Load environment variables for vLLM endpoints
        self.benchmark_url = os.getenv('BENCHMARK_URL', '')
        self.timesliced_url = os.getenv('TIMESLICED_URL', '') 
        self.benchmark_model = os.getenv('BENCHMARK_MODEL', 'Qwen/Qwen3-4B')
        self.timesliced_model = os.getenv('TIMESLICED_MODEL', 'Qwen/Qwen3-4B')
        self.benchmark_api_key = os.getenv('BENCHMARK_API_KEY', '')
        self.timesliced_api_key = os.getenv('TIMESLICED_API_KEY', '')
        
        if not self.benchmark_url or not self.timesliced_url:
            raise ValueError("Please set BENCHMARK_URL and TIMESLICED_URL in .env file")
        
        self.endpoints = {
            'benchmark': {
                'url': self.benchmark_url,
                'model': self.benchmark_model,
                'headers': {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.benchmark_api_key}' if self.benchmark_api_key else None
                }
            },
            'timesliced': {
                'url': self.timesliced_url,
                'model': self.timesliced_model,
                'headers': {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.timesliced_api_key}' if self.timesliced_api_key else None
                }
            }
        }
        
        # Clean headers if no auth
        for endpoint in self.endpoints.values():
            if not endpoint['headers']['Authorization']:
                del endpoint['headers']['Authorization']
        
        # Test configuration
        self.concurrency_levels = [1, 2, 5, 10, 20, 50]
        self.requests_per_level = 100
        self.warmup_requests = 10
        self.timeout_seconds = 60
        
        # Dataset configuration
        self.dataset_name = "IzzulGod/indonesian-conversation"
        self.cache_dir = "./cache"
        self.processed_data_file = os.path.join(self.cache_dir, "processed_conversations.pkl")
        
        # Model configuration
        self.max_tokens = 150
        self.temperature = 0.7
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None
        
        self.experiment_payloads = []
        self.results = []
        
        os.makedirs(self.cache_dir, exist_ok=True)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: rough estimation
            return int(len(text.split()) * 1.3)

    def load_and_preprocess_data(self) -> List[str]:
        """Load dataset and preprocess by removing last assistant messages"""
        if os.path.exists(self.processed_data_file):
            print(f"[CACHE] Loading processed data from {self.processed_data_file}")
            with open(self.processed_data_file, 'rb') as f:
                return pickle.load(f)
        
        print(f"[DATASET] Downloading and processing {self.dataset_name}")
        dataset = load_dataset(self.dataset_name, split='train')
        
        processed_prompts = []
        
        for item in tqdm(dataset, desc="Processing conversations"):
            messages = item['messages']
            
            # Remove the last assistant message to create realistic prompts
            if len(messages) >= 2 and messages[-1]['role'] == 'assistant':
                # Create conversation context without the final assistant response
                conversation_context = []
                for msg in messages[:-1]:
                    conversation_context.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
                
                # Convert to a format suitable for API calls
                if conversation_context:
                    processed_prompts.append(conversation_context)
        
        print(f"[DATASET] Processed {len(processed_prompts)} conversation prompts")
        
        # Cache the processed data
        with open(self.processed_data_file, 'wb') as f:
            pickle.dump(processed_prompts, f)
        
        return processed_prompts

    def create_experiment_payloads(self, prompts: List[List[Dict]]) -> None:
        """Create preloaded experiment payloads for consistent testing"""
        print(f"[EXPERIMENT] Creating experiment payloads from {len(prompts)} prompts")
        
        total_needed = sum(self.requests_per_level for _ in self.concurrency_levels) * 2  # *2 for both endpoints
        
        # Create payloads by cycling through prompts
        self.experiment_payloads = []
        for i in tqdm(range(total_needed), desc="Creating experiment payloads"):
            prompt = prompts[i % len(prompts)]
            
            payload = {
                "messages": prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": False
            }
            
            # Calculate input tokens
            input_text = " ".join([msg['content'] for msg in prompt])
            input_tokens = self.count_tokens(input_text)
            
            self.experiment_payloads.append({
                'payload': payload,
                'input_tokens': input_tokens,
                'input_text': input_text
            })
        
        print(f"[EXPERIMENT] Created {len(self.experiment_payloads)} experiment payloads")

    async def make_request(self, session: aiohttp.ClientSession, endpoint: str, 
                          experiment_payload: Dict, request_id: int) -> Dict:
        """Make a single request to vLLM endpoint"""
        start_time = time.time()
        endpoint_config = self.endpoints[endpoint]
        
        url = f"{endpoint_config['url']}/v1/chat/completions"
        
        # Add model to payload for this specific endpoint
        payload_with_model = experiment_payload['payload'].copy()
        payload_with_model['model'] = endpoint_config['model']
        
        try:
            async with session.post(
                url,
                json=payload_with_model,
                headers=endpoint_config['headers'],
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
            ) as response:
                first_byte_time = time.time()
                response_data = await response.json()
                end_time = time.time()
                
                total_response_time = (end_time - start_time) * 1000
                time_to_first_byte = (first_byte_time - start_time) * 1000
                
                if response.status == 200:
                    # Extract token information
                    usage = response_data.get('usage', {})
                    completion_tokens = usage.get('completion_tokens', 0)
                    prompt_tokens = usage.get('prompt_tokens', experiment_payload['input_tokens'])
                    total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
                    
                    # Calculate token-level metrics
                    tokens_per_second = completion_tokens / (total_response_time / 1000) if total_response_time > 0 else 0
                    
                    # Extract generated text for analysis
                    generated_text = ""
                    if 'choices' in response_data and response_data['choices']:
                        generated_text = response_data['choices'][0].get('message', {}).get('content', '')
                    
                    return {
                        'success': True,
                        'request_id': request_id,
                        'endpoint': endpoint,
                        'total_response_time': total_response_time,
                        'time_to_first_byte': time_to_first_byte,
                        'status_code': response.status,
                        'error': None,
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'total_tokens': total_tokens,
                        'tokens_per_second': tokens_per_second,
                        'input_tokens': experiment_payload['input_tokens'],
                        'generated_text': generated_text,
                        'generated_tokens_actual': self.count_tokens(generated_text) if generated_text else 0,
                        'response_data': response_data
                    }
                else:
                    error_text = await response.text()
                    return {
                        'success': False,
                        'request_id': request_id,
                        'endpoint': endpoint,
                        'total_response_time': total_response_time,
                        'time_to_first_byte': time_to_first_byte,
                        'status_code': response.status,
                        'error': f"HTTP {response.status}: {error_text}",
                        'prompt_tokens': experiment_payload['input_tokens'],
                        'completion_tokens': 0,
                        'total_tokens': experiment_payload['input_tokens'],
                        'tokens_per_second': 0,
                        'input_tokens': experiment_payload['input_tokens'],
                        'generated_text': '',
                        'generated_tokens_actual': 0,
                        'response_data': None
                    }
                    
        except asyncio.TimeoutError:
            return {
                'success': False,
                'request_id': request_id,
                'endpoint': endpoint,
                'total_response_time': self.timeout_seconds * 1000,
                'time_to_first_byte': self.timeout_seconds * 1000,
                'status_code': 0,
                'error': 'Request timeout',
                'prompt_tokens': experiment_payload['input_tokens'],
                'completion_tokens': 0,
                'total_tokens': experiment_payload['input_tokens'],
                'tokens_per_second': 0,
                'input_tokens': experiment_payload['input_tokens'],
                'generated_text': '',
                'generated_tokens_actual': 0,
                'response_data': None
            }
        except Exception as e:
            return {
                'success': False,
                'request_id': request_id,
                'endpoint': endpoint,
                'total_response_time': (time.time() - start_time) * 1000,
                'time_to_first_byte': 0,
                'status_code': 0,
                'error': str(e),
                'prompt_tokens': experiment_payload['input_tokens'],
                'completion_tokens': 0,
                'total_tokens': experiment_payload['input_tokens'],
                'tokens_per_second': 0,
                'input_tokens': experiment_payload['input_tokens'],
                'generated_text': '',
                'generated_tokens_actual': 0,
                'response_data': None
            }

    async def run_concurrent_requests(self, session: aiohttp.ClientSession, endpoint: str,
                                    concurrency: int, total_requests: int, 
                                    payload_offset: int) -> List[Dict]:
        """Run concurrent requests for a specific load level"""
        print(f"[CONCURRENT] {endpoint.upper()}: {concurrency} workers, {total_requests} requests")
        
        results = []
        request_queue = asyncio.Queue()
        
        # Create progress bar for this concurrency level
        pbar = tqdm(total=total_requests, desc=f"{endpoint.upper()} C{concurrency}", 
                   unit="req", leave=False)
        
        # Fill the queue with request tasks
        for i in range(total_requests):
            payload_idx = (payload_offset + i) % len(self.experiment_payloads)
            await request_queue.put((i, self.experiment_payloads[payload_idx]))
        
        async def worker(_: int):
            while not request_queue.empty():
                try:
                    request_id, experiment_payload = await asyncio.wait_for(request_queue.get(), timeout=0.1)
                    result = await self.make_request(session, endpoint, experiment_payload, request_id)
                    results.append(result)
                    pbar.update(1)
                    request_queue.task_done()
                except asyncio.TimeoutError:
                    break
        
        # Start workers
        workers = [worker(i) for i in range(concurrency)]
        start_time = time.time()
        await asyncio.gather(*workers)
        test_duration = time.time() - start_time
        
        pbar.close()
        
        # Add test metadata to results
        for result in results:
            result['concurrency_level'] = concurrency
            result['test_duration'] = test_duration
        
        return results

    async def warmup(self, session: aiohttp.ClientSession, endpoint: str):
        """Perform warmup requests"""
        print(f"[WARMUP] {endpoint.upper()}: {self.warmup_requests} requests")
        warmup_results = await self.run_concurrent_requests(
            session, endpoint, 1, self.warmup_requests, 0
        )
        successful = sum(1 for r in warmup_results if r['success'])
        print(f"[WARMUP] {endpoint.upper()}: {successful}/{self.warmup_requests} successful")

    async def test_endpoint(self, endpoint: str) -> List[Dict]:
        """Test a single endpoint across all concurrency levels"""
        print(f"\n[ENDPOINT] Testing {endpoint.upper()}")
        print(f"[ENDPOINT] URL: {self.endpoints[endpoint]['url']}")
        
        endpoint_results = []
        payload_offset = 0
        
        connector = aiohttp.TCPConnector(limit=1000, limit_per_host=1000)
        async with aiohttp.ClientSession(connector=connector) as session:
            await self.warmup(session, endpoint)
            
            for concurrency in tqdm(self.concurrency_levels, desc=f"Testing {endpoint.upper()}", leave=True):
                print(f"\n[TEST] {endpoint.upper()}: Concurrency {concurrency}")
                
                results = await self.run_concurrent_requests(
                    session, endpoint, concurrency, self.requests_per_level, payload_offset
                )
                
                payload_offset += self.requests_per_level
                endpoint_results.extend(results)
                
                # Calculate and display metrics
                successful = [r for r in results if r['success']]
                failed = [r for r in results if not r['success']]
                
                if successful:
                    response_times = [r['total_response_time'] for r in successful]
                    tokens_per_second = [r['tokens_per_second'] for r in successful]
                    
                    avg_response_time = statistics.mean(response_times)
                    p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times)
                    avg_tokens_per_sec = statistics.mean(tokens_per_second)
                    
                    test_duration = results[0]['test_duration'] if results else 0
                    rps = len(successful) / test_duration if test_duration > 0 else 0
                    
                    print(f"[METRICS] Success: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
                    print(f"[METRICS] Avg Response Time: {avg_response_time:.2f}ms, P95: {p95_response_time:.2f}ms")
                    print(f"[METRICS] RPS: {rps:.2f}, Avg Tokens/sec: {avg_tokens_per_sec:.2f}")
                else:
                    print(f"[METRICS] All requests failed: {len(failed)}")
        
        return endpoint_results

    async def run_sequential_tests(self):
        """Run tests sequentially for both endpoints"""
        print("[MAIN] Starting sequential vLLM Time-Slice Comparison Tests")
        print(f"[CONFIG] Concurrency levels: {self.concurrency_levels}")
        print(f"[CONFIG] Requests per level: {self.requests_per_level}")
        print(f"[CONFIG] Benchmark Model: {self.benchmark_model}")
        print(f"[CONFIG] Timesliced Model: {self.timesliced_model}")
        
        # Test benchmark endpoint first
        print("\n" + "="*80)
        benchmark_results = await self.test_endpoint('benchmark')
        
        # Test time-sliced endpoint second  
        print("\n" + "="*80)
        timesliced_results = await self.test_endpoint('timesliced')
        
        # Combine results
        self.results = benchmark_results + timesliced_results
        print(f"\n[MAIN] All tests completed. Total results: {len(self.results)}")

    def save_results(self, filename_prefix: str = "timeslice_test"):
        """Save results to CSV and JSON files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        csv_filename = f"{output_dir}/{filename_prefix}_{timestamp}.csv"
        json_filename = f"{output_dir}/{filename_prefix}_{timestamp}.json"
        
        # Save CSV
        with open(csv_filename, 'w', newline='') as csvfile:
            if self.results:
                fieldnames = list(self.results[0].keys())
                fieldnames = [f for f in fieldnames if f != 'response_data']  # Exclude large response data
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in tqdm(self.results, desc="Saving CSV", leave=False):
                    row = {k: v for k, v in result.items() if k in fieldnames}
                    writer.writerow(row)
        
        # Save JSON
        results_for_json = []
        for result in tqdm(self.results, desc="Preparing JSON", leave=False):
            result_copy = result.copy()
            result_copy.pop('response_data', None)  # Remove large response data
            results_for_json.append(result_copy)
        
        with open(json_filename, 'w') as jsonfile:
            json.dump(results_for_json, jsonfile, indent=2, default=str)
        
        print("\nResults saved to:")
        print(f"  - CSV: {csv_filename}")
        print(f"  - JSON: {json_filename}")

    def generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        if not self.results:
            print("No results to analyze")
            return
        
        print("\n" + "="*100)
        print("VLLM TIME-SLICING VS BENCHMARK DEPLOYMENT ANALYSIS")
        print("="*100)
        
        # Separate results by endpoint
        benchmark_results = [r for r in self.results if r['endpoint'] == 'benchmark']
        timesliced_results = [r for r in self.results if r['endpoint'] == 'timesliced']
        
        print(f"\n{'Endpoint':<12} {'Concurrency':<12} {'Success %':<10} {'Avg RT (ms)':<12} "
              f"{'P95 RT (ms)':<12} {'RPS':<10} {'Tokens/sec':<12}")
        print("-" * 100)
        
        for endpoint_name, endpoint_results in [('BENCHMARK', benchmark_results), ('TIMESLICED', timesliced_results)]:
            for concurrency in self.concurrency_levels:
                level_results = [r for r in endpoint_results if r['concurrency_level'] == concurrency]
                if level_results:
                    successful = [r for r in level_results if r['success']]
                    success_rate = len(successful) / len(level_results) * 100
                    
                    if successful:
                        response_times = [r['total_response_time'] for r in successful]
                        tokens_per_second = [r['tokens_per_second'] for r in successful]
                        
                        avg_rt = statistics.mean(response_times)
                        p95_rt = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times)
                        avg_tps = statistics.mean(tokens_per_second)
                        
                        test_duration = level_results[0]['test_duration']
                        rps = len(successful) / test_duration if test_duration > 0 else 0
                    else:
                        avg_rt = p95_rt = avg_tps = rps = 0
                    
                    print(f"{endpoint_name:<12} {concurrency:<12} {success_rate:<10.1f} {avg_rt:<12.2f} "
                          f"{p95_rt:<12.2f} {rps:<10.2f} {avg_tps:<12.2f}")
        
        # Performance comparison
        print("\nPERFORMANCE COMPARISON SUMMARY:")
        print("-" * 50)
        
        if benchmark_results and timesliced_results:
            benchmark_successful = [r for r in benchmark_results if r['success']]
            timesliced_successful = [r for r in timesliced_results if r['success']]
            
            if benchmark_successful and timesliced_successful:
                # Overall averages
                benchmark_avg_rt = statistics.mean([r['total_response_time'] for r in benchmark_successful])
                timesliced_avg_rt = statistics.mean([r['total_response_time'] for r in timesliced_successful])
                
                benchmark_avg_tps = statistics.mean([r['tokens_per_second'] for r in benchmark_successful])
                timesliced_avg_tps = statistics.mean([r['tokens_per_second'] for r in timesliced_successful])
                
                # Calculate total throughput
                benchmark_total_duration = sum([r['test_duration'] for r in benchmark_results]) / len(self.concurrency_levels)
                timesliced_total_duration = sum([r['test_duration'] for r in timesliced_results]) / len(self.concurrency_levels)
                
                benchmark_rps = len(benchmark_successful) / benchmark_total_duration if benchmark_total_duration > 0 else 0
                timesliced_rps = len(timesliced_successful) / timesliced_total_duration if timesliced_total_duration > 0 else 0
                
                print("Benchmark Deployment:")
                print(f"  - Average Response Time: {benchmark_avg_rt:.2f}ms")
                print(f"  - Average Tokens/sec: {benchmark_avg_tps:.2f}")
                print(f"  - Overall RPS: {benchmark_rps:.2f}")
                
                print("\nTime-Sliced Deployment:")
                print(f"  - Average Response Time: {timesliced_avg_rt:.2f}ms")  
                print(f"  - Average Tokens/sec: {timesliced_avg_tps:.2f}")
                print(f"  - Overall RPS: {timesliced_rps:.2f}")
                
                # Performance difference analysis
                rt_diff_pct = ((timesliced_avg_rt - benchmark_avg_rt) / benchmark_avg_rt) * 100
                tps_diff_pct = ((timesliced_avg_tps - benchmark_avg_tps) / benchmark_avg_tps) * 100
                rps_diff_pct = ((timesliced_rps - benchmark_rps) / benchmark_rps) * 100
                
                print("\nTime-Slicing Impact:")
                print(f"  - Response Time Change: {rt_diff_pct:+.1f}%")
                print(f"  - Tokens/sec Change: {tps_diff_pct:+.1f}%") 
                print(f"  - RPS Change: {rps_diff_pct:+.1f}%")
                
                if abs(rt_diff_pct) < 10 and abs(tps_diff_pct) < 10:
                    print("  - CONCLUSION: Time-slicing has minimal performance impact (<10%)")
                elif rt_diff_pct > 10 or tps_diff_pct < -10:
                    print("  - CONCLUSION: Time-slicing shows performance degradation")
                else:
                    print("  - CONCLUSION: Mixed performance impact - detailed analysis needed")

async def main():
    parser = argparse.ArgumentParser(description='vLLM Time-Slicing vs Benchmark Stress Test')
    parser.add_argument('--output', '-o', default='vllm_timeslice_comparison',
                       help='Output filename prefix')
    args = parser.parse_args()
    
    tester = VLLMTimeSliceStressTester()
    
    try:
        # Load and preprocess data
        prompts = tester.load_and_preprocess_data()
        if not prompts:
            print("No data loaded. Exiting.")
            return
        
        # Create experiment payloads
        tester.create_experiment_payloads(prompts)
        
        # Run tests
        await tester.run_sequential_tests()
        
        # Save and analyze results
        tester.save_results(args.output)
        tester.generate_analysis_report()
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        if tester.results:
            tester.save_results(args.output)
            tester.generate_analysis_report()
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())