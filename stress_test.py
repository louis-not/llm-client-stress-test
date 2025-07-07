#!/usr/bin/env python3

import asyncio
import aiohttp
import json
import csv
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import statistics
import argparse
import sys
import os
from dotenv import load_dotenv

class LLMStressTester:
    def __init__(self):
        load_dotenv()
        
        ollama_url = os.getenv('OLLAMA_URL', '')
        ollama_auth = os.getenv('OLLAMA_AUTH', '')
        vllm_url = os.getenv('VLLM_URL', '')
        vllm_auth = os.getenv('VLLM_AUTH', '')
        
        self.endpoints = {
            'ollama': {
                'url': ollama_url,
                'headers': {'Content-Type': 'application/json'},
                'auth': ollama_auth if ollama_auth else None
            },
            'vllm': {
                'url': vllm_url,
                'headers': {
                    'Content-Type': 'application/json',
                    'Authorization': vllm_auth
                },
                'auth': vllm_auth
            }
        }
        
        self.test_prompts = [
            "Kamu siapa ya?",
            "Kamu bisa bantuin aku apa sih?",
            "Jelaskan tentang cuaca hari ini",
            "Bagaimana cara memasak nasi goreng?",
            "Apa yang kamu ketahui tentang Indonesia?"
        ]
        
        self.concurrency_levels = [1, 5, 10, 25, 50, 100, 200, 500]
        self.requests_per_level = 100
        self.warmup_requests = 10
        self.timeout_seconds = 30
        
        self.results = []

    async def make_request(self, session: aiohttp.ClientSession, endpoint: str, prompt: str) -> Dict:
        """Make a single request to an endpoint"""
        start_time = time.time()
        endpoint_config = self.endpoints[endpoint]
        
        try:
            payload = {
                "model": "qwen3:4b" if endpoint == "ollama" else "Qwen/Qwen3-4B",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,
                "temperature": 0.7
            }
            
            if endpoint == "ollama":
                url = f"{endpoint_config['url']}/chat/completions"
            else:
                url = f"{endpoint_config['url']}/v1/chat/completions"
            
            print(f"[REQUEST] {endpoint.upper()} - Sending request to: {url}")
            print(f"[REQUEST] Prompt: {prompt[:50]}...")
            
            async with session.post(
                url,
                json=payload,
                headers=endpoint_config['headers'],
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
            ) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    print(f"[SUCCESS] {endpoint.upper()} - Response received in {response_time:.2f}ms")
                    return {
                        'success': True,
                        'response_time': response_time,
                        'status_code': response.status,
                        'error': None,
                        'response_data': data
                    }
                else:
                    error_text = await response.text()
                    print(f"[ERROR] {endpoint.upper()} - HTTP {response.status}: {error_text[:100]}...")
                    return {
                        'success': False,
                        'response_time': response_time,
                        'status_code': response.status,
                        'error': f"HTTP {response.status}: {error_text}",
                        'response_data': None
                    }
                    
        except asyncio.TimeoutError:
            print(f"[TIMEOUT] {endpoint.upper()} - Request timed out after {self.timeout_seconds}s")
            return {
                'success': False,
                'response_time': self.timeout_seconds * 1000,
                'status_code': 0,
                'error': 'Request timeout',
                'response_data': None
            }
        except Exception as e:
            print(f"[EXCEPTION] {endpoint.upper()} - {str(e)}")
            return {
                'success': False,
                'response_time': (time.time() - start_time) * 1000,
                'status_code': 0,
                'error': str(e),
                'response_data': None
            }

    async def run_concurrent_requests(self, session: aiohttp.ClientSession, endpoint: str, 
                                    concurrency: int, total_requests: int) -> List[Dict]:
        """Run concurrent requests for a specific load level"""
        print(f"[CONCURRENT] Starting {concurrency} concurrent workers for {total_requests} requests on {endpoint.upper()}")
        results = []
        prompt_index = 0
        
        async def worker(worker_id: int):
            nonlocal prompt_index
            worker_requests = 0
            while len(results) < total_requests:
                if len(results) >= total_requests:
                    break
                    
                current_prompt = self.test_prompts[prompt_index % len(self.test_prompts)]
                prompt_index += 1
                worker_requests += 1
                
                print(f"[WORKER-{worker_id}] Starting request #{worker_requests} (total: {len(results)+1}/{total_requests})")
                result = await self.make_request(session, endpoint, current_prompt)
                results.append(result)
                print(f"[WORKER-{worker_id}] Completed request #{worker_requests} - {'SUCCESS' if result['success'] else 'FAILED'}")
        
        tasks = [worker(i) for i in range(min(concurrency, total_requests))]
        await asyncio.gather(*tasks)
        
        print(f"[CONCURRENT] Completed all {len(results)} requests for {endpoint.upper()}")
        return results[:total_requests]

    async def warmup(self, session: aiohttp.ClientSession, endpoint: str):
        """Perform warmup requests"""
        print(f"[WARMUP] Starting warmup for {endpoint.upper()} with {self.warmup_requests} requests...")
        warmup_start = time.time()
        warmup_results = await self.run_concurrent_requests(
            session, endpoint, 1, self.warmup_requests
        )
        warmup_duration = time.time() - warmup_start
        successful_warmups = sum(1 for r in warmup_results if r['success'])
        print(f"[WARMUP] Completed in {warmup_duration:.2f}s: {successful_warmups}/{self.warmup_requests} successful")

    async def test_endpoint(self, endpoint: str) -> List[Dict]:
        """Test a single endpoint across all concurrency levels"""
        print(f"\n[ENDPOINT] Starting test for {endpoint.upper()} framework...")
        print(f"[ENDPOINT] Target URL: {self.endpoints[endpoint]['url']}")
        endpoint_results = []
        
        connector = aiohttp.TCPConnector(limit=1000, limit_per_host=1000)
        async with aiohttp.ClientSession(connector=connector) as session:
            await self.warmup(session, endpoint)
            
            for concurrency in self.concurrency_levels:
                print(f"\n[CONCURRENCY] Testing concurrency level: {concurrency}")
                print(f"[CONCURRENCY] Will send {self.requests_per_level} requests with {concurrency} concurrent workers")
                
                start_time = time.time()
                results = await self.run_concurrent_requests(
                    session, endpoint, concurrency, self.requests_per_level
                )
                end_time = time.time()
                
                test_duration = end_time - start_time
                successful_requests = [r for r in results if r['success']]
                failed_requests = [r for r in results if not r['success']]
                
                print(f"[RESULTS] Test completed in {test_duration:.2f}s")
                print(f"[RESULTS] Successful: {len(successful_requests)}, Failed: {len(failed_requests)}")
                
                if successful_requests:
                    response_times = [r['response_time'] for r in successful_requests]
                    avg_response_time = statistics.mean(response_times)
                    p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times)
                    rps = len(successful_requests) / test_duration
                    print(f"[METRICS] Avg Response Time: {avg_response_time:.2f}ms, P95: {p95_response_time:.2f}ms")
                else:
                    avg_response_time = 0
                    p95_response_time = 0
                    rps = 0
                    print(f"[METRICS] No successful requests to analyze")
                
                success_rate = (len(successful_requests) / len(results)) * 100
                print(f"[METRICS] RPS: {rps:.2f}, Success Rate: {success_rate:.1f}%")
                
                error_summary = {}
                for failed in failed_requests:
                    error_type = failed['error'] or 'Unknown error'
                    error_summary[error_type] = error_summary.get(error_type, 0) + 1
                
                if error_summary:
                    print(f"[ERRORS] Error breakdown:")
                    for error_type, count in error_summary.items():
                        print(f"[ERRORS]   {error_type}: {count} occurrences")
                
                result = {
                    'concurrency_level': concurrency,
                    'framework': endpoint,
                    'avg_response_time': avg_response_time,
                    'p95_response_time': p95_response_time,
                    'rps': rps,
                    'success_rate': success_rate,
                    'total_requests': len(results),
                    'successful_requests': len(successful_requests),
                    'failed_requests': len(failed_requests),
                    'error_summary': error_summary,
                    'test_duration': test_duration
                }
                
                endpoint_results.append(result)
        
        print(f"\n[ENDPOINT] Completed all tests for {endpoint.upper()} framework")
        return endpoint_results

    async def run_all_tests(self):
        """Run tests for both endpoints concurrently"""
        print("[MAIN] Starting LLM Framework Stress Testing...")
        print(f"[CONFIG] Test configuration:")
        print(f"[CONFIG]   - Concurrency levels: {self.concurrency_levels}")
        print(f"[CONFIG]   - Requests per level: {self.requests_per_level}")
        print(f"[CONFIG]   - Warmup requests: {self.warmup_requests}")
        print(f"[CONFIG]   - Timeout: {self.timeout_seconds}s")
        print(f"[CONFIG]   - Test prompts: {len(self.test_prompts)} prompts")
        print(f"[CONFIG]   - Running both frameworks CONCURRENTLY for faster testing")
        
        # Run both endpoints concurrently using asyncio.gather
        print(f"\n[MAIN] Starting concurrent tests for both OLLAMA and VLLM endpoints...")
        
        async def test_with_error_handling(endpoint):
            try:
                print(f"[MAIN] Starting tests for {endpoint.upper()} endpoint...")
                results = await self.test_endpoint(endpoint)
                print(f"[MAIN] Completed tests for {endpoint.upper()} endpoint")
                return results
            except Exception as e:
                print(f"[ERROR] Error testing {endpoint}: {e}")
                import traceback
                traceback.print_exc()
                return []
        
        # Run both frameworks concurrently
        results_list = await asyncio.gather(
            test_with_error_handling('ollama'),
            test_with_error_handling('vllm'),
            return_exceptions=True
        )
        
        # Collect results from both endpoints
        for results in results_list:
            if isinstance(results, list):
                self.results.extend(results)
            else:
                print(f"[ERROR] Unexpected result type: {type(results)}")
        
        print(f"\n[MAIN] All tests completed. Total results: {len(self.results)}")

    def save_results(self, filename_prefix: str = "test"):
        """Save results to CSV and JSON files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        import os
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        csv_filename = f"{output_dir}/{filename_prefix}_{timestamp}.csv"
        json_filename = f"{output_dir}/{filename_prefix}_{timestamp}.json"
        
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = [
                'concurrency_level', 'framework', 'avg_response_time', 
                'p95_response_time', 'rps', 'success_rate', 'total_requests',
                'successful_requests', 'failed_requests', 'test_duration'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {k: v for k, v in result.items() if k in fieldnames}
                writer.writerow(row)
        
        with open(json_filename, 'w') as jsonfile:
            json.dump(self.results, jsonfile, indent=2, default=str)
        
        print(f"\nResults saved to:")
        print(f"  - CSV: {csv_filename}")
        print(f"  - JSON: {json_filename}")

    def generate_report(self):
        """Generate a performance comparison report"""
        if not self.results:
            print("No results to analyze")
            return
        
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON REPORT")
        print("="*80)
        
        print(f"\n{'Concurrency':<12} {'Framework':<10} {'Avg RT (ms)':<12} {'P95 RT (ms)':<12} "
              f"{'RPS':<10} {'Success %':<10} {'Errors':<10}")
        print("-" * 80)
        
        for result in self.results:
            error_count = result['failed_requests']
            print(f"{result['concurrency_level']:<12} {result['framework']:<10} "
                  f"{result['avg_response_time']:<12.2f} {result['p95_response_time']:<12.2f} "
                  f"{result['rps']:<10.2f} {result['success_rate']:<10.1f} {error_count:<10}")
        
        print("\nSCALABILITY ANALYSIS:")
        print("-" * 40)
        
        for framework in ['ollama', 'vllm']:
            framework_results = [r for r in self.results if r['framework'] == framework]
            if framework_results:
                print(f"\n{framework.upper()} Framework:")
                
                max_rps = max(r['rps'] for r in framework_results)
                best_concurrency = next(r['concurrency_level'] for r in framework_results if r['rps'] == max_rps)
                
                breaking_point = None
                for result in framework_results:
                    if result['success_rate'] < 90:
                        breaking_point = result['concurrency_level']
                        break
                
                print(f"  - Peak RPS: {max_rps:.2f} at concurrency {best_concurrency}")
                if breaking_point:
                    print(f"  - Breaking point: {breaking_point} concurrent requests")
                else:
                    print(f"  - No breaking point found (success rate >90% at all levels)")
        
        print("\nPRODUCTION RECOMMENDATIONS:")
        print("-" * 40)
        
        ollama_results = [r for r in self.results if r['framework'] == 'ollama']
        vllm_results = [r for r in self.results if r['framework'] == 'vllm']
        
        if ollama_results and vllm_results:
            ollama_avg_rps = statistics.mean([r['rps'] for r in ollama_results])
            vllm_avg_rps = statistics.mean([r['rps'] for r in vllm_results])
            
            ollama_avg_rt = statistics.mean([r['avg_response_time'] for r in ollama_results])
            vllm_avg_rt = statistics.mean([r['avg_response_time'] for r in vllm_results])
            
            better_throughput = "vLLM" if vllm_avg_rps > ollama_avg_rps else "Ollama"
            better_latency = "vLLM" if vllm_avg_rt < ollama_avg_rt else "Ollama"
            
            print(f"- Better throughput: {better_throughput} (avg RPS: {max(ollama_avg_rps, vllm_avg_rps):.2f})")
            print(f"- Better latency: {better_latency} (avg RT: {min(ollama_avg_rt, vllm_avg_rt):.2f}ms)")
            
            if vllm_avg_rps > ollama_avg_rps and vllm_avg_rt < ollama_avg_rt:
                print("- Recommendation: vLLM for production (better in both metrics)")
            elif ollama_avg_rps > vllm_avg_rps and ollama_avg_rt < vllm_avg_rt:
                print("- Recommendation: Ollama for production (better in both metrics)")
            else:
                print("- Recommendation: Choose based on priority (throughput vs latency)")

async def main():
    parser = argparse.ArgumentParser(description='LLM Framework Stress Testing')
    parser.add_argument('--output', '-o', default='llm_stress_test', 
                       help='Output filename prefix (default: llm_stress_test)')
    args = parser.parse_args()
    
    tester = LLMStressTester()
    
    try:
        await tester.run_all_tests()
        tester.save_results(args.output)
        tester.generate_report()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        if tester.results:
            tester.save_results(args.output)
            tester.generate_report()
    except Exception as e:
        print(f"Error during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())