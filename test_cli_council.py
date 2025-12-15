#!/usr/bin/env python3
"""Test the CLI-based LLM Council with Claude, Codex, and Gemini."""

import asyncio
import sys
import time
from datetime import datetime

# Add backend to path
sys.path.insert(0, '.')

async def test_individual_providers():
    """Test each CLI provider individually."""
    from backend.cli_providers import query_cli_provider, get_available_providers

    print("=" * 60)
    print("Testing Individual CLI Providers")
    print("=" * 60)

    providers = get_available_providers()
    print(f"Available providers: {providers}")

    test_prompt = "In exactly one sentence, what is 2+2?"

    results = {}
    for provider in providers:
        print(f"\n[{provider}] Querying...")
        start = time.time()
        result = await query_cli_provider(provider, test_prompt, timeout=60.0)
        elapsed = time.time() - start

        if result:
            content = result.get('content', '')[:200]
            print(f"  Response ({elapsed:.1f}s): {content}...")
            results[provider] = True
        else:
            print(f"  FAILED ({elapsed:.1f}s)")
            results[provider] = False

    print("\n" + "=" * 60)
    print("Provider Test Results:")
    for provider, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {provider}: {status}")

    return all(results.values())


async def test_parallel_query():
    """Test parallel querying of all providers."""
    from backend.cli_providers import query_providers_parallel

    print("\n" + "=" * 60)
    print("Testing Parallel Query")
    print("=" * 60)

    providers = ["claude", "codex", "gemini"]
    test_prompt = "What is the capital of France? Answer in one word."

    print(f"Querying {len(providers)} providers in parallel...")
    start = time.time()
    results = await query_providers_parallel(providers, test_prompt)
    elapsed = time.time() - start

    print(f"Total time: {elapsed:.1f}s")

    success_count = 0
    for provider, result in results.items():
        if result:
            content = result.get('content', '')[:100]
            print(f"  [{provider}] {content}")
            success_count += 1
        else:
            print(f"  [{provider}] FAILED")

    print(f"\nSuccess: {success_count}/{len(providers)}")
    return success_count == len(providers)


async def test_stage1():
    """Test Stage 1 - Collect responses."""
    from backend.council import stage1_collect_responses

    print("\n" + "=" * 60)
    print("Testing Stage 1 - Collect Responses")
    print("=" * 60)

    query = "Explain the concept of recursion in programming in 2-3 sentences."

    print(f"Query: {query}")
    print("Collecting responses from council...")

    start = time.time()
    results = await stage1_collect_responses(query)
    elapsed = time.time() - start

    print(f"Time: {elapsed:.1f}s")
    print(f"Responses received: {len(results)}")

    for result in results:
        model = result.get('display_name', result['model'])
        response = result['response'][:150] + "..." if len(result['response']) > 150 else result['response']
        print(f"\n[{model}]")
        print(f"  {response}")

    return len(results) >= 2  # At least 2 providers should respond


async def test_full_council():
    """Test full 3-stage council process."""
    from backend.council import run_full_council

    print("\n" + "=" * 60)
    print("Testing Full 3-Stage Council")
    print("=" * 60)

    query = "What are the key differences between Python and JavaScript for web development?"

    print(f"Query: {query}")
    print("\nRunning full council deliberation...")
    print("(This may take 2-3 minutes as each stage queries all providers)")

    start = time.time()
    stage1, stage2, stage3, metadata = await run_full_council(query)
    elapsed = time.time() - start

    print(f"\nTotal time: {elapsed:.1f}s")

    # Stage 1 Summary
    print("\n--- STAGE 1: Individual Responses ---")
    for result in stage1:
        model = result.get('display_name', result['model'])
        response_len = len(result['response'])
        print(f"  [{model}] {response_len} chars")

    # Stage 2 Summary
    print("\n--- STAGE 2: Peer Rankings ---")
    for result in stage2:
        model = result.get('display_name', result['model'])
        parsed = result.get('parsed_ranking', [])
        print(f"  [{model}] Ranking: {' > '.join(parsed)}")

    # Aggregate Rankings
    print("\n--- AGGREGATE RANKINGS ---")
    for rank in metadata.get('aggregate_rankings', []):
        model = rank.get('display_name', rank['model'])
        avg = rank['average_rank']
        votes = rank['rankings_count']
        print(f"  {model}: avg rank {avg} ({votes} votes)")

    # Stage 3 Summary
    print("\n--- STAGE 3: Chairman's Synthesis ---")
    chairman = stage3.get('display_name', stage3['model'])
    synthesis = stage3['response'][:300] + "..." if len(stage3['response']) > 300 else stage3['response']
    print(f"  Chairman: {chairman}")
    print(f"  {synthesis}")

    return len(stage1) >= 2 and len(stage2) >= 2


async def main():
    print(f"\nLLM Council CLI Test Suite")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)

    tests = [
        ("Individual Providers", test_individual_providers),
        ("Parallel Query", test_parallel_query),
        ("Stage 1", test_stage1),
        # ("Full Council", test_full_council),  # Uncomment to run full test
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = await test_func()
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            results[name] = False

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
