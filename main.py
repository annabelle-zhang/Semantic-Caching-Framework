import os
from semantic_cache import SemanticCache
from test_suite import SemanticCacheEvaluator

# runs test_suite.py
def run_test_suite():
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("ERROR: Please set GOOGLE_API_KEY environment variable")
        return
    
    # initializing cache
    cache = SemanticCache(
        api_key=api_key,
        similarity_threshold=0.85,
        context_window=4,
        max_cache_size=1000
    )
    
    print("\nCache Configuration:")
    print(f"  Similarity Threshold: {cache.similarity_threshold}")
    print(f"  Context Window: {cache.context_window} turns")
    print(f"  Max Cache Size: {cache.max_cache_size} entries")
    print(f"  Embedding Model: {cache.embedding_model}")
    print(f"  Generation Model: gemini-2.5-flash")
    
    # imported class and function from test_suite.py
    evaluator = SemanticCacheEvaluator(cache)
    evaluator.run_all_tests()
    
    print("\nResults are above!")

# demo
def demo():
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("ERROR: Please set GOOGLE_API_KEY environment variable")
        return
    
    print("\nSEMANTIC CACHE DEMO")
    
    cache = SemanticCache(api_key=api_key)
    session_id = "demo"
    
    print("\nSimulating a conversation about climate change, air quality, and farming:")
    
    queries = [
        "What are the causes of climate change?",
        "How does climate change affect the air?",
        "What about the impact on agriculture?",
        "Can you explain how it affects farming?",  # should be similar to query 3
        "What's climate change's effect on the atmosphere?",  # should be similar to query 2
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        
        result = cache.query(session_id, query)
        
        print(f"Response: {result['response'][:200]}...")
        print(f"\n{'[CACHE HIT]' if result['cache_hit'] else '[CACHE MISS]'}")
        
        if result['cache_hit']:
            print(f"Similarity: {result['similarity']:.4f}")
            print(f"Cached in {result['latency_ms']:.2f}ms")
        else:
            print(f"Cache missed, generated new response in {result['latency_ms']:.2f}ms")
    
    # final metrics
    print("\nSESSION METRICS:")
    
    metrics = cache.get_metrics()
    print(f"Total Queries: {metrics['total_queries']}")
    print(f"Cache Hit Rate: {metrics['hit_rate']*100:.1f}%")
    print(f"LLM Calls Avoided: {metrics['llm_calls_avoided']}")
    print(f"Estimated Time Saved: {metrics['latency_saved_ms']/1000:.2f}s")

# custom queries
def custom_queries():
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("ERROR: Please set GOOGLE_API_KEY environment variable")
        return
    
    print("\nCUSTOM QUERY MODE")
    print("Enter queries (type 'quit' to exit, 'metrics' to see stats)")
    
    cache = SemanticCache(api_key=api_key)
    session_id = "custom_session"
    
    while True:
        print("\n")
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_input.lower() == 'metrics':
            metrics = cache.get_metrics()
            print("\nCurrent Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
            continue
        
        if not user_input:
            continue
        
        result = cache.query(session_id, user_input)
        
        print(f"\nAI: {result['response']}")
        print(f"\n[{'CACHE HIT' if result['cache_hit'] else 'CACHE MISS'}]", end="")
        
        if result['cache_hit']:
            print(f" (similarity: {result['similarity']:.3f})")
        else:
            print()
    
    print("\nSESSION METRICS:")
    metrics = cache.get_metrics()
    print(f"  Queries: {metrics['total_queries']}")
    print(f"  Hit Rate: {metrics['hit_rate']*100:.1f}%")
    print(f"  LLM Calls Saved: {metrics['llm_calls_avoided']}")


def main():
    print("Semantic Caching Framework")
    
    print("Select a mode:\n")
    print("  1. Run Test Suite")
    print("  2. Demo")
    print("  3. Custom Queries")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        run_test_suite()
    elif choice == '2':
        demo()
    elif choice == '3':
        custom_queries()
    else:
        print("Invalid choice. Please run again and select 1-4.")


if __name__ == "__main__":
    main()