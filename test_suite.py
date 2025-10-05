from semantic_cache import SemanticCache


class SemanticCacheEvaluator:
    
    def __init__(self, cache: SemanticCache):
        self.cache = cache
    
    # test 1:
    def test_identical_caching(self):
        print("TEST 1: identical queries")
        
        session_id = "test_identical"
        query = "What is the impact of climate change on corn yields?"
        
        # first query - should miss
        result1 = self.cache.query(session_id, query)
        print(f"Query 1: {query}")
        print(f"  Cache Hit: {result1['cache_hit']}")
        print(f"  Latency: {result1['latency_ms']:.2f}ms")
        
        # second identical query - should hit
        result2 = self.cache.query(session_id, query)
        print(f"\nQuery 2: {query}")
        print(f"  Cache Hit: {result2['cache_hit']}")
        print(f"  Similarity: {result2['similarity']:.4f}")
        print(f"  Latency: {result2['latency_ms']:.2f}ms")
        print(f"  Speedup: {result1['latency_ms']/result2['latency_ms']:.2f}x")

    # test 2:    
    def test_semantic_similarity(self):
        print("\nTEST 2: semantic similarities")
        
        session_id = "test_semantic"
        queries = [
            "What is the impact of climate change on corn yields?",
            "How does global warming affect the productivity of maize crops?"
        ]
        
        for i, query in enumerate(queries, 1):
            result = self.cache.query(session_id, query)
            print(f"\nQuery {i}: {query}")
            print(f"  Cache Hit: {result['cache_hit']}")
            if result['cache_hit']:
                print(f"  Similarity: {result['similarity']:.4f}")
            print(f"  Latency: {result['latency_ms']:.2f}ms")
    
    # test 3: conversation
    def test_conversational_context(self):
        print("\nTEST 3: conversational caching")
        
        session_id = "test_conversation"
        
        conversation = [
            "What is the impact of climate change on corn yields in the US?",
            "What about wheat?",  # depends on previous context
            "How about in Europe?", # needs to answer about climate change
            "What about wheat?",  # the same as query 2, should hit cache
        ]
        
        for i, query in enumerate(conversation, 1):
            result = self.cache.query(session_id, query)
            print(f"\nTurn {i}: {query}")
            print(f"  Cache Hit: {result['cache_hit']}")
            if result['cache_hit']:
                print(f"  Similarity: {result['similarity']:.4f}")
            print(f"  Latency: {result['latency_ms']:.2f}ms")
    
    # test 4: session isolation
    def test_cross_session_isolation(self):
        print("\nTEST 4: session isolation")
        
        query = "What crops are affected by climate change?"
        
        # session 1
        result1 = self.cache.query("session_1", query)
        print(f"Session 1, Query 1: {query}")
        print(f"  Cache Hit: {result1['cache_hit']}")
        
        # session 2 - different context, should not hit session 1's cache
        result2 = self.cache.query("session_2", query)
        print(f"\nSession 2, Query 1: {query}")
        print(f"  Cache Hit: {result2['cache_hit']}")
        print(f"  (this should be False - different session context)")
    
    # test 5: similarity thresholds
    def test_threshold_sensitivity(self):
        print("\nTEST 5: threshold analysis")
        
        thresholds = [0.75, 0.80, 0.85, 0.90, 0.95]
        
        test_pairs = [
            ("What is climate change?", "What is global warming?"),
            ("How does rain form?", "What causes precipitation?"),
            ("Explain photosynthesis", "How do plants make food?")
        ]
        
        print("\nSimilarity scores for query pairs:")
        for q1, q2 in test_pairs:
            emb1 = self.cache._get_embedding(q1)
            emb2 = self.cache._get_embedding(q2)
            similarity = self.cache._cosine_similarity(emb1, emb2)
            print(f"\nQ1: {q1}")
            print(f"Q2: {q2}")
            print(f"Similarity: {similarity:.4f}")
            
            for thresh in thresholds:
                hit = "HIT" if similarity >= thresh else "MISS"
                print(f"  Threshold {thresh:.2f}: {hit}")
    
    # test 6: many queries
    def test_repeated(self):
        print("\nTEST 6: repeated queries")
        
        import time
        
        session_id = "load_test"
        base_queries = [
            "What is machine learning?",
            "Explain neural networks",
            "What is deep learning?",
            "How do transformers work?",
            "What is attention mechanism?"
        ]
        
        all_queries = []
        for query in base_queries:
            all_queries.append(query)
            # create new queries with word variations
            all_queries.append(query.replace("What", "Can you explain what"))
            all_queries.append(query.replace("?", " in simple terms?"))
        
        print(f"Running {len(all_queries)} queries...")
        start = time.time()
        
        for query in all_queries:
            self.cache.query(session_id, query)
        
        elapsed = time.time() - start
        
        print(f"Total time: {elapsed:.2f}s")
        print(f"Avg time per query: {elapsed/len(all_queries)*1000:.2f}ms")
        
        metrics = self.cache.get_metrics()
        print(f"\nCache hit rate: {metrics['hit_rate']*100:.1f}%")
        print(f"LLM calls avoided: {metrics['llm_calls_avoided']}")
    
    def run_all_tests(self):
        print("\nSEMANTIC CACHE TEST SUITE:")
        
        self.test_identical_caching()
        self.test_semantic_similarity()
        self.test_conversational_context()
        self.test_cross_session_isolation()
        self.test_threshold_sensitivity()
        self.test_repeated()
        
        # metrics
        print("\nSESSION METRICS:")
        metrics = self.cache.get_metrics()
        
        print(f"\nTotal Queries: {metrics['total_queries']}")
        print(f"Cache Hits: {metrics['cache_hits']}")
        print(f"Cache Misses: {metrics['cache_misses']}")
        print(f"Hit Rate: {metrics['hit_rate']*100:.2f}%")
        print(f"\nLLM Calls Made: {metrics['llm_calls']}")
        print(f"LLM Calls Avoided: {metrics['llm_calls_avoided']}")
        print(f"Tokens Generated: {metrics['tokens_generated']}")
        print(f"Tokens Saved: {metrics['tokens_saved']}")
        print(f"\nAvg Latency: {metrics['avg_latency_ms']:.2f}ms")
        print(f"Total Latency Saved: {metrics['latency_saved_ms']/1000:.2f}s")
        
        # cost estimation
        cost_per_token = 0.30 / 1_000_000  # google token cost (for gemini)
        cost_without_cache = (metrics['tokens_generated'] + metrics['tokens_saved']) * cost_per_token
        cost_with_cache = metrics['tokens_generated'] * cost_per_token
        savings = cost_without_cache - cost_with_cache
        
        print(f"\nCost Analysis:")
        print(f"Cost without cache: ${cost_without_cache:.6f}")
        print(f"Cost with cache: ${cost_with_cache:.6f}")
        print(f"Savings: ${savings:.6f} ({savings/cost_without_cache*100:.1f}%)")
        
