import numpy as np
from typing import Dict, List, Optional, Tuple
import google.generativeai as genai
from datetime import datetime
import time
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
# this is a pair of a query and a response
class CacheEntry:
    session_id: str
    query: str
    context: str
    context_embedding: np.ndarray
    response: str
    timestamp: float
    hit_count: int = 0
    
    # conversion to dictionary
    def to_dict(self):
        d = asdict(self)
        d['context_embedding'] = self.context_embedding.tolist()
        return d

# this tracks metrics
@dataclass
class CacheMetrics:
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    llm_calls: int = 0
    total_latency_ms: float = 0
    latency_saved_ms: float = 0
    tokens_generated: int = 0
    tokens_saved: int = 0
    
    @property
    def hit_rate(self) -> float:
        return self.cache_hits / self.total_queries if self.total_queries > 0 else 0
    
    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_queries if self.total_queries > 0 else 0
    
    def to_dict(self):
        return {
            **asdict(self),
            'hit_rate': self.hit_rate,
            'avg_latency_ms': self.avg_latency_ms,
            'llm_calls_avoided': self.cache_hits
        }


class SemanticCache:
    
    def __init__(
        self, 
        api_key: str,
        similarity_threshold: float = 0.85,
        context_window: int = 4,
        max_cache_size: int = 1000,
        model_name: str = "gemini-2.5-flash"
    ):

        genai.configure(api_key=api_key)
        self.embedding_model = "models/text-embedding-004"
        self.generation_model = genai.GenerativeModel(model_name)
        
        self.similarity_threshold = similarity_threshold # see readme for information
        self.context_window = context_window # see readme for information
        self.max_cache_size = max_cache_size # see readme for information
        
        # storage
        self.cache: List[CacheEntry] = []
        self.sessions: Dict[str, List[Dict]] = defaultdict(list)
        self.metrics = CacheMetrics()
        
    # create embedding using Gemini    
    def _get_embedding(self, text: str) -> np.ndarray:
        result = genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_document"
        )
        return np.array(result['embedding'])
    
    # build context for conversation using the turn limitations for cache
    def _build_context(self, session_id: str, current_query: str) -> str:
        history = self.sessions[session_id]
        
        # get recent history within context window
        recent_history = history[-self.context_window:] if len(history) > 0 else []
        
        # build context string
        context_parts = []
        for i, turn in enumerate(recent_history):
            turn_num = len(history) - len(recent_history) + i + 1
            context_parts.append(f"[Turn {turn_num}] {turn['query']}")
            if 'response' in turn:
                context_parts.append(f"[Response {turn_num}] {turn['response']}")
        
        # add current query
        context_parts.append(f"[Current Query] {current_query}")
        
        return " ".join(context_parts)
    
    # similarity is based on the cosine between two vectors
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # search the cache for similar entries
    def _find_similar_cache_entry(
        self, 
        query_embedding: np.ndarray,
        session_id: str
    ) -> Optional[Tuple[CacheEntry, float]]:

        best_match = None
        best_similarity = -1
        
        for entry in self.cache:
            # only compare within same session for context consistency
            if entry.session_id != session_id:
                continue
                
            similarity = self._cosine_similarity(
                query_embedding, 
                entry.context_embedding
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry
        
        if best_similarity >= self.similarity_threshold:
            return (best_match, best_similarity)
        return None
    
    # eviction policy (denoted in readme)
    def _evict_if_needed(self):
        if len(self.cache) >= self.max_cache_size:
            # sort by hit_count (least used) then timestamp (oldest)
            self.cache.sort(key=lambda x: (x.hit_count, x.timestamp))
            # remove bottom 10% to avoid constant eviction
            evict_count = max(1, self.max_cache_size // 10)
            self.cache = self.cache[evict_count:]
    
    def _call_llm(self, query: str) -> Tuple[str, int, float]:
        start_time = time.time()
        response = self.generation_model.generate_content(query)
        latency_ms = (time.time() - start_time) * 1000
        
        # rough token estimate: ~4 chars per token
        estimated_tokens = len(response.text) // 4
        
        return response.text, estimated_tokens, latency_ms
    
    # session_id is unique identifier for each conversation
    # user_query is current user's query
    # returned dictionary contains response (AI response), cache_hit and from_cache (true or false),
    # if cache_hit true: similarity score and session_id
    # if cache_hit false: latency_ms (the time it took for a response) and session_id
    def query(self, session_id: str, user_query: str) -> Dict:

        start_time = time.time()
        self.metrics.total_queries += 1
        
        # build conversation with context
        context = self._build_context(session_id, user_query)
        context_embedding = self._get_embedding(context)
        
        # search the cache
        cache_result = self._find_similar_cache_entry(context_embedding, session_id)
        
        if cache_result:
            # this means the cache hit
            entry, similarity = cache_result
            entry.hit_count += 1
            
            # update metrics
            self.metrics.cache_hits += 1
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.total_latency_ms += latency_ms
            
            # estimate latency saved 
            estimated_llm_latency = 1000  # rough estimate, most LLM calls take 500-2500 ms
            self.metrics.latency_saved_ms += estimated_llm_latency
            
            # estimate tokens saved
            estimated_tokens = len(entry.response) // 4 # divide by 4 chars per token
            self.metrics.tokens_saved += estimated_tokens
            
            # update session history
            self.sessions[session_id].append({
                'query': user_query,
                'response': entry.response,
                'cache_hit': True,
                'timestamp': time.time()
            })
            
            return {
                'response': entry.response,
                'cache_hit': True,
                'from_cache': True,
                'similarity': float(similarity),
                'latency_ms': latency_ms,
                'session_id': session_id
            }
        
        else:
            # cache missed --> call LLM
            self.metrics.cache_misses += 1
            self.metrics.llm_calls += 1
            
            response_text, tokens, llm_latency_ms = self._call_llm(context)
            total_latency_ms = (time.time() - start_time) * 1000
            
            self.metrics.total_latency_ms += total_latency_ms
            self.metrics.tokens_generated += tokens
            
            # add to cache
            cache_entry = CacheEntry(
                session_id=session_id,
                query=user_query,
                context=context,
                context_embedding=context_embedding,
                response=response_text,
                timestamp=time.time()
            )
            self.cache.append(cache_entry)
            self._evict_if_needed()
            
            # update session history
            self.sessions[session_id].append({
                'query': user_query,
                'response': response_text,
                'cache_hit': False,
                'timestamp': time.time()
            })
            
            return {
                'response': response_text,
                'cache_hit': False,
                'from_cache': False,
                'similarity': 0.0,
                'latency_ms': total_latency_ms,
                'session_id': session_id
            }
    
    def get_metrics(self) -> Dict:
        return self.metrics.to_dict()
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        return self.sessions[session_id]
    
    def clear_cache(self):
        self.cache = []
        
    # clear cache for a specific session
    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]