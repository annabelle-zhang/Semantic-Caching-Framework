# Semantic-Caching-Framework
For DSRS application

In order to use my code, you have to save your Gemini API key to your environment. Open Windows Command Prompt or Powershell and do this --> SETX GOOGLE_API_KEY "[your API key]" and then reload everything and the code should run with your newly saved key. In my code, I saved API keys to the variable GOOGLE_API_KEY, so it has to be exactly that. For future reference, if the Gemini models change, you'll also have to change the code in semantic_cache.py (at model_name). Additionally, if you don't have google.generativeai downloaded to your computer, you'll have to run this command --> pip install google-generativeai.

To run the main python code, simply open your terminal and type this --> python main.py

It'll give you three options, to run an evaluation suite, do a demo (based on the document questions), and custom queries.

The evaluation suite uses test_suite.py to show some example queries and examples of my code.
Test 1 in the suite shows that identical queries will be a cache hit.
Test 2 shows that when words are similar (corn and maize), there will be a cache hit.
Test 3 shows conversational cache hits.
Test 4 shows that when the sessions are different, the cache will not be a hit, since they are saved separately.
Test 5 shows similarity thresholds for pairs of queries that are fairly similar (demonstrating where similarity threshold is optimally placed).
Test 6 is a performance test with multiple queries based on a set of starter questions (that get changed later with phrases substituted into the queries to produce similar questions).
Finally, the metrics (from the assignment directions) for all 6 tests are printed as well.

The demo simulates a conversation about agriculture and climate change (from the instructions document).
Here's an example of my code demo's output:

            PS C:\Users\annab\OneDrive\Desktop\Semantic-Caching-Framework> python main.py
            Semantic Caching Framework
            Select a mode:

            1. Run Test Suite
            2. Demo
            3. Custom Queries

            Enter choice (1-3): 2

            SEMANTIC CACHE DEMO

            Simulating a conversation about climate change, air quality, and farming:

            Query 1: What are the causes of climate change?
            Response: The overwhelming scientific consensus is that the primary cause of the rapid climate change observed over the past century, particularly since the mid-20th century, is human activity. These activities...

            [CACHE MISS]
            Cache missed, generated new response in 14569.89ms

            Query 2: How does climate change affect the air?
            Response: Climate change significantly alters the air in several fundamental ways, impacting its temperature, composition, quality, and the weather patterns it generates. These changes are largely a direct cons...

            [CACHE MISS]
            Cache missed, generated new response in 11343.58ms

            Query 3: What about the impact on agriculture?
            Response: Climate change significantly alters the air in several fundamental ways, impacting its temperature, composition, quality, and the weather patterns it generates. These changes are largely a direct cons...

            [CACHE HIT]
            Similarity: 0.9796
            Cached in 225.57ms

            Query 4: Can you explain how it affects farming?
            Response: Climate change significantly alters the air in several fundamental ways, impacting its temperature, composition, quality, and the weather patterns it generates. These changes are largely a direct cons...

            [CACHE HIT]
            Similarity: 0.9796
            Cached in 184.54ms

            Query 5: What's climate change's effect on the atmosphere?
            Response: Climate change significantly alters the air in several fundamental ways, impacting its temperature, composition, quality, and the weather patterns it generates. These changes are largely a direct cons...

            [CACHE HIT]
            Similarity: 0.9796
            Cached in 225.10ms

            SESSION METRICS:
            Total Queries: 5
            Cache Hit Rate: 60.0%
            LLM Calls Avoided: 3
            Estimated Time Saved: 3.00s

As you can see, the demo has similar fourth and fifth questions, and the semantic cache acknowledges that, bringing out the cached answers.

For the custom queries, the code outputs if the questions you input are a cache hit or a cache miss, so you can test the system with your own prompts, and see the caching system work.

I decided to set the similarity threshold selection at 0.85 based off of testing. Even when the prompt is the exact same, similarity sometimes only registers a little bit above 0.85:
The text below is an output of my code.
        Query 1: What is the impact of climate change on corn yields?
        Cache Hit: False
        Latency: 14506.74ms

        Query 2: What is the impact of climate change on corn yields?
        Cache Hit: True
        Similarity: 0.8593
        Latency: 313.54ms
        Speedup: 46.27x


I have a context window (that I've set to 4) to determine how many queries the AI should observe in advance for context. I think that most queries are likely to be resolved wtihin 3-4 exchanges, so that's a good number. However, if the conversations are very complex, the window should be increased, and if conversations don't need much context, the window can be decreased. I decided to keep it on the smaller end to reduce the noise presented to the AI and prevent the embedding from becoming overwhelmed with old content. 
The cache is a list that is in the memory of the code, and it uses a linear search. The insert process is done in O(1) time, but the search is O(n) where n is the size of the cache, so as entries increase, the process will start to slow down. This means that with the proposed millions of queries, there would likely be a time complexity issue with the cache.
As for the eviction, the cache simply removes the least recently used entries once it is full. Likely if the items have not been used in a while, they are no longer relevant to the user, so it's best to preserve memory by removing items that are no longer relevant. I set the limit for items in the cache to 1000, which is on the lower end, but with a higher limit of items in the cache, the code would be slower (see linear search above). At 1000 items, the search time is just around 30-80 ms.