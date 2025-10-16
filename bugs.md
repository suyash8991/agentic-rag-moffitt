# Moffitt Agentic RAG System - Known Bugs and Issues

## Empty Name Fields in Vector Database (FIXED)

### Issue Description
When searching for specific researchers by name like "Theresa Boyle", the system failed to find relevant results despite the data being present in the Chroma DB. In the search results, researcher names appeared empty:

```
Researcher:
Program: Malignant Hematology
```

### Investigation Results
After analyzing the database contents, we found that all 867 chunks in the vector database had empty name fields in their metadata:

```
Found 0 unique researcher names in 867 chunks
Found 867 chunks with empty name fields
```

However, the names were actually present in:
1. The profile URLs (e.g., "https://www.moffitt.org/research-science/researchers/ahmad-tarhini")
2. The text content of the chunks (e.g., "Name: , MD, PhD")
3. A separate `researcher_name` field that was added to the source data but not incorporated in the vector database

### Root Cause
The issue was in how researcher chunks were stored in the Chroma DB. When the vector database was created, the name fields in the metadata were not being populated correctly, resulting in empty strings being stored in the database. Additionally, the newly added `researcher_name` field in the source JSON data was not being utilized.

### Impact
1. Searches for specific researchers by name failed to return relevant results
2. The hybrid search functionality didn't work properly for name searches
3. Name-based filtering in the search tools was ineffective
4. Agent responses contained empty researcher names in the output

### Fix Implemented
We implemented a comprehensive solution that properly incorporates the `researcher_name` field into all aspects of the system:

1. **Updated Data Models**:
   - Added `researcher_name` field to `ResearcherProfile` model
   - Added `researcher_name` field to `ResearcherChunk` model
   - Updated `to_document()` method to include `researcher_name` in metadata

2. **Improved Chunk Creation**:
   - Modified `create_researcher_chunks()` to include `researcher_name` in all chunks
   - Updated `deduplicate_chunks()` to handle the new field

3. **Enhanced Vector Database**:
   - Updated metadata creation to include `researcher_name` field
   - Created a script to rebuild the vector database

4. **Improved Search**:
   - Enhanced keyword search to prioritize matches in `researcher_name` field
   - Added bonuses for exact and partial matches on researcher names
   - Updated `ResearcherSearchTool` to prioritize the `researcher_name` field

The solution is fully documented in [name_search_fix.md](name_search_fix.md) with instructions on rebuilding the database.

### Example Code Changes

**Updated keyword search with name matching:**
```python
# Check for exact or partial matches in researcher_name
if researcher_name:
    if query_lower == researcher_name:  # Exact match
        name_match_bonus = 2.0  # Maximum bonus
    elif query_lower in researcher_name:  # Substring match
        name_match_bonus = 1.5
    else:  # Check individual keywords
        name_match_bonus = sum(0.5 for keyword in keywords if keyword in researcher_name)
```

**Updated name prioritization in search results:**
```python
# Prioritize researcher_name field, fall back to name or extract from URL/text if empty
display_name = doc.metadata.get("researcher_name", "").strip()

if not display_name:
    # Fall back to name field
    display_name = doc.metadata.get("name", "").strip()

    if not display_name:
        # Try to extract from URL or text
        # ...
```

## Euron API Streaming Compatibility Issue (FIXED)

### Issue Description
When using the Euron.ai API with the LangChain ReAct agent, the following error occurs:

```
NotImplementedError: Streaming is not currently supported for Euron API
```

This happens because the LangChain ReAct agent attempts to use streaming functionality for generating responses, but our custom Euron integration initially did not support this feature.

### Error Message
```
Agent invocation error: NotImplementedError: Streaming is not currently supported for Euron API
Traceback: Traceback (most recent call last):
  File "...streamlit\components\chat.py", line 171, in invoke_agent
    result = agent.invoke({"input": query})
  ...
  File "...models\euron_chat.py", line 211, in _stream
    raise NotImplementedError(
NotImplementedError: Streaming is not currently supported for Euron API
```

### Root Cause
The Euron API doesn't natively support streaming, but the LangChain ReAct agent framework expects all LLM providers to implement a _stream method. Our initial implementation raised a NotImplementedError when streaming was requested.

### Fix
1. Modified the `_stream` method in `euron_chat.py` to provide a non-streaming fallback that yields the full response at once, making it compatible with LangChain's streaming expectations:

```python
def _stream(self, messages, stop=None, **kwargs) -> Iterator[ChatGeneration]:
    """
    Simulate streaming for compatibility with LangChain agents.

    Euron API doesn't natively support streaming, so we generate the full response
    and yield it as a single chunk.
    """
    logger.info("Streaming requested for Euron API, using non-streaming fallback")

    # Get the full response using the regular generate method
    chat_result = self._generate(messages, stop=stop, **kwargs)

    # If there's a generation, yield it as a single chunk
    if chat_result.generations:
        yield chat_result.generations[0]
```

2. Updated the agent creation code to explicitly disable streaming when using the Euron provider:

```python
if llm_provider == LLMProvider.EURON:
    logger.info("Using Euron provider with streaming disabled")
    llm = get_llm_model(
        provider=llm_provider,
        model_name=model_name,
        temperature=temperature,
        stream=False  # Explicitly disable streaming for Euron
    )
```

This combination of changes ensures compatibility with the LangChain ReAct agent pattern while still leveraging the Euron API for text generation.

## Groq API Retry Parameters Issue (FIXED)

### Issue Description
When using custom retry parameters with the Groq API, the following error occurs:
```
TypeError: Completions.create() got an unexpected keyword argument 'retry_min_seconds'
```

### Root Cause
The Groq client implementation in `langchain_groq` does not support custom retry parameters like `retry_min_seconds`, `retry_max_seconds`, and `retry_multiplier` that we tried to add to improve rate limit handling. These parameters are not part of the Groq client's API.

### Fix
Removed the unsupported retry parameters from the Groq client initialization, keeping only the `max_retries` parameter which is supported:
```python
llm = ChatGroq(
    model=model_to_try,
    temperature=temperature,
    streaming=stream,
    groq_api_key=groq_api_key,
    max_retries=5,  # Maximum 5 retries for rate limits
    **kwargs
)
```

## ReAct Agent Formatting Issues

### Issue Description
The LangChain ReAct agent is failing to correctly format tool calls, leading to repeated errors and inability to execute tools properly. This prevents the agent from retrieving information about researchers, which is core to the system's functionality.

### Error Patterns

When attempting to execute a query like "who is Theresa Boyle?", the agent repeatedly produces the following error:

```
Utilize ResearcherSearch to find information about Theresa Boyle. is not a valid tool, try one of [ResearcherSearch, DepartmentFilter, ProgramFilter, InterestMatch, Collaboration].
```

This indicates that the agent is attempting to use a full sentence as a tool name rather than using just the tool name as required by LangChain's ReAct framework.

### Problematic Agent Response Format

The agent incorrectly formats tool calls in the following way:

```
Action: Utilize ResearcherSearch to find information about Theresa Boyle.
Action Input: ResearcherSearch: Theresa Boyle
```

But the correct format should be:

```
Action: ResearcherSearch
Action Input: Theresa Boyle
```

### Root Causes

1. The `AGENT_PROMPT_TEMPLATE` in `agent.py` doesn't provide clear examples of the correct tool call format expected by LangChain's ReAct agent.
2. The tool usage guidance in the prompt mentions tools but doesn't specify the exact format for invoking them.
3. The agent is not adapting its format despite repeated error messages, suggesting a prompt or instruction issue.

## API Rate Limiting Issues

### Issue Description
The system experiences frequent rate limiting when making requests to the Groq API:

```
INFO:httpx:HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
INFO:groq._base_client:Retrying request to /openai/v1/chat/completions in 13.000000 seconds
```

### Impact
The rate limiting:
1. Slows down the agent's responses
2. Creates unnecessary retries of the same failed formatting
3. May lead to incomplete or interrupted agent reasoning

### Root Causes
1. No rate limiting strategy beyond simple retries
2. Repeated failed attempts with incorrect formatting cause excessive API calls
3. No fallback mechanism to an alternative provider when rate limits are hit

## Reflection Behavior on Failed Executions

### Issue Description
The reflection module appears to function correctly but operates on failed agent executions where no actual information was retrieved using the tools.

### Example
In the user-facing output, the reflection acknowledges the lack of information:

```
Upon reflection, I realize that my preliminary answer did not directly address the user's question about Theresa Boyle. I did not provide any specific details about Theresa Boyle, and my response was incomplete.
```

### Impact
While the reflection module is working as designed, it's reflecting on failed tool executions, which limits its usefulness. The reflection is essentially analyzing an empty or failed response rather than enhancing actual retrieved information.

## Recommended Fixes

### 1. Update Agent Prompt Template
Modify the `AGENT_PROMPT_TEMPLATE` in `agent.py` to include explicit examples of correct tool call formatting:

```python
AGENT_PROMPT_TEMPLATE = """
{system_message}

You have access to the following tools:

{tools}

When using tools, you MUST use the following format:
Thought: I need to find information about X.
Action: ResearcherSearch  <- use ONLY the tool name
Action Input: search query here  <- provide only the input for the tool

Use the tools to answer the user's query. Follow these guidelines:
1. For general researcher searches, use ResearcherSearch
2. To find researchers in a specific department, use DepartmentFilter
3. To find researchers in a specific program, use ProgramFilter
4. To find researchers with similar interests, use InterestMatch
5. To discover potential collaborations, use Collaboration

Remember to provide source information and always be helpful and accurate.

User Query: {input}

{agent_scratchpad}
"""
```

### 2. Improve Rate Limit Handling
Implement more sophisticated rate limit handling:
- Add exponential backoff for retries
- Implement provider switching when rate limits are hit (e.g., fall back to OpenAI or Ollama)
- Cache frequent queries to reduce API calls

### 3. Enhance Error Recovery
Add logic to detect and recover from tool formatting errors:
- Monitor for repeated error patterns
- Implement correction logic when the same error occurs multiple times
- Add a fallback mechanism to handle cases where tools consistently fail

### 4. Update LangChain Dependencies
Ensure the project is using the latest compatible versions of LangChain libraries, as the ReAct agent implementation may have changed in recent versions.