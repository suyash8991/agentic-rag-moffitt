# Detailed Explanation of researcher_search.py

The `researcher_search.py` file implements the main search tool for finding researchers. It's a critical component that handles both name-based and topic-based searches.

## File Overview

**File Path**: `src/moffitt_rag/tools/researcher_search.py`

**Purpose**: Implements a tool for searching researchers by expertise, interests, or name using both semantic and keyword search.

**Key Components**:
- Helper functions for name extraction and snippet generation
- ResearcherSearchTool class implementing LangChain's BaseTool
- Rate limiting logic to prevent excessive searches
- Name detection algorithms
- Result formatting utilities

## Helper Functions

### extract_name_from_url

```python
def extract_name_from_url(url: str) -> str:
    """
    Extract researcher name from profile URL.

    Args:
        url (str): The profile URL

    Returns:
        str: The extracted name
    """
    if not url:
        return ""

    # Extract the last part of the URL path which often contains the name
    path_parts = url.rstrip('/').split('/')
    if len(path_parts) > 0:
        name_slug = path_parts[-1]
        # Convert from slug format to name format
        name = name_slug.replace('-', ' ').title()
        return name
    return ""
```

**Purpose**: This function extracts a researcher's name from their profile URL.

**Process**:
1. Splits the URL by '/' to get the path components
2. Takes the last component, which is typically the name in slug format (e.g., "theresa-boyle")
3. Converts the slug to a readable name by:
   - Replacing hyphens with spaces
   - Applying title case formatting
4. Returns the extracted name or an empty string if extraction fails

**Usage**:
```python
name = extract_name_from_url("https://www.moffitt.org/research-science/researchers/theresa-boyle")
# Returns "Theresa Boyle"
```

### extract_name_from_text

```python
def extract_name_from_text(text: str) -> str:
    """
    Extract researcher name from text content.

    Args:
        text (str): The text content that might contain a name

    Returns:
        str: The extracted name
    """
    # Try to find name in a "Name: " pattern
    name_match = re.search(r'Name:\s*([^,\n]+)', text)
    if name_match:
        name = name_match.group(1).strip()
        if name and name != ',':  # Check if name is non-empty and not just a comma
            return name

    # Try to find a name at the beginning followed by degrees
    degrees_match = re.search(r'^([^,]+),\s*([A-Z][A-Za-z]*\.?(?:\s+[A-Z][A-Za-z]*\.?)*)', text)
    if degrees_match:
        return degrees_match.group(1).strip()

    return ""
```

**Purpose**: Extracts a researcher's name from text content using regex patterns.

**Process**:
1. First tries to find a name in the "Name: [name]" pattern
   - Uses regular expression `r'Name:\s*([^,\n]+)'` to capture the name
   - Ensures the name is non-empty and not just a comma
2. If that fails, tries to find a name at the beginning followed by degrees
   - Uses pattern `r'^([^,]+),\s*([A-Z][A-Za-z]*\.?(?:\s+[A-Z][A-Za-z]*\.?)*)'`
   - This matches patterns like "John Smith, PhD"
3. Returns the extracted name or an empty string if extraction fails

**Usage**:
```python
name = extract_name_from_text("Name: John Smith, PhD\nTitle: Professor")
# Returns "John Smith"
```

### extract_relevant_snippet

```python
def extract_relevant_snippet(text: str, query: str, max_length: int = 200) -> str:
    """
    Extract the most relevant snippet from a text based on a query.

    Args:
        text (str): The text to extract from
        query (str): The query to search for
        max_length (int, optional): Maximum length of the snippet. Defaults to 200.

    Returns:
        str: The most relevant snippet
    """
    # If text is short enough, return the whole thing
    if len(text) <= max_length:
        return text

    # Split the query into terms
    query_terms = set(re.findall(r'\w+', query.lower()))

    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Score each sentence based on query term matches
    scored_sentences = []
    for sentence in sentences:
        # Count the number of query terms in the sentence
        sentence_terms = set(re.findall(r'\w+', sentence.lower()))
        score = len(query_terms.intersection(sentence_terms))
        scored_sentences.append((sentence, score))

    # Sort sentences by score (descending)
    scored_sentences.sort(key=lambda x: x[1], reverse=True)

    # Take the top sentences until we reach the max length
    snippet = ""
    for sentence, _ in scored_sentences:
        if len(snippet) + len(sentence) + 1 <= max_length:
            snippet += sentence + " "
        else:
            break

    # If no good matches, just take the beginning of the text
    if not snippet:
        snippet = text[:max_length].rsplit(" ", 1)[0] + "..."

    return snippet.strip()
```

**Purpose**: Extracts a relevant snippet from text based on the search query.

**Process**:
1. If text is already shorter than max_length, returns the entire text
2. Splits the query and text into terms using regex
3. Splits the text into sentences
4. Scores each sentence based on the number of query terms it contains
5. Sorts sentences by score (highest first)
6. Takes the top-scoring sentences until the max_length is reached
7. If no good matches are found, falls back to using the beginning of the text

**Usage**:
```python
snippet = extract_relevant_snippet(
    "Dr. Smith focuses on immunotherapy for melanoma. He has published extensively on cancer genomics.",
    "immunotherapy cancer",
    max_length=100
)
# Returns "Dr. Smith focuses on immunotherapy for melanoma. He has published extensively on cancer genomics."
```

## ResearcherSearchTool Class

```python
class ResearcherSearchTool(BaseTool):
    """
    Tool for searching researchers by their expertise, interests, or background.

    This tool uses hybrid search (combining vector search with keyword search)
    to find the most relevant researchers for a given query.
    """

    name: str = "ResearcherSearch"
    description: str = "Search for researchers by their expertise, interests, or background"

    # Class-level tracking of queries and attempts
    _query_attempts = {}
    _max_attempts_per_query = 3  # Maximum attempts for the same query
```

### Rate Limiting Logic

```python
def _check_rate_limit(self, query: str) -> bool:
    """
    Check if the query has exceeded the rate limit.

    Args:
        query (str): The search query

    Returns:
        bool: True if the query is allowed, False if it should be rate-limited
    """
    # Normalize query for comparison (lowercase and remove extra spaces)
    normalized_query = ' '.join(query.lower().split())

    # Get current count for this query
    current_count = self._query_attempts.get(normalized_query, 0)

    # Check if we've exceeded the limit
    if current_count >= self._max_attempts_per_query:
        logger.warning(f"Rate limit exceeded for query: {query} ({current_count} attempts)")
        return False

    # Increment the count
    self._query_attempts[normalized_query] = current_count + 1
    logger.info(f"Query attempt {current_count + 1}/{self._max_attempts_per_query} for: {query}")

    return True
```

**Purpose**: Prevents excessive search attempts for the same query.

**Process**:
1. Normalizes the query (lowercase, remove extra spaces)
2. Checks if the query has been attempted too many times (default: 3)
3. If limit is exceeded, returns False to block the search
4. Otherwise, increments the attempt count and returns True
5. Uses class-level static dictionary to track attempts across multiple calls

### Main Run Method

```python
def _run(self, query: str) -> str:
    """
    Run the tool with the given query.

    Args:
        query (str): The search query

    Returns:
        str: The search results formatted as a string
    """
    logger.info(f"Searching for researchers matching: {query}")

    # Check if we've exceeded the rate limit for this query
    if not self._check_rate_limit(query):
        # Return a helpful message suggesting how to proceed
        return (
            f"You've made multiple attempts searching for '{query}' without success. "
            f"Based on the available data, this researcher may not be in the database. "
            # ...
        )

    # Determine if this is likely a name search or a topic search
    # ...
```

### Name Detection Logic

```python
# Method 1: Check for phrases that indicate name searches
name_search_indicators = ["who is", "find", "about", "researcher named", "dr.", "doctor", "professor"]
is_name_search_by_phrase = any(indicator in query_lower for indicator in name_search_indicators)

# Method 2: Check for capitalized words which likely indicate proper names
words = query.split()
capitalized_words = [word for word in words if len(word) > 1 and word[0].isupper() and word.lower() not in ["what", "who", "where", "when", "why", "how"]]
has_capitalized_words = len(capitalized_words) >= 1
is_short_query = len(words) <= 3
is_name_search_by_capitalization = has_capitalized_words and is_short_query

# Method 3: Check for specific known researchers by name
important_researchers = [
    "theresa boyle", "ahmad tarhini", "noemi andor", "eric padron",
    "john cleveland", "aleksandra karolak"
]
is_known_researcher = any(researcher in query_lower for researcher in important_researchers)

# Combine all detection methods
is_name_search = is_name_search_by_phrase or is_name_search_by_capitalization or is_known_researcher

# For name searches, give more weight to keyword matching by using a lower alpha
alpha = 0.3 if is_name_search else 0.7
```

**Purpose**: Determines if the query is likely a name search or a topic search.

**Methods**:
1. **Phrase Indicators**: Checks for phrases like "who is", "about", "researcher named"
2. **Capitalized Words**: Identifies proper names (capitalized words)
3. **Known Researchers**: Checks against a list of known researcher names
4. **Combined Detection**: Combines all methods to make the final determination
5. **Alpha Adjustment**: Sets alpha=0.3 for name searches (favoring keyword matching) or alpha=0.7 for topic searches (favoring semantic search)

### Direct Lookup for Known Researchers

```python
# Direct lookup optimization for known researchers
results = None
if is_known_researcher and matched_researcher:
    # If this is a known researcher, do a direct metadata lookup first
    logger.info(f"Performing direct lookup for known researcher: {matched_researcher}")
    from ..db.vector_store import get_or_create_vector_db

    # Get all chunks from the database
    db = get_or_create_vector_db()
    db_results = db.get()
    texts = db_results['documents']
    metadatas = db_results['metadatas']

    # Find exact and partial matches in researcher_name field
    direct_matches = []
    partial_matches = []

    for i, metadata in enumerate(metadatas):
        researcher_name = metadata.get("researcher_name", "").lower()
        name_field = metadata.get("name", "").lower()

        # Check for exact matches first
        if matched_researcher == researcher_name or matched_researcher == name_field:
            direct_matches.append(Document(page_content=texts[i], metadata=metadata))
        # Then check for partial matches
        elif matched_researcher in researcher_name or matched_researcher in name_field:
            partial_matches.append(Document(page_content=texts[i], metadata=metadata))

    # If we found direct matches, use them
    if direct_matches:
        logger.info(f"Found {len(direct_matches)} direct matches for {matched_researcher}")
        results = direct_matches[:5]  # Limit to 5 matches
    # Otherwise, try partial matches if available
    elif partial_matches:
        logger.info(f"Found {len(partial_matches)} partial matches for {matched_researcher}")
        results = partial_matches[:5]  # Limit to 5 matches
```

**Purpose**: Optimizes searches for known researchers by bypassing the hybrid search.

**Process**:
1. For known researchers, performs a direct lookup in the database
2. Gets all documents and their metadata from the database
3. Searches for exact matches in the metadata fields (researcher_name, name)
4. If no exact matches, tries partial matches
5. Returns the matches if found, otherwise continues to hybrid search

### Hybrid Search Fallback

```python
# If no direct matches or not a known researcher, use hybrid search
if not results:
    # Use the hybrid search functionality to find relevant researchers
    results = hybrid_search(
        query=query,
        k=5,
        alpha=alpha  # Adjusted based on query type
    )
```

**Purpose**: Falls back to hybrid search when direct lookup fails or for non-known researchers.

**Process**:
1. Calls hybrid_search with the query and appropriate alpha value
2. Uses alpha=0.3 for name searches (from earlier detection)
3. Uses alpha=0.7 for topic searches (from earlier detection)
4. Passes k=5 to get 5 results

### Special Handling for No Results

```python
if not results:
    # Check if this was a search for a known researcher
    known_researchers = {
        "theresa boyle": "Theresa Boyle",
        "ahmad tarhini": "Ahmad Tarhini",
        "noemi andor": "Noemi Andor",
        "eric padron": "Eric Padron",
        "john cleveland": "John Cleveland",
        "aleksandra karolak": "Aleksandra Karolak"
    }

    # Check if the query was for a known researcher by name
    is_known_researcher_query = False
    matched_name = None

    for known_name_lower, display_name in known_researchers.items():
        if known_name_lower in query_lower:
            is_known_researcher_query = True
            matched_name = display_name
            break

    if is_known_researcher_query and matched_name:
        # For known researchers, provide a more specific message
        return (
            f"The researcher {matched_name} is mentioned in our database of known researchers, "
            f"but no detailed profile information was found. This may be due to one of the following reasons:\n"
            f"1. The researcher's profile may not be in the current database index\n"
            f"2. The researcher may be mentioned in relation to other researchers' work\n"
            f"3. There might be a technical issue with retrieving the profile\n\n"
            f"Would you like information about another researcher or research area instead?"
        )
    else:
        # Generic "not found" message
        return f"No researchers found matching the query: {query}"
```

**Purpose**: Provides helpful responses when no results are found.

**Process**:
1. For known researchers, provides a specific message explaining why the search might have failed
2. For other queries, provides a generic "not found" message
3. Uses a dictionary of known researchers to format the message with the correct name

### Result Formatting

```python
formatted_results = []
for doc in results:
    researcher_id = doc.metadata["researcher_id"]
    profile_url = doc.metadata.get("profile_url", "")

    # Prioritize researcher_name field, fall back to name or extract from URL/text if empty
    display_name = doc.metadata.get("researcher_name", "").strip()

    if not display_name:
        # Fall back to name field
        display_name = doc.metadata.get("name", "").strip()

        if not display_name:
            # Try to extract name from URL
            url_name = extract_name_from_url(profile_url)

            # Try to extract name from text if URL extraction failed
            text_name = extract_name_from_text(doc.page_content)

            # Use the best name we could find
            display_name = text_name or url_name or "Unknown Researcher"
            logger.info(f"Extracted name '{display_name}' for researcher_id {researcher_id} from {'text' if text_name else 'URL'}")

    program = doc.metadata.get("program", "Unknown Program")

    # Extract the most relevant snippet from the content
    snippet = extract_relevant_snippet(doc.page_content, query)

    formatted_results.append(
        f"Researcher: {display_name}\n"
        f"Program: {program}\n"
        f"Relevance: {snippet}\n"
        f"Profile: {profile_url}\n"
    )

return "\n".join(formatted_results)
```

**Purpose**: Formats search results into a readable string for the agent.

**Process**:
1. Extracts metadata from each document (researcher_id, profile_url)
2. Prioritizes the researcher_name field for display
3. Falls back to the name field if researcher_name is empty
4. If both are empty, tries to extract the name from URL or text
5. Gets the program information
6. Extracts a relevant snippet based on the query
7. Formats each result with researcher name, program, relevance snippet, and profile URL
8. Joins all results with newlines

## Async Support

The tool also includes an async implementation for compatibility with async workflows:

```python
async def _arun(self, query: str) -> str:
    """
    Run the tool asynchronously with the given query.

    Args:
        query (str): The search query

    Returns:
        str: The search results formatted as a string
    """
    # For now, just call the synchronous version
    return self._run(query)
```

Currently, this simply calls the synchronous version, but it could be extended to provide true async behavior in the future.

## Key Challenges Addressed

The `researcher_search.py` implementation addresses several key challenges:

1. **Name vs. Topic Detection**: Accurately distinguishes between name searches and topic searches
2. **Search Optimization**: Adjusts search parameters based on query type
3. **Name Extraction**: Extracts researcher names from various sources (URL, text)
4. **Rate Limiting**: Prevents excessive search attempts for the same query
5. **Result Relevance**: Extracts and highlights relevant content based on the query
6. **Missing Data Handling**: Gracefully handles missing data fields
7. **Known Researcher Handling**: Provides special handling for known researchers

These features combine to create a robust and effective search tool for the Moffitt RAG system.