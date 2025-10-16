# ChromaDB Chunking Process

## Chunking Strategy

Researcher profiles are divided into chunks to optimize retrieval. The chunking strategy is implemented in `create_researcher_chunks` in `src/moffitt_rag/data/loader.py`:

1. **Unique ID Generation**:
   - Creates a unique identifier for each researcher based on name, URL, program, and department
   - Generates a hash to ensure uniqueness

2. **Core Information Chunk**:
   - Basic information about the researcher (name, title, program, department, overview)
   - Stored as a single chunk with `chunk_type="core"`

3. **Research Interests Chunk**:
   - Research interests are stored as a separate chunk with `chunk_type="interests"`
   - Allows targeted retrieval of researchers by their interests

4. **Publications Chunks**:
   - Publications are grouped into multiple chunks based on size
   - Each group becomes a chunk with `chunk_type="publications"`
   - Publications are chunked to avoid exceeding the maximum chunk size (1024 characters by default)

5. **Grant Chunks**:
   - Grants are similarly grouped into chunks based on size
   - Each group becomes a chunk with `chunk_type="grants"`

## Chunk ID Structure

Each chunk has a unique ID with the format:
```
{researcher_id}_{hash_prefix}_{chunk_type}[_{index}]
```

For example:
- `24764_8a5b_core` - Core information for researcher with ID 24764
- `24764_8a5b_interests` - Research interests for the same researcher
- `24764_8a5b_pubs_0` - First chunk of publications
- `24764_8a5b_pubs_1` - Second chunk of publications

## Chunking Code Example

```python
def create_researcher_chunks(profile: ResearcherProfile, chunk_size: int = 1024) -> List[ResearcherChunk]:
    chunks = []

    # Create a unique identifier
    unique_data = f"{profile.name}_{profile.profile_url}_{profile.primary_program or ''}_{profile.department or ''}"
    unique_hash = hashlib.md5(unique_data.encode()).hexdigest()[:8]
    prefix = f"{profile.researcher_id}_{unique_hash[:4]}_"

    # Core information chunk
    core_info = "\n".join([
        f"Name: {profile.full_name}",
        f"Title: {profile.title}" if profile.title else "",
        f"Program: {profile.primary_program}" if profile.primary_program else "",
        f"Department: {profile.department}" if profile.department else "",
        f"Overview: {profile.overview}" if profile.overview else ""
    ])

    core_chunk = ResearcherChunk(
        chunk_id=f"{prefix}core",
        text=core_info,
        researcher_id=profile.researcher_id,
        name=profile.name,
        researcher_name=profile.researcher_name,
        program=profile.primary_program,
        department=profile.department,
        research_interests=profile.research_interests,
        chunk_type="core",
        profile_url=profile.profile_url
    )
    chunks.append(core_chunk)

    # Add interests chunk
    if profile.research_interests:
        interests_text = "Research Interests: " + "\n".join(profile.research_interests)
        interests_chunk = ResearcherChunk(
            chunk_id=f"{prefix}interests",
            text=interests_text,
            researcher_id=profile.researcher_id,
            name=profile.name,
            researcher_name=profile.researcher_name,
            program=profile.primary_program,
            department=profile.department,
            research_interests=profile.research_interests,
            chunk_type="interests",
            profile_url=profile.profile_url
        )
        chunks.append(interests_chunk)

    # Add publication chunks
    if profile.publications:
        # Group publications into chunks
        pub_chunks = []
        current_chunk = []
        current_size = 0

        for pub in profile.publications:
            pub_text = "\n".join([
                f"Title: {pub.title}",
                f"Authors: {pub.authors}" if pub.authors else "",
                f"Journal: {pub.journal}" if pub.journal else "",
                f"Year: {pub.year}" if pub.year else ""
            ])

            # If adding this publication would exceed the chunk size, create a new chunk
            if current_size + len(pub_text) > chunk_size and current_chunk:
                pub_chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(pub_text)
            current_size += len(pub_text)

        # Add any remaining publications
        if current_chunk:
            pub_chunks.append("\n\n".join(current_chunk))

        # Create chunks for each group
        for i, chunk_text in enumerate(pub_chunks):
            pub_chunk = ResearcherChunk(
                chunk_id=f"{prefix}pubs_{i}",
                text=f"Publications:\n{chunk_text}",
                researcher_id=profile.researcher_id,
                name=profile.name,
                researcher_name=profile.researcher_name,
                program=profile.primary_program,
                department=profile.department,
                research_interests=profile.research_interests,
                chunk_type="publications",
                profile_url=profile.profile_url
            )
            chunks.append(pub_chunk)

    # Similar chunking for grants
    # ...

    return chunks
```

## Chunk Model Definition

Chunks are represented by the `ResearcherChunk` Pydantic model in `src/moffitt_rag/data/models.py`:

```python
class ResearcherChunk(BaseModel):
    """
    A chunk of a researcher profile for retrieval.

    This model represents a chunk of text from a researcher profile
    along with metadata for retrieval.
    """

    chunk_id: str
    text: str
    researcher_id: str
    name: str
    researcher_name: str = ""  # Added field to match JSON structure
    program: Optional[str] = None
    department: Optional[str] = None
    research_interests: List[str] = Field(default_factory=list)
    chunk_type: str  # 'core', 'interests', 'publications', 'grants'
    profile_url: str
```

## Deduplication Process

Before adding chunks to ChromaDB, a deduplication process ensures all chunk IDs are unique:

```python
def deduplicate_chunks(chunks: List[ResearcherChunk]) -> List[ResearcherChunk]:
    """
    Make chunk IDs unique by adding a counter suffix when duplicates are found.
    """
    id_count = {}
    deduplicated_chunks = []

    for chunk in chunks:
        # If this ID has been seen before, add a suffix
        if chunk.chunk_id in id_count:
            id_count[chunk.chunk_id] += 1
            new_id = f"{chunk.chunk_id}_{id_count[chunk.chunk_id]}"

            # Create a new chunk with the updated ID
            new_chunk = ResearcherChunk(
                chunk_id=new_id,
                text=chunk.text,
                researcher_id=chunk.researcher_id,
                name=chunk.name,
                researcher_name=chunk.researcher_name,
                program=chunk.program,
                department=chunk.department,
                research_interests=chunk.research_interests,
                chunk_type=chunk.chunk_type,
                profile_url=chunk.profile_url
            )
            deduplicated_chunks.append(new_chunk)
        else:
            id_count[chunk.chunk_id] = 1
            deduplicated_chunks.append(chunk)

    return deduplicated_chunks
```

## Database Integration

When creating the ChromaDB database, the chunks are processed as follows:

1. Text content is extracted for embedding
2. Metadata is prepared (converting lists to strings for Chroma compatibility)
3. Chunk IDs are used as document IDs
4. The data is passed to ChromaDB for storage

```python
# Create texts and metadatas
texts = []
metadatas = []
ids = []

for chunk in chunks:
    texts.append(chunk.text)
    # Convert any list fields to strings to ensure compatibility with Chroma
    research_interests_str = "; ".join(chunk.research_interests) if chunk.research_interests else ""
    metadatas.append({
        "researcher_id": chunk.researcher_id,
        "name": chunk.name,
        "researcher_name": chunk.researcher_name,
        "program": chunk.program,
        "department": chunk.department,
        "research_interests": research_interests_str,  # Convert list to string
        "chunk_type": chunk.chunk_type,
        "profile_url": chunk.profile_url,
    })
    ids.append(chunk.chunk_id)

# Create the vector database
db = Chroma.from_texts(
    texts=texts,
    embedding=embedding_function,
    metadatas=metadatas,
    ids=ids,
    persist_directory=settings.vector_db_dir,
    collection_name=settings.collection_name,
)
```

## Benefits of Chunking

The chunking strategy offers several advantages:

1. **Improved Retrieval Relevance**: Smaller chunks allow more precise retrieval of relevant information
2. **Targeted Searches**: Chunking by type (core, interests, publications, grants) enables focused searches
3. **Optimized Token Usage**: Smaller chunks reduce token consumption when using LLMs
4. **Better Context Management**: Related information is kept together in logical chunks
5. **Efficient Updates**: Only affected chunks need to be updated when researcher information changes