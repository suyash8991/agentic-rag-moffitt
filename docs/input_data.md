# Input Data and ChromaDB Storage

## Data Structure Overview

The Moffitt Agentic RAG system uses researcher profile data that's stored as JSON files in the `data/processed/` directory. Each researcher profile contains information such as:

- **Biographical information**: Name, degrees, title, department, program
- **Research information**: Overview, research interests, associations
- **Publications and grants**: List of publications and research grants
- **Education and contact**: Educational background and contact information

Here's an example of a researcher profile JSON structure:

```json
{
  "profile_url": "https://www.moffitt.org/research-science/researchers/ahmad-tarhini",
  "last_updated": "2025-10-14T22:06:29.132821",
  "researcher_id": "24764",
  "name": "",
  "researcher_name": "Ahmad Tarhini",
  "degrees": ["MD", "PhD"],
  "title": "",
  "primary_program": "Cutaneous Oncology",
  "research_program": "Molecular Medicine Program,Immuno-Oncology Program",
  "overview": "As a clinical and translational physician-scientist...",
  "research_interests": ["As a clinical and translational physician-scientist..."],
  "associations": [
    "Cutaneous Oncology",
    "Immunology",
    "Melanoma & Skin Cancer Center of Excellence",
    "Molecular Medicine Program",
    "Immuno-Oncology Program"
  ],
  "publications": [
    {
      "title": "Long GV, Nair N, Marbach D...",
      "pubmed_id": "40993242",
      "year": "2025"
    }
  ],
  "grants": [
    {
      "description": "Title: Predictors of Immunotherapeutic Benefits..."
    }
  ],
  "education": [
    {
      "type": "Medical School",
      "institution": "Lithuanian University of Health Sciences",
      "degree": "MD"
    }
  ],
  "photo_url": "https://www.moffitt.org/globalassets/images/researchers_bio/TarhiniAhmad_24764.jpg",
  "contact": {
    "contact_url": "https://eforms.moffitt.org/ContactResearchersForm?PERID=24764"
  },
  "content_hash": "702622078a5967c70161a3489d63054e536fcec39bf6cbae8f30ec78c9007da9"
}
```

## Data Loading Process

The data loading process is implemented in `src/moffitt_rag/data/loader.py`:

1. **Individual profile loading** (`load_researcher_profile`):
   - Opens a JSON file
   - Parses it into a `ResearcherProfile` Pydantic model

2. **Batch loading** (`load_all_researcher_profiles`):
   - Lists all JSON files in the processed directory
   - Loads each file using the individual loader
   - Returns a list of `ResearcherProfile` objects

```python
# Example of loading all profiles
profiles = load_all_researcher_profiles()
```

## ChromaDB Storage

Researcher profiles are stored in ChromaDB, a vector database that enables semantic search:

1. **Database Creation** (`create_vector_db` in `src/moffitt_rag/db/vector_store.py`):
   - Takes chunks of researcher profiles (created during chunking process)
   - Embeds the text content using the HuggingFace SentenceTransformers model
   - Stores the embeddings along with metadata in ChromaDB

2. **Database Loading** (`load_vector_db`):
   - Loads an existing database from disk
   - Connects it with the embedding function

3. **Access Pattern** (`get_or_create_vector_db`):
   - Attempts to load an existing database
   - If not found, creates a new one

```python
# Example of getting or creating the vector database
db = get_or_create_vector_db()
```

## Database Structure

The ChromaDB database is organized as follows:

- **Collection**: `moffitt_researchers` (configurable in settings)
- **Documents**: Textual content of researcher profiles, chunked for efficient retrieval
- **Embeddings**: Vector representations of the text chunks
- **Metadata**: Additional information for filtering and retrieval, including:
  - `researcher_id`: Unique identifier for the researcher
  - `name`: Researcher's name
  - `researcher_name`: Another field for researcher's name (added to fix issues)
  - `program`: Researcher's primary program
  - `department`: Researcher's department
  - `research_interests`: Research interests (converted to string)
  - `chunk_type`: Type of chunk (core, interests, publications, grants)
  - `profile_url`: URL to the researcher's profile

## Data Model Implementation

The data models are defined in `src/moffitt_rag/data/models.py` using Pydantic:

```python
class ResearcherProfile(BaseModel):
    # Required fields
    researcher_id: str
    profile_url: str
    content_hash: str
    last_updated: datetime

    # Biographical information
    name: str = ""
    researcher_name: str = ""  # Added field to match JSON structure
    degrees: List[str] = Field(default_factory=list)
    title: Optional[str] = None
    primary_program: Optional[str] = None
    research_program: Optional[str] = None
    department: Optional[str] = None

    # Research information
    overview: Optional[str] = None
    research_interests: List[str] = Field(default_factory=list)
    associations: List[str] = Field(default_factory=list)

    # Publications and grants
    publications: List[Publication] = Field(default_factory=list)
    grants: List[Grant] = Field(default_factory=list)

    # Education and contact
    education: List[Education] = Field(default_factory=list)
    contact: Optional[Contact] = None

    # Additional fields
    photo_url: Optional[str] = None
```

The model includes methods for text conversion (`to_text()`) and document creation (`to_document()`) used in the embedding process.