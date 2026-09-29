import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

from langchain_groq import ChatGroq


# ---------------------------------------------------------
# Load API key
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# Define the structure we want from the LLM
# ---------------------------------------------------------

class Entity(BaseModel):

    name: str = Field(
        description="Name of the entity"
    )

    type: str = Field(
        description="Type of entity"
    )


class Relationship(BaseModel):

    source: str = Field(
        description="Source entity name"
    )

    relation: str = Field(
        description="Relationship between the entities"
    )

    target: str = Field(
        description="Target entity name"
    )


class Extraction(BaseModel):

    entities: List[Entity]

    relationships: List[Relationship]


# ---------------------------------------------------------
# Load chunks
# ---------------------------------------------------------

with open(
    "chunks.json",
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)


# ---------------------------------------------------------
# Find ISE chunk
# ---------------------------------------------------------

chunk = next(
    c for c in chunks
    if c["chunk_id"] == "chunk_612"
)


print("\nSOURCE:")
print(chunk["source_url"])

print("\nCONTENT:")
print(chunk["content"])


# ---------------------------------------------------------
# Create Groq model
# ---------------------------------------------------------

llm = ChatGroq(

    model="openai/gpt-oss-20b",

    temperature=0,

    max_tokens=2000
)


# ---------------------------------------------------------
# Force structured output
# ---------------------------------------------------------

structured_llm = llm.with_structured_output(
    Extraction
)


# ---------------------------------------------------------
# Prompt
# ---------------------------------------------------------

prompt = f"""
You are building a knowledge graph from the official
NMAM Institute of Technology (NMAMIT) website.

Extract ONLY information explicitly stated in the text.

Do NOT invent entities or relationships.

IMPORTANT:

1. Identify real entities.
2. Use precise entity types.
3. Only create relationships that are supported
   by the text.
4. Do not classify broad concepts as ResearchArea
   unless the text clearly describes them as research.
5. Do not create relationships based on assumptions.
6. If a placement record does not provide a person's name,
   DO NOT invent one. Represent it as a PlacementRecord.

Possible entity types include:

Department
Person
Program
Course
ResearchArea
ResearchTopic
Facility
Organization
Company
Event
Placement
Location
Achievement
Alumni
PlacementRecord
Company
JobRole
Location
PlacementStatistic
Alumni

Possible relationships include:

OFFERS
HAS_FACULTY
WORKS_AT
WORKS_ON
HAS_PROGRAM
HAS_FACILITY
HAS_PLACEMENT
PLACED_AT
PARTICIPATED_IN
LOCATED_IN
ESTABLISHED_IN
HAS_PLACEMENT_RECORD
PLACED_AT
HAS_ROLE
LOCATED_IN
HAS_PLACEMENT_RATE
RECRUITED_BY

Website URL:
{chunk["source_url"]}

Page title:
{chunk["page_title"]}

Department:
{chunk.get("department")}

Text:
{chunk["content"]}
"""


# ---------------------------------------------------------
# Extract
# ---------------------------------------------------------

result = structured_llm.invoke(prompt)


# ---------------------------------------------------------
# Print result
# ---------------------------------------------------------

print("\n")
print("=" * 60)
print("ENTITIES")
print("=" * 60)

for entity in result.entities:

    print(
        f"{entity.name} "
        f"({entity.type})"
    )


print("\n")
print("=" * 60)
print("RELATIONSHIPS")
print("=" * 60)

for relationship in result.relationships:

    print(
        f"{relationship.source}"
        f" --[{relationship.relation}]--> "
        f"{relationship.target}"
    )