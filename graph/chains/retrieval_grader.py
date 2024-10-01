from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm= ChatOpenAI(temperature=0)

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str =Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

structured_llm_grader= llm.with_structured_output(GradeDocuments)

system_prompt="""
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. 
\n If the document contains keyword or semantic meaning related to question, grade it as relevant.
\n Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
"""


grade_prompt= ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Retrieved document: {document} User question: {question}")
    ]
)

retrival_grader= grade_prompt | structured_llm_grader
