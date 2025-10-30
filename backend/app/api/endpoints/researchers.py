"""
Researcher API endpoints.

This module provides endpoints for retrieving researcher information.
Uses dependency injection for the researcher service.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ...models.researcher import ResearcherList, ResearcherProfileSummary, ResearcherProfileDetail
from ...core.security import get_api_key
from ..dependencies import common_parameters, ResearcherServiceDep

router = APIRouter()


class DepartmentList(BaseModel):
    """Department list response model."""
    departments: List[str]


class ProgramList(BaseModel):
    """Program list response model."""
    programs: List[str]


@router.get("/researchers", response_model=ResearcherList)
async def read_researchers(
    service: ResearcherServiceDep,
    commons: dict = Depends(common_parameters),
    department: Optional[str] = None,
    program: Optional[str] = None,
    api_key: str = Depends(get_api_key),
):
    """
    Retrieve a list of researchers with optional filtering.

    Args:
        service: Injected researcher service
        commons: Common pagination parameters
        department: Optional department filter
        program: Optional program filter
        api_key: API key for authentication

    Returns:
        ResearcherList: List of researchers with pagination info
    """
    try:
        researchers = service.list_researchers(
            skip=commons["skip"],
            limit=commons["limit"],
            department=department,
            program=program,
        )
        return researchers
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/researchers/{researcher_id}", response_model=ResearcherProfileDetail)
async def read_researcher(
    researcher_id: str,
    service: ResearcherServiceDep,
    api_key: str = Depends(get_api_key),
):
    """
    Retrieve details for a specific researcher.

    Args:
        researcher_id: ID of the researcher
        service: Injected researcher service
        api_key: API key for authentication

    Returns:
        ResearcherProfileDetail: Detailed researcher profile
    """
    researcher = service.get_researcher_by_id(researcher_id)
    if researcher is None:
        raise HTTPException(status_code=404, detail=f"Researcher {researcher_id} not found")
    return researcher


@router.get("/departments", response_model=DepartmentList)
async def read_departments(
    service: ResearcherServiceDep,
    api_key: str = Depends(get_api_key),
):
    """
    Retrieve a list of all departments.

    Args:
        service: Injected researcher service
        api_key: API key for authentication

    Returns:
        DepartmentList: List of department names
    """
    try:
        departments = service.list_departments()
        return {"departments": departments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/programs", response_model=ProgramList)
async def read_programs(
    service: ResearcherServiceDep,
    api_key: str = Depends(get_api_key),
):
    """
    Retrieve a list of all research programs.

    Args:
        service: Injected researcher service
        api_key: API key for authentication

    Returns:
        ProgramList: List of program names
    """
    try:
        programs = service.list_programs()
        return {"programs": programs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))