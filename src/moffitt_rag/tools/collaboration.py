"""
CollaborationTool for the Moffitt Agentic RAG system.

This module implements a tool for discovering potential collaborations
between researchers or departments.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set

from langchain.tools import BaseTool

from ..data.loader import load_all_researcher_profiles
from ..data.models import ResearcherProfile
from ..models.embeddings import embed_query, embedding_similarity
from ..config.config import get_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CollaborationTool(BaseTool):
    """
    Tool for discovering potential collaborations between researchers or departments.

    This tool can find potential collaborations based on:
    - Interdepartmental connections
    - Researcher similarity
    - Complementary research interests
    """

    name = "Collaboration"
    description = "Find potential collaborations between researchers or departments"

    def _run(self, query: str) -> str:
        """
        Run the tool with the given query.

        Args:
            query (str): The query, which can be:
                         - "between [dept1] and [dept2]" for interdepartmental collaborations
                         - "for [researcher]" for collaborators for a specific researcher
                         - General query for suggested collaborations

        Returns:
            str: The collaboration suggestions formatted as a string
        """
        # Check the type of collaboration query
        if "between" in query.lower() and any(term in query.lower() for term in ["department", "program"]):
            return self._find_interdepartmental_collaborations(query)
        elif any(term in query.lower() for term in ["for", "with"]) and not "between" in query.lower():
            # Looking for collaborators for a specific researcher
            return self._find_researcher_collaborations(query)
        else:
            # General collaboration suggestions
            return self._suggest_collaborations(query)

    def _find_interdepartmental_collaborations(self, query: str) -> str:
        """
        Find potential collaborations between two departments or programs.

        Args:
            query (str): The query containing "between [dept1] and [dept2]"

        Returns:
            str: The potential collaborations formatted as a string
        """
        logger.info(f"Finding interdepartmental collaborations: {query}")

        # Extract the two departments/programs
        match = re.search(r"between\s+([A-Za-z\s&]+)\s+and\s+([A-Za-z\s&]+)", query.lower())
        if not match:
            return "Please specify two departments or programs, e.g., 'Find collaborations between Biostatistics and Cancer Epidemiology'"

        dept1 = match.group(1).strip()
        dept2 = match.group(2).strip()

        # Load all profiles
        profiles = load_all_researcher_profiles()

        # Find researchers in each department/program
        researchers_dept1 = []
        researchers_dept2 = []

        for profile in profiles:
            # Check first department/program
            is_in_dept1 = False
            if profile.department and dept1.lower() in profile.department.lower():
                is_in_dept1 = True
            elif profile.primary_program and dept1.lower() in profile.primary_program.lower():
                is_in_dept1 = True

            # Check second department/program
            is_in_dept2 = False
            if profile.department and dept2.lower() in profile.department.lower():
                is_in_dept2 = True
            elif profile.primary_program and dept2.lower() in profile.primary_program.lower():
                is_in_dept2 = True

            # Add to respective lists
            if is_in_dept1:
                researchers_dept1.append(profile)
            if is_in_dept2:
                researchers_dept2.append(profile)

        # Check if we found researchers in both departments
        if not researchers_dept1:
            return f"No researchers found in '{dept1}'. Please check the department/program name."
        if not researchers_dept2:
            return f"No researchers found in '{dept2}'. Please check the department/program name."

        # Find potential collaborations based on research interests
        collaborations = self._find_collaborations_between_groups(researchers_dept1, researchers_dept2)

        # Format results
        if not collaborations:
            return f"No strong potential collaborations found between {dept1} and {dept2}"

        formatted_results = [
            f"Potential collaborations between {dept1} and {dept2}:\n"
        ]

        for r1, r2, similarity, common_interests in collaborations[:5]:
            formatted_results.append(f"- {r1.name} ({dept1}) and {r2.name} ({dept2})")

            if common_interests:
                interest_str = " & ".join(common_interests[:2])
                formatted_results.append(f"  Common research areas: {interest_str}")

        return "\n".join(formatted_results)

    def _find_researcher_collaborations(self, query: str) -> str:
        """
        Find potential collaborators for a specific researcher.

        Args:
            query (str): The query containing "for [researcher]" or "with [researcher]"

        Returns:
            str: The potential collaborators formatted as a string
        """
        logger.info(f"Finding collaborators for a specific researcher: {query}")

        # Extract the researcher name
        match = None
        if "for" in query.lower():
            match = re.search(r"for\s+([A-Za-z\s\.-]+)", query.lower())
        elif "with" in query.lower():
            match = re.search(r"with\s+([A-Za-z\s\.-]+)", query.lower())

        if not match:
            return "Please specify a researcher name, e.g., 'Find collaborators for John Cleveland'"

        researcher_name = match.group(1).strip()

        # Load all profiles
        profiles = load_all_researcher_profiles()

        # Find the named researcher
        target_profile = None
        for profile in profiles:
            if researcher_name.lower() in profile.name.lower():
                target_profile = profile
                break

        if not target_profile:
            return f"Could not find a researcher named '{researcher_name}'. Please check the name and try again."

        # Find potential collaborators
        collaborators = self._find_collaborators_for_researcher(target_profile, profiles)

        # Format results
        if not collaborators:
            return f"No strong potential collaborators found for {target_profile.name}"

        formatted_results = [
            f"Potential collaborators for {target_profile.name} ({target_profile.primary_program or 'Unknown Program'}):\n"
        ]

        for profile, similarity, common_interests in collaborators[:5]:
            formatted_results.append(
                f"- {profile.name} ({profile.primary_program or 'Unknown Program'})"
            )

            if common_interests:
                interest_str = " & ".join(common_interests[:2])
                formatted_results.append(f"  Common research areas: {interest_str}")

        return "\n".join(formatted_results)

    def _suggest_collaborations(self, query: str) -> str:
        """
        Suggest potential collaborations based on a general query.

        Args:
            query (str): The general collaboration query

        Returns:
            str: The collaboration suggestions formatted as a string
        """
        logger.info(f"Suggesting collaborations based on query: {query}")

        # Extract key research areas from the query
        # This is a simple approach - in a real system, you might use
        # entity extraction or topic modeling
        research_areas = [term.strip() for term in query.split(',')]
        if len(research_areas) == 1 and len(research_areas[0].split()) > 3:
            # If it's not a comma-separated list but a longer text,
            # extract key phrases using a simplified approach
            words = query.lower().split()
            research_areas = []
            for i in range(len(words) - 1):
                if words[i] in ["cancer", "research", "study", "therapy", "treatment", "analysis"]:
                    research_areas.append(" ".join(words[i:i+2]))

        if not research_areas:
            return "Please specify research areas or departments for collaboration suggestions."

        # Load all profiles
        profiles = load_all_researcher_profiles()

        # Group researchers by department/program
        dept_researchers = {}
        for profile in profiles:
            if profile.department:
                dept_researchers.setdefault(profile.department, []).append(profile)
            elif profile.primary_program:
                dept_researchers.setdefault(profile.primary_program, []).append(profile)

        # Find departments with researchers matching the research areas
        matching_depts = []
        for dept, researchers in dept_researchers.items():
            for area in research_areas:
                for researcher in researchers:
                    interests = " ".join(researcher.research_interests) if researcher.research_interests else ""
                    overview = researcher.overview or ""
                    if area.lower() in interests.lower() or area.lower() in overview.lower():
                        matching_depts.append(dept)
                        break

        if len(matching_depts) < 2:
            # Not enough matching departments for interdepartmental suggestions
            return "Could not find multiple departments matching the research areas. Please try a more specific query."

        # Take the top 2 matching departments
        dept1 = matching_depts[0]
        dept2 = matching_depts[1]

        # Find collaborations between these departments
        collaborations = self._find_collaborations_between_groups(
            dept_researchers.get(dept1, []),
            dept_researchers.get(dept2, [])
        )

        # Format results
        if not collaborations:
            return f"No strong potential collaborations found for the research areas: {', '.join(research_areas)}"

        formatted_results = [
            f"Suggested collaborations for research in {', '.join(research_areas)}:\n"
        ]

        for r1, r2, similarity, common_interests in collaborations[:5]:
            formatted_results.append(
                f"- {r1.name} ({r1.primary_program or r1.department or 'Unknown'}) and "
                f"{r2.name} ({r2.primary_program or r2.department or 'Unknown'})"
            )

            if common_interests:
                interest_str = " & ".join(common_interests[:2])
                formatted_results.append(f"  Common research areas: {interest_str}")

        return "\n".join(formatted_results)

    def _find_collaborations_between_groups(
        self,
        group1: List[ResearcherProfile],
        group2: List[ResearcherProfile]
    ) -> List[Tuple[ResearcherProfile, ResearcherProfile, float, List[str]]]:
        """
        Find potential collaborations between two groups of researchers.

        Args:
            group1 (List[ResearcherProfile]): The first group of researchers
            group2 (List[ResearcherProfile]): The second group of researchers

        Returns:
            List[Tuple[ResearcherProfile, ResearcherProfile, float, List[str]]]:
                List of tuples containing:
                - First researcher
                - Second researcher
                - Similarity score
                - List of common research interests
        """
        collaborations = []

        for r1 in group1:
            for r2 in group2:
                # Skip if same researcher
                if r1.researcher_id == r2.researcher_id:
                    continue

                # Calculate similarity between research interests
                r1_interests = " ".join(r1.research_interests) if r1.research_interests else ""
                if r1.overview:
                    r1_interests += " " + r1.overview

                r2_interests = " ".join(r2.research_interests) if r2.research_interests else ""
                if r2.overview:
                    r2_interests += " " + r2.overview

                similarity = 0.0
                if r1_interests and r2_interests:
                    # Calculate similarity using embeddings
                    try:
                        r1_embedding = embed_query(r1_interests)
                        r2_embedding = embed_query(r2_interests)
                        similarity = embedding_similarity(r1_embedding, r2_embedding)
                    except Exception as e:
                        logger.warning(f"Error calculating similarity: {e}")
                        # Fall back to a simpler method
                        common_words = set(r1_interests.lower().split()) & set(r2_interests.lower().split())
                        total_words = set(r1_interests.lower().split()) | set(r2_interests.lower().split())
                        similarity = len(common_words) / len(total_words) if total_words else 0

                # Find common research topics
                common_interests = self._find_common_interests(r1, r2)

                # Only include if there's meaningful similarity
                if similarity > 0.5 or common_interests:
                    collaborations.append((r1, r2, similarity, common_interests))

        # Sort by similarity score
        collaborations.sort(key=lambda x: x[2], reverse=True)
        return collaborations

    def _find_collaborators_for_researcher(
        self,
        researcher: ResearcherProfile,
        all_profiles: List[ResearcherProfile]
    ) -> List[Tuple[ResearcherProfile, float, List[str]]]:
        """
        Find potential collaborators for a specific researcher.

        Args:
            researcher (ResearcherProfile): The target researcher
            all_profiles (List[ResearcherProfile]): All researcher profiles

        Returns:
            List[Tuple[ResearcherProfile, float, List[str]]]:
                List of tuples containing:
                - Collaborator profile
                - Similarity score
                - List of common research interests
        """
        collaborators = []

        # Create embedding for the researcher's interests
        r1_interests = " ".join(researcher.research_interests) if researcher.research_interests else ""
        if researcher.overview:
            r1_interests += " " + researcher.overview

        r1_embedding = None
        if r1_interests:
            try:
                r1_embedding = embed_query(r1_interests)
            except Exception as e:
                logger.warning(f"Error creating embedding for target researcher: {e}")

        for profile in all_profiles:
            # Skip if same researcher
            if profile.researcher_id == researcher.researcher_id:
                continue

            # Skip if same department (look for cross-departmental collaborations)
            if researcher.department and profile.department == researcher.department:
                continue

            # Calculate similarity between research interests
            r2_interests = " ".join(profile.research_interests) if profile.research_interests else ""
            if profile.overview:
                r2_interests += " " + profile.overview

            similarity = 0.0
            if r1_interests and r2_interests and r1_embedding is not None:
                # Calculate similarity using embeddings
                try:
                    r2_embedding = embed_query(r2_interests)
                    similarity = embedding_similarity(r1_embedding, r2_embedding)
                except Exception as e:
                    logger.warning(f"Error calculating similarity: {e}")
                    # Fall back to a simpler method
                    common_words = set(r1_interests.lower().split()) & set(r2_interests.lower().split())
                    total_words = set(r1_interests.lower().split()) | set(r2_interests.lower().split())
                    similarity = len(common_words) / len(total_words) if total_words else 0

            # Find common research topics
            common_interests = self._find_common_interests(researcher, profile)

            # Only include if there's meaningful similarity
            if similarity > 0.5 or common_interests:
                collaborators.append((profile, similarity, common_interests))

        # Sort by similarity score
        collaborators.sort(key=lambda x: x[1], reverse=True)
        return collaborators

    def _find_common_interests(
        self,
        r1: ResearcherProfile,
        r2: ResearcherProfile
    ) -> List[str]:
        """
        Find common research interests between two researchers.

        Args:
            r1 (ResearcherProfile): The first researcher
            r2 (ResearcherProfile): The second researcher

        Returns:
            List[str]: List of common research interests
        """
        r1_interests = r1.research_interests or []
        r2_interests = r2.research_interests or []

        # Direct overlap (exact matches)
        common_exact = set(r1_interests) & set(r2_interests)
        if common_exact:
            return list(common_exact)

        # Look for partial matches using term overlap
        common_interests = []
        for i1 in r1_interests:
            for i2 in r2_interests:
                i1_terms = set(i1.lower().split())
                i2_terms = set(i2.lower().split())

                # Check if there's significant term overlap
                common_terms = i1_terms & i2_terms
                if len(common_terms) >= 2 or (len(common_terms) == 1 and len(i1_terms) <= 3 and len(i2_terms) <= 3):
                    # Use the shorter interest as the common one
                    common_interest = i1 if len(i1) < len(i2) else i2
                    common_interests.append(common_interest)

        return common_interests

    async def _arun(self, query: str) -> str:
        """
        Run the tool asynchronously with the given query.

        Args:
            query (str): The query to search for

        Returns:
            str: The search results formatted as a string
        """
        # For now, just call the synchronous version
        return self._run(query)