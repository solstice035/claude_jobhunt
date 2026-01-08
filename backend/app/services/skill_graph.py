"""
Skill Graph Service - Skill Relationship Graph and Inference

This service builds and traverses a graph of skill relationships to enable
skill inference. If a candidate knows "Kubernetes", we can infer they likely
know "Docker" and "containerization" concepts.

Graph Structure:
- Nodes: Skills (with type attribute: technical, soft, domain)
- Edges: Relationships with types:
  - "broader": Points to more general concept (k8s -> containerization)
  - "narrower": Points to more specific concept (containerization -> k8s)
  - "requires": Prerequisite skill (k8s -> docker)
  - "related": Similar/complementary skill (react -> javascript)

Inference Rules:
1. Broader skills: If you know X, you understand its broader concepts
2. Required skills: If you know X, you likely know its prerequisites
3. Related skills: Weaker inference for similar skills

Usage:
    graph = SkillGraph()
    graph.add_skill("kubernetes", skill_type="technical")
    graph.add_skill("docker", skill_type="technical")
    graph.add_relationship("kubernetes", "docker", relation_type="requires")

    inferred = graph.infer_skills({"kubernetes"})
    # Returns {"kubernetes", "docker"}
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import logging

# Try to import networkx, use simple dict-based fallback if not available
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

logger = logging.getLogger(__name__)


@dataclass
class SkillRelation:
    """
    Represents a relationship between two skills.

    Attributes:
        source: Source skill name
        target: Target skill name
        relation_type: Type of relationship (broader, narrower, requires, related)
        weight: Strength of relationship (0-1, default 1.0)
    """
    source: str
    target: str
    relation_type: str
    weight: float = 1.0


class SkillGraph:
    """
    Graph-based skill relationship manager.

    Builds a directed graph of skills and their relationships,
    enabling skill inference for improved job matching.

    The graph supports multiple relationship types:
    - broader: More general concepts (Python -> Programming)
    - narrower: More specific concepts (Programming -> Python)
    - requires: Prerequisites (Kubernetes -> Docker)
    - related: Similar/complementary (React -> JavaScript)

    Attributes:
        _graph: NetworkX DiGraph or dict-based fallback
        _skills: Set of skill names in graph
    """

    def __init__(self) -> None:
        """Initialize empty skill graph."""
        if HAS_NETWORKX:
            self._graph = nx.DiGraph()
        else:
            # Fallback: adjacency list
            self._adjacency: Dict[str, List[SkillRelation]] = {}
        self._skills: Set[str] = set()
        self._skill_types: Dict[str, str] = {}

    def add_skill(self, name: str, skill_type: str = "technical") -> None:
        """
        Add a skill node to the graph.

        Args:
            name: Skill name (will be lowercased)
            skill_type: Classification (technical, soft, domain, tool)
        """
        name_lower = name.lower().strip()
        self._skills.add(name_lower)
        self._skill_types[name_lower] = skill_type

        if HAS_NETWORKX:
            self._graph.add_node(name_lower, type=skill_type)
        else:
            if name_lower not in self._adjacency:
                self._adjacency[name_lower] = []

    def has_skill(self, name: str) -> bool:
        """Check if a skill exists in the graph."""
        return name.lower().strip() in self._skills

    def add_relationship(
        self,
        source: str,
        target: str,
        relation_type: str,
        weight: float = 1.0
    ) -> None:
        """
        Add a relationship between two skills.

        Args:
            source: Source skill name
            target: Target skill name
            relation_type: Type of relationship (broader, narrower, requires, related)
            weight: Relationship strength (0-1)
        """
        source_lower = source.lower().strip()
        target_lower = target.lower().strip()

        # Ensure both skills exist
        if source_lower not in self._skills:
            self.add_skill(source_lower)
        if target_lower not in self._skills:
            self.add_skill(target_lower)

        relation = SkillRelation(
            source=source_lower,
            target=target_lower,
            relation_type=relation_type,
            weight=weight
        )

        if HAS_NETWORKX:
            self._graph.add_edge(
                source_lower,
                target_lower,
                relation=relation_type,
                weight=weight
            )
        else:
            if source_lower not in self._adjacency:
                self._adjacency[source_lower] = []
            self._adjacency[source_lower].append(relation)

    def get_relationships(self, skill: str) -> List[SkillRelation]:
        """
        Get all outgoing relationships from a skill.

        Args:
            skill: Skill name to get relationships for

        Returns:
            List of SkillRelation objects
        """
        skill_lower = skill.lower().strip()

        if HAS_NETWORKX:
            relations = []
            if skill_lower in self._graph:
                for target in self._graph.successors(skill_lower):
                    edge_data = self._graph.edges[skill_lower, target]
                    relations.append(SkillRelation(
                        source=skill_lower,
                        target=target,
                        relation_type=edge_data.get("relation", "related"),
                        weight=edge_data.get("weight", 1.0)
                    ))
            return relations
        else:
            return self._adjacency.get(skill_lower, [])

    def infer_skills(
        self,
        explicit_skills: Set[str],
        include_related: bool = False,
        max_depth: int = 2
    ) -> Set[str]:
        """
        Expand skill set with inferred skills.

        Inference rules:
        1. Add broader skills (if you know k8s, you know containerization)
        2. Add required skills (if you know k8s, you know docker)
        3. Optionally add related skills (if you know react, you know javascript)

        Args:
            explicit_skills: Set of explicitly stated skills
            include_related: Whether to include related skills (weaker inference)
            max_depth: Maximum traversal depth for transitive inference

        Returns:
            Expanded set including explicit and inferred skills
        """
        # Normalize input skills
        normalized = {s.lower().strip() for s in explicit_skills}
        inferred = set(normalized)

        # Process each explicit skill
        for skill in normalized:
            if skill not in self._skills:
                continue

            # BFS for inference with depth limit
            visited = set()
            queue = [(skill, 0)]

            while queue:
                current, depth = queue.pop(0)
                if current in visited or depth > max_depth:
                    continue
                visited.add(current)

                for relation in self.get_relationships(current):
                    # Infer broader skills
                    if relation.relation_type == "broader":
                        inferred.add(relation.target)
                        if depth < max_depth:
                            queue.append((relation.target, depth + 1))

                    # Infer required/prerequisite skills
                    elif relation.relation_type == "requires":
                        inferred.add(relation.target)

                    # Optionally infer related skills
                    elif relation.relation_type == "related" and include_related:
                        inferred.add(relation.target)

        return inferred

    def get_similar_skills(self, skill: str, limit: int = 10) -> List[str]:
        """
        Find skills similar/related to the given skill.

        Args:
            skill: Skill to find similar skills for
            limit: Maximum number of results

        Returns:
            List of similar skill names
        """
        skill_lower = skill.lower().strip()
        similar = []

        for relation in self.get_relationships(skill_lower):
            if relation.relation_type == "related":
                similar.append(relation.target)

        return similar[:limit]

    def calculate_skill_distance(self, skill1: str, skill2: str) -> float:
        """
        Calculate semantic distance between two skills.

        Uses shortest path length in the graph as a proxy for relatedness.
        Lower distance = more related.

        Args:
            skill1: First skill
            skill2: Second skill

        Returns:
            Distance (0 = same skill, inf = unrelated)
        """
        s1 = skill1.lower().strip()
        s2 = skill2.lower().strip()

        if s1 == s2:
            return 0.0

        if not HAS_NETWORKX:
            # Simple fallback: check direct relationship
            for rel in self.get_relationships(s1):
                if rel.target == s2:
                    return 1.0
            return float('inf')

        try:
            # Use undirected view for distance calculation
            path_length = nx.shortest_path_length(
                self._graph.to_undirected(), s1, s2
            )
            return float(path_length)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float('inf')

    @classmethod
    def from_esco_data(cls, esco_skills: List[dict]) -> "SkillGraph":
        """
        Build skill graph from ESCO skill data.

        Args:
            esco_skills: List of ESCO skill dictionaries with keys:
                - uri, preferred_label, skill_type, broader_skills,
                  narrower_skills, related_skills

        Returns:
            Populated SkillGraph
        """
        graph = cls()

        # Create URI to label mapping
        uri_to_label: Dict[str, str] = {}
        for skill in esco_skills:
            uri = skill.get("uri", "")
            label = skill.get("preferred_label", "").lower()
            if uri and label:
                uri_to_label[uri] = label

        # Add all skills first
        for skill in esco_skills:
            label = skill.get("preferred_label", "").lower()
            skill_type = skill.get("skill_type", "skill")
            if label:
                graph.add_skill(label, skill_type)

        # Add relationships
        for skill in esco_skills:
            label = skill.get("preferred_label", "").lower()
            if not label:
                continue

            # Broader skills
            for broader_uri in skill.get("broader_skills", []):
                broader_label = uri_to_label.get(broader_uri)
                if broader_label:
                    graph.add_relationship(label, broader_label, "broader")

            # Narrower skills
            for narrower_uri in skill.get("narrower_skills", []):
                narrower_label = uri_to_label.get(narrower_uri)
                if narrower_label:
                    graph.add_relationship(label, narrower_label, "narrower")

            # Related skills
            for related_uri in skill.get("related_skills", []):
                related_label = uri_to_label.get(related_uri)
                if related_label:
                    graph.add_relationship(label, related_label, "related")

        return graph

    def get_skill_count(self) -> int:
        """Return number of skills in graph."""
        return len(self._skills)

    def get_relationship_count(self) -> int:
        """Return number of relationships in graph."""
        if HAS_NETWORKX:
            return self._graph.number_of_edges()
        else:
            return sum(len(rels) for rels in self._adjacency.values())


# Pre-built tech skill graph with common relationships
def build_default_skill_graph() -> SkillGraph:
    """
    Build a default skill graph with common tech skill relationships.

    This provides a fallback when ESCO data is not available.

    Returns:
        SkillGraph with common tech skill relationships
    """
    graph = SkillGraph()

    # Languages and their ecosystems
    languages = {
        "python": ["django", "flask", "fastapi", "pandas", "numpy", "pytorch"],
        "javascript": ["react", "angular", "vue", "node.js", "typescript"],
        "java": ["spring", "spring boot", "hibernate"],
        "go": ["gin", "echo"],
        "rust": [],
    }

    # Add languages and their frameworks
    for lang, frameworks in languages.items():
        graph.add_skill(lang, "technical")
        graph.add_skill("programming", "technical")
        graph.add_relationship(lang, "programming", "broader")

        for framework in frameworks:
            graph.add_skill(framework, "technical")
            graph.add_relationship(framework, lang, "requires")

    # Cloud and DevOps
    graph.add_skill("kubernetes", "technical")
    graph.add_skill("docker", "technical")
    graph.add_skill("containerization", "technical")
    graph.add_relationship("kubernetes", "docker", "requires")
    graph.add_relationship("kubernetes", "containerization", "broader")
    graph.add_relationship("docker", "containerization", "broader")

    # Cloud providers
    for provider in ["aws", "azure", "gcp"]:
        graph.add_skill(provider, "technical")
        graph.add_skill("cloud computing", "technical")
        graph.add_relationship(provider, "cloud computing", "broader")

    # Related frontend frameworks
    graph.add_relationship("react", "angular", "related")
    graph.add_relationship("react", "vue", "related")
    graph.add_relationship("angular", "vue", "related")

    # Databases
    for db in ["postgresql", "mysql", "mongodb", "redis"]:
        graph.add_skill(db, "technical")
        graph.add_skill("databases", "technical")
        graph.add_relationship(db, "databases", "broader")

    return graph


# Global default graph instance
_default_graph: Optional[SkillGraph] = None


def get_default_skill_graph() -> SkillGraph:
    """Get or create the default skill graph."""
    global _default_graph
    if _default_graph is None:
        _default_graph = build_default_skill_graph()
    return _default_graph
