#!/usr/bin/env python3
"""
ESCO Skills Import Script

Downloads and imports ESCO (European Skills, Competences, Qualifications and
Occupations) skills data into the local database.

Data Source: https://esco.ec.europa.eu/
Format: CSV files from ESCO download portal

This script can:
1. Download ESCO CSV files from the official portal
2. Parse CSV files and extract skill data
3. Import skills into the esco_skills table
4. Create a seed file with common tech skills (for offline/testing use)

Usage:
    # Import from local CSV file
    python scripts/import_esco.py --csv-file path/to/skills.csv

    # Generate seed data (common tech skills)
    python scripts/import_esco.py --seed

    # Verify import
    python scripts/import_esco.py --verify
"""

import asyncio
import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.database import Base
from app.models.esco import ESCOSkill
from app.config import get_settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

settings = get_settings()


# ==============================================================================
# Seed Data - Common Tech Skills
# ==============================================================================

SEED_SKILLS = [
    # Programming Languages
    {
        "uri": "esco:skill/python",
        "preferred_label": "Python",
        "alt_labels": ["Python programming", "Python3", "py"],
        "description": "Programming using Python language for software development, data analysis, and automation",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/programming"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/django", "esco:skill/flask", "esco:skill/fastapi"]
    },
    {
        "uri": "esco:skill/javascript",
        "preferred_label": "JavaScript",
        "alt_labels": ["JS", "ECMAScript", "ES6", "JavaScript programming"],
        "description": "Programming using JavaScript for web development and application development",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/programming"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/typescript", "esco:skill/react", "esco:skill/nodejs"]
    },
    {
        "uri": "esco:skill/typescript",
        "preferred_label": "TypeScript",
        "alt_labels": ["TS", "TypeScript programming"],
        "description": "Programming using TypeScript, a typed superset of JavaScript",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/javascript"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/javascript", "esco:skill/react", "esco:skill/angular"]
    },
    {
        "uri": "esco:skill/java",
        "preferred_label": "Java",
        "alt_labels": ["Java programming", "JVM", "J2EE"],
        "description": "Programming using Java for enterprise applications and Android development",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/programming"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/spring", "esco:skill/kotlin"]
    },
    {
        "uri": "esco:skill/go",
        "preferred_label": "Go",
        "alt_labels": ["Golang", "Go programming"],
        "description": "Programming using Go for systems programming and cloud services",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/programming"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/kubernetes", "esco:skill/docker"]
    },
    {
        "uri": "esco:skill/rust",
        "preferred_label": "Rust",
        "alt_labels": ["Rust programming", "Rustlang"],
        "description": "Programming using Rust for systems programming with memory safety",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/programming"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/go"]
    },
    {
        "uri": "esco:skill/programming",
        "preferred_label": "Programming",
        "alt_labels": ["Software development", "Coding", "Computer programming"],
        "description": "Writing computer programs to solve problems and automate tasks",
        "skill_type": "skill",
        "broader_skills": [],
        "narrower_skills": ["esco:skill/python", "esco:skill/javascript", "esco:skill/java", "esco:skill/go"],
        "related_skills": []
    },

    # Frontend Frameworks
    {
        "uri": "esco:skill/react",
        "preferred_label": "React",
        "alt_labels": ["ReactJS", "React.js", "React framework"],
        "description": "Building user interfaces using React JavaScript library",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/frontend-development"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/javascript", "esco:skill/typescript", "esco:skill/angular", "esco:skill/vue"]
    },
    {
        "uri": "esco:skill/angular",
        "preferred_label": "Angular",
        "alt_labels": ["AngularJS", "Angular framework"],
        "description": "Building web applications using Angular framework",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/frontend-development"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/typescript", "esco:skill/react", "esco:skill/vue"]
    },
    {
        "uri": "esco:skill/vue",
        "preferred_label": "Vue.js",
        "alt_labels": ["Vue", "VueJS", "Vue framework"],
        "description": "Building user interfaces using Vue.js framework",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/frontend-development"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/javascript", "esco:skill/react", "esco:skill/angular"]
    },
    {
        "uri": "esco:skill/frontend-development",
        "preferred_label": "Frontend Development",
        "alt_labels": ["Front-end development", "UI development", "Client-side development"],
        "description": "Building user interfaces for web and mobile applications",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/web-development"],
        "narrower_skills": ["esco:skill/react", "esco:skill/angular", "esco:skill/vue"],
        "related_skills": ["esco:skill/backend-development"]
    },

    # Backend Frameworks
    {
        "uri": "esco:skill/django",
        "preferred_label": "Django",
        "alt_labels": ["Django framework", "Django REST framework", "DRF"],
        "description": "Building web applications using Django Python framework",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/backend-development"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/python", "esco:skill/flask", "esco:skill/fastapi"]
    },
    {
        "uri": "esco:skill/flask",
        "preferred_label": "Flask",
        "alt_labels": ["Flask framework", "Flask Python"],
        "description": "Building web applications using Flask Python microframework",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/backend-development"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/python", "esco:skill/django", "esco:skill/fastapi"]
    },
    {
        "uri": "esco:skill/fastapi",
        "preferred_label": "FastAPI",
        "alt_labels": ["Fast API", "FastAPI framework"],
        "description": "Building APIs using FastAPI Python framework",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/backend-development"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/python", "esco:skill/django", "esco:skill/flask"]
    },
    {
        "uri": "esco:skill/spring",
        "preferred_label": "Spring",
        "alt_labels": ["Spring framework", "Spring Boot", "SpringBoot"],
        "description": "Building enterprise applications using Spring Java framework",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/backend-development"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/java", "esco:skill/kotlin"]
    },
    {
        "uri": "esco:skill/nodejs",
        "preferred_label": "Node.js",
        "alt_labels": ["NodeJS", "Node", "Node.js runtime"],
        "description": "Building server-side applications using Node.js JavaScript runtime",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/backend-development"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/javascript", "esco:skill/express"]
    },
    {
        "uri": "esco:skill/express",
        "preferred_label": "Express.js",
        "alt_labels": ["Express", "ExpressJS"],
        "description": "Building web applications using Express.js Node.js framework",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/nodejs"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/nodejs", "esco:skill/javascript"]
    },
    {
        "uri": "esco:skill/backend-development",
        "preferred_label": "Backend Development",
        "alt_labels": ["Back-end development", "Server-side development"],
        "description": "Building server-side logic and APIs for applications",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/web-development"],
        "narrower_skills": ["esco:skill/django", "esco:skill/flask", "esco:skill/spring", "esco:skill/nodejs"],
        "related_skills": ["esco:skill/frontend-development"]
    },
    {
        "uri": "esco:skill/web-development",
        "preferred_label": "Web Development",
        "alt_labels": ["Web application development", "Website development"],
        "description": "Building websites and web applications",
        "skill_type": "skill",
        "broader_skills": [],
        "narrower_skills": ["esco:skill/frontend-development", "esco:skill/backend-development"],
        "related_skills": []
    },

    # Cloud & DevOps
    {
        "uri": "esco:skill/aws",
        "preferred_label": "AWS",
        "alt_labels": ["Amazon Web Services", "Amazon AWS", "AWS cloud"],
        "description": "Using Amazon Web Services cloud platform for infrastructure and services",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/cloud-computing"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/azure", "esco:skill/gcp"]
    },
    {
        "uri": "esco:skill/azure",
        "preferred_label": "Azure",
        "alt_labels": ["Microsoft Azure", "Azure cloud"],
        "description": "Using Microsoft Azure cloud platform for infrastructure and services",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/cloud-computing"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/aws", "esco:skill/gcp"]
    },
    {
        "uri": "esco:skill/gcp",
        "preferred_label": "Google Cloud Platform",
        "alt_labels": ["GCP", "Google Cloud", "Google Cloud Services"],
        "description": "Using Google Cloud Platform for infrastructure and services",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/cloud-computing"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/aws", "esco:skill/azure"]
    },
    {
        "uri": "esco:skill/cloud-computing",
        "preferred_label": "Cloud Computing",
        "alt_labels": ["Cloud infrastructure", "Cloud services"],
        "description": "Using cloud platforms for computing, storage, and services",
        "skill_type": "skill",
        "broader_skills": [],
        "narrower_skills": ["esco:skill/aws", "esco:skill/azure", "esco:skill/gcp"],
        "related_skills": ["esco:skill/devops"]
    },
    {
        "uri": "esco:skill/docker",
        "preferred_label": "Docker",
        "alt_labels": ["Docker containers", "Containerization", "Docker Engine"],
        "description": "Building and running applications in Docker containers",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/containerization"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/kubernetes", "esco:skill/devops"]
    },
    {
        "uri": "esco:skill/kubernetes",
        "preferred_label": "Kubernetes",
        "alt_labels": ["K8s", "K8", "Container orchestration"],
        "description": "Managing containerized applications using Kubernetes",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/containerization"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/docker", "esco:skill/devops"]
    },
    {
        "uri": "esco:skill/containerization",
        "preferred_label": "Containerization",
        "alt_labels": ["Container technology", "Containers"],
        "description": "Packaging applications in containers for consistent deployment",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/devops"],
        "narrower_skills": ["esco:skill/docker", "esco:skill/kubernetes"],
        "related_skills": []
    },
    {
        "uri": "esco:skill/terraform",
        "preferred_label": "Terraform",
        "alt_labels": ["Infrastructure as Code", "IaC", "HashiCorp Terraform"],
        "description": "Managing infrastructure as code using Terraform",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/devops"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/aws", "esco:skill/azure", "esco:skill/gcp"]
    },
    {
        "uri": "esco:skill/devops",
        "preferred_label": "DevOps",
        "alt_labels": ["Development Operations", "DevOps engineering"],
        "description": "Combining software development and IT operations practices",
        "skill_type": "skill",
        "broader_skills": [],
        "narrower_skills": ["esco:skill/containerization", "esco:skill/terraform"],
        "related_skills": ["esco:skill/cloud-computing"]
    },

    # Databases
    {
        "uri": "esco:skill/postgresql",
        "preferred_label": "PostgreSQL",
        "alt_labels": ["Postgres", "PSQL", "PostgreSQL database"],
        "description": "Using PostgreSQL relational database management system",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/databases"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/mysql", "esco:skill/sql"]
    },
    {
        "uri": "esco:skill/mysql",
        "preferred_label": "MySQL",
        "alt_labels": ["MySQL database", "MariaDB"],
        "description": "Using MySQL relational database management system",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/databases"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/postgresql", "esco:skill/sql"]
    },
    {
        "uri": "esco:skill/mongodb",
        "preferred_label": "MongoDB",
        "alt_labels": ["Mongo", "MongoDB database", "NoSQL"],
        "description": "Using MongoDB document database",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/databases"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/redis"]
    },
    {
        "uri": "esco:skill/redis",
        "preferred_label": "Redis",
        "alt_labels": ["Redis cache", "Redis database"],
        "description": "Using Redis in-memory data store for caching and messaging",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/databases"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/mongodb"]
    },
    {
        "uri": "esco:skill/sql",
        "preferred_label": "SQL",
        "alt_labels": ["Structured Query Language", "Database queries"],
        "description": "Writing SQL queries for database operations",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/databases"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/postgresql", "esco:skill/mysql"]
    },
    {
        "uri": "esco:skill/databases",
        "preferred_label": "Databases",
        "alt_labels": ["Database management", "Data storage"],
        "description": "Managing and querying databases for data storage",
        "skill_type": "skill",
        "broader_skills": [],
        "narrower_skills": ["esco:skill/postgresql", "esco:skill/mysql", "esco:skill/mongodb", "esco:skill/redis"],
        "related_skills": []
    },

    # AI/ML
    {
        "uri": "esco:skill/machine-learning",
        "preferred_label": "Machine Learning",
        "alt_labels": ["ML", "Predictive modeling", "Statistical modeling"],
        "description": "Building machine learning models for prediction and analysis",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/data-science"],
        "narrower_skills": ["esco:skill/deep-learning"],
        "related_skills": ["esco:skill/python", "esco:skill/pytorch", "esco:skill/tensorflow"]
    },
    {
        "uri": "esco:skill/deep-learning",
        "preferred_label": "Deep Learning",
        "alt_labels": ["Neural networks", "DL", "Deep neural networks"],
        "description": "Building deep neural networks for complex pattern recognition",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/machine-learning"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/pytorch", "esco:skill/tensorflow"]
    },
    {
        "uri": "esco:skill/pytorch",
        "preferred_label": "PyTorch",
        "alt_labels": ["Torch", "PyTorch framework"],
        "description": "Building machine learning models using PyTorch framework",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/machine-learning"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/tensorflow", "esco:skill/python"]
    },
    {
        "uri": "esco:skill/tensorflow",
        "preferred_label": "TensorFlow",
        "alt_labels": ["TF", "TensorFlow framework", "Keras"],
        "description": "Building machine learning models using TensorFlow framework",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/machine-learning"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/pytorch", "esco:skill/python"]
    },
    {
        "uri": "esco:skill/data-science",
        "preferred_label": "Data Science",
        "alt_labels": ["Data analysis", "Analytics", "Data scientist"],
        "description": "Analyzing data to extract insights and build predictive models",
        "skill_type": "skill",
        "broader_skills": [],
        "narrower_skills": ["esco:skill/machine-learning"],
        "related_skills": ["esco:skill/python"]
    },

    # Soft Skills
    {
        "uri": "esco:skill/leadership",
        "preferred_label": "Leadership",
        "alt_labels": ["Team leadership", "Leading teams", "People leadership"],
        "description": "Leading and guiding teams to achieve objectives",
        "skill_type": "skill",
        "broader_skills": [],
        "narrower_skills": ["esco:skill/mentoring"],
        "related_skills": ["esco:skill/communication", "esco:skill/stakeholder-management"]
    },
    {
        "uri": "esco:skill/communication",
        "preferred_label": "Communication",
        "alt_labels": ["Written communication", "Verbal communication", "Presentation skills"],
        "description": "Effectively communicating ideas and information",
        "skill_type": "skill",
        "broader_skills": [],
        "narrower_skills": [],
        "related_skills": ["esco:skill/leadership", "esco:skill/stakeholder-management"]
    },
    {
        "uri": "esco:skill/stakeholder-management",
        "preferred_label": "Stakeholder Management",
        "alt_labels": ["Stakeholder engagement", "Client management", "Stakeholder relations"],
        "description": "Managing relationships and expectations with stakeholders",
        "skill_type": "skill",
        "broader_skills": [],
        "narrower_skills": [],
        "related_skills": ["esco:skill/communication", "esco:skill/leadership"]
    },
    {
        "uri": "esco:skill/mentoring",
        "preferred_label": "Mentoring",
        "alt_labels": ["Coaching", "Mentorship", "Training others"],
        "description": "Guiding and developing others' skills and careers",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/leadership"],
        "narrower_skills": [],
        "related_skills": ["esco:skill/leadership"]
    },
    {
        "uri": "esco:skill/agile",
        "preferred_label": "Agile",
        "alt_labels": ["Agile methodology", "Scrum", "Agile development", "Sprint planning"],
        "description": "Working with Agile methodologies for iterative development",
        "skill_type": "skill",
        "broader_skills": ["esco:skill/project-management"],
        "narrower_skills": [],
        "related_skills": []
    },
    {
        "uri": "esco:skill/project-management",
        "preferred_label": "Project Management",
        "alt_labels": ["PM", "Programme management", "Project planning"],
        "description": "Planning, executing, and managing projects",
        "skill_type": "skill",
        "broader_skills": [],
        "narrower_skills": ["esco:skill/agile"],
        "related_skills": ["esco:skill/leadership"]
    },
]


# ==============================================================================
# Database Operations
# ==============================================================================

async def get_async_session() -> AsyncSession:
    """Create an async database session."""
    database_url = settings.database_url.replace("sqlite:///", "sqlite+aiosqlite:///")
    engine = create_async_engine(database_url, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return async_session()


async def import_seed_data(session: AsyncSession) -> int:
    """Import seed skills into database."""
    count = 0

    for skill_data in SEED_SKILLS:
        # Check if skill already exists
        existing = await session.execute(
            select(ESCOSkill).where(ESCOSkill.uri == skill_data["uri"])
        )
        if existing.scalars().first():
            logger.debug(f"Skill already exists: {skill_data['preferred_label']}")
            continue

        skill = ESCOSkill(
            uri=skill_data["uri"],
            preferred_label=skill_data["preferred_label"],
            alt_labels=skill_data["alt_labels"],
            description=skill_data["description"],
            skill_type=skill_data["skill_type"],
            broader_skills=skill_data["broader_skills"],
            narrower_skills=skill_data["narrower_skills"],
            related_skills=skill_data["related_skills"],
        )
        session.add(skill)
        count += 1
        logger.info(f"Imported: {skill.preferred_label}")

    await session.commit()
    return count


async def import_csv_file(session: AsyncSession, csv_path: Path) -> int:
    """Import skills from ESCO CSV file."""
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return 0

    count = 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            uri = row.get("conceptUri", row.get("uri", ""))
            if not uri:
                continue

            # Check if skill already exists
            existing = await session.execute(
                select(ESCOSkill).where(ESCOSkill.uri == uri)
            )
            if existing.scalars().first():
                continue

            # Parse alt labels
            alt_labels_str = row.get("altLabels", row.get("alt_labels", ""))
            alt_labels = [s.strip() for s in alt_labels_str.split("\n") if s.strip()] if alt_labels_str else []

            skill = ESCOSkill(
                uri=uri,
                preferred_label=row.get("preferredLabel", row.get("preferred_label", "")),
                alt_labels=alt_labels,
                description=row.get("description", ""),
                skill_type=row.get("skillType", row.get("skill_type", "skill")),
                broader_skills=[],  # Would need separate relationship file
                narrower_skills=[],
                related_skills=[],
            )
            session.add(skill)
            count += 1

            if count % 100 == 0:
                logger.info(f"Imported {count} skills...")
                await session.commit()

    await session.commit()
    return count


async def verify_import(session: AsyncSession) -> None:
    """Verify import by checking skill counts."""
    result = await session.execute(select(ESCOSkill))
    skills = result.scalars().all()

    logger.info(f"Total skills in database: {len(skills)}")

    # Sample skills
    logger.info("\nSample skills:")
    for skill in skills[:5]:
        logger.info(f"  - {skill.preferred_label} ({skill.skill_type})")
        if skill.alt_labels:
            logger.info(f"    Alt labels: {', '.join(skill.alt_labels[:3])}")


# ==============================================================================
# Main
# ==============================================================================

async def main() -> None:
    parser = argparse.ArgumentParser(description="Import ESCO skills data")
    parser.add_argument("--csv-file", type=Path, help="Path to ESCO CSV file")
    parser.add_argument("--seed", action="store_true", help="Import seed tech skills")
    parser.add_argument("--verify", action="store_true", help="Verify import")

    args = parser.parse_args()

    session = await get_async_session()

    try:
        if args.seed:
            logger.info("Importing seed skills...")
            count = await import_seed_data(session)
            logger.info(f"Imported {count} seed skills")

        elif args.csv_file:
            logger.info(f"Importing from CSV: {args.csv_file}")
            count = await import_csv_file(session, args.csv_file)
            logger.info(f"Imported {count} skills from CSV")

        elif args.verify:
            await verify_import(session)

        else:
            # Default: import seed data
            logger.info("No arguments provided, importing seed skills...")
            count = await import_seed_data(session)
            logger.info(f"Imported {count} seed skills")

    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(main())
