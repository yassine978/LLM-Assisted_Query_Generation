"""Load sample data into Neo4j for Phase 4.

This script creates a social network graph with:
- Person nodes (users)
- Company nodes (employers)
- Technology nodes (skills)
- FRIENDS_WITH relationships
- WORKS_AT relationships
- KNOWS_SKILL relationships
"""

import codecs
import sys
from pathlib import Path

# Configure UTF-8 output for Windows console
if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, errors="replace")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, errors="replace")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from neo4j import GraphDatabase
from src.utils.config import get_settings

# Neo4j connection settings
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"


def clear_database(tx):
    """Clear all nodes and relationships."""
    tx.run("MATCH (n) DETACH DELETE n")
    print("✓ Cleared existing data")


def create_constraints(tx):
    """Create constraints and indexes."""
    # Create uniqueness constraints
    tx.run("CREATE CONSTRAINT person_email IF NOT EXISTS FOR (p:Person) REQUIRE p.email IS UNIQUE")
    tx.run("CREATE CONSTRAINT company_name IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE")
    tx.run("CREATE CONSTRAINT tech_name IF NOT EXISTS FOR (t:Technology) REQUIRE t.name IS UNIQUE")
    print("✓ Created constraints")


def create_people(tx):
    """Create Person nodes."""
    people = [
        {
            "name": "Alice Johnson",
            "email": "alice.j@email.com",
            "age": 28,
            "city": "New York",
            "country": "USA",
            "role": "Software Engineer"
        },
        {
            "name": "Bob Smith",
            "email": "bob.s@email.com",
            "age": 34,
            "city": "San Francisco",
            "country": "USA",
            "role": "Data Scientist"
        },
        {
            "name": "Charlie Brown",
            "email": "charlie.b@email.com",
            "age": 45,
            "city": "London",
            "country": "UK",
            "role": "Engineering Manager"
        },
        {
            "name": "Diana Prince",
            "email": "diana.p@email.com",
            "age": 29,
            "city": "New York",
            "country": "USA",
            "role": "UX Designer"
        },
        {
            "name": "Eve Martinez",
            "email": "eve.m@email.com",
            "age": 31,
            "city": "Boston",
            "country": "USA",
            "role": "Product Manager"
        },
        {
            "name": "Frank Wilson",
            "email": "frank.w@email.com",
            "age": 52,
            "city": "Seattle",
            "country": "USA",
            "role": "CTO"
        },
        {
            "name": "Grace Lee",
            "email": "grace.l@email.com",
            "age": 26,
            "city": "San Francisco",
            "country": "USA",
            "role": "Frontend Developer"
        },
        {
            "name": "Henry Davis",
            "email": "henry.d@email.com",
            "age": 38,
            "city": "Austin",
            "country": "USA",
            "role": "DevOps Engineer"
        }
    ]

    for person in people:
        tx.run("""
            CREATE (p:Person {
                name: $name,
                email: $email,
                age: $age,
                city: $city,
                country: $country,
                role: $role
            })
        """, **person)

    print(f"✓ Created {len(people)} people")


def create_companies(tx):
    """Create Company nodes."""
    companies = [
        {
            "name": "TechCorp",
            "industry": "Technology",
            "size": "Large",
            "founded": 2010,
            "location": "San Francisco"
        },
        {
            "name": "DataSystems Inc",
            "industry": "Data Analytics",
            "size": "Medium",
            "founded": 2015,
            "location": "New York"
        },
        {
            "name": "CloudNine Solutions",
            "industry": "Cloud Services",
            "size": "Large",
            "founded": 2012,
            "location": "Seattle"
        },
        {
            "name": "StartupHub",
            "industry": "Incubator",
            "size": "Small",
            "founded": 2020,
            "location": "Austin"
        }
    ]

    for company in companies:
        tx.run("""
            CREATE (c:Company {
                name: $name,
                industry: $industry,
                size: $size,
                founded: $founded,
                location: $location
            })
        """, **company)

    print(f"✓ Created {len(companies)} companies")


def create_technologies(tx):
    """Create Technology nodes."""
    technologies = [
        {"name": "Python", "category": "Programming Language", "popularity": "High"},
        {"name": "JavaScript", "category": "Programming Language", "popularity": "High"},
        {"name": "Java", "category": "Programming Language", "popularity": "High"},
        {"name": "React", "category": "Frontend Framework", "popularity": "High"},
        {"name": "Node.js", "category": "Backend Framework", "popularity": "High"},
        {"name": "Docker", "category": "DevOps", "popularity": "High"},
        {"name": "Kubernetes", "category": "DevOps", "popularity": "Medium"},
        {"name": "PostgreSQL", "category": "Database", "popularity": "High"},
        {"name": "MongoDB", "category": "Database", "popularity": "High"},
        {"name": "Machine Learning", "category": "AI/ML", "popularity": "High"},
        {"name": "AWS", "category": "Cloud Platform", "popularity": "High"},
        {"name": "Neo4j", "category": "Database", "popularity": "Medium"}
    ]

    for tech in technologies:
        tx.run("""
            CREATE (t:Technology {
                name: $name,
                category: $category,
                popularity: $popularity
            })
        """, **tech)

    print(f"✓ Created {len(technologies)} technologies")


def create_relationships(tx):
    """Create relationships between nodes."""

    # FRIENDS_WITH relationships
    friendships = [
        ("alice.j@email.com", "bob.s@email.com", 2020),
        ("alice.j@email.com", "diana.p@email.com", 2021),
        ("bob.s@email.com", "charlie.b@email.com", 2018),
        ("bob.s@email.com", "grace.l@email.com", 2022),
        ("charlie.b@email.com", "frank.w@email.com", 2015),
        ("diana.p@email.com", "grace.l@email.com", 2021),
        ("eve.m@email.com", "alice.j@email.com", 2019),
        ("eve.m@email.com", "henry.d@email.com", 2020),
        ("henry.d@email.com", "frank.w@email.com", 2017)
    ]

    for email1, email2, since in friendships:
        tx.run("""
            MATCH (p1:Person {email: $email1})
            MATCH (p2:Person {email: $email2})
            CREATE (p1)-[:FRIENDS_WITH {since: $since}]->(p2)
            CREATE (p2)-[:FRIENDS_WITH {since: $since}]->(p1)
        """, email1=email1, email2=email2, since=since)

    print(f"✓ Created {len(friendships) * 2} friendship relationships")

    # WORKS_AT relationships
    employment = [
        ("alice.j@email.com", "TechCorp", 2022, "Software Engineer"),
        ("bob.s@email.com", "TechCorp", 2020, "Data Scientist"),
        ("charlie.b@email.com", "DataSystems Inc", 2018, "Engineering Manager"),
        ("diana.p@email.com", "DataSystems Inc", 2021, "UX Designer"),
        ("eve.m@email.com", "CloudNine Solutions", 2019, "Product Manager"),
        ("frank.w@email.com", "CloudNine Solutions", 2015, "CTO"),
        ("grace.l@email.com", "TechCorp", 2022, "Frontend Developer"),
        ("henry.d@email.com", "StartupHub", 2020, "DevOps Engineer")
    ]

    for email, company, year, position in employment:
        tx.run("""
            MATCH (p:Person {email: $email})
            MATCH (c:Company {name: $company})
            CREATE (p)-[:WORKS_AT {since: $year, position: $position}]->(c)
        """, email=email, company=company, year=year, position=position)

    print(f"✓ Created {len(employment)} employment relationships")

    # KNOWS_SKILL relationships
    skills = [
        ("alice.j@email.com", "Python", 8),
        ("alice.j@email.com", "JavaScript", 7),
        ("alice.j@email.com", "Docker", 6),
        ("bob.s@email.com", "Python", 9),
        ("bob.s@email.com", "Machine Learning", 8),
        ("bob.s@email.com", "PostgreSQL", 7),
        ("charlie.b@email.com", "Java", 9),
        ("charlie.b@email.com", "Kubernetes", 8),
        ("charlie.b@email.com", "AWS", 9),
        ("diana.p@email.com", "JavaScript", 8),
        ("diana.p@email.com", "React", 9),
        ("diana.p@email.com", "Node.js", 6),
        ("eve.m@email.com", "Python", 6),
        ("eve.m@email.com", "AWS", 7),
        ("frank.w@email.com", "Python", 9),
        ("frank.w@email.com", "Java", 9),
        ("frank.w@email.com", "Kubernetes", 9),
        ("frank.w@email.com", "AWS", 10),
        ("grace.l@email.com", "JavaScript", 9),
        ("grace.l@email.com", "React", 10),
        ("grace.l@email.com", "Docker", 7),
        ("henry.d@email.com", "Docker", 10),
        ("henry.d@email.com", "Kubernetes", 10),
        ("henry.d@email.com", "AWS", 9),
        ("henry.d@email.com", "Python", 7)
    ]

    for email, tech, level in skills:
        tx.run("""
            MATCH (p:Person {email: $email})
            MATCH (t:Technology {name: $tech})
            CREATE (p)-[:KNOWS_SKILL {level: $level}]->(t)
        """, email=email, tech=tech, level=level)

    print(f"✓ Created {len(skills)} skill relationships")


def get_statistics(tx):
    """Get database statistics."""
    result = tx.run("""
        MATCH (n)
        RETURN labels(n)[0] as label, count(*) as count
        ORDER BY count DESC
    """)

    print("\n📊 Database Statistics:")
    print("=" * 50)
    for record in result:
        print(f"  {record['label']}: {record['count']} nodes")

    result = tx.run("""
        MATCH ()-[r]->()
        RETURN type(r) as type, count(*) as count
        ORDER BY count DESC
    """)

    print("\nRelationships:")
    for record in result:
        print(f"  {record['type']}: {record['count']} relationships")


def main():
    """Main function to load all sample data."""
    print("\n" + "=" * 60)
    print("  Loading Sample Data into Neo4j - Social Network Graph")
    print("=" * 60 + "\n")

    # Connect to Neo4j
    print(f"Connecting to Neo4j at {NEO4J_URI}...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        # Verify connection
        driver.verify_connectivity()
        print("✓ Connected successfully\n")

        with driver.session() as session:
            # Clear existing data
            print("Step 1: Clearing existing data...")
            session.execute_write(clear_database)

            # Create constraints
            print("\nStep 2: Creating constraints...")
            session.execute_write(create_constraints)

            # Create nodes
            print("\nStep 3: Creating nodes...")
            session.execute_write(create_people)
            session.execute_write(create_companies)
            session.execute_write(create_technologies)

            # Create relationships
            print("\nStep 4: Creating relationships...")
            session.execute_write(create_relationships)

            # Show statistics
            print("\nStep 5: Gathering statistics...")
            session.execute_read(get_statistics)

        print("\n" + "=" * 60)
        print("  ✅ Sample data loaded successfully!")
        print("=" * 60)
        print("\n💡 Access Neo4j Browser at: http://localhost:7474")
        print("   Username: neo4j")
        print("   Password: password123\n")

        print("🔍 Try these Cypher queries:")
        print("   1. MATCH (n) RETURN n LIMIT 25")
        print("   2. MATCH (p:Person)-[:FRIENDS_WITH]->(friend) RETURN p, friend")
        print("   3. MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.name, c.name")
        print("   4. MATCH (p:Person)-[:KNOWS_SKILL]->(t:Technology) RETURN p.name, t.name, t.category")
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        driver.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
