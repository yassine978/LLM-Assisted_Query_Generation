"""Load RDF data into Apache Jena Fuseki."""

import requests
from pathlib import Path

# Fuseki configuration
FUSEKI_URL = "http://localhost:3030"
DATASET_NAME = "dataset"  # Default dataset name
UPDATE_ENDPOINT = f"{FUSEKI_URL}/{DATASET_NAME}/update"
DATA_ENDPOINT = f"{FUSEKI_URL}/{DATASET_NAME}/data"

def clear_dataset():
    """Clear all data from the dataset."""
    print("Clearing existing data...")
    query = "DELETE WHERE { ?s ?p ?o }"
    response = requests.post(
        UPDATE_ENDPOINT,
        data={"update": query},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    if response.status_code == 200 or response.status_code == 204:
        print("✓ Dataset cleared")
    else:
        print(f"⚠️  Clear failed: {response.status_code} - {response.text}")

def load_ttl_file(file_path):
    """Load a Turtle file into Fuseki."""
    print(f"\nLoading {file_path.name}...")

    with open(file_path, 'rb') as f:
        response = requests.post(
            DATA_ENDPOINT,
            data=f,
            headers={"Content-Type": "text/turtle"}
        )

    if response.status_code == 200 or response.status_code == 201:
        print(f"✓ Loaded {file_path.name}")
        return True
    else:
        print(f"❌ Failed to load {file_path.name}: {response.status_code}")
        print(f"   Response: {response.text}")
        return False

def count_triples():
    """Count total triples in the dataset."""
    query_endpoint = f"{FUSEKI_URL}/{DATASET_NAME}/sparql"
    query = "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }"

    response = requests.post(
        query_endpoint,
        data={"query": query},
        headers={"Accept": "application/sparql-results+json"}
    )

    if response.status_code == 200:
        result = response.json()
        count = result['results']['bindings'][0]['count']['value']
        print(f"\n📊 Total triples in dataset: {count}")
    else:
        print(f"⚠️  Could not count triples: {response.status_code}")

def main():
    """Load sample RDF data."""
    print("=" * 70)
    print("LOADING RDF DATA INTO APACHE JENA FUSEKI")
    print("=" * 70)

    # Check Fuseki is running
    try:
        response = requests.get(f"{FUSEKI_URL}/$/ping")
        if response.status_code != 200:
            print("❌ Fuseki is not responding!")
            print("   Start it with: docker-compose up -d fuseki")
            return
        print("✓ Fuseki is running\n")
    except Exception as e:
        print(f"❌ Cannot connect to Fuseki: {e}")
        print("   Start it with: docker-compose up -d fuseki")
        return

    # Clear existing data
    clear_dataset()

    # Load files
    rdf_dir = Path(__file__).parent
    files_to_load = [
        rdf_dir / "sample_wikidata.ttl",
        # Uncomment to also load simple test data:
        # rdf_dir / "simple_test.ttl",
    ]

    success_count = 0
    for file_path in files_to_load:
        if file_path.exists():
            if load_ttl_file(file_path):
                success_count += 1
        else:
            print(f"⚠️  File not found: {file_path}")

    # Count total triples
    count_triples()

    print("\n" + "=" * 70)
    print(f"✓ LOADED {success_count}/{len(files_to_load)} FILES")
    print("=" * 70)
    print("\n💡 Access Fuseki web UI: http://localhost:3030")
    print("   Dataset name: dataset")
    print("\n🔍 Test with SPARQL query:")
    print("   SELECT * WHERE { ?s ?p ?o } LIMIT 10")

if __name__ == "__main__":
    main()
