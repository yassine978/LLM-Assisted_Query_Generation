"""Load sample RDF data into Apache Jena Fuseki.

This script loads the sample Wikidata RDF dataset into Fuseki for testing.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import requests
from src.utils.config import get_settings


def load_rdf_data():
    """Load RDF data into Fuseki."""
    settings = get_settings()

    # Fuseki endpoints
    base_url = settings.fuseki_endpoint
    dataset = settings.fuseki_dataset
    data_endpoint = f"{base_url}/{dataset}/data"
    query_endpoint = f"{base_url}/{dataset}/sparql"

    # Authentication
    auth = None
    if settings.fuseki_username and settings.fuseki_password:
        auth = (settings.fuseki_username, settings.fuseki_password)

    print("="*60)
    print("LOADING RDF DATA INTO FUSEKI")
    print("="*60)

    # Check if Fuseki is running
    print(f"\n1. Checking Fuseki connection at {base_url}...")
    try:
        response = requests.get(f"{base_url}/$/ping", timeout=5)
        if response.status_code == 200:
            print("   Fuseki is running!")
        else:
            print(f"   Fuseki returned status {response.status_code}")
    except Exception as e:
        print(f"   ERROR: Cannot connect to Fuseki: {e}")
        print("\n   Make sure Fuseki is running:")
        print("   docker-compose --profile phase6 up -d fuseki")
        print("   Wait ~10 seconds for Fuseki to start")
        return False

    # Check if dataset exists
    print(f"\n2. Checking if dataset '{dataset}' exists...")
    try:
        response = requests.get(f"{base_url}/$/datasets", auth=auth, timeout=5)
        datasets_info = response.json()

        dataset_exists = False
        for ds in datasets_info.get("datasets", []):
            ds_name = ds.get("ds.name", "")
            if ds_name.strip("/") == dataset:
                dataset_exists = True
                break

        if dataset_exists:
            print(f"   Dataset '{dataset}' exists!")
        else:
            print(f"   Dataset '{dataset}' not found. Creating it...")
            # Create dataset
            create_payload = {
                "dbName": dataset,
                "dbType": "tdb2"
            }
            response = requests.post(
                f"{base_url}/$/datasets",
                auth=auth,
                data=create_payload,
                timeout=10
            )
            if response.status_code in [200, 201]:
                print(f"   Dataset '{dataset}' created successfully!")
            else:
                print(f"   ERROR: Failed to create dataset: {response.status_code}")
                print(f"   Response: {response.text}")
                return False

    except Exception as e:
        print(f"   ERROR checking/creating dataset: {e}")
        return False

    # Load RDF data
    print(f"\n3. Loading RDF data from sample_wikidata.ttl...")
    ttl_file = Path(__file__).parent / "sample_wikidata.ttl"

    if not ttl_file.exists():
        print(f"   ERROR: File not found: {ttl_file}")
        return False

    try:
        with open(ttl_file, "rb") as f:
            rdf_data = f.read()

        # Upload to Fuseki
        headers = {"Content-Type": "text/turtle; charset=utf-8"}
        response = requests.post(
            data_endpoint,
            auth=auth,
            data=rdf_data,
            headers=headers,
            timeout=30
        )

        if response.status_code in [200, 201, 204]:
            print("   RDF data loaded successfully!")
        else:
            print(f"   ERROR: Failed to load data: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"   ERROR loading data: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify data was loaded
    print(f"\n4. Verifying data...")
    try:
        # Count triples
        count_query = "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }"
        response = requests.post(
            query_endpoint,
            auth=auth,
            data={"query": count_query},
            headers={"Accept": "application/sparql-results+json"},
            timeout=10
        )

        if response.status_code == 200:
            results = response.json()
            count = results["results"]["bindings"][0]["count"]["value"]
            print(f"   Total triples: {count}")

            # Count entities (persons)
            person_query = """
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            SELECT (COUNT(DISTINCT ?person) as ?count)
            WHERE {
                ?person wdt:P31 wd:Q5
            }
            """
            response = requests.post(
                query_endpoint,
                auth=auth,
                data={"query": person_query},
                headers={"Accept": "application/sparql-results+json"},
                timeout=10
            )
            if response.status_code == 200:
                results = response.json()
                count = results["results"]["bindings"][0]["count"]["value"]
                print(f"   Persons (humans): {count}")

            # Count physicists
            physicist_query = """
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            SELECT (COUNT(DISTINCT ?physicist) as ?count)
            WHERE {
                ?physicist wdt:P106 wd:Q169470
            }
            """
            response = requests.post(
                query_endpoint,
                auth=auth,
                data={"query": physicist_query},
                headers={"Accept": "application/sparql-results+json"},
                timeout=10
            )
            if response.status_code == 200:
                results = response.json()
                count = results["results"]["bindings"][0]["count"]["value"]
                print(f"   Physicists: {count}")

        else:
            print(f"   ERROR verifying data: {response.status_code}")

    except Exception as e:
        print(f"   ERROR during verification: {e}")

    # Summary
    print("\n" + "="*60)
    print("DATA LOADING COMPLETE")
    print("="*60)
    print(f"\nFuseki Web UI: {base_url}")
    print(f"SPARQL Endpoint: {query_endpoint}")
    print(f"Dataset: {dataset}")
    print(f"\nSample SPARQL Query:")
    print("""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>

    SELECT ?person ?name WHERE {
        ?person wdt:P106 wd:Q169470 .  # occupation: physicist
        ?person rdfs:label ?name .
    }
    LIMIT 10
    """)

    print("\nYou can test this at: " + base_url + "/#/dataset/" + dataset + "/query")
    print("="*60)

    return True


if __name__ == "__main__":
    try:
        success = load_rdf_data()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
