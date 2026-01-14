"""Load sample data into HBase for testing.

This script populates HBase with sample tables demonstrating:
- Column families
- Row keys (user IDs, timestamps, etc.)
- Multiple column qualifiers
- Wide-column storage patterns
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import happybase
from src.utils.config import get_settings


def load_sample_data():
    """Load sample data into HBase."""
    settings = get_settings()

    # Connect to HBase via Thrift
    connection = happybase.Connection(
        host=settings.hbase_host,
        port=settings.hbase_port,
        protocol=settings.hbase_thrift_protocol
    )

    print("Connected to HBase")
    # Note: version attribute may not be available in all happybase versions
    try:
        print(f"HBase version: {connection.version}")
    except AttributeError:
        print("HBase connection established")

    # Clean up existing tables
    print("\nCleaning up existing tables...")
    existing_tables = [t.decode('utf-8') if isinstance(t, bytes) else t for t in connection.tables()]
    tables_to_delete = ['users', 'products', 'orders', 'sensors']

    for table_name in tables_to_delete:
        if table_name in existing_tables or table_name.encode('utf-8') in connection.tables():
            print(f"  Disabling and deleting table: {table_name}")
            connection.disable_table(table_name)
            connection.delete_table(table_name)

    # 1. Create Users table
    print("\n1. Creating 'users' table...")
    connection.create_table(
        'users',
        {
            'profile': dict(),      # Basic profile info
            'contact': dict(),      # Contact information
            'preferences': dict()   # User preferences
        }
    )

    users_table = connection.table('users')

    # Insert user data
    users_data = [
        {
            'row': 'user_1001',
            'data': {
                b'profile:name': b'Alice Johnson',
                b'profile:age': b'28',
                b'profile:city': b'New York',
                b'contact:email': b'alice@example.com',
                b'contact:phone': b'+1-555-0101',
                b'preferences:theme': b'dark',
                b'preferences:language': b'en'
            }
        },
        {
            'row': 'user_1002',
            'data': {
                b'profile:name': b'Bob Smith',
                b'profile:age': b'35',
                b'profile:city': b'San Francisco',
                b'contact:email': b'bob@example.com',
                b'contact:phone': b'+1-555-0102',
                b'preferences:theme': b'light',
                b'preferences:language': b'en'
            }
        },
        {
            'row': 'user_1003',
            'data': {
                b'profile:name': b'Charlie Davis',
                b'profile:age': b'42',
                b'profile:city': b'Austin',
                b'contact:email': b'charlie@example.com',
                b'contact:phone': b'+1-555-0103',
                b'preferences:theme': b'dark',
                b'preferences:language': b'es'
            }
        },
        {
            'row': 'user_1004',
            'data': {
                b'profile:name': b'Diana Lee',
                b'profile:age': b'29',
                b'profile:city': b'Seattle',
                b'contact:email': b'diana@example.com',
                b'contact:phone': b'+1-555-0104',
                b'preferences:theme': b'light',
                b'preferences:language': b'zh'
            }
        }
    ]

    for user in users_data:
        users_table.put(user['row'], user['data'])

    print(f"   Loaded {len(users_data)} users")

    # 2. Create Products table
    print("\n2. Creating 'products' table...")
    connection.create_table(
        'products',
        {
            'info': dict(),          # Product information
            'pricing': dict(),       # Price-related data
            'inventory': dict()      # Stock levels
        }
    )

    products_table = connection.table('products')

    products_data = [
        {
            'row': 'product_101',
            'data': {
                b'info:name': b'Laptop',
                b'info:category': b'electronics',
                b'info:brand': b'TechCorp',
                b'pricing:price': b'999.99',
                b'pricing:currency': b'USD',
                b'inventory:stock': b'15',
                b'inventory:warehouse': b'WH-01'
            }
        },
        {
            'row': 'product_102',
            'data': {
                b'info:name': b'Mouse',
                b'info:category': b'electronics',
                b'info:brand': b'TechCorp',
                b'pricing:price': b'29.99',
                b'pricing:currency': b'USD',
                b'inventory:stock': b'50',
                b'inventory:warehouse': b'WH-01'
            }
        },
        {
            'row': 'product_103',
            'data': {
                b'info:name': b'Desk',
                b'info:category': b'furniture',
                b'info:brand': b'HomeCo',
                b'pricing:price': b'299.99',
                b'pricing:currency': b'USD',
                b'inventory:stock': b'8',
                b'inventory:warehouse': b'WH-02'
            }
        }
    ]

    for product in products_data:
        products_table.put(product['row'], product['data'])

    print(f"   Loaded {len(products_data)} products")

    # 3. Create Orders table
    print("\n3. Creating 'orders' table...")
    connection.create_table(
        'orders',
        {
            'order': dict(),         # Order details
            'customer': dict(),      # Customer info
            'shipping': dict()       # Shipping details
        }
    )

    orders_table = connection.table('orders')

    orders_data = [
        {
            'row': 'order_2023_001',
            'data': {
                b'order:total': b'999.99',
                b'order:status': b'delivered',
                b'order:date': b'2023-12-01',
                b'customer:user_id': b'user_1001',
                b'customer:name': b'Alice Johnson',
                b'shipping:address': b'123 Main St, New York, NY',
                b'shipping:method': b'express'
            }
        },
        {
            'row': 'order_2023_002',
            'data': {
                b'order:total': b'29.99',
                b'order:status': b'shipped',
                b'order:date': b'2023-12-15',
                b'customer:user_id': b'user_1002',
                b'customer:name': b'Bob Smith',
                b'shipping:address': b'456 Oak Ave, San Francisco, CA',
                b'shipping:method': b'standard'
            }
        }
    ]

    for order in orders_data:
        orders_table.put(order['row'], order['data'])

    print(f"   Loaded {len(orders_data)} orders")

    # 4. Create Sensors table (time-series data)
    print("\n4. Creating 'sensors' table...")
    connection.create_table(
        'sensors',
        {
            'data': dict(),          # Sensor readings
            'metadata': dict()       # Sensor metadata
        }
    )

    sensors_table = connection.table('sensors')

    sensors_data = [
        {
            'row': 'sensor_temp_001_1640000000',
            'data': {
                b'data:value': b'22.5',
                b'data:unit': b'celsius',
                b'metadata:location': b'warehouse-1',
                b'metadata:type': b'temperature'
            }
        },
        {
            'row': 'sensor_temp_001_1640003600',
            'data': {
                b'data:value': b'23.1',
                b'data:unit': b'celsius',
                b'metadata:location': b'warehouse-1',
                b'metadata:type': b'temperature'
            }
        },
        {
            'row': 'sensor_humid_002_1640000000',
            'data': {
                b'data:value': b'45.0',
                b'data:unit': b'percent',
                b'metadata:location': b'warehouse-2',
                b'metadata:type': b'humidity'
            }
        }
    ]

    for sensor in sensors_data:
        sensors_table.put(sensor['row'], sensor['data'])

    print(f"   Loaded {len(sensors_data)} sensor readings")

    # Print summary
    print("\n" + "="*60)
    print("DATA LOADING COMPLETE")
    print("="*60)

    # List all tables
    all_tables = connection.tables()
    print(f"\nTotal tables created: {len(all_tables)}")
    print("Tables:")
    for table in all_tables:
        table_name = table.decode('utf-8') if isinstance(table, bytes) else table
        row_count = sum(1 for _ in connection.table(table_name).scan(limit=1000))
        print(f"  - {table_name}: {row_count} rows")

    print(f"\nHBase Host: {settings.hbase_host}:{settings.hbase_port}")
    print(f"Protocol: {settings.hbase_thrift_protocol}")

    print("\nSample HBase commands to test:")
    print("  # Python happybase")
    print("  import happybase")
    print(f"  conn = happybase.Connection('{settings.hbase_host}', {settings.hbase_port})")
    print("  table = conn.table('users')")
    print("  print(table.row('user_1001'))")

    connection.close()


if __name__ == "__main__":
    try:
        load_sample_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure HBase is running:")
        print("  docker-compose --profile phase6 up -d hbase")
        print("\nWait ~30 seconds for HBase to fully start, then try again.")
