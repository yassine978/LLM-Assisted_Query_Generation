"""Load sample data into Redis for testing.

This script populates Redis with various data types to demonstrate:
- String keys
- Hash data structures
- List data structures
- Set data structures
- Sorted Set data structures
- JSON data (if RedisJSON module available)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import redis
from src.utils.config import get_settings

def load_sample_data():
    """Load sample data into Redis."""
    settings = get_settings()

    # Connect to Redis
    r = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password if settings.redis_password else None,
        db=settings.redis_db,
        decode_responses=True
    )

    print("Connected to Redis")

    # Clear existing data (optional)
    r.flushdb()
    print("Cleared existing data")

    # 1. String keys - User session tokens
    print("\n1. Loading user session tokens...")
    r.set("session:user:1001", "abc123xyz", ex=3600)  # Expires in 1 hour
    r.set("session:user:1002", "def456uvw", ex=3600)
    r.set("session:user:1003", "ghi789rst", ex=3600)
    print("   Loaded 3 session tokens")

    # 2. String keys - Configuration values
    print("\n2. Loading configuration...")
    r.set("config:max_connections", "100")
    r.set("config:timeout", "30")
    r.set("config:feature:new_ui", "enabled")
    print("   Loaded 3 configuration values")

    # 3. Hash - User profiles
    print("\n3. Loading user profiles...")
    r.hset("user:1001", mapping={
        "username": "alice",
        "email": "alice@example.com",
        "age": "28",
        "city": "New York",
        "premium": "true"
    })
    r.hset("user:1002", mapping={
        "username": "bob",
        "email": "bob@example.com",
        "age": "35",
        "city": "San Francisco",
        "premium": "false"
    })
    r.hset("user:1003", mapping={
        "username": "charlie",
        "email": "charlie@example.com",
        "age": "42",
        "city": "Austin",
        "premium": "true"
    })
    r.hset("user:1004", mapping={
        "username": "diana",
        "email": "diana@example.com",
        "age": "29",
        "city": "Seattle",
        "premium": "false"
    })
    print("   Loaded 4 user profiles")

    # 4. Hash - Product catalog
    print("\n4. Loading product catalog...")
    r.hset("product:101", mapping={
        "name": "Laptop",
        "price": "999.99",
        "category": "electronics",
        "stock": "15",
        "rating": "4.5"
    })
    r.hset("product:102", mapping={
        "name": "Mouse",
        "price": "29.99",
        "category": "electronics",
        "stock": "50",
        "rating": "4.2"
    })
    r.hset("product:103", mapping={
        "name": "Desk",
        "price": "299.99",
        "category": "furniture",
        "stock": "8",
        "rating": "4.7"
    })
    print("   Loaded 3 products")

    # 5. Lists - Recent activity logs
    print("\n5. Loading activity logs...")
    r.rpush("logs:user:1001", "Login from 192.168.1.1")
    r.rpush("logs:user:1001", "Viewed product 101")
    r.rpush("logs:user:1001", "Added product 101 to cart")
    r.rpush("logs:user:1001", "Completed purchase")
    r.rpush("logs:user:1002", "Login from 10.0.0.5")
    r.rpush("logs:user:1002", "Searched for 'laptop'")
    r.rpush("logs:user:1002", "Viewed product 101")
    print("   Loaded activity logs for 2 users")

    # 6. Sets - User tags/interests
    print("\n6. Loading user interests...")
    r.sadd("interests:user:1001", "technology", "gaming", "music")
    r.sadd("interests:user:1002", "sports", "travel", "photography")
    r.sadd("interests:user:1003", "cooking", "fitness", "technology")
    print("   Loaded interests for 3 users")

    # 7. Sets - Product tags
    print("\n7. Loading product tags...")
    r.sadd("tags:product:101", "laptop", "computer", "portable", "work")
    r.sadd("tags:product:102", "mouse", "wireless", "ergonomic")
    r.sadd("tags:product:103", "desk", "furniture", "home-office")
    print("   Loaded tags for 3 products")

    # 8. Sorted Sets - Leaderboard
    print("\n8. Loading leaderboard...")
    r.zadd("leaderboard:game1", {"alice": 1500, "bob": 1200, "charlie": 1800, "diana": 1350})
    r.zadd("leaderboard:game2", {"alice": 2100, "bob": 1950, "charlie": 2300})
    print("   Loaded 2 leaderboards")

    # 9. Sorted Sets - Popular products by views
    print("\n9. Loading product views...")
    r.zadd("popular:products", {"product:101": 1250, "product:102": 890, "product:103": 450})
    print("   Loaded product view counts")

    # 10. Counters
    print("\n10. Loading counters...")
    r.set("counter:page_views", "15432")
    r.set("counter:signups_today", "23")
    r.set("counter:active_sessions", "156")
    print("   Loaded 3 counters")

    # 11. Cache entries with expiration
    print("\n11. Loading cache entries...")
    r.setex("cache:api:weather:nyc", 300, '{"temp": 72, "condition": "sunny"}')  # 5 min
    r.setex("cache:api:stocks:AAPL", 60, '{"price": 185.50, "change": +2.3}')  # 1 min
    print("   Loaded 2 cache entries with expiration")

    # 12. Key with namespace patterns
    print("\n12. Loading namespace patterns...")
    r.set("app:user:settings:1001", '{"theme": "dark", "notifications": true}')
    r.set("app:user:settings:1002", '{"theme": "light", "notifications": false}')
    r.set("app:system:version", "2.5.1")
    r.set("app:system:maintenance_mode", "false")
    print("   Loaded namespaced keys")

    # Print summary
    print("\n" + "="*60)
    print("DATA LOADING COMPLETE")
    print("="*60)

    # Get database info
    info = r.info("keyspace")
    db_info = info.get(f"db{settings.redis_db}", {})
    keys_count = db_info.get("keys", 0)

    print(f"\nTotal keys loaded: {keys_count}")
    print(f"Redis database: {settings.redis_db}")
    print(f"Redis host: {settings.redis_host}:{settings.redis_port}")

    # Show sample key patterns
    print("\nKey Patterns:")
    patterns = ["session:*", "user:*", "product:*", "logs:*", "interests:*",
                "tags:*", "leaderboard:*", "counter:*", "cache:*", "app:*"]

    for pattern in patterns:
        count = len(r.keys(pattern))
        if count > 0:
            print(f"  {pattern}: {count} keys")

    print("\nSample commands to test:")
    print("  GET session:user:1001")
    print("  HGETALL user:1001")
    print("  LRANGE logs:user:1001 0 -1")
    print("  SMEMBERS interests:user:1001")
    print("  ZRANGE leaderboard:game1 0 -1 WITHSCORES")

    r.close()

if __name__ == "__main__":
    try:
        load_sample_data()
    except redis.ConnectionError as e:
        print(f"Error: Could not connect to Redis: {e}")
        print("\nMake sure Redis is running:")
        print("  docker-compose up -d redis")
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
