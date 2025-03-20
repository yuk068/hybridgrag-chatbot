from neo4j import GraphDatabase

def is_database_populated(uri="bolt://localhost:7687", user="neo4j", password="your_password"):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) AS node_count")
        node_count = result.single()["node_count"]
        
    driver.close()
    return node_count > 0