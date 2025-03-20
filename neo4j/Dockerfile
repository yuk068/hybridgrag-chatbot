FROM neo4j:5.22.0

# Enable necessary plugins
ENV NEO4JLABS_PLUGINS '[ "apoc", "neo4j-search" ]'
ENV NEO4J_dbms_security_procedures_unrestricted 'apoc.*,db.index.*'

# Copy APOC plugin jar file to the plugins directory
COPY ./apoc-5.22.0-core.jar /var/lib/neo4j/plugins

# Expose necessary ports
EXPOSE 7474 7687
