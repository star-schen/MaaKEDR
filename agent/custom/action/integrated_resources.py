from maa.agent.agent_server import AgentServer
from utils.resource_collection import IntegratedResourceFarm

AgentServer.register_custom_action("IntegratedResourceFarm", IntegratedResourceFarm())
