from dataclasses import dataclass
from pydantic_graph import GraphRunContext, BaseNode, Graph, End

@dataclass
class NodeA(BaseNode[int]):
    track_number: int
    async def run(self, ctx: GraphRunContext) -> BaseNode:
        print("Calling A")
        return NodeB(self.track_number)

@dataclass
class NodeB(BaseNode[int]):
    track_number: int
    async def run(self, ctx: GraphRunContext) -> BaseNode | End:
        print("Calling B")
        if self.track_number == 1:
            return End(f"Stop at Node B with value --> {self.track_number}")
        else:
            return NodeC(self.track_number)

@dataclass
class NodeC(BaseNode[int]):
    track_number: int
    async def run(self, ctx: GraphRunContext) -> BaseNode | End:
        print("Calling C")
        return End(f"Stop at Node C with value --> {self.track_number}")

graph = Graph(nodes=[NodeA, NodeB, NodeC])

result, history = graph.run_sync(start_node=NodeA(track_number=1))

print("#" * 40)
print("History:")
for history_part in history:
    print(history_part)
    print()

print("-" * 40)
print("Result:", result)
