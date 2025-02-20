from __future__ import annotations
from dataclasses import dataclass, field
import os
import nest_asyncio
nest_asyncio.apply()  # Allow nested event loops (required for Gradio)

from pydantic import BaseModel, EmailStr
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.format_as_xml import format_as_xml
from pydantic_ai.messages import ModelMessage
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
import gradio as gr

# ----------------------------------------------------------------------------------
# DATA MODELS
# ----------------------------------------------------------------------------------

@dataclass
class User:
    name: str
    email: EmailStr
    interests: list[str]

@dataclass
class State:
    user: User
    write_agent_messages: list[ModelMessage] = field(default_factory=list)

@dataclass
class Email:
    subject: str
    body: str

class EmailRequiresWrite(BaseModel):
    feedback: str

class Email0k(BaseModel):
    pass

# ----------------------------------------------------------------------------------
# OPENAI AGENTS
# ----------------------------------------------------------------------------------

# Retrieve the API key securely from the environment or replace with your key.
openai_api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")

# Create model instances for the agents.
email_writer_model = OpenAIModel("gpt-3.5-turbo", api_key=openai_api_key)
email_writer_agent = Agent(
    email_writer_model,
    result_type=Email,
    system_prompt="Write a welcome email for people onboarding new employees to the company name ABC. The sender's name is Karthik",
)

feedback_model = OpenAIModel("gpt-3.5-turbo", api_key=openai_api_key)
feedback_agent = Agent(
    feedback_model,
    result_type=EmailRequiresWrite | Email0k,
    system_prompt=(
        "Review the email and provide feedback; the email must reference "
        "the user's specific interests and qualities."
    ),
)

# ----------------------------------------------------------------------------------
# GRAPH NODES
# ----------------------------------------------------------------------------------

@dataclass
class WriteEmail(BaseNode[State]):
    """
    Node that generates or rewrites an email based on feedback.
    If no feedback is provided, it generates an initial email.
    If feedback is provided, it rewrites the email accordingly.
    """
    email_feedback: str | None = None

    async def run(self, ctx: GraphRunContext[State]) -> "Feedback":
        print("-" * 50)
        print("WriteEmail call fired. Email feedback:", self.email_feedback)
        print()

        if self.email_feedback:
            # If we have feedback, rewrite the email accordingly
            prompt = (
                f"Rewrite the email for the user:\n"
                f"{format_as_xml(ctx.state.user)}\n"
                f"Feedback: {self.email_feedback}"
            )
        else:
            # If no feedback, generate an initial email
            user_xml = """
            <examples>
                <name>John Doe</name>
                <email>john.joe@example.com</email>
            </examples>
            """
            prompt = (
                "Write a welcome email for the user:\n"
                f"{user_xml}"
            )

        # Run the email writer agent to produce (or rewrite) the email
        result = await email_writer_agent.run(prompt)
        print("Email Writer Agent result:", result)
        # Optionally store the messages if you want to track them
        ctx.state.write_agent_messages = result.all_messages()

        # result.data should be an Email instance as per email_writer_agent.result_type
        return Feedback(email=result.data)

@dataclass
class Feedback(BaseNode[State, None, Email]):
    """
    Node that receives an email and uses the feedback agent to determine
    if further revisions are needed or if the email is acceptable.
    """
    email: Email

    async def run(self, ctx: GraphRunContext[State]) -> WriteEmail | End[Email]:
        print("Feedback call fired. Email object received:", self.email)
        print()

        # Prepare the prompt for the feedback agent
        prompt = format_as_xml({"user": ctx.state.user, "email": self.email})
        result = await feedback_agent.run(prompt)
        print("Feedback result received. Feedback result:", result.data)

        # If feedback indicates a rewrite is required, pass feedback to WriteEmail
        if isinstance(result.data, EmailRequiresWrite):
            return WriteEmail(email_feedback=result.data.feedback)
        else:
            # Otherwise, end the graph execution with the final email
            return End(self.email)

# ----------------------------------------------------------------------------------
# BUILD THE GRAPH
# ----------------------------------------------------------------------------------

feedback_graph = Graph(nodes=[WriteEmail, Feedback])

# ----------------------------------------------------------------------------------
# GRADIO INTERFACE FUNCTION
# ----------------------------------------------------------------------------------

def generate_email(name: str, email: str, interests: str) -> str:
    """
    Create a User and State object from the input values,
    run the graph synchronously, and return the final email text.
    """
    interests_list = [interest.strip() for interest in interests.split(",") if interest.strip()]
    user_obj = User(name=name, email=email, interests=interests_list)
    state_obj = State(user=user_obj)

    # Run the graph synchronously. (This call blocks until complete.)
    final_email, history = feedback_graph.run_sync(start_node=WriteEmail(), state=state_obj)
    
    # Combine the subject and body for display.
    return f"Subject: {final_email.subject}\n\n{final_email.body}"

# ----------------------------------------------------------------------------------
# LAUNCH GRADIO INTERFACE
# ----------------------------------------------------------------------------------

iface = gr.Interface(
    fn=generate_email,
    inputs=[
        gr.Textbox(label="Name"),
        gr.Textbox(label="Email"),
        gr.Textbox(label="Interests (comma separated)"),
    ],
    outputs=gr.Textbox(label="Generated Email"),
    title="Welcome Email Generator",
    description="Enter user details to generate a welcome email using OpenAI.",
)

iface.launch(share=True)
