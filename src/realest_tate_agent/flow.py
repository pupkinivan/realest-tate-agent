from datetime import datetime, timedelta
import json
from typing import TypedDict, Literal, List, Dict, Any, Optional

from langchain.globals import set_verbose
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from realest_tate_agent.ai_models import LlmFactory, LlmTier

llm = LlmFactory().instantiate_llm(LlmTier.STANDARD, temperature=0.0)


class AgentState(TypedDict):
    user_type: Optional[Literal["owner", "resident"]]
    owner_details: Optional[Dict[str, Any]]
    resident_preferences: Optional[Dict[str, Any]]
    properties: Optional[List[Dict[str, Any]]]
    inspection_date: Optional[str]
    current_step: str
    messages: List[str]
    human_input: Optional[str]


MOCK_PROPERTIES = [
    {
        "id": 1,
        "address": "123 Oak Street, Downtown",
        "bedrooms": 2,
        "bathrooms": 2,
        "price": 2500,
        "area": "Downtown",
        "features": ["Parking", "Balcony", "Pet-friendly"],
    },
    {
        "id": 2,
        "address": "456 Pine Avenue, Suburbs",
        "bedrooms": 3,
        "bathrooms": 2,
        "price": 3200,
        "area": "Suburbs",
        "features": ["Garden", "Garage", "Near schools"],
    },
    {
        "id": 3,
        "address": "789 Maple Drive, City Center",
        "bedrooms": 1,
        "bathrooms": 1,
        "price": 1800,
        "area": "City Center",
        "features": ["Modern", "Gym access", "Public transport"],
    },
    {
        "id": 4,
        "address": "321 Elm Street, Riverside",
        "bedrooms": 4,
        "bathrooms": 3,
        "price": 4500,
        "area": "Riverside",
        "features": ["River view", "Large kitchen", "2-car garage"],
    },
]


def detect_user_type(state: AgentState) -> AgentState:
    """Detect whether the user is looking to rent or owns a property."""
    state["current_step"] = "detect_user_type"
    question = "AI: Are you a property owner looking to rent out your home, or are you looking to rent a property? (owner/resident)"
    state["messages"].append(question)
    user_input = input(
        "AI: Are you a property owner looking to rent out your home, or are you looking to rent a property? (owner/resident)\nYou: "
    ).strip()
    state["human_input"] = user_input
    state["messages"].append(f"User: {user_input}")

    response = (
        llm.invoke(
            f"""You are a real estate agent. Based on the user's input, determine if they are an owner or resident.
        User input: {user_input}.
        "Only respond with 'owner' or 'resident'."""
        )
        .content.strip()
        .lower()
    )

    if "owner" == response:
        state["user_type"] = "owner"
        state["current_step"] = "collect_owner_details"
    elif "resident" == response:
        state["user_type"] = "resident"
        state["current_step"] = "collect_resident_preferences"
    else:
        state["messages"].append("Please specify 'owner' or 'resident'")
        state["current_step"] = "detect_user_type"

    return state


def route_user_type(state: AgentState) -> str:
    """Route user to the appropriate branch based on their type"""
    user_type = state.get("user_type", "detect_user_type")

    if user_type == "owner":
        return "owner"
    elif user_type == "resident":
        return "resident"
    else:
        return "not_a_valid_type"


def collect_owner_details(state: AgentState) -> AgentState:
    """Collect the owner user's details"""
    question = """Please provide your home details in the following format:
Full Name: [Your Name]
Contact: [Phone or Email]
Home Address: [Full Address]
Utilities On: [Yes/No]
Home Vacant: [Yes/No]

You: """
    state["messages"].append(question)
    user_input = input(question).strip()
    state["human_input"] = user_input
    state["messages"].append(f"You: {user_input}")

    # Use an LLM to parse the details
    result = llm.invoke(
        f"""Extract the owner details from the following input and format as a
        JSON object:
        
        {user_input}
        
        Expected fields: full_name (the user's full name, a string), contact_info (phone
        number or email address, a string), home_address (home address, a string),
        has_utilities (whether the utitilies are working, boolean), and
        is_vacant (whether the home is vacant, boolean).
        
        Return ONLY a valid JSON with those keys."""
    )
    try:
        details = json.loads(result.content)
    except json.JSONDecodeError:
        for i in range(3):
            try:
                result = llm.invoke(
                    f"""The following JSON is not valid. Please fix it:
                    
                    {result}
                    
                    Return ONLY a valid JSON string with the keys:
                    full_name (str), contact_info (str), home_address (str),
                    has_utilities (bool), is_vacant (bool).
                    
                    Do not add any prefix or formatting to the JSON.
                    """
                )
                details = json.loads(result.content)
                break
            except json.JSONDecodeError:
                if i == 2:
                    raise ValueError(
                        "Failed to parse owner details after multiple attempts; result is:\n"
                        + result.content
                    )
                details = {}

    state["owner_details"] = details
    return state


def schedule_inspection(state: AgentState) -> AgentState:
    """Schedule inspection for vacant property with utilities"""
    state["current_step"] = "schedule_inspection"
    # Mock inspection date for demonstration purposes
    # TODO: In a real scenario, this would be pulled from a calendar or user suggestion
    delta = timedelta(days=1)
    MOCK_DATE = (datetime.now() + delta).strftime("%Y-%m-%d %H:%M")
    question = (
        "Looks like your home is ready for an inspection! Let's try to "
        f"schedule it. How does {MOCK_DATE} sound? "
        "If that's not good, just suggest a new one and we'll make it work!"
    )
    state["messages"].append(question)
    user_input = input(question + "\n\nYou: ").strip()
    state["human_input"] = user_input

    # Have the LLM parse the user input to extract the date
    inspection_input = llm.invoke(
        f"""You are an assistant trying to set up an inspection date for a property.
        You previously proposed a date in this message:
        <assistant_message>
        {question}
        </assistant_message>
        
        The user replied as follows; extract the inspection date from their message:
        <user_message>
        {user_input}
        </user_message>
        
        The date should be in the format YYYY-MM-DD HH:MM.
        Return ONLY the date string in that format."""
    ).content.strip()
    state["inspection_date"] = inspection_input

    return state


def confirm_owner_details(state: AgentState) -> AgentState:
    """Confirm owner details and inspection date"""
    details = state.get("owner_details", {})
    inspection_date = state.get("inspection_date")

    confirmation = "Confirmation of Details:\n"
    for key, value in details.items():
        confirmation += f"- {key}: {value}\n"

    if inspection_date:
        confirmation += f"- Inspection Date: {inspection_date}\n"

    confirmation += "\nThank you for providing your property details!"

    state["messages"].append(confirmation)
    return state


def collect_resident_preferences(state: AgentState) -> AgentState:
    """Collect resident preferences"""
    state["current_step"] = "collect_resident_preferences"
    question = (
        "Please provide your rental preferences:\n"
        "Number of bedrooms: [1, 2, 3, 4+]\n"
        "Preferred city/area: [Area name]\n"
        "Budget: [Monthly budget in $]\n"
        "Additional preferences: [Any specific requirements]"
    )
    state["messages"].append(question)
    user_input = input(question + "\n\nYou: ").strip()
    state["messages"].append(user_input)
    state["human_input"] = user_input
    state["resident_preferences"] = user_input

    return state


def match_properties(state: AgentState) -> AgentState:
    """Match preferences with available properties"""
    state["current_step"] = "match_properties"
    # Have the LLM compare the user's preferences with the mock properties
    # TODO: replace mock properties with actual logic. E.g., interpolating the preferences in a query to a DB.
    matching_properties = llm.invoke(
        f"""You are a real estate agent. Based on the user's preferences, select the properties
        that match their criteria.
        User preferences: {state["resident_preferences"]}.

        The listings are as follows:
        {json.dumps(MOCK_PROPERTIES, indent=2)}
        
        Return suggestions for properties the user might like, based on their parameters and location.
        Keep the JSON format I just gave you, only filter out properties that don't match."""
    ).content
    state["messages"].append(matching_properties)
    state["properties"] = matching_properties
    return state


def show_properties(state: AgentState) -> AgentState:
    """Step 3.3: Show detailed information about properties"""
    state["current_step"] = "show_properties"
    # Have the LLM format the output in an apealing way for the user

    pretty_message = llm.invoke(
        f"""You are a real estate agent. Based on the user's preferences, you
        were given a list of potential properties.

        User preferences: {state["resident_preferences"]}.

        The listings are as follows:
        {state["properties"]}.
        
        You have to make the properties sound appealing and highlight why they
        are relevant and aligned with what the user requested."""
    ).content
    state["messages"].append(pretty_message)
    return state


def route_owner_details(state: AgentState) -> str:
    """Determine next step after processing owner details"""
    # Check if home is vacant and utilities are on
    details = state.get("owner_details")
    utilities_on = details.get("has_utilities", False)
    home_vacant = details.get("is_vacant", False)

    if utilities_on and home_vacant:
        return "ready_for_inspection"
    else:
        return "not_ready"


def instantiate_workflow():
    """Create and return the LangGraph graph flow."""

    # Initialize graph with state
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("detect_user_type", detect_user_type)
    # Owner nodes
    workflow.add_node("collect_owner_details", collect_owner_details)
    workflow.add_node("schedule_inspection", schedule_inspection)
    workflow.add_node("confirm_owner_details", confirm_owner_details)
    # Resident nodes
    workflow.add_node("collect_resident_preferences", collect_resident_preferences)
    workflow.add_node("match_properties", match_properties)
    workflow.add_node("show_properties", show_properties)

    # Starting node
    workflow.set_entry_point("detect_user_type")

    # Define steps sequence
    workflow.add_conditional_edges(
        "detect_user_type",
        route_user_type,
        {
            "owner": "collect_owner_details",
            "resident": "collect_resident_preferences",
            "not_a_valid_type": "detect_user_type",  # Feed back for errors
        },
    )

    # Owners branch
    # workflow.add_edge("collect_owner_details", "route_owner_details")
    workflow.add_conditional_edges(
        "collect_owner_details",
        route_owner_details,
        {
            "ready_for_inspection": "schedule_inspection",
            "not_ready": "confirm_owner_details",
        },
    )
    workflow.add_edge("schedule_inspection", "confirm_owner_details")

    # Residents branch
    workflow.add_edge("collect_resident_preferences", "match_properties")
    workflow.add_edge("match_properties", "show_properties")

    # Add ending states
    workflow.add_edge("confirm_owner_details", END)
    workflow.add_edge("show_properties", END)

    # Compile the graph
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    app.get_graph().draw_mermaid_png(output_file_path="./graph.png")
    return app


def run_pipeline():
    """Run the real estate agent pipeline"""
    graph = instantiate_workflow()

    # Initialize the state
    state = AgentState(
        user_type=None,
        owner_details=None,
        resident_preferences=None,
        properties=None,
        inspection_date=None,
        current_step="start",
        messages=[],
        human_input=None,
    )

    config = {
        "configurable": {"thread_id": "realest-tate-session"},
        "recursion_limit": 10,
    }

    print("Welcome to Realest Tate Agent!")
    print("I'm here to help you with your real estate needs.\n")

    for chunk in graph.stream(
        state,
        stream_mode="updates",
        # stream_mode="debug",
        config=config,
    ):
        continue
        # For inspecting:
        # from pprint import pprint
        # pprint(chunk)
        # print(chunk["messages"][-1])
        # print("-" * 50)

    print(
        list(chunk.values())[0]["messages"][-1]
    )  # Print the message from the last graph step

    print("\n🎉 Thank you for using Realest Tate Agent!")


if __name__ == "__main__":
    run_pipeline()
