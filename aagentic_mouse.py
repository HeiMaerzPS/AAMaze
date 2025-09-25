"""
AAgenticMouse - Step 1: Class structure with imports and method stubs
"""

# Standard library imports
import json
import re
import os
import logging
import time
from typing import TypedDict, List, Optional, Dict, Any, Tuple, Literal
import streamlit as st

# Third-party imports for the agentic framework
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Our maze components
from aamaze_mouse import AAMaze, AAMouse, get_default_maze, render_with_mouse

# Constants and configuration
MODEL = 'gpt-5-nano'  # Default OpenAI model 'gpt-4.1-nano'
LOG_LEVEL = logging.WARNING
__version__ = '20250925_1557'

# Regular expression to parse LLM action responses
ACTION_RE = re.compile(
    r'^Action:\s*([A-Za-z_]\w*)\((\{.*?\})?\)\s*$',
    re.DOTALL | re.MULTILINE
)

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s"
)

# State definition for the LangGraph workflow
class MazeState(TypedDict):
    """State structure that flows through the agent's decision-making graph"""
    scan_info: Dict             # Latest sensor scan - single source of truth
    history: List[str]           # Recent action history for LLM context
    strategy: str                # Natural language strategy
    last_result: Optional[Dict]  # Result from last operation
    step_budget: int             # Maximum steps allowed


class AAgenticMouse:
    """
    Agentic wrapper for maze-solving mouse using LangGraph workflow.
    
    Encapsulates the decision-making process and provides a simple step-by-step
    interface for external control and visualization.
    """
    
    def __init__(
        self, 
        maze: AAMaze, 
        mouse: AAMouse, 
        strategy: str,
        use_llm: bool = True,
        step_budget: int = 500,
        model: str = MODEL
    ):
        """
        Initialize the agentic mouse with maze environment and strategy.
        
        Args:
            maze: The AAMaze instance containing the environment
            mouse: The AAMouse instance that will navigate
            strategy: Natural language description of solving strategy
            use_llm: Whether to use LLM or rule-based decision making
            step_budget: Maximum steps before timeout
            model: OpenAI model to use if use_llm=True
        """
        try:
            # Store core components
            self.maze = maze
            self.mouse = mouse
            self.strategy = strategy
            self.use_llm = use_llm
            self.step_budget = step_budget
            self.model = model
            self.start_time = time.time()

            # Initialize LLM if needed
            self.llm = None
            if self.use_llm:
                API_TOKEN = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
                self.llm = ChatOpenAI(model=self.model, temperature=0, api_key=API_TOKEN)

            # State management
            self.reasoning_output = 'Start the AAgent'
            self.current_state = None
            self.graph_app = None
            self.is_initialized = False

            # Logging
            self.logger = logger
        except Exception as e:
            self.logger.exception(f"{type(e).__name__}: {e}")
            raise
        
        self.logger.debug(f"AAgenticMouse initialized with strategy: {strategy[:50]}...")

    def _perform_scan(self) -> Dict:
        """
        Perform a comprehensive scan of the mouse's current environment.

        Gathers all sensor data into a structured dictionary that serves as the
        single source of truth for the agent's decision-making process.

        Returns:
            Dict: Structured sensor data with keys:
                - at_goal: bool - whether mouse has reached the goal
                - goal_visible: bool - whether goal is in line of sight
                - goal_dir: Optional[str] - relative direction to goal ('ahead'/'left'/'right'/None)
                - walls: List[str] - blocked directions (['ahead', 'left', 'right'])
                - unvisited: List[str] - unvisited directions (['ahead', 'left', 'right'])
                - visited_count: Dict[str, int] - visit counts for each direction
                - absolute_direction: str - compass direction mouse is facing ('N'/'E'/'S'/'W')
                - loop: bool - whether mouse appears to be in a loop
                - stuck: bool - whether mouse appears to be stuck
        """
        # Gather all sensor readings from the mouse
        at_goal = self.mouse.sense_at_goal()
        goal_dir = self.mouse.sense_goal_direction()  # Returns relative direction or None
        unvisited = self.mouse.sense_unvisited_directions()  # Returns ['ahead', 'left', 'right']
        visited = self.mouse.sense_visited_directions()  # Returns {'direction': count}
        absolute_direction = self.mouse.sense_absolute_direction()  # Returns 'N'/'E'/'S'/'W'
        walls = self.mouse.sense_walls()  # Returns ['ahead', 'left', 'right']
        loop = self.mouse.sense_loop_detected()
        stuck = self.mouse.sense_stuck()

        # Package all sensor data into structured format
        scan_info = {
            "at_goal": at_goal,
            "goal_visible": goal_dir is not None,
            "goal_dir": goal_dir,
            "walls": walls,
            "unvisited": unvisited,
            "visited_count": visited,
            "absolute_direction": absolute_direction,
            "loop": loop,
            "stuck": stuck,
        }

        return scan_info

    def _format_scan_observation(self, scan_info: Dict) -> str:
        """
        Convert structured scan data into human-readable observation string.

        Args:
            scan_info: Dictionary from _perform_scan() containing sensor data

        Returns:
            str: Formatted observation string for LLM prompts and logging
        """
        # Start with core status information
        obs_parts = [f"at_goal={scan_info['at_goal']}"]

        # Add goal direction only if goal is visible
        if scan_info['goal_visible'] and scan_info['goal_dir']:
            obs_parts.append(f"goal_direction={scan_info['goal_dir']}")

        # Always include walls (even if empty list)
        walls_str = ', '.join(scan_info['walls']) if scan_info['walls'] else 'none'
        obs_parts.append(f"walls=[{walls_str}]")

        # Always include unvisited directions (empty list means all explored)
        unvisited_str = ', '.join(scan_info['unvisited']) if scan_info['unvisited'] else 'none'
        obs_parts.append(f"unvisited=[{unvisited_str}]")

        # Always include visited counts (empty dict means no revisited neighbors)
        if scan_info['visited_count']:
            visited_items = [f"{direction}={count}" for direction, count in scan_info['visited_count'].items()]
            visited_str = ', '.join(visited_items)
            obs_parts.append(f"visited_count=[{visited_str}]")
        else:
            obs_parts.append(f"visited_count=[none]")

        # Always include absolute direction for orientation
        obs_parts.append(f"facing={scan_info['absolute_direction']}")

        # Combine all parts with consistent formatting
        return "**Scan:** " + ", ".join(obs_parts)

    def _build_workflow(self) -> StateGraph:
        """
        Build the simplified LangGraph workflow with single decision node.

        This creates a minimal workflow with just one node that takes scan data
        and strategy, then directly returns the movement direction to take.

        Returns:
            StateGraph: Compiled workflow graph ready for execution
        """
        # Create the workflow graph
        workflow = StateGraph(MazeState)

        # Add single decision-making node
        workflow.add_node("agent_decision", self._agent_decision_node)

        # Simple flow: start -> decide -> end
        workflow.set_entry_point("agent_decision")
        workflow.add_edge("agent_decision", END)

        # Compile the workflow into executable form
        compiled_workflow = workflow.compile()

        self.logger.debug("Simplified LangGraph workflow built and compiled")
        return compiled_workflow

    def _agent_decision_node(self, state: MazeState) -> MazeState:
        """
        Single agent decision node: Analyzes situation and directly decides movement direction.

        This is the core "intelligence" of the agent. It takes the current scan data,
        strategy, and history, then decides which direction to move: ahead, left, right, or backtrack.

        Args:
            state: Current maze state with scan_info, history, strategy, etc.

        Returns:
            Updated state with chosen direction stored in last_result
        """
        self.logger.debug("=== AGENT DECISION ===")

        # Get current situation
        scan_info = state["scan_info"]
        strategy = state["strategy"]
        history = state.get("history", [])

        # Log current situation for debugging
        observation = self._format_scan_observation(scan_info)
        self.logger.debug(f"Current observation: {observation}")
        self.logger.debug(f"Recent history: {history[-3:] if history else 'none'}")

        # Make the decision (this is where the "thinking" happens)
        if self.use_llm and self.llm:
            # LLM-based decision making (we'll implement this later)
            chosen_direction = self._make_llm_decision(scan_info, strategy, history)
        else:
            # Rule-based decision making for now
            chosen_direction = self._make_rule_based_decision(scan_info, strategy, history)

        self.logger.debug(f"Agent decision: move '{chosen_direction}'")

        # Store the decision in the state
        updated_state = state.copy()
        updated_state["last_result"] = {
            "node": "agent_decision",
            "chosen_direction": chosen_direction,
            "reasoning": f"Based on: {observation}"
        }

        return updated_state

    def _make_rule_based_decision(self, scan_info: Dict, strategy: str, history: List[str]) -> str:
        """
        Rule-based decision making - simple but effective maze-solving logic.

        This method implements a straightforward strategy:
        1. If goal visible -> move toward goal
        2. If unvisited paths available -> explore them (prefer ahead > left > right)
        3. If all neighbors visited -> backtrack

        Args:
            scan_info: Current sensor readings
            strategy: Natural language strategy (not used in rule-based, but kept for consistency)
            history: Recent action history for context

        Returns:
            str: Direction to move ("ahead", "left", "right", "backtrack")
        """
        self.logger.debug("Using rule-based decision making")

        self.reasoning_output = self._format_scan_observation(scan_info=scan_info)

        # If we can see the goal directly, move toward it
        if scan_info["goal_visible"] and scan_info["goal_dir"]:
            goal_dir = scan_info["goal_dir"]
            self.logger.debug(f"Goal visible - moving {goal_dir}")
            return goal_dir

        # If we have unvisited directions, explore them (prioritize ahead > left > right)
        unvisited = scan_info.get("unvisited", [])
        if unvisited:
            # Prioritize directions: ahead first, then left, then right
            for preferred_dir in ["ahead", "left", "right"]:
                if preferred_dir in unvisited:
                    self.logger.debug(f"Exploring unvisited direction: {preferred_dir}")
                    return preferred_dir

        # If all neighbors are visited, we need to backtrack to find new paths
        self.logger.debug("All neighbors visited - backtracking")
        return "backtrack"

    def _make_llm_decision(self, scan_info: Dict, strategy: str, history: List[str]) -> str:
        """
        LLM-based decision making using natural language reasoning.

        This method creates a detailed prompt with the current situation and strategy,
        sends it to the LLM, and parses the response to extract a movement decision.

        Args:
            scan_info: Current sensor readings from _perform_scan()
            strategy: Natural language strategy to follow
            history: Recent action history for context

        Returns:
            str: Direction to move ("ahead", "left", "right", "backtrack")
        """
        self.logger.debug("Using LLM-based decision making")

        try:
            # Create the current observation string for the LLM
            current_observation = self._format_scan_observation(scan_info)

            # Create recent history context (last 3 actions)
            recent_history = history[-3:] if len(history) >= 3 else history
            history_text = ", ".join(recent_history) if recent_history else "just started"

            # Build comprehensive prompt for the LLM
            prompt = f"""You are navigating a maze. Your task is to follow this specific strategy:

STRATEGY:
"{strategy}"

CURRENT SITUATION:
{current_observation}

RECENT ACTIONS: {history_text}

AVAILABLE MOVES:
- "ahead" - move forward in current facing direction
- "left" - turn left and move forward  
- "right" - turn right and move forward
- "backtrack" - retrace steps to previous position

SENSOR INFORMATION GUIDE:
- at_goal: whether you have reached the maze exit or goal
- goal_direction: direction of the goal if it is in line of sight, only when it is in sight
- walls: which directions are blocked by walls, a dead end is defined by walls ahead, left, and right
- unvisited: which directions lead to places you've never been
- visited_count: how many times you've been to neighboring positions (represents your "marks")
- facing: your current compass direction (N/E/S/W)

INSTRUCTIONS:
1. Read your strategy carefully and understand what it tells you to do
2. Analyze your current situation using the sensor information
3. Apply your strategy to decide which move to make
4. Explain your reasoning step by step
5. Choose exactly ONE move from the available options

RESPONSE FORMAT:
Reasoning: [Your step-by-step analysis applying the strategy to current situation]
Decision: [ahead/left/right/backtrack]"""


            self.logger.debug(f"LLM Prompt:\n{prompt}")

            # Send prompt to LLM and get response
            time_start = time.time()
            response = self.llm.invoke(prompt)
            time_end = time.time()
            total_usage = {'llm_time': round(time_end - time_start, 2)}
            for tkey in ['total_tokens', 'prompt_tokens', 'completion_tokens']:
                total_usage[tkey] = response.response_metadata['token_usage'][tkey]
            total_usage['reasoning_tokens'] = response.response_metadata['token_usage']['completion_tokens_details'][
                'reasoning_tokens']

            llm_output = response.content if hasattr(response, 'content') else str(response)

            self.reasoning_output = f"{self._format_scan_observation(scan_info=scan_info)}\n{llm_output}"
            self.logger.debug(f"LLM Response:\n{llm_output}")

            # Parse the LLM response to extract the decision
            chosen_direction = self._parse_llm_response(llm_output)

            self.logger.debug(f"Extracted LLM decision: '{chosen_direction}'")

            return chosen_direction

        except Exception as e:
            # If anything goes wrong with LLM, fall back to rule-based
            self.logger.warning(f"LLM decision failed ({e}), falling back to rule-based")
            return self._make_rule_based_decision(scan_info, strategy, history)

    def _parse_llm_response(self, llm_output: str) -> str:
        """
        Parse LLM response to extract the chosen direction.

        Looks for patterns like "Decision: ahead" or just "ahead" in the response.
        Falls back to rule-based decision if parsing fails.

        Args:
            llm_output: Raw text response from the LLM

        Returns:
            str: Extracted direction ("ahead", "left", "right", "backtrack")
        """
        # Clean up the output
        output_clean = llm_output.strip().lower()

        # Valid directions the LLM can choose
        valid_directions = ["ahead", "left", "right", "backtrack"]

        # Try to find "Decision: [direction]" pattern first
        decision_pattern = r'decision:\s*([a-zA-Z]+)'
        decision_match = re.search(decision_pattern, output_clean)

        if decision_match:
            extracted = decision_match.group(1).strip()
            if extracted in valid_directions:
                self.logger.debug(f"Found decision pattern: '{extracted}'")
                return extracted

        # If no decision pattern, look for any valid direction in the text
        for direction in valid_directions:
            if direction in output_clean:
                # Make sure it's a whole word, not part of another word
                word_pattern = rf'\b{direction}\b'
                if re.search(word_pattern, output_clean):
                    self.logger.debug(f"Found direction word: '{direction}'")
                    return direction

        # If we can't parse anything valid, log the issue and fall back
        self.logger.warning(f"Could not parse valid direction from LLM output: {llm_output[:100]}...")

        # Use a simple fallback decision (this could be improved)
        return "ahead"

    def _initialize_workflow(self):
        """
        Lazy initialization of the LangGraph workflow and initial state.

        This method sets up the workflow graph and creates the initial state
        with fresh scan data from the starting position.
        """
        if self.is_initialized:
            return

        self.logger.debug("Initializing agentic workflow...")

        # Build the workflow graph
        self.graph_app = self._build_workflow()

        # Perform initial scan from starting position
        initial_scan = self._perform_scan()

        # Create initial state
        self.current_state = {
            "scan_info": initial_scan,
            "history": [],
            "strategy": self.strategy,
            "last_result": None,
            "step_budget": self.step_budget
        }

        # Log initial situation
        initial_obs = self._format_scan_observation(initial_scan)
        self.logger.debug(f"Initial state: {initial_obs}")

        self.is_initialized = True
        self.logger.debug("Workflow initialization complete")

    def step(self) -> Tuple[bool, int, Dict, Optional[str]]:
        """
        Execute one decision-making cycle of the agent.

        Flow:
        1. Initialize workflow if needed (includes initial scan)
        2. Agent decides direction based on current scan + strategy
        3. Mouse moves in chosen direction (if possible)
        4. Perform new scan from new position
        5. Update history and prepare for next cycle

        Returns:
            Tuple containing:
            - bool: True if mouse has reached the goal
            - int: Total number of steps taken by the mouse
            - Dict: Render data from maze.prepare_render(mouse) for visualization
        """
        # Initialize workflow on first call
        if not self.is_initialized:
            self._initialize_workflow()

        lcl_start = time.time()
        # Check if we're already at goal or out of budget
        if self.current_state["scan_info"]["at_goal"]:
            self.logger.debug("Already at goal!")
            at_goal = True
            total_steps = self.mouse.steps
            render_data = self.maze.prepare_render(mouse=self.mouse)
            return at_goal, total_steps, render_data

        if self.mouse.steps >= self.step_budget:
            self.logger.debug(f"Step budget exceeded ({self.step_budget})")
            at_goal = False
            total_steps = self.mouse.steps
            render_data = self.maze.prepare_render(mouse=self.mouse)
            return at_goal, total_steps, render_data

        # === AGENT DECISION PHASE ===
        # Run the agent through the workflow to decide direction
        self.logger.debug(f"\n--- STEP {self.mouse.steps + 1} ---")

        # Invoke the workflow to get agent's decision
        workflow_result = self.graph_app.invoke(self.current_state)

        # Extract the chosen direction from workflow result
        agent_result = workflow_result.get("last_result", {})
        chosen_direction = agent_result.get("chosen_direction", "ahead")

        # === MOUSE MOVEMENT PHASE ===
        # Execute the agent's chosen direction
        movement_success = self.mouse.move(direction=chosen_direction)
        self.logger.debug(f"Mouse move '{chosen_direction}': {'SUCCESS' if movement_success else 'BLOCKED'}")

        # === SCANNING PHASE ===
        # Perform fresh scan from new position
        fresh_scan = self._perform_scan()
        new_observation = self._format_scan_observation(fresh_scan)
        self.logger.debug(f"New position scan: {new_observation}")

        # === STATE UPDATE PHASE ===
        # Update history with what just happened
        history = self.current_state.get("history", []).copy()
        action_result = f"{chosen_direction}({'success' if movement_success else 'blocked'})"
        history.append(action_result)

        # Keep only last 5 moves for context (prevents history from growing infinitely)
        if len(history) > 5:
            history = history[-5:]

        # Update current state with fresh information
        self.current_state = {
            "scan_info": fresh_scan,
            "history": history,
            "strategy": self.strategy,
            "last_result": {
                "chosen_direction": chosen_direction,
                "movement_success": movement_success,
                "step_completed": True
            },
            "step_budget": self.step_budget
        }

        # === RETURN PHASE ===
        # Prepare return values
        at_goal = fresh_scan["at_goal"]
        total_steps = self.mouse.steps
        render_data = self.maze.prepare_render(mouse=self.mouse)

        # Log completion status
        if at_goal:
            lcl_goal = 'Goal Reached!'
            self.logger.debug(f"GOAL REACHED! Total steps: {total_steps}")
        else:
            lcl_goal = "Searching"
        lcl_end = time.time()
        lcl_mtime, lcl_ttime = lcl_end - lcl_start, lcl_end - self.start_time
        total_steps = f"{lcl_goal}, Total Steps: {self.mouse.steps:>3}, Total Time: {lcl_ttime:.1f} seconds, Thinking Time: {lcl_mtime:.1f} seconds"
        return at_goal, total_steps, render_data, self.reasoning_output

    def render_at_start(self):
        return self.maze.prepare_render(mouse=self.mouse)

    def step_old(self) -> Tuple[bool, int, Dict]:
        """
        Execute one decision-making cycle of the agent.
        
        Returns:
            Tuple containing:
            - bool: True if mouse has reached the goal
            - int: Total number of steps taken by the mouse
            - Dict: Render data from maze.prepare_render(mouse) for visualization
        """
        # TODO: Implementation will go here
        # For now, return placeholder values
        at_goal = self.mouse.sense_at_goal()
        total_steps = self.mouse.steps
        render_data = self.maze.prepare_render(mouse=self.mouse)
        
        self.logger.debug(f"Step executed: at_goal={at_goal}, steps={total_steps}")
        
        return at_goal, total_steps, render_data


def main():
    try:
        print(f"start maze v.{__version__}")
        load_dotenv()
        # Create maze and mouse
        maze, start, goal = get_default_maze()
        maze_obj = AAMaze(maze=maze, start=start, goal=goal)
        mouse = AAMouse(aamaze=maze_obj)

        # Define strategy
        strategy = (
            "Walk into the maze and keep moving forward until you cannot go further. "
            "Each time you walk down a passage, make a small mark on it. "
            "If you reach a dead end, turn around and go back the way you came."
        )

        # Create agentic mouse
        agent = AAgenticMouse(
            maze=maze_obj,
            mouse=mouse,
            strategy=strategy,
            use_llm=True  # Start with rule-based for testing
        )

        # Test the step method
        print(render_with_mouse(maze=maze_obj, mouse=mouse))
        at_goal, steps, render_data, reasoning_str = agent.step()
        # scan_result = agent._perform_scan()
        # formatted_scan = agent._format_scan_observation(scan_info=scan_result)

        # Continue until goal reached
        while not at_goal:
            print(f"Mouse steps: {steps},\nreasoning: '{reasoning_str}'\n")
            print(render_with_mouse(maze=maze_obj, mouse=mouse))

            # Safety check - avoid infinite loops
            if steps >= agent.step_budget:
                print(f"\nStep budget exceeded ({agent.step_budget} steps)")
                break

            # Execute next step
            at_goal, steps, render_data, reasoning_str = agent.step()

        # Show final result
        print(f"\nMouse steps: {steps}")
        print(render_with_mouse(maze=maze_obj, mouse=mouse))

        if at_goal:
            print(f"\nSUCCESS! Goal reached in {steps} mouse steps!")
        else:
            print(f"\nStopped after {steps} mouse steps and {agent.agent_steps} agent decisions")

        return True
    except Exception as e:
        logger.exception(f"Error in main: {e}")
        return False



if __name__ == "__main__":
    main()
