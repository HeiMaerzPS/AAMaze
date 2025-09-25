import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle  # Missing import for render_payload_matplotlib
import numpy as np
from aamaze_mouse import AAMaze, AAMouse, get_default_maze, MAX_STEPS
from aagentic_mouse import AAgenticMouse

MODEL = 'gpt-5-nano'
__version__ = '20250924_1907'


def render_maze_matplotlib(maze_obj, mouse_obj):
    """Render maze using matplotlib with walls, free spaces, start, goal, and mouse."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Get maze dimensions
    rows, cols = maze_obj.maze.shape

    # Display the maze (0=white/free, 1=black/wall)
    ax.imshow(maze_obj.maze, cmap='gray_r', interpolation='nearest')

    # Mark visited cells (light gray dots) - equivalent to "." in ASCII
    for pos in mouse_obj.visited:
        if pos != mouse_obj.pos and pos != maze_obj.start and pos != maze_obj.goal:
            r, c = pos
            ax.add_patch(plt.Circle((c, r), 0.15, color='lightgray', alpha=0.7))

    # Mark trail (yellow stars) - equivalent to "*" in ASCII
    for pos in mouse_obj.trail:
        if pos != mouse_obj.pos and pos != maze_obj.start and pos != maze_obj.goal:
            r, c = pos
            ax.plot(c, r, marker='*', color='gold', markersize=12, alpha=0.8)

    # Mark start position (green circle)
    start_r, start_c = maze_obj.start
    ax.add_patch(plt.Circle((start_c, start_r), 0.3, color='green', alpha=0.8))
    ax.text(start_c, start_r, 'S', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    # Mark goal position (red circle)
    goal_r, goal_c = maze_obj.goal
    ax.add_patch(plt.Circle((goal_c, goal_r), 0.3, color='red', alpha=0.8))
    ax.text(goal_c, goal_r, 'G', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    # Mark mouse position (blue circle)
    mouse_r, mouse_c = mouse_obj.pos
    ax.add_patch(plt.Circle((mouse_c, mouse_r), 0.3, color='blue', alpha=0.8))
    ax.text(mouse_c, mouse_r, 'M', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    # Set up the plot
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)  # Invert y-axis to match array indexing
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title('AAMaze')

    plt.tight_layout()
    return fig


def main():
    st.set_page_config(layout="wide")
    st.title("AAMaze")

    # Initialize maze and mouse in session state (so they persist)
    if 'maze_obj' not in st.session_state:
        maze, start, goal = get_default_maze()
        st.session_state.maze_obj = AAMaze(maze=maze, start=start, goal=goal)
        st.session_state.mouse_obj = AAMouse(aamaze=st.session_state.maze_obj, max_steps=MAX_STEPS)

    # # Initialize solving state management variables
    # if 'solving_active' not in st.session_state:
    #     st.session_state.solving_active = False
    #
    # if 'solve_complete' not in st.session_state:
    #     st.session_state.solve_complete = False
    #
    # if 'completion_reason' not in st.session_state:
    #     st.session_state.completion_reason = ""

    if 'agent' not in st.session_state:
        st.session_state.agent = None

    st.session_state.at_goal, st.session_state.steps, st.session_state.reasoning_str = False, 0, ""

    # Split screen into two columns
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Maze Solving Strategy")

        user_input = st.text_area(
            "Enter your maze-solving strategy in natural language:",
            height=200,
            placeholder="Example: Keep moving forward when possible. If blocked, turn right, then left, then backtrack. Always prefer unvisited paths...",
            key='strategy'
        )

        # Simple start button with state-aware behavior
        if st.button("Start Solving"):
            if user_input.strip():
                # Reset all state for fresh solving attempt
                st.session_state.solving_active = True
                st.session_state.solve_complete = False
                st.session_state.completion_reason = ""

                # Reset maze and mouse to starting positions
                maze, start, goal = get_default_maze()
                st.session_state.maze_obj = AAMaze(maze=maze, start=start, goal=goal)
                st.session_state.mouse_obj = AAMouse(aamaze=st.session_state.maze_obj, max_steps=MAX_STEPS)

                # Instantiate AAgenticMouse with the user's strategy
                agent = AAgenticMouse(
                    maze=st.session_state.maze_obj,
                    mouse=st.session_state.mouse_obj,
                    strategy=user_input.strip(),
                    use_llm=True,
                    step_budget=500,
                    model=MODEL
                )

                # Store the agent in session state
                st.session_state.agent = agent

                # st.success("AAgenticMouse instantiated successfully! Solving will begin...")
                # st.write(f"Strategy: {user_input.strip()[:100]}...")

            else:
                st.error("Please enter a strategy first.")

        reasoning_box = st.empty()

    with right_col:
        if st.session_state.agent:
            maze_placeholder = st.empty()

            reasoning_box.markdown(f"**AAgenticMouse** will run through the maze")

            # Initial render after instantiation
            render_data = st.session_state.agent.render_at_start()
            fig = render_payload_matplotlib(render_data)
            maze_placeholder.pyplot(fig)
            plt.close(fig)

            while not st.session_state.at_goal and st.session_state.steps < 500:
                at_goal, steps, render_data, reasoning_str = st.session_state.agent.step()
                st.session_state.at_goal = at_goal
                st.session_state.steps = steps
                st.session_state.reasoning_str = reasoning_str

                # update left-column stream
                reasoning_box.markdown(f"**Last Step Reasoning:**\n {reasoning_str}")

                fig = render_payload_matplotlib(render_data)
                maze_placeholder.pyplot(fig)
                plt.close(fig)

        #         st.write(st.session_state.reasoning_str)
        # else:
        #     st.write(f"waiting for user input")

    st.divider()

    with st.expander('monsters hide here'):
        st.write(st.session_state)


def render_payload_matplotlib(payload):
    def _as_tuples(points):
        return [tuple(p) for p in points]

    payload["background"]["array"] = np.array(payload["background"]["array"])

    for k in ("trail_dark", "trail_light"):
        if payload.get(k) and "points" in payload[k]:
            payload[k]["points"] = _as_tuples(payload[k]["points"])

    for k in ("start", "goal", "mouse"):
        if payload.get(k) and "rc" in payload[k]:
            payload[k]["rc"] = tuple(payload[k]["rc"])

    rows = payload["extent"]["rows"];
    cols = payload["extent"]["cols"]
    arr = payload["background"]["array"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(arr, cmap="gray_r", interpolation="nearest")

    # no edgecolor, no linewidth
    def dot(c, r, rad, fc, alpha=1, z=3, txt=None, fs=12):
        ax.add_patch(Circle((c, r), rad, facecolor=fc, edgecolor="none", linewidth=0, alpha=alpha, zorder=z))
        if txt:
            ax.text(c, r, txt, ha="center", va="center", color="black", weight="bold", zorder=z + 1, fontsize=fs)

    studs = payload.get("studs")
    if studs and studs.get("points"):
        for r, c in _as_tuples(studs["points"]):
            dot(c, r, studs["radius"], studs["color"], alpha=studs.get("alpha", 0.6), z=2)

    for r, c in payload["trail_light"]["points"]:
        dot(c, r, payload["trail_light"]["radius"], payload["trail_light"]["facecolor"], z=3)
    for r, c in payload["trail_dark"]["points"]:
        dot(c, r, payload["trail_dark"]["radius"], payload["trail_dark"]["facecolor"], z=4)

    sr, sc = payload["start"]["rc"]
    dot(sc, sr, payload["start"]["radius"], payload["start"]["facecolor"], z=5)
    gr, gc = payload["goal"]["rc"]
    dot(gc, gr, payload["goal"]["radius"], payload["goal"]["facecolor"], z=6)
    mr, mc = payload["mouse"]["rc"]
    dot(mc, mr, payload["mouse"]["radius"], payload["mouse"]["facecolor"], z=7,
        txt=payload["mouse"]["label"]["text"], fs=8)

    ax.set_xlim(-0.5, cols - 0.5);
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xticks([]);
    ax.set_yticks([])
    ax.set_aspect("equal");
    for sp in ax.spines.values():
        sp.set_visible(False)
    plt.tight_layout();
    return fig


if __name__ == "__main__":
    main()