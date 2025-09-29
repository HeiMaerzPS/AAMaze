import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle  # Missing import for render_payload_matplotlib
import numpy as np
import time
from aamaze_mouse import AAMaze, AAMouse, get_default_maze, MAX_STEPS
from aagentic_mouse import AAgenticMouse

MODEL = 'gpt-4.1-nano' #'gpt-5-nano' 'gpt-4.1-nano'
MAX_STEPS = 128
__version__ = '20250929_0944'


def main():
    st.set_page_config(layout="wide")
    st.image("AA_large.png", width=64)
    st.title("The AA AI Breakdown Challenge")

    if 'app_version' not in st.session_state:
        st.session_state.app_version = __version__

    if 'llm_model' not in st.session_state:
        try:
            st.session_state.llm_model = st.secrets.get("LLM_MODEL") or MODEL
        except Exception as e:
            st.session_state.llm_model = MODEL

    if 'max_steps' not in st.session_state:
        try:
            st.session_state.max_steps = int(st.secrets.get("MAX_STEPS") or MAX_STEPS)
        except Exception as e:
            st.session_state.max_steps = MAX_STEPS

    if 'steps' not in st.session_state:
        st.session_state.steps = 0

    # Initialize maze and mouse in session state (so they persist)
    if 'maze_obj' not in st.session_state:
        maze, start, goal = get_default_maze()
        st.session_state.maze_obj = AAMaze(maze=maze, start=start, goal=goal)
        st.session_state.mouse_obj = AAMouse(aamaze=st.session_state.maze_obj, max_steps=MAX_STEPS)

    if 'agent' not in st.session_state:
        st.session_state.agent = None

    st.session_state.at_goal, st.session_state.reasoning_str = False, ""

    # Split screen into two columns
    left_col, right_col = st.columns([5,3])

    with left_col:
        user_input = st.text_area(
            "Reach the customer as fast as you can",
            height=160,
            placeholder="Enter your strategy using natural language",
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
                    step_budget=st.session_state.max_steps,
                    model=st.session_state.llm_model
                )

                # Store the agent in session state
                st.session_state.agent = agent

                # st.success("AAgenticMouse instantiated successfully! Solving will begin...")
                # st.write(f"Strategy: {user_input.strip()[:100]}...")

            else:
                st.error("Please enter a strategy first.")

        reasoning_box = st.empty()
        # st.divider()
        # mouse_state_box = st.empty()

    with right_col:
        if st.session_state.agent:
            maze_placeholder = st.empty()

            mouse_state_box = st.empty()

            mouse_state = st.session_state.agent.get_status()
            # st.session_state.mouse_state = mouse_state
            mouse_state_box.markdown(mouse_state)
            reasoning_str=f"**AAgenticMouse** will crawl the maze using {st.session_state.llm_model}"
            reasoning_box.markdown(reasoning_str)

            # Initial render after instantiation
            render_data = st.session_state.agent.render_at_start()
            fig = render_payload_matplotlib(render_data)
            maze_placeholder.pyplot(fig)
            plt.close(fig)

            while not st.session_state.at_goal and st.session_state.steps < st.session_state.max_steps:
                st.session_state.steps += 1
                at_goal, score, render_data, reasoning_str = st.session_state.agent.step()
                st.session_state.at_goal = at_goal
                # st.session_state.steps = steps
                st.session_state.reasoning_str = reasoning_str

                # update left-column stream
                mouse_state = st.session_state.agent.get_status()
                # st.session_state.mouse_state = mouse_state
                mouse_state_box.markdown(mouse_state)
                reasoning_box.markdown(reasoning_str)
                # time.sleep(0.1)

                fig = render_payload_matplotlib(render_data)
                maze_placeholder.pyplot(fig)
                plt.close(fig)

            if not st.session_state.at_goal and st.session_state.steps >= st.session_state.max_steps:
                st.error(f"Stopped: {st.session_state.steps}")

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