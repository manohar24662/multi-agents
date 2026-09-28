# scripts/debug_visualize.py
"""
Standalone debug script for visualizing the agentic_langgraph graph structure.
Run with: python -m agentic_langgraph.scripts.debug_visualize
"""
import logging

from agentic_langgraph.graph import create_graph

logging.basicConfig(level=logging.INFO)


def visualize():
    compiled_graph = create_graph()
    graph_repr = compiled_graph.get_graph()

    try:
        print(graph_repr.draw_ascii())
    except Exception as error:
        print(f"Skipped ASCII export: {error}")

    try:
        png_bytes = graph_repr.draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_bytes)
        print("Saved graph.png")
    except Exception as e:
        print(f"Skipped PNG export: {e}")


if __name__ == "__main__":
    visualize()
