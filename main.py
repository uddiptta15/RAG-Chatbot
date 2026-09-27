from src.graph import build_graph

def main():
    graph = build_graph()          # compiles the LangGraph once
    history = []                    # conversation history for this session

    print("Course assistant ready. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break

        result = graph.invoke({
            "query": user_input,
            "history": history,
        })

        print(f"\nAssistant: {result['answer']}")
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": result["answer"]})

if __name__ == "__main__":
    main()