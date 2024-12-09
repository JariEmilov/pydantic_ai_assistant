from data_analysis_agent import DataAnalysisAgent
import os
from pathlib import Path

def main():
    # Initialize the agent
    agent = DataAnalysisAgent()
    
    # Example usage with a sample dataset
    # Replace 'your_data.csv' with your actual data file
    print("Please provide the path to your data file (CSV, Excel):")
    file_path = input().strip()
    
    try:
        # Load the data
        agent.load_data(file_path)
        
        # Create dashboard filename based on input file
        dashboard_file = f"dashboard_{Path(file_path).stem}.html"
        
        while True:
            print("\nWhat would you like to do?")
            print("1. Get basic statistics")
            print("2. Create interactive dashboard")
            print("3. Chat about the data")
            print("4. Generate a comprehensive report")
            print("5. View conversation history")
            print("6. Exit")
            
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == "1":
                stats = agent.get_basic_stats()
                print("\nBasic Statistics:")
                print(f"Dataset shape: {stats['shape']}")
                print(f"Columns: {stats['columns']}")
                print("\nMissing values:")
                for col, count in stats['missing_values'].items():
                    print(f"{col}: {count}")
                
            elif choice == "2":
                print("\nCreating interactive dashboard...")
                agent.create_dashboard(dashboard_file)
                
            elif choice == "3":
                print("\nChat Mode - Type 'exit' to return to main menu")
                while True:
                    question = input("\nYou: ").strip()
                    if question.lower() == 'exit':
                        break
                    
                    response = agent.chat_with_data(question)
                    print("\nAssistant:", response)
                
            elif choice == "4":
                report = agent.generate_report()
                print("\nData Analysis Report:")
                print(report)
                
            elif choice == "5":
                conversation_summary = agent.get_conversation_summary()
                print("\nConversation History:")
                print(conversation_summary)
                
            elif choice == "6":
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please try again.")
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 