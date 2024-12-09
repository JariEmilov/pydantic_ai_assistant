from typing import Any, Dict, List, Optional, Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field
import os
from pathlib import Path
import requests
import json
from io import StringIO
import numpy as np
from datetime import datetime

class DataAnalysisAgent(BaseModel):
    data: Optional[pd.DataFrame] = None
    file_path: Optional[str] = None
    column_descriptions: Dict[str, str] = Field(default_factory=dict)
    ollama_url: str = "http://localhost:11434/api/generate"
    model_name: str = "llama2"  # default model
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    data_summary: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True  # Allow pandas DataFrame
        
    def _generate_data_summary(self) -> str:
        """Generate a comprehensive summary of the data."""
        if self.data is None:
            return ""
            
        buffer = StringIO()
        self.data.info(buf=buffer)
        data_info = buffer.getvalue()
        
        summary = [
            "Data Summary:",
            f"- Shape: {self.data.shape}",
            f"- Columns: {list(self.data.columns)}",
            "\nColumn Details:"
        ]
        
        for column, description in self.column_descriptions.items():
            summary.append(f"- {column}: {description}")
            
        summary.append("\nSample Data:")
        summary.append(self.data.head().to_string())
        
        summary.append("\nBasic Statistics:")
        summary.append(self.data.describe().to_string())
        
        return "\n".join(summary)

    def load_data(self, file_path: str) -> None:
        """Load data from various file formats."""
        try:
            # Remove quotes if present
            file_path = file_path.strip('"').strip("'")
            self.file_path = file_path
            
            # Convert to Path object for better path handling
            path = Path(file_path)
            file_extension = path.suffix.lower()
            
            try:
                if file_extension == '.csv':
                    self.data = pd.read_csv(path, encoding='utf-8')
                elif file_extension in ['.xlsx', '.xls']:
                    self.data = pd.read_excel(path, engine='openpyxl')
                else:
                    raise ValueError(f"Unsupported file format: {file_extension}")
                
                # Check if data was loaded successfully
                if self.data is None or self.data.empty:
                    raise ValueError("No data was loaded from the file")
                
                self._analyze_columns()
                self.data_summary = self._generate_data_summary()
                
                # Add initial system message to conversation history
                self.conversation_history = [{
                    "role": "system",
                    "content": f"You are a data analysis expert. You have loaded a dataset with the following information:\n{self.data_summary}\n\nYou should maintain context of our conversation and refer back to previous insights when relevant."
                }]
                
                print(f"Successfully loaded data with shape: {self.data.shape}")
                
            except UnicodeDecodeError:
                # Try different encoding for CSV files
                if file_extension == '.csv':
                    self.data = pd.read_csv(path, encoding='latin1')
                    self._analyze_columns()
                else:
                    raise
                    
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def _analyze_columns(self) -> None:
        """Analyze and store information about each column."""
        if self.data is None:
            return

        for column in self.data.columns:
            dtype = str(self.data[column].dtype)
            unique_count = self.data[column].nunique()
            missing_count = self.data[column].isnull().sum()
            
            description = f"Type: {dtype}, Unique values: {unique_count}, Missing values: {missing_count}"
            self.column_descriptions[column] = description

    def get_basic_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the dataset."""
        if self.data is None:
            raise ValueError("No data loaded")
        
        return {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "numeric_summary": self.data.describe().to_dict(),
            "missing_values": self.data.isnull().sum().to_dict()
        }

    def _identify_column_types(self) -> Dict[str, str]:
        """Identify the type of each column for visualization purposes."""
        if self.data is None:
            return {}
            
        column_types = {}
        for column in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[column]):
                if self.data[column].nunique() < 10:
                    column_types[column] = 'categorical_numeric'
                else:
                    column_types[column] = 'continuous'
            elif pd.api.types.is_datetime64_any_dtype(self.data[column]):
                column_types[column] = 'datetime'
            else:
                if self.data[column].nunique() < 10:
                    column_types[column] = 'categorical'
                else:
                    column_types[column] = 'text'
        return column_types

    def create_dashboard(self, output_file: str = "data_dashboard.html") -> None:
        """Create an interactive dashboard with Plotly."""
        if self.data is None:
            raise ValueError("No data loaded")

        # Identify column types
        column_types = self._identify_column_types()
        
        # Get insights about what to visualize
        insights_prompt = f"""Analyze this dataset and suggest 3-5 key aspects to visualize:

Column Information:
{json.dumps(column_types, indent=2)}

Data Sample:
{self.data.head().to_string()}

Statistical Summary:
{self.data.describe().to_string()}

Suggest specific visualizations that would reveal important business insights."""

        insights = self._query_ollama(insights_prompt)
        
        # Create figures list to store all plots
        figures = []
        
        # 1. Distribution plots for continuous variables
        continuous_vars = [col for col, type_ in column_types.items() if type_ == 'continuous']
        if continuous_vars:
            for var in continuous_vars:
                fig = px.histogram(
                    self.data, 
                    x=var,
                    title=f'Distribution of {var}',
                    template='plotly_white'
                )
                figures.append(fig)

        # 2. Bar plots for categorical variables
        categorical_vars = [col for col, type_ in column_types.items() 
                          if type_ in ['categorical', 'categorical_numeric']]
        if categorical_vars:
            for var in categorical_vars:
                value_counts = self.data[var].value_counts()
                fig = px.bar(
                    x=value_counts.index.astype(str),
                    y=value_counts.values,
                    title=f'Distribution of {var}',
                    template='plotly_white'
                )
                fig.update_xaxes(tickangle=45)
                figures.append(fig)

        # 3. Correlation heatmap for numeric variables
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = self.data[numeric_cols].corr()
            fig = px.imshow(
                correlation_matrix,
                title='Correlation Heatmap',
                template='plotly_white',
                aspect='auto',
                color_continuous_scale='RdBu'
            )
            figures.append(fig)

        # 4. Time series plots if datetime columns exist
        datetime_cols = [col for col, type_ in column_types.items() if type_ == 'datetime']
        if datetime_cols and len(numeric_cols) > 0:
            for date_col in datetime_cols:
                for numeric_col in numeric_cols[:3]:
                    fig = px.line(
                        self.data,
                        x=date_col,
                        y=numeric_col,
                        title=f'{numeric_col} over Time',
                        template='plotly_white'
                    )
                    figures.append(fig)

        # 5. Box plots for categorical vs continuous variables
        if categorical_vars and continuous_vars:
            for cat_var in categorical_vars[:2]:
                for cont_var in continuous_vars[:3]:
                    fig = px.box(
                        self.data,
                        x=cat_var,
                        y=cont_var,
                        title=f'{cont_var} by {cat_var}',
                        template='plotly_white'
                    )
                    fig.update_xaxes(tickangle=45)
                    figures.append(fig)

        # Create the dashboard layout
        dashboard = go.Figure()

        # Add a button to toggle between plots
        steps = []
        for i in range(len(figures)):
            step = {
                'args': [
                    [{'visible': [False] * len(figures)}],
                    {'showlegend': True}
                ],
                'label': f'Plot {i+1}',
                'method': 'update'
            }
            step['args'][0][0]['visible'] = [j == i for j in range(len(figures))]
            steps.append(step)

        # Add all traces from individual figures
        for fig in figures:
            for trace in fig.data:
                dashboard.add_trace(trace)

        # Make only the first plot visible initially
        for i in range(len(dashboard.data)):
            dashboard.data[i].visible = (i < len(figures[0].data))

        # Update layout with dropdown menu
        dashboard.update_layout(
            updatemenus=[{
                'buttons': steps,
                'direction': 'down',
                'showactive': True,
                'x': 0.1,
                'y': 1.15,
                'xanchor': 'left',
                'yanchor': 'top'
            }],
            title={
                'text': 'Interactive Data Analysis Dashboard',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            template='plotly_white',
            height=800
        )

        # Add AI insights as annotations
        dashboard.add_annotation(
            text=f"AI Insights:<br>{insights}",
            xref="paper",
            yref="paper",
            x=0,
            y=-0.2,
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )

        # Save the dashboard
        dashboard.write_html(
            output_file,
            include_plotlyjs=True,
            full_html=True
        )
        
        print(f"\nDashboard has been created and saved as '{output_file}'")
        print("Open this file in your web browser to explore the interactive dashboard.")

    def _query_ollama(self, prompt: str) -> str:
        """Query Ollama API with a prompt."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            return f"Error querying Ollama: {str(e)}"

    def analyze_question(self, question: str) -> str:
        """Analyze the data based on a user question using Ollama."""
        if self.data is None:
            raise ValueError("No data loaded")

        # Get data info using StringIO
        buffer = StringIO()
        self.data.info(buf=buffer)
        data_info = buffer.getvalue()
        
        # Get sample and description
        data_sample = self.data.head().to_string()
        data_description = self.data.describe().to_string()
        
        # Add column information
        columns_info = "\nColumn Details:\n"
        for column, description in self.column_descriptions.items():
            columns_info += f"{column}: {description}\n"

        # Construct the prompt
        prompt = f"""You are a data analysis expert. Please analyze the following dataset and question:

Data Info:
{data_info}

Sample Data:
{data_sample}

Statistical Description:
{data_description}

Column Details:
{columns_info}

Question: {question}

Please provide a detailed analysis focusing on business insights and patterns in the data."""

        return self._query_ollama(prompt)

    def generate_report(self) -> str:
        """Generate a comprehensive report about the dataset."""
        if self.data is None:
            raise ValueError("No data loaded")

        report = []
        report.append("# Data Analysis Report")
        report.append("\n## Dataset Overview")
        report.append(f"- Number of rows: {self.data.shape[0]}")
        report.append(f"- Number of columns: {self.data.shape[1]}")
        
        report.append("\n## Column Information")
        for column, description in self.column_descriptions.items():
            report.append(f"\n### {column}")
            report.append(description)
            
        report.append("\n## Statistical Summary")
        report.append(self.data.describe().to_string())
        
        report.append("\n## Missing Values Analysis")
        missing_values = self.data.isnull().sum()
        report.append(missing_values.to_string())
        
        # Get AI insights using Ollama
        prompt = f"""Given this dataset summary, provide key insights and recommendations:

{report}

Please provide 3-5 key insights about this data."""
        
        ai_insights = self._query_ollama(prompt)
        report.append("\n## AI Insights")
        report.append(ai_insights)
        
        return "\n".join(report) 

    def chat_with_data(self, user_message: str) -> str:
        """Have a contextual conversation about the data."""
        if self.data is None:
            raise ValueError("No data loaded")
            
        # Add user message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Construct the prompt with conversation history
        prompt = "You are a data analysis expert having a conversation about a dataset. "
        prompt += "Use the conversation history to maintain context and refer to previous insights when relevant.\n\n"
        
        # Add conversation history
        for message in self.conversation_history[-5:]:  # Last 5 messages for context
            role = message["role"]
            content = message["content"]
            prompt += f"{role}: {content}\n\n"
            
        # Add the current question
        prompt += f"Based on the conversation history and the data, please provide a detailed response to the last question."
        
        # Get response from Ollama
        response = self._query_ollama(prompt)
        
        # Add assistant's response to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return response

    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation history."""
        if not self.conversation_history:
            return "No conversation history available."
            
        summary = ["Conversation History:"]
        for message in self.conversation_history[1:]:  # Skip the initial system message
            timestamp = message.get("timestamp", "")
            summary.append(f"\n[{message['role']}] {timestamp}")
            summary.append(message['content'])
            summary.append("-" * 50)
            
        return "\n".join(summary) 