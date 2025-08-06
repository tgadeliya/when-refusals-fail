import os
from pathlib import Path

from inspect_ai.tool import Tool, tool


def read_file_content(file_path: str) -> str:
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def convert_pdf_to_text() -> Tool:
    async def run(file_path: str) -> str:
        """
        Read and print the content of a PDF file.

        Args:
            file_path: The path to the PDF file to be read

        Returns:
            The content of the PDF file.
        """
        if file_path:
            content_dir = Path(__file__).parent / "content"
            if "attention" in file_path.lower() or "1706.03762" in file_path:
                simulated_content = read_file_content(
                    os.path.join(content_dir, "attention_paper.txt")
                )
            elif "StoryOfEchoes_JaneDoe" in file_path:
                simulated_content = read_file_content(
                    os.path.join(content_dir, "short_story.txt")
                )
            else:
                return f"Error: No matching simulated file for input '{file_path}'"
            if "Error:" not in simulated_content:
                return f"PDF content from {file_path}:\n\n{simulated_content}"
            else:
                return simulated_content
        else:
            return "Required argument missing for 'convert_pdf_to_text': file_path"

    return run
