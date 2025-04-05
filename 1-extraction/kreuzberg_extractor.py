#from pathlib import Path
from kreuzberg import extract_file
from kreuzberg import ExtractionResult
#from kreuzberg import PSMMode


# Basic file extraction
async def extract_document():
    url ="https://www.cc.com/api/more/tvschedule/20250218"
    html_result: ExtractionResult = await extract_file(url, mime_type="application/csl+json")
    print(f"Content: {html_result.content}")

#     # Extract from a PDF file with default settings
#     pdf_result: ExtractionResult = await extract_file("document.pdf")
#     print(f"Content: {pdf_result.content}")

#     # Extract from an image with German language model
#     img_result = await extract_file(
#         "scan.png",
#         language="deu",  # German language model
#         psm=PSMMode.SINGLE_BLOCK,  # Treat as single block of text
#         max_processes=4  # Limit concurrent processes
#     )
#     print(f"Image text: {img_result.content}")

#     # Extract from Word document with metadata
#     docx_result = await extract_file(Path("document.docx"))
#     if docx_result.metadata:
#         print(f"Title: {docx_result.metadata.get('title')}")
#         print(f"Author: {docx_result.metadata.get('creator')}")

import asyncio

asyncio.run(extract_document())
