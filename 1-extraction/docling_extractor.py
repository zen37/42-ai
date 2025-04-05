from docling.document_converter import DocumentConverter

converter = DocumentConverter()

# --------------------------------------------------------------
# Basic PDF extraction
# --------------------------------------------------------------

# result = converter.convert("https://arxiv.org/pdf/2408.09869")

# document = result.document
# markdown_output = document.export_to_markdown()
# json_output = document.export_to_dict()

# print(markdown_output)

# --------------------------------------------------------------
# Basic HTML extraction
# --------------------------------------------------------------

# url = "bbc.html"
url = "https://www.cc.com/api/more/tvschedule/20250218"
# url = "https://en.wikipedia.org/wiki/Duck"
result = converter.convert(url)

document = result.document
print(document)


# markdown_output = document.export_to_markdown()
# print(markdown_output)
# import json

# json_output = document.export_to_dict()  # Get structured data as a dictionary
# print(json_output)
#print(json.dumps(json_output, indent=4))  # Pretty-print as JSON
