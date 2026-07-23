from pypdf import PdfReader
import asyncio
import os

from llama_cloud import AsyncLlamaCloud

PDF_FOLDER = "pdfs-back"
TEXT_OUTPUT_FOLDER = "texts-l"

async def process_pdf(file_path):
    client = AsyncLlamaCloud(api_key="llx-REDACTED")

    file_obj = await client.files.create(file=file_path, purpose="parse")

    result = await client.parsing.parse(
        file_id=file_obj.id,
        version="latest",
        tier="agentic",
        expand=["markdown_full"],
    )

    return result.markdown_full

def pdf_to_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

def convention_run():
    for pdf in os.listdir(PDF_FOLDER):
        if pdf.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, pdf)
            text = pdf_to_text(pdf_path)
            txt_path = os.path.join("texts", pdf.replace(".pdf", ".txt"))
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

async def run_all():
    for pdf in os.listdir(PDF_FOLDER):
        if pdf.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, pdf)
            text = await process_pdf(pdf_path)
            txt_path = os.path.join(TEXT_OUTPUT_FOLDER, pdf.replace(".pdf", ".md"))
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

if __name__ == "__main__":
    asyncio.run(run_all())