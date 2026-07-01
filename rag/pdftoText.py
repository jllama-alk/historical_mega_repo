from pypdf import PdfReader
import asyncio
import os

from llama_cloud import AsyncLlamaCloud

async def process_pdf(file_path):
    client = AsyncLlamaCloud(api_key="llx-BRg54Rz6hpadc3mD0GYVZgYeH1dDhwT9sPm68MXPoVwIVry5")

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
    for pdf in os.listdir("pdfs"):
        if pdf.endswith(".pdf"):
            pdf_path = os.path.join("pdfs", pdf)
            text = pdf_to_text(pdf_path)
            txt_path = os.path.join("texts", pdf.replace(".pdf", ".txt"))
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

async def run_all():
    for pdf in os.listdir("pdfs"):
        if pdf.endswith(".pdf"):
            pdf_path = os.path.join("pdfs", pdf)
            text = await process_pdf(pdf_path)
            txt_path = os.path.join("texts-l", pdf.replace(".pdf", ".txt"))
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

if __name__ == "__main__":
    asyncio.run(run_all())