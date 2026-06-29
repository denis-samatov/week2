"""OCR/LLM pipeline based on the original Colab notebook `week2.ipynb`.

Pipeline:
1. preprocess images with OpenCV;
2. convert PDF/images to Markdown and JSON via Docling + RapidOCR;
3. optionally generate a short Russian analytical report via Ollama Cloud.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Iterable

import cv2

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption
from docling_core.types.doc import ImageRefMode
from llama_index.llms.openai_like import OpenAILike


ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}
DEFAULT_MODEL = "gemma4:31b-cloud"


def get_ollama_api_key() -> str:
    """Return Ollama API key from environment or Google Colab secret storage."""
    api_key = os.getenv("OLLAMA_API_KEY")
    if api_key:
        return api_key

    try:
        from google.colab import userdata  # type: ignore

        api_key = userdata.get("ocr_olama")
        if api_key:
            return api_key
    except Exception:
        pass

    raise ValueError(
        "Ollama API key is not configured. Set OLLAMA_API_KEY or add Colab secret 'ocr_olama'."
    )


def preprocess_image(image_path: str | Path) -> str:
    """Sharpen and binarize an image before OCR; returns temporary file path."""
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpen = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)
    threshold = cv2.adaptiveThreshold(
        sharpen,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        15,
    )

    output_path = Path(tempfile.gettempdir()) / f"preprocessed_{image_path.name}"
    cv2.imwrite(str(output_path), threshold)
    return str(output_path)


def build_converter() -> DocumentConverter:
    """Create Docling converter with OCR, formula enrichment and picture extraction."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_formula_enrichment = True
    pipeline_options.generate_picture_images = True
    pipeline_options.ocr_options = RapidOcrOptions()

    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
    }
    return DocumentConverter(format_options=format_options)


def run_ocr_pipeline(
    file_path: str | Path,
    converter: DocumentConverter,
    output_dir: str | Path,
    assets: bool = False,
) -> tuple[str, dict]:
    """Convert a document/image to Markdown and Docling JSON dictionary."""
    path = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_image = path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    target_path = str(path)
    temp_file_to_clean: str | None = None

    if is_image:
        target_path = preprocess_image(path)
        temp_file_to_clean = target_path

    try:
        result = converter.convert(target_path)
    finally:
        if temp_file_to_clean and os.path.exists(temp_file_to_clean):
            os.remove(temp_file_to_clean)

    doc_dict = result.document.export_to_dict()

    if assets:
        assets_dir = output_dir / f"{path.stem}_assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / f"{path.stem}.md"

        result.document.save_as_markdown(
            filename=md_path,
            image_mode=ImageRefMode.REFERENCED,
            artifacts_dir=assets_dir,
        )
        markdown_text = md_path.read_text(encoding="utf-8")
    else:
        markdown_text = result.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
        (output_dir / f"{path.stem}.md").write_text(markdown_text, encoding="utf-8")

    return markdown_text, doc_dict


def run_llm_extraction(markdown_text: str, model_name: str = DEFAULT_MODEL) -> str:
    """Generate a short Russian analytical report from Markdown text via Ollama Cloud."""
    api_key = get_ollama_api_key()
    llm = OpenAILike(
        model=model_name,
        api_key=api_key,
        api_base="https://ollama.com/v1",
        is_chat_model=True,
        temperature=0.1,
        max_tokens=2048,
        timeout=360.0,
    )

    prompt = f"""
Перед тобой структурированное представление документа в формате Markdown.
Твоя задача — внимательно проанализировать его и составить краткий аналитический отчет/резюме на русском языке (1-2 абзаца).

Сфокусируйся на:
- Основной цели документа / исследования.
- Ключевых выводах, результатах или метриках.

Важные требования:
- Пиши только связный текст отчета на русском языке.
- Не добавляй вводных фраз ("Вот ваш отчет:", "Конечно, я помогу" и т.д.).

Документ для анализа:
---
{markdown_text}
---
"""
    response = llm.complete(prompt)
    return response.text.strip()


def iter_input_files(input_dir: str | Path) -> Iterable[Path]:
    input_dir = Path(input_dir)
    return sorted(
        file_path
        for file_path in input_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTENSIONS
    )


def process_documents(
    input_dir: str | Path,
    markdown_dir: str | Path,
    reports_dir: str | Path | None = None,
    assets: bool = False,
    run_llm: bool = False,
    model_name: str = DEFAULT_MODEL,
) -> None:
    converter = build_converter()
    markdown_dir = Path(markdown_dir)
    markdown_dir.mkdir(parents=True, exist_ok=True)

    reports_path = Path(reports_dir) if reports_dir else None
    if reports_path:
        reports_path.mkdir(parents=True, exist_ok=True)

    files_to_process = list(iter_input_files(input_dir))
    print(f"Found files to process: {len(files_to_process)}")

    for file_path in files_to_process:
        print(f"\nProcessing file: {file_path.name}")
        start_time = time.time()

        markdown_text, doc_dict = run_ocr_pipeline(
            file_path=file_path,
            converter=converter,
            output_dir=markdown_dir,
            assets=assets,
        )

        json_output_path = markdown_dir / f"{file_path.stem}.json"
        json_output_path.write_text(
            json.dumps(doc_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        elapsed = time.time() - start_time
        print(f"OCR completed in {elapsed:.2f} sec")
        print(f"Markdown/JSON saved to: {markdown_dir}")

        if run_llm:
            if reports_path is None:
                raise ValueError("reports_dir must be provided when run_llm=True")
            report = run_llm_extraction(markdown_text, model_name=model_name)
            report_path = reports_path / f"{file_path.stem}_report.txt"
            report_path.write_text(report, encoding="utf-8")
            print(f"Report saved to: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR + LLM document processing pipeline")
    parser.add_argument("--input-dir", required=True, help="Directory with PDF/images")
    parser.add_argument("--markdown-dir", required=True, help="Directory for Markdown/JSON OCR results")
    parser.add_argument("--reports-dir", default=None, help="Directory for LLM reports")
    parser.add_argument("--assets", action="store_true", help="Save extracted images as referenced assets")
    parser.add_argument("--run-llm", action="store_true", help="Generate LLM reports from Markdown")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama Cloud model name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_documents(
        input_dir=args.input_dir,
        markdown_dir=args.markdown_dir,
        reports_dir=args.reports_dir,
        assets=args.assets,
        run_llm=args.run_llm,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
