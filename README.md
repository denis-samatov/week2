# Week 2 OCR/LLM Pipeline

Проект содержит конвейер для обработки PDF и изображений:

1. OCR и структурирование документов через **Docling** + **RapidOCR**.
2. Предобработка изображений через OpenCV.
3. Экспорт результата в Markdown и JSON.
4. Генерация краткого аналитического отчёта на русском языке через **Ollama Cloud** (`gemma4:31b-cloud`) с интерфейсом LlamaIndex/OpenAI-like API.

Исходный notebook сохранён в `notebooks/week2.ipynb`.

## Структура

```text
.
├── notebooks/
│   └── week2.ipynb
├── src/
│   └── ocr_llm_pipeline.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Переменные окружения

Для генерации отчётов через Ollama Cloud задайте API-ключ:

```bash
export OLLAMA_API_KEY="your_api_key"
```

В Google Colab можно использовать секрет `ocr_olama`, как в исходном notebook.

## Запуск

Пример обработки документов из папки `ocr_samples`:

```bash
python src/ocr_llm_pipeline.py \
  --input-dir ./ocr_samples \
  --markdown-dir ./md_results \
  --reports-dir ./final_results \
  --assets \
  --run-llm
```

Поддерживаемые форматы: `.pdf`, `.jpg`, `.jpeg`, `.png`.

## Что создаётся

Для каждого входного документа:

- Markdown-файл с распознанной структурой;
- JSON-файл с объектным представлением Docling;
- папка с изображениями/артефактами, если включён режим `--assets`;
- текстовый аналитический отчёт, если включён `--run-llm`.

## Примечания

- Для Colab-версии используйте исходный notebook `notebooks/week2.ipynb`.
- Для локального запуска используйте `src/ocr_llm_pipeline.py`.
- В репозиторий не следует добавлять реальные документы, результаты OCR, API-ключи и временные файлы.
