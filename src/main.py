from concurrent.futures import ThreadPoolExecutor, as_completed

from bs4 import BeautifulSoup

from helpers.page_helpers import get_pages_to_embed, upsert_embedding
from services.EmbeddingModel import embedding_model
from services.KafkaService import kafka_service
from services.Logging import LoggingService
from services.PostgresConnector import postgres_connector
from services.S3Bucket import minimized_pages_bucket

logger = LoggingService.get_logger("main")


def process_page(page) -> None:
    try:
        logger.info("Processing page: %s", page.url)
        raw_content = minimized_pages_bucket.download(page.s3_key)
        text_content = BeautifulSoup(raw_content, "html.parser").get_text()
        embedding = embedding_model.get_embedding(text_content).tolist()
        upsert_embedding(postgres_connector, page.url, embedding)
        kafka_service.send_success_message({"url": page.url})
    except Exception as e:
        logger.exception("Failed to process page: %s. Error: %s", page.url, str(e))
        kafka_service.send_failed_message({"url": page.url, "error": str(e)})


def main():
    unembedded_pages = get_pages_to_embed(postgres_connector)
    total = len(unembedded_pages)

    if total == 0:
        logger.info("No pages to process.")
        return

    max_workers = min(8, total)
    logger.info("Processing %d pages with %d workers", total, max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_page, page): page for page in unembedded_pages}
        for completed, future in enumerate(as_completed(futures), start=1):
            page = futures[future]
            try:
                future.result()
                logger.info("Progress: %d/%d pages completed (last: %s)", completed, total, page.url)
            except Exception:
                logger.exception("Failed to process page: %s", page.url)
                logger.info("Progress: %d/%d pages completed (last failed: %s)", completed, total, page.url)


if __name__ == "__main__":
    main()
