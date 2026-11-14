import config
from models.Page import Page
from services.PostgresConnector import PostgresConnector


def get_pages_to_embed(postgres: PostgresConnector) -> list[Page]:
    query = """
            SELECT p.url, \
                   p.s3_key
            FROM page_inferred_labels pil
                     JOIN page_classifications pc
                          ON pc.url = pil.url
                              AND pc.type = 'product'
                     JOIN pages p
                          ON p.url = pil.url
                              AND p.is_active = TRUE
                              AND p.s3_key NOT IN ('NOT_CRAWLED', 'REDIRECT')
            WHERE pil.product_price IS NOT NULL
              AND pil.product_title IS NOT NULL
              AND NOT EXISTS (SELECT 1 \
                              FROM products_embeddings_finetuned_768_2 pe \
                              WHERE pe.url = pil.url)
            ORDER BY pil.last_inferred_at
                LIMIT %s; \
            """

    result = postgres.run_query(query, (config.get_page_retrieval_count(),))

    return [Page(row[0], row[1]) for row in result]


def upsert_embedding(postgres_connector: PostgresConnector, url: str, embedding: list[float]) -> None:
    query = """
            INSERT INTO products_embeddings_finetuned_768_2 (url, embedding)
            VALUES (%s, %s) ON CONFLICT (url) DO
            UPDATE
                SET embedding = EXCLUDED.embedding,
                last_updated_at = NOW(); \
            """

    postgres_connector.execute(query, (url, embedding))
