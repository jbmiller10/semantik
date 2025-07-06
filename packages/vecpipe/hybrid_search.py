#!/usr/bin/env python3
"""
Hybrid search implementation for Qdrant
Combines vector similarity search with text-based filtering
"""

import logging
import re
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchText

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """Hybrid search engine combining vector and text search"""

    def __init__(self, host: str, port: int, collection_name: str = "work_docs"):
        self.client = QdrantClient(url=f"http://{host}:{port}")
        self.collection_name = collection_name

    def extract_keywords(self, query: str) -> list[str]:
        """Extract meaningful keywords from query"""
        # Remove common stop words
        stop_words = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "up",
            "down",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
        }

        # Split query into words and filter
        words = re.findall(r"\w+", query.lower())
        return [word for word in words if word not in stop_words and len(word) > 2]

    def build_text_filter(self, keywords: list[str], mode: str = "any") -> Filter | None:
        """Build Qdrant filter for text matching"""
        if not keywords:
            return None

        conditions = []
        for keyword in keywords:
            # Create condition for text matching in content field
            condition = FieldCondition(key="content", match=MatchText(text=keyword))
            conditions.append(condition)

        if mode == "all":
            # All keywords must match
            return Filter(must=conditions)
        # Any keyword can match
        return Filter(should=conditions)

    def hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        limit: int = 10,
        keyword_mode: str = "any",
        score_threshold: float | None = None,
        hybrid_mode: str = "filter",  # "filter" or "rerank"
    ) -> list[dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and text matching

        Args:
            query_vector: Embedding vector for the query
            query_text: Original query text
            limit: Number of results to return
            keyword_mode: "any" or "all" for keyword matching
            score_threshold: Minimum similarity score threshold
            hybrid_mode: "filter" uses Qdrant filters, "rerank" does post-processing

        Returns:
            List of search results with scores
        """
        try:
            if hybrid_mode == "filter":
                # Extract keywords from query
                keywords = self.extract_keywords(query_text)
                logger.info(f"Extracted keywords: {keywords}")

                # Build text filter
                text_filter = self.build_text_filter(keywords, keyword_mode)

                # Perform vector search with text filter
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=text_filter,
                    limit=limit,
                    score_threshold=score_threshold,
                )

                # Convert to standard format
                results = []
                for hit in search_result:
                    result = {
                        "id": str(hit.id),
                        "score": hit.score,
                        "payload": hit.payload,
                        "matched_keywords": keywords if text_filter else [],
                    }
                    results.append(result)

                return results

            # rerank mode
            # First get more candidates using vector search
            candidate_limit = limit * 3  # Get 3x candidates for reranking

            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=candidate_limit,
                score_threshold=score_threshold,
            )

            # Extract keywords
            keywords = self.extract_keywords(query_text)

            # Score and rerank based on keyword matches
            scored_results = []
            for hit in search_result:
                content = hit.payload.get("content", "").lower()

                # Count keyword matches
                keyword_score = 0
                matched_keywords = []
                for keyword in keywords:
                    if keyword in content:
                        keyword_score += 1
                        matched_keywords.append(keyword)

                # Combine vector score with keyword score
                # Normalize keyword score (0-1 range)
                normalized_keyword_score = keyword_score / len(keywords) if keywords else 0

                # Weighted combination (70% vector, 30% keywords)
                combined_score = 0.7 * hit.score + 0.3 * normalized_keyword_score

                scored_results.append(
                    {
                        "id": str(hit.id),
                        "score": hit.score,
                        "keyword_score": normalized_keyword_score,
                        "combined_score": combined_score,
                        "payload": hit.payload,
                        "matched_keywords": matched_keywords,
                    }
                )

            # Sort by combined score and return top results
            scored_results.sort(key=lambda x: x["combined_score"], reverse=True)
            return scored_results[:limit]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise

    def search_by_keywords(self, keywords: list[str], limit: int = 10, mode: str = "any") -> list[dict[str, Any]]:
        """
        Search using only keywords (no vector similarity)

        Args:
            keywords: List of keywords to search for
            limit: Number of results to return
            mode: "any" or "all" for keyword matching

        Returns:
            List of search results
        """
        try:
            # Build text filter
            text_filter = self.build_text_filter(keywords, mode)

            if not text_filter:
                return []

            # Scroll through results with filter (no vector)
            results = self.client.scroll(collection_name=self.collection_name, scroll_filter=text_filter, limit=limit)

            # Convert to standard format
            output = []
            for point in results[0]:  # results is a tuple (points, next_offset)
                output.append({"id": str(point.id), "payload": point.payload, "matched_keywords": keywords})

            return output

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            raise

    def close(self):
        """Close the client connection"""
        self.client.close()
