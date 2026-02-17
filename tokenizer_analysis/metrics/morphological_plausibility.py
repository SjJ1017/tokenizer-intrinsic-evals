"""
Morphological plausibility metrics for tokenizer analysis.

This metric evaluates tokenizers by generating tokenized segmentations for a
language-specific UniMorph/UniSeg gold file and comparing them via
`evaluate_segmentations` from morph_tok_eval.align.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
import json
import logging
import os
import tempfile

import numpy as np

from .base import BaseMetrics
from ..core.input_types import TokenizedData
from ..core.input_providers import InputProvider, RawTokenizationProvider
from ..loaders.constants import FLORES_to_ISO639_2

logger = logging.getLogger(__name__)

try:
    from morph_tok_eval.align import evaluate_segmentations
    MORPH_PLAUSIBILITY_AVAILABLE = True
except ImportError:
    logger.warning(
        "Morphological plausibility library not available. Metrics will be disabled."
    )
    MORPH_PLAUSIBILITY_AVAILABLE = False


class MorphologicalPlausibilityMetrics(BaseMetrics):
    """Morphological plausibility metrics for tokenizer evaluation."""

    def __init__(
        self,
        input_provider: InputProvider,
        data_dir: str = "dataset/morph_plausibility",
        language_subset: Optional[List[str]] = None,
        thresholds: Optional[List[float]] = None,
        iterations: int = 100,
        model: str = "IBM1",
    ):
        super().__init__(input_provider)

        if not MORPH_PLAUSIBILITY_AVAILABLE:
            raise ImportError(
                "Morphological plausibility requires morph_tok_eval.align"
            )

        if not isinstance(input_provider, RawTokenizationProvider):
            raise ValueError(
                "Morphological plausibility requires RawTokenizationProvider"
            )

        self.data_dir = data_dir
        self.language_subset = language_subset
        self.thresholds = thresholds or [0.1]
        self.iterations = iterations
        self.model = model

        available = input_provider.get_languages()
        if language_subset is None:
            self.target_languages = available
        else:
            self.target_languages = [lang for lang in language_subset if lang in available]

        missing_langs = [lang for lang in (language_subset or []) if lang not in available]
        if missing_langs:
            logger.warning(
                "Requested languages not available in input provider: %s",
                missing_langs,
            )

        logger.info(
            "Morphological plausibility metrics initialized for %d languages: %s",
            len(self.target_languages),
            self.target_languages,
        )

    def _group_metrics_by_threshold(self, test_metrics: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        grouped: Dict[str, Dict[str, float]] = {}
        for key, value in test_metrics.items():
            if not key.startswith("test-"):
                continue
            parts = key.split("-")
            if len(parts) < 4:
                continue
            model = parts[-1]
            threshold = parts[-2]
            metric_key = "-".join(parts[1:-2] + [model])
            grouped.setdefault(threshold, {})[metric_key] = value
        return grouped

    def _suppress_logging(self):
        prev_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        return prev_disable

    def _unicode_safe_tokenize(self, tokenizer: Any, text: str) -> List[str]:
        try:
            if hasattr(tokenizer, "encode"):
                encoding = tokenizer.encode(text)
                if hasattr(encoding, "offsets"):
                    offset_mapping = list(encoding.offsets)
                elif hasattr(encoding, "offset_mapping"):
                    offset_mapping = list(encoding.offset_mapping)
                else:
                    offset_mapping = []
            else:
                offset_mapping = []

            if not offset_mapping and hasattr(tokenizer, "__call__"):
                encoding = tokenizer(
                    text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                offset_mapping = list(encoding.get("offset_mapping", []))

            for i in range(len(offset_mapping) - 1):
                this_start, this_end = offset_mapping[i]
                next_start, next_end = offset_mapping[i + 1]
                if this_end != next_start:
                    offset_mapping[i + 1] = (this_end, next_end)

            if offset_mapping:
                return [text[start:end] for start, end in offset_mapping]
        except Exception:
            pass

        return list(text)

    def _tokenize_unimorph(self, input_file: str, output_file: str, tokenizer: Any) -> int:
        count = 0
        with open(input_file, encoding="UTF-8") as f_in, open(
            output_file, "w", encoding="UTF-8"
        ) as f_out:
            for line in f_in:
                word, tag, _segments = line.split("\t")
                tokenized = self._unicode_safe_tokenize(tokenizer, word)
                if "".join(tokenized) != word:
                    tokenized = list(word)
                segments = "|".join(tokenized)
                print(word, tag, segments, sep="\t", file=f_out)
                count += 1
        return count

    def _resolve_gold_file(self, language: str) -> Optional[str]:
        filename = FLORES_to_ISO639_2.get(language)
        if not filename:
            logger.warning("No gold file mapping for language: %s", language)
            return None
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            path = os.path.join(self.data_dir, f"{filename}.tsv")
            if not os.path.exists(path):
                logger.warning("Gold file not found for %s at %s", language, path)
                return None
        return path

    def compute(
        self, tokenized_data: Optional[Dict[str, List[TokenizedData]]] = None
    ) -> Dict[str, Any]:
        if not MORPH_PLAUSIBILITY_AVAILABLE:
            return {
                "morphological_plausibility": {
                    "error": "Morphological plausibility library not available"
                }
            }

        logger.info("Computing morphological plausibility metrics...")

        results = {
            "per_tokenizer": {},
            "metadata": {
                "thresholds": self.thresholds,
                "iterations": self.iterations,
                "model": self.model,
                "data_dir": self.data_dir,
                "target_languages": self.target_languages,
            },
        }

        for tok_name in self.tokenizer_names:
            logger.info("Evaluating morphological plausibility for tokenizer: %s", tok_name)
            tok_results = {
                "per_language": {},
                "summary": {
                    "languages_evaluated": 0,
                    "total_samples": 0,
                    "thresholds": {}
                },
            }

            try:
                tokenizer_wrapper = self.input_provider.get_tokenizer(tok_name)
                underlying = tokenizer_wrapper.get_underlying_tokenizer()
                if underlying is None:
                    tok_results["error"] = "No underlying tokenizer available"
                    results["per_tokenizer"][tok_name] = tok_results
                    continue

                total_samples = 0

                for lang in self.target_languages:
                    gold_file = self._resolve_gold_file(lang)
                    if not gold_file:
                        continue
                    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tsv")
                    os.close(tmp_fd)
                    try:
                        num_samples = self._tokenize_unimorph(
                            gold_file, tmp_path, underlying
                        )
                        total_samples += num_samples

                        prev_disable = self._suppress_logging()
                        try:
                            res, _model = evaluate_segmentations(
                                gold_file=gold_file,
                                test_file=tmp_path,
                                thresholds=self.thresholds,
                                iterations=self.iterations,
                                model=self.model,
                                skip_gold_train=True,
                            )
                        finally:
                            logging.disable(prev_disable)

                        test_only = {k: v for k, v in res.items() if k.startswith("test-")}
                        grouped = self._group_metrics_by_threshold(test_only)
                        tok_results["per_language"][lang] = {
                            "thresholds": grouped,
                            "num_samples": num_samples,
                        }
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

                tok_results["summary"]["languages_evaluated"] = len(
                    tok_results["per_language"]
                )
                tok_results["summary"]["total_samples"] = total_samples

                # Aggregate per-threshold metrics across languages
                per_threshold_values: Dict[str, Dict[str, List[float]]] = {}
                for lang_data in tok_results["per_language"].values():
                    for threshold, metrics in lang_data.get("thresholds", {}).items():
                        for metric_key, value in metrics.items():
                            per_threshold_values.setdefault(threshold, {}).setdefault(metric_key, []).append(value)

                for threshold, metrics in per_threshold_values.items():
                    tok_results["summary"]["thresholds"][threshold] = {}
                    for metric_key, values in metrics.items():
                        tok_results["summary"]["thresholds"][threshold][f"avg_{metric_key}"] = float(np.mean(values))
                        tok_results["summary"]["thresholds"][threshold][f"avg_{metric_key}_std"] = float(np.std(values))

                results["per_tokenizer"][tok_name] = tok_results
                logger.info(
                    "Morphological plausibility completed for %s: %d languages, %d samples",
                    tok_name,
                    tok_results["summary"]["languages_evaluated"],
                    tok_results["summary"]["total_samples"],
                )

            except Exception as exc:
                logger.error(
                    "Error evaluating morphological plausibility for %s: %s",
                    tok_name,
                    exc,
                )
                results["per_tokenizer"][tok_name] = {
                    "error": str(exc),
                    "per_language": {},
                    "summary": {
                        "languages_evaluated": 0,
                        "total_samples": 0,
                    },
                }

        return {"morphological_plausibility": results}

    def print_results(self, results: Dict[str, Any], per_lang: bool = False) -> None:
        if "morphological_plausibility" not in results:
            return

        data = results["morphological_plausibility"]
        if "error" in data:
            print("\nMORPHOLOGICAL PLAUSIBILITY")
            print("-" * 40)
            print(f"Error: {data['error']}")
            return

        print("\n" + "=" * 60)
        print("MORPHOLOGICAL PLAUSIBILITY RESULTS")
        print("=" * 60)

        for tok_name in self.tokenizer_names:
            tok_data = data.get("per_tokenizer", {}).get(tok_name)
            if not tok_data:
                continue
            if "error" in tok_data:
                print(f"{tok_name:20}: Error - {tok_data['error']}")
                continue

            summary = tok_data.get("summary", {})
            print(f"\n{tok_name}:")
            print(f"  Languages: {summary.get('languages_evaluated', 0)}")
            print(f"  Samples: {summary.get('total_samples', 0)}")

        if per_lang:
            print("\nPER-LANGUAGE RESULTS")
            print("-" * 60)
            for tok_name in self.tokenizer_names:
                tok_data = data.get("per_tokenizer", {}).get(tok_name)
                if not tok_data or not tok_data.get("per_language"):
                    continue
                print(f"\n{tok_name}:")
                for lang, metrics in tok_data["per_language"].items():
                    print(f"  {lang}: {json.dumps(metrics, indent=2)}")

        print("\n" + "=" * 60)
