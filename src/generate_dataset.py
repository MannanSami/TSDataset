import asyncio
import json
import os
import random

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from bug_injector import BugInjector
from config import (
    DEFAULT_DATASET_PATH,
    MAX_CONCURRENT_SESSIONS,
    OPENCODE_PORT,
    logger,
)
from conversation_generator import ConversationGenerator
from opencode_client import OpenCodeServer
from validators import (
    compute_quality_score,
    validate_bug_fixing_sample,
    validate_code_gen_sample,
)

load_dotenv()

# Task categories for code generation (model will generate specific tasks from these)
CODE_GENERATION_TASK_KINDS = [
    "data structures and algorithms",
    "utility functions and helpers",
    "design patterns implementation",
    "async/promise handling",
    "state management",
    "type-safe API clients",
    "event handling systems",
    "form validation and processing",
    "data transformation and parsing",
    "generic collections and containers",
    "builder and factory patterns",
    "reactive programming patterns",
]


async def process_single_sample(
    idx: int,
    n_samples: int,
    dataset,
    bug_injector: BugInjector,
    conversation_gen: ConversationGenerator,
    is_code_gen: bool,
    is_multi_turn: bool,
    is_multi_bug: bool,
    persona: str,
    max_retries: int = 3,
) -> dict | None:
    """Process a single sample and return the data entry, or None on failure.

    Retries up to max_retries times on transient failures (e.g. LLM call errors).
    """
    tag = f"S{idx}"

    for attempt in range(1, max_retries + 1):
        try:
            result = await _process_single_sample_attempt(
                idx, n_samples, dataset, bug_injector, conversation_gen,
                is_code_gen, is_multi_turn, is_multi_bug, persona, tag, attempt,
            )
            if result is not None:
                return result
        except Exception as e:
            logger.error(f"[{tag}] Attempt {attempt}/{max_retries} raised: {e}")
        if attempt < max_retries:
            wait = 2 ** attempt
            logger.warning(f"[{tag}] Attempt {attempt}/{max_retries} failed, retrying in {wait}s...")
            await asyncio.sleep(wait)

    logger.error(f"[{tag}] All {max_retries} attempts failed, skipping sample")
    return None


async def _process_single_sample_attempt(
    idx, n_samples, dataset, bug_injector, conversation_gen,
    is_code_gen, is_multi_turn, is_multi_bug, persona, tag, attempt,
) -> dict | None:
    """Single attempt at processing a sample. Returns None on failure."""

    logger.info(f"[{tag}] Processing (index {idx}, attempt {attempt})")

    if is_code_gen:
        # CODE GENERATION SCENARIO
        logger.info(
            f"[{tag}] Type: Code Generation ({'Multi' if is_multi_turn else 'Single'}-turn)"
        )
        logger.info(f"[{tag}] Persona: {persona}")

        task_kind = random.choice(CODE_GENERATION_TASK_KINDS)

        logger.info(f"[{tag}] Step 1: Generating TypeScript code ({task_kind})...")

        try:
            code_data = await bug_injector.generate_typescript_code(task_kind)

            if "Failed" in code_data["code"]:
                logger.warning(f"[{tag}] Code generation failed, skipping sample...")
                return None

            logger.success(f"[{tag}] Code generated ({len(code_data['code'])} chars)")
        except Exception as e:
            logger.error(f"[{tag}] Error generating code: {e}")
            return None

        logger.info(f"[{tag}] Step 2: Generating conversation...")
        try:
            num_turns = random.randint(2, 4) if is_multi_turn else 2
            conversation = (
                await conversation_gen.generate_code_generation_conversation(
                    generated_code_data=code_data,
                    persona=persona,
                    num_turns=num_turns,
                )
            )
            logger.success(f"[{tag}] Generated {len(conversation)} turns")
        except Exception as e:
            logger.error(f"[{tag}] Error generating conversation: {e}")
            return None

        actual_turns = len(conversation) // 2 if len(conversation) % 2 == 0 else (len(conversation) + 1) // 2
        is_multi_turn = actual_turns > 1

        data_entry = {
            "conversation": conversation,
            "metadata": {
                "sample_index": idx,
                "conversation_type": "code_generation",
                "is_multi_turn": is_multi_turn,
                "persona": persona,
                "task": code_data["task"],
                "task_kind": code_data.get("task_kind", task_kind),
                "generated_code": code_data["code"],
                "features": code_data.get("features", []),
                "num_turns": actual_turns,
            },
        }

    else:
        # BUG FIXING SCENARIO
        example = dataset[idx]
        original_code = example["code"]

        logger.info(
            f"[{tag}] Type: Bug Fixing ({'Multi' if is_multi_turn else 'Single'}-turn, {'Multiple' if is_multi_bug else 'Single'} bug)"
        )
        logger.info(f"[{tag}] Persona: {persona}, Function: {example['func_name']}")

        # Step 1: Inject bug(s)
        if is_multi_bug:
            num_bugs = random.randint(2, 3)
            logger.info(f"[{tag}] Step 1: Injecting {num_bugs} bugs...")
            try:
                bug_data = await bug_injector.inject_multiple_bugs(
                    original_code, num_bugs=num_bugs
                )

                if (
                    bug_data["buggy_code"] == original_code
                    or bug_data.get("num_bugs", 0) == 0
                ):
                    logger.warning(f"[{tag}] Bug injection failed, skipping sample...")
                    return None

                logger.success(
                    f"[{tag}] {bug_data.get('num_bugs', num_bugs)} bugs injected"
                )
            except Exception as e:
                logger.error(f"[{tag}] Error injecting bugs: {e}")
                return None
        else:
            logger.info(f"[{tag}] Step 1: Injecting single bug...")
            try:
                bug_data = await bug_injector.inject_bug(original_code)

                if (
                    bug_data["buggy_code"] == original_code
                    and "Failed" in bug_data["bug_description"]
                ):
                    logger.warning(f"[{tag}] Bug injection failed, skipping sample...")
                    return None

                logger.success(f"[{tag}] Bug injected: {bug_data['bug_category']}")
            except Exception as e:
                logger.error(f"[{tag}] Error injecting bug: {e}")
                return None

        logger.info(f"[{tag}] Step 2: Generating debugging conversation...")
        try:
            if is_multi_turn:
                num_turns = random.randint(2, 6)
                conversation = await conversation_gen.generate_multi_turn_synthetic(
                    bug_data=bug_data, persona=persona, num_turns=num_turns
                )
            else:
                conversation = await conversation_gen.generate_single_turn_synthetic(
                    bug_data=bug_data, persona=persona
                )
            logger.success(f"[{tag}] Generated {len(conversation)} turns")
        except Exception as e:
            logger.error(f"[{tag}] Error generating conversation: {e}")
            return None

        actual_turns = len(conversation) // 2 if len(conversation) % 2 == 0 else (len(conversation) + 1) // 2
        is_multi_turn = actual_turns > 1

        if is_multi_bug:
            metadata = {
                "sample_index": idx,
                "conversation_type": "bug_fixing",
                "is_multi_turn": is_multi_turn,
                "is_multi_bug": True,
                "persona": persona,
                "original_code": original_code,
                "buggy_code": bug_data["buggy_code"],
                "bugs": bug_data.get("bugs", []),
                "num_bugs": bug_data.get("num_bugs", 0),
                "all_bug_categories": bug_data.get("all_bug_categories", []),
                "expected_errors": bug_data.get("expected_errors", ""),
                "source_repo": example["repo"],
                "source_path": example["path"],
                "func_name": example["func_name"],
                "docstring": example.get("docstring", ""),
                "num_turns": actual_turns,
            }
        else:
            metadata = {
                "sample_index": idx,
                "conversation_type": "bug_fixing",
                "is_multi_turn": is_multi_turn,
                "is_multi_bug": False,
                "persona": persona,
                "original_code": original_code,
                "buggy_code": bug_data["buggy_code"],
                "bug_category": bug_data["bug_category"],
                "bug_description": bug_data["bug_description"],
                "expected_error": bug_data["expected_error"],
                "source_repo": example["repo"],
                "source_path": example["path"],
                "func_name": example["func_name"],
                "docstring": example.get("docstring", ""),
                "num_turns": actual_turns,
            }

        data_entry = {"conversation": conversation, "metadata": metadata}

    # --- Validate and score the sample ---
    conv_type = data_entry["metadata"].get("conversation_type", "")
    if conv_type == "bug_fixing":
        is_valid, issues = validate_bug_fixing_sample(data_entry)
    else:
        is_valid, issues = validate_code_gen_sample(data_entry)

    quality_score = compute_quality_score(data_entry)
    data_entry["metadata"]["quality_score"] = round(quality_score, 2)
    data_entry["metadata"]["validation_passed"] = is_valid
    data_entry["metadata"]["validation_issues"] = issues

    if not is_valid:
        logger.warning(f"[{tag}] Validation issues: {issues}")

    return data_entry


async def generate_samples(
    n_samples: int = 5,
    dataset_path: str = DEFAULT_DATASET_PATH,
    output_dir: str = "../output",
    resume: bool = True,
    code_gen_ratio: float = 0.2,
    multi_bug_ratio: float = 0.4,
    multi_turn_ratio: float = 0.6,
    save_parquet: bool = True,
):
    """
    Generate synthetic TypeScript dataset with multiple conversation types.

    Uses async concurrency with a semaphore to run multiple samples in parallel.
    """
    logger.info(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path, split="train")

    logger.info(f"Dataset loaded with {len(dataset)} samples")
    logger.info("Configuration:")
    logger.info(f"  Code generation ratio: {code_gen_ratio * 100}%")
    logger.info(f"  Multi-bug ratio: {multi_bug_ratio * 100}%")
    logger.info(f"  Multi-turn ratio: {multi_turn_ratio * 100}%")
    logger.info(f"  Concurrent sessions: {MAX_CONCURRENT_SESSIONS}")

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "dataset_samples.json")

    generated_data = []
    completed_indices = set()

    if resume and os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                generated_data = json.load(f)
            completed_indices = {
                entry["metadata"]["sample_index"]
                for entry in generated_data
                if "sample_index" in entry.get("metadata", {})
            }
            logger.info(f"Resuming: found {len(generated_data)} existing samples")
        except Exception as e:
            logger.warning(f"Could not load existing samples: {e}")
            generated_data = []
            completed_indices = set()

    logger.info(
        f"Generating {n_samples} total samples ({len(generated_data)} already done, {n_samples - len(generated_data)} to go)..."
    )

    random.seed()
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)

    # Collect all available indices (not just n_samples worth) so workers
    # can backfill when samples fail.
    available_indices = [idx for idx in all_indices if idx not in completed_indices]

    def save_current_state():
        """Save the current generated_data to JSON file (atomic write)."""
        tmp_path = json_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(generated_data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, json_path)
            return True
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            return False

    # Build a queue of (idx, config) for workers to pull from.
    # Pre-decide types per index so random state is deterministic.
    work_queue: asyncio.Queue = asyncio.Queue()
    for idx in available_indices:
        is_code_gen = random.random() < code_gen_ratio
        is_multi_turn = random.random() < multi_turn_ratio
        is_multi_bug = (not is_code_gen) and (random.random() < multi_bug_ratio)
        persona = random.choice(["beginner", "intermediate", "advanced"])
        work_queue.put_nowait((idx, is_code_gen, is_multi_turn, is_multi_bug, persona))

    remaining = n_samples - len(generated_data)

    async with OpenCodeServer(port=OPENCODE_PORT) as server:
        attach_url = f"http://localhost:{OPENCODE_PORT}"
        bug_injector = BugInjector(attach_url=attach_url)
        conversation_gen = ConversationGenerator(attach_url=attach_url)

        save_lock = asyncio.Lock()
        progress = tqdm(
            total=n_samples,
            initial=len(generated_data),
            desc="Generating samples",
        )

        async def worker():
            while True:
                # Check if we already have enough
                if len(generated_data) >= n_samples:
                    break
                try:
                    idx, is_code_gen, is_multi_turn, is_multi_bug, persona = (
                        work_queue.get_nowait()
                    )
                except asyncio.QueueEmpty:
                    break

                result = await process_single_sample(
                    idx=idx,
                    n_samples=n_samples,
                    dataset=dataset,
                    bug_injector=bug_injector,
                    conversation_gen=conversation_gen,
                    is_code_gen=is_code_gen,
                    is_multi_turn=is_multi_turn,
                    is_multi_bug=is_multi_bug,
                    persona=persona,
                )

                if result is not None:
                    async with save_lock:
                        if len(generated_data) >= n_samples:
                            break
                        generated_data.append(result)
                        await asyncio.to_thread(save_current_state)
                        progress.update(1)
                        logger.success(
                            f"Sample {len(generated_data)}/{n_samples} completed "
                            f"(quality: {result['metadata'].get('quality_score', 0):.2f})"
                        )

        workers = [
            asyncio.create_task(worker())
            for _ in range(MAX_CONCURRENT_SESSIONS)
        ]

        try:
            await asyncio.gather(*workers)
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.warning("Interrupted! Cancelling remaining workers and saving...")
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            save_current_state()
            progress.close()
            raise KeyboardInterrupt

        progress.close()

    logger.info("=" * 60)
    logger.info(f"Successfully generated {len(generated_data)} samples")
    logger.info(f"Saved JSON to: {json_path}")

    # Statistics
    if len(generated_data) > 0:
        code_gen_count = sum(
            1
            for entry in generated_data
            if entry["metadata"].get("conversation_type") == "code_generation"
        )
        bug_fix_count = sum(
            1
            for entry in generated_data
            if entry["metadata"].get("conversation_type") == "bug_fixing"
        )
        multi_turn_count = sum(
            1
            for entry in generated_data
            if entry["metadata"].get("is_multi_turn", False)
        )
        multi_bug_count = sum(
            1
            for entry in generated_data
            if entry["metadata"].get("is_multi_bug", False)
        )

        logger.info("Dataset Statistics:")
        logger.info(f"  Code generation: {code_gen_count}")
        logger.info(f"  Bug fixing: {bug_fix_count}")
        logger.info(f"    - Multi-bug: {multi_bug_count}")
        logger.info(f"    - Single bug: {bug_fix_count - multi_bug_count}")
        logger.info(f"  Multi-turn: {multi_turn_count}")
        logger.info(f"  Single-turn: {len(generated_data) - multi_turn_count}")

    # Return DataFrame for optional parquet processing
    df = None
    if len(generated_data) > 0:
        try:
            df_data = []
            for entry in generated_data:
                conversation = entry["conversation"]
                meta = entry["metadata"]
                row = {
                    "conversation": json.dumps(conversation),
                    "conversation_type": meta.get("conversation_type", "unknown"),
                    "is_multi_turn": meta.get("is_multi_turn", False),
                    "persona": meta.get("persona", "unknown"),
                    "num_turns": meta.get("num_turns", 0),
                    "num_messages": len(conversation),
                    "total_chars": sum(len(m.get("content", "")) for m in conversation),
                    "quality_score": meta.get("quality_score", 0.0),
                    "validation_passed": meta.get("validation_passed", False),
                }

                if meta.get("conversation_type") == "bug_fixing":
                    row.update(
                        {
                            "original_code": meta.get("original_code", ""),
                            "buggy_code": meta.get("buggy_code", ""),
                            "is_multi_bug": meta.get("is_multi_bug", False),
                            "bug_category": meta.get("bug_category", ""),
                            "all_bug_categories": json.dumps(
                                meta.get("all_bug_categories", [])
                            ),
                            "expected_error": meta.get(
                                "expected_error", meta.get("expected_errors", "")
                            ),
                            "source_repo": meta.get("source_repo", ""),
                            "func_name": meta.get("func_name", ""),
                        }
                    )
                else:
                    row.update(
                        {
                            "task": meta.get("task", ""),
                            "task_kind": meta.get("task_kind", ""),
                            "generated_code": meta.get("generated_code", ""),
                            "is_multi_bug": False,
                            "bug_category": "",
                            "all_bug_categories": "[]",
                            "expected_error": "",
                            "source_repo": "",
                            "func_name": "",
                        }
                    )

                df_data.append(row)

            df = pd.DataFrame(df_data)

            # Save parquet
            if save_parquet:
                parquet_path = os.path.join(output_dir, "dataset.parquet")
                df.to_parquet(parquet_path, index=False)
                logger.info(f"Saved Parquet to: {parquet_path}")
        except Exception as e:
            logger.warning(f"Could not save Parquet: {e}")
            df = None

    logger.success("Dataset generation complete!")
    logger.info(f"  Total samples: {len(generated_data)}")
    logger.info(f"  Output directory: {output_dir}")

    return generated_data, df


def create_train_val_split(
    df: pd.DataFrame, output_dir: str = "../output", val_ratio: float = 0.1
):
    """
    Split dataset into train and validation sets with stratification.

    Stratifies on: conversation_type + persona + is_multi_turn.
    Falls back to non-stratified split if any stratum is too small.
    """
    from sklearn.model_selection import train_test_split

    stratify_key = (
        df["conversation_type"].astype(str)
        + "_"
        + df["persona"].astype(str)
        + "_"
        + df["is_multi_turn"].astype(str)
    )

    try:
        train_df, val_df = train_test_split(
            df, test_size=val_ratio, random_state=42, stratify=stratify_key
        )
        logger.info("(stratified split succeeded)")
    except ValueError as e:
        logger.warning(f"Stratified split failed ({e}), falling back to random split")
        train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=42)

    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "validation.parquet")

    train_df.to_parquet(train_path, index=False)  # type: ignore
    val_df.to_parquet(val_path, index=False)  # type: ignore

    logger.info("Train/Val split complete:")
    logger.info(f"  Train: {len(train_df)} samples -> {train_path}")
    logger.info(f"  Val: {len(val_df)} samples -> {val_path}")

    return train_df, val_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic TypeScript dataset with bug fixing and code generation scenarios",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-n", "--samples", type=int, default=10, help="Number of samples to generate"
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="../output",
        help="Output directory for generated files",
    )

    parser.add_argument(
        "-d",
        "--dataset-path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Path to the source TypeScript dataset",
    )

    parser.add_argument(
        "--code-gen-ratio",
        type=float,
        default=0.2,
        help="Ratio of code generation samples (0.0-1.0)",
    )

    parser.add_argument(
        "--multi-bug-ratio",
        type=float,
        default=0.4,
        help="Ratio of multi-bug samples in bug fixing scenarios (0.0-1.0)",
    )

    parser.add_argument(
        "--multi-turn-ratio",
        type=float,
        default=0.5,
        help="Ratio of multi-turn conversations (0.0-1.0)",
    )

    parser.add_argument(
        "--no-parquet",
        action="store_true",
        help="Skip parquet file generation (only save JSON)",
    )

    parser.add_argument(
        "--no-resume", action="store_true", help="Start fresh, ignore existing samples"
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio for train/val split",
    )

    parser.add_argument(
        "--show-sample",
        action="store_true",
        help="Show a sample conversation at the end",
    )

    args = parser.parse_args()

    logger.info(f"Starting dataset generation for {args.samples} samples...")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Parquet generation: {'disabled' if args.no_parquet else 'enabled'}")
    logger.info(f"Resume mode: {'disabled' if args.no_resume else 'enabled'}")
    logger.info(f"Concurrent sessions: {MAX_CONCURRENT_SESSIONS}")
    logger.info("Progress is saved after each sample. Press Ctrl+C to interrupt safely.")

    try:
        generated_data, df = asyncio.run(
            generate_samples(
                n_samples=args.samples,
                dataset_path=args.dataset_path,
                output_dir=args.output_dir,
                resume=not args.no_resume,
                code_gen_ratio=args.code_gen_ratio,
                multi_bug_ratio=args.multi_bug_ratio,
                multi_turn_ratio=args.multi_turn_ratio,
                save_parquet=not args.no_parquet,
            )
        )

        # Create train/val split if we have data and a DataFrame
        if df is not None and len(df) > 1:
            train_df, val_df = create_train_val_split(
                df, output_dir=args.output_dir, val_ratio=args.val_ratio
            )

        # Show sample conversation if requested
        if args.show_sample and generated_data:
            logger.info("=" * 60)
            logger.info("Sample conversation example:")
            logger.info("=" * 60)
            logger.info(json.dumps(generated_data[0], indent=2)[:2000])

    except KeyboardInterrupt:
        logger.warning("Generation interrupted by user")
        logger.info("Your progress has been saved. Run the script again to resume.")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        logger.info("Your progress has been saved in dataset_samples.json")
